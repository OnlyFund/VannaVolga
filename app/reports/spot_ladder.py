"""Spot-ladder risk report calculation and rendering."""

import logging
from copy import copy, deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.progress import current

import numpy as np
import pandas as pd

from app.curve_roll import next_business_day
from pricing.greek_catalog import columns as greek_columns
from pricing.greeks import (
    _greek_inputs, calculate_extra_greeks, calculate_greeks, calculate_forward_theta,
    forward_delta_factor, smile_path_delta_gamma,
)
from pricing.pricing import price_option_with_vol
from pricing.fx_forward import FxForwardPosition
from helper_functions.premium_quotes import conventional_premium_ccy
from helper_functions.cashflows import signed_premium_cashflow
from helper_functions.input_resolvers import parse_notional, resolve_strike
from smile_perturbed_market import PerturbedMarket


GREEK_NAMES = ["delta", "gamma", "vega", "vanna", "volga", "theta", "rega", "sega"]
QUOTES = ("v_atm", "rr_25", "rr_10", "bf_25", "bf_10")
logger = logging.getLogger(__name__)


def _has_dynamics(mods):
    if mods is None:
        return False
    get = mods.get if isinstance(mods, dict) else (lambda name, default: getattr(mods, name, default))
    return any(float(get(name, 0.0) or 0.0) != 0.0 for name in ("beta_atm", "beta_rr", "beta_bf"))


def _option_greeks(base_market, market, next_market, row, pricing_model, atm_type, delta_base,
                   mods, next_valuation, mtm):
    """An option's Greeks at one ladder spot, and the market they were taken on.

    Greeks use the smile the betas imply at that spot (``_smile_moved_markets``),
    except smile delta and gamma, which are derivatives along the ladder's own
    smile path from today's spot: the deltas shown at each spot then add up
    to the P&L between spots. Either step falls back to the plain scenario
    market should it fail.
    """
    greek_market, greek_next = _smile_moved_markets(
        base_market, market, next_market, row, mods, pricing_model, atm_type, delta_base)
    # A vol Greek that cannot be computed is blank on the level (as an
    # extra is), not the level's failure.
    greeks = calculate_greeks(
        greek_market, row, None, pricing_model, atm_type, delta_base, mods,
        next_market=greek_next, next_valuation=next_valuation, option_value_term=mtm,
        problems={},
    )
    if _has_dynamics(mods) and float(market.s) != float(base_market.s):
        try:
            greeks["delta"], greeks["gamma"] = smile_path_delta_gamma(
                base_market, market, row, mods, pricing_model, atm_type, delta_base)
        except Exception as exc:
            logger.warning("Ladder smile-path delta failed at spot %s: %s", market.s, exc)
    return greeks, greek_market, greek_next


def move_quotes_with_spot(base_market, market, next_market, row, mods, pricing_model,
                          atm_type, delta_base):
    """``(market, next_market)`` with the vol quotes moved by the smile dynamics
    from ``base_market``'s spot to ``market``'s; raises if the dynamics fail.

    The strike's vol on the moved quotes is the smile-path vol the ladder's
    P&L uses (``_solve_smile_vol``), so pricing on them gives the same value.
    The Portfolio's spot edit reprices this way too (``app.pricer``).
    """
    from surface_models.smile_extension import SmileExtension
    common = _greek_inputs(base_market, row, pricing_model, atm_type, delta_base)
    shifted = deepcopy(mods)
    if isinstance(shifted, dict):
        shifted["shifted_spot"] = float(market.s)
    else:
        setattr(shifted, "shifted_spot", float(market.s))
    extension = SmileExtension(
        market=base_market, K=common["k"], expiry_date=common["expiry_date"],
        valuation_date=common["valuation_date"], CallPut=common["CallPut"], mods=shifted,
        pricing_model=pricing_model, atm_conv=common["atm_type"],
        deltaBase=common["deltaBase"], reference_spot=float(base_market.s),
    )
    skews = extension.get_local_skews()
    quotes = extension.extend_wings_convexity(extension.get_new_pillar_vols(skews), skews)
    change = {name: float(quotes[name]) - float(getattr(base_market, name)) for name in QUOTES}
    if not all(np.isfinite(value) for value in change.values()):
        raise ValueError("smile dynamics gave non-finite quotes")
    moved = market.copy_with(**{name: getattr(market, name) + change[name] for name in QUOTES})
    if next_market is not None:
        next_market = next_market.copy_with(
            **{name: getattr(next_market, name) + change[name] for name in QUOTES})
    return moved, next_market


def _smile_moved_markets(base_market, market, next_market, row, mods, pricing_model,
                         atm_type, delta_base):
    """The scenario's markets with their vol quotes moved by the smile dynamics.

    A level's P&L is priced on the smile the betas imply at its spot
    (``_solve_smile_vol``); its Greeks use that same smile, so the delta and
    gamma shown at each spot follow the P&L. The next-day market (theta) moves
    by the same quote change. At the base spot, without dynamics, or if the
    dynamics fail, the quotes stay as quoted.
    """
    if not _has_dynamics(mods) or float(market.s) == float(base_market.s):
        return market, next_market
    try:
        return move_quotes_with_spot(base_market, market, next_market, row, mods,
                                     pricing_model, atm_type, delta_base)
    except Exception as exc:
        logger.warning("Ladder smile dynamics failed at spot %s; Greeks on quoted smile: %s",
                       market.s, exc)
        return market, next_market


def _prepare_specs(rows, valuation_date, resolve_expiry, market_for_expiry,
                   atm_type, delta_base):
    specs = []
    errors = []
    for index, source_row in rows:
        try:
            row = source_row.copy()
            row["valuation"] = valuation_date
            row["expiry"] = resolve_expiry(row["expiry"])
            row["notional"] = parse_notional(row["notional"])
            base_market = market_for_expiry(row["expiry"])
            row["strike"] = resolve_strike(base_market, row, atm_type, delta_base)
            specs.append((int(index), row))
        except Exception as exc:
            errors.append(f"Row {int(index) + 1}: {exc}")
    return specs, errors


def _next_valuation(specs, valuation_date):
    """Next business day on the portfolio pair's calendar (theta horizon)."""
    pair = next(
        (str(row.get("currency_pair", "")).strip().upper() for _, row in specs), ""
    )
    return next_business_day(pair, valuation_date)


def _cached_market(cache, market_for_expiry, expiry, as_of, spot):
    key = (
        pd.Timestamp(expiry).date(),
        pd.Timestamp(as_of).date() if as_of is not None else None,
        float(spot),
    )
    if key not in cache:
        cache[key] = market_for_expiry(
            expiry,
            as_of=as_of,
            allow_short_end=as_of is not None,
            spot_value=spot,
        )
    return cache[key]


def _solve_smile_vol(
    base_market, scenario_market, row, pricing_model, atm_type, delta_base, mods,
):
    """Solve scenario volatility using the same dynamics as smile delta."""
    smile = PerturbedMarket(
        base_market=base_market,
        mods=mods,
        K=float(row["strike"]),
        expiry_date=row["expiry"].isoformat(),
        valuation_date=row["valuation"].isoformat(),
        CallPut=row["call_put"],
        deltaBase=delta_base,
        atm_type=atm_type,
        pricing_model=pricing_model,
    )
    vol = smile.shift_spot(new_spot=float(scenario_market.s))
    vol = float(vol)
    if not np.isfinite(vol) or vol <= 0.0:
        raise ValueError("Non-finite or non-positive ladder volatility")
    return vol


def _empty_level(spot, extras=()):
    level = {
        "spot": spot,
        "pnl_base": 0.0,
    }
    for column in greek_columns([*GREEK_NAMES, *extras]):
        level[column] = 0.0
    return level


def _add_option_greeks(level, greeks, market, row, extras, pricing_model, atm_type,
                       delta_base, mods, next_market, next_valuation):
    """Add one option's Greeks, and the requested extras, to a ladder level.

    An extra that cannot be computed makes that level's cell blank (NaN)
    rather than failing the level.
    """
    for greek_name in GREEK_NAMES:
        for column in (f"fixed_{greek_name}", greek_name):
            value = greeks[column]
            level[column] += np.nan if value is None else value     # unavailable: blank
    if not extras:
        return
    values, _ = calculate_extra_greeks(
        market, row, extras, greeks, pricing_model, atm_type, delta_base, mods,
        next_market=next_market, next_valuation=next_valuation,
    )
    for column, value in values.items():
        level[column] += np.nan if value is None else float(value)


def _scenario_worker_count(scenario_workers, scenario_count):
    """Return a bounded thread count for independent ladder scenarios."""
    if scenario_workers is not None:
        if not isinstance(scenario_workers, int) or scenario_workers < 1:
            raise ValueError("scenario_workers must be a positive integer or None")
        return min(scenario_workers, scenario_count)
    # Smile recalibration invokes NumPy, SciPy, and Numba kernels that can
    # already use native threads. A small outer pool avoids oversubscription.
    return min(scenario_count, 2)


def _run_scenarios(scenarios, scenario_workers, calculate_scenario, stage="spot ladder"):
    """Run pure, independent scenario calculations in stable input order.

    Progress (``app.progress.current``) is reported on this thread as each
    scenario finishes, never from the workers.
    """
    if not scenarios:
        return []
    report = current()
    total = len(scenarios)
    worker_count = _scenario_worker_count(scenario_workers, total)
    if worker_count == 1:
        results = []
        for done, scenario in enumerate(scenarios, start=1):
            results.append(calculate_scenario(scenario))
            report(done, total, stage)
        return results
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(calculate_scenario, scenario) for scenario in scenarios]
        for done, _ in enumerate(as_completed(futures), start=1):
            report(done, total, stage)
        return [future.result() for future in futures]


def _add_option_pnl(level, row, mtm_term, spot):
    """Add an option's total base-currency P&L to a ladder level."""
    pair = str(row.get("currency_pair", "")).strip().upper()
    if len(pair) != 6 or spot <= 0.0:
        raise ValueError("Option currency pair and scenario spot must be valid")
    raw_premium = row.get("premium", 0.0)
    if raw_premium is None or pd.isna(raw_premium):
        raise ValueError("Premium must be entered before calculating ladder P&L")
    buy_sell = str(row.get("buy_sell", "BUY")).strip().upper()
    premium_ccy = str(row.get("premium_ccy", pair[3:])).strip().upper()
    if premium_ccy not in {pair[:3], pair[3:]}:
        raise ValueError("Premium currency must match the pair's base or term currency")

    signed_premium = signed_premium_cashflow(raw_premium, buy_sell)
    if premium_ccy == pair[:3]:
        premium_base = signed_premium
    else:
        premium_base = signed_premium / spot
    level["pnl_base"] += premium_base + mtm_term / spot


def _add_forward_pnl(level, position, market, valuation_date,
                     next_market=None, next_valuation=None):
    """Add the forward's base P&L, spot delta and rolled theta."""
    if isinstance(position, dict):
        position = FxForwardPosition(**position)
    value = position.value(market, valuation_date)
    base_amount = float(position.base_amount)
    level["pnl_base"] += value["mtm_base"]
    value_date = getattr(position, "value_date", None)
    delta = base_amount * forward_delta_factor(market, position.expiry, valuation_date,
                                               delivery=value_date)
    level["fixed_delta"] += delta
    level["delta"] += delta
    from pricing.greeks import calculate_forward_rate_risks
    for key, risk in calculate_forward_rate_risks(
            position, market, valuation_date, names=level).items():
        if key in level and level[key] is not None:
            level[key] += risk
    if next_market is not None or value_date is not None:
        # A forward has no smile: its roll goes into sticky and smile theta.
        if next_market is None:
            # A deal settling before the next business day loses its whole MTM.
            theta = -value["mtm_term"]
            if conventional_premium_ccy(position.currency_pair) != position.currency_pair[3:]:
                theta = -value["mtm_base"]
        else:
            theta = calculate_forward_theta(
                position, market, next_market, valuation_date, next_valuation
            )
        level["fixed_theta"] += theta
        level["theta"] += theta
        # A forward's roll is carry, when the ladder shows the theta parts.
        for key in ("fixed_theta_carry", "theta_carry"):
            if key in level and level[key] is not None:
                level[key] += theta


def _forward_markets(position, expiry, market_for_expiry, market_for_delivery,
                     next_valuation, ladder_spot):
    """``(market, next-day market)`` of one forward at a ladder spot."""
    value_date = (position.get("value_date") if isinstance(position, dict)
                  else getattr(position, "value_date", None))
    if value_date is None:
        return (market_for_expiry(expiry, spot_value=ladder_spot),
                market_for_expiry(expiry, as_of=next_valuation, allow_short_end=True,
                                  spot_value=ladder_spot))
    if market_for_delivery is None:
        raise ValueError("a value-dated forward needs the market at its delivery date")
    after = next_valuation is not None and pd.Timestamp(next_valuation) < pd.Timestamp(value_date)
    return (market_for_delivery(value_date, spot_value=ladder_spot),
            market_for_delivery(value_date, as_of=next_valuation, spot_value=ladder_spot)
            if after else None)


def _calculate_reference_ladder(
    rows, ladder_spots, valuation_date, market_for_expiry, resolve_expiry,
    pricing_model, atm_type, delta_base, mods, forward_positions=(),
    scenario_workers=None, extras=(), market_for_delivery=None,
):
    """Run the original scalar ladder path without prepared-row shortcuts."""
    errors = []

    # Expiry resolution, notional parsing, and strike resolution only depend
    # on each row and the base (non-ladder-bumped) spot via
    # market_for_expiry(row["expiry"]) -- they do not depend on ladder_spot.
    # Precompute them once per row instead of redundantly recomputing them
    # on every spot iteration below; this changes nothing mathematically
    # since the inputs to resolve_strike are identical on every iteration.
    specs, spec_errors = _prepare_specs(
        rows, valuation_date, resolve_expiry, market_for_expiry, atm_type, delta_base
    )
    errors.extend(spec_errors)
    next_valuation = _next_valuation(specs, valuation_date)
    base_markets = {
        index: market_for_expiry(row["expiry"])
        for index, row in specs
    }
    prepared_forwards = [
        (
            position,
            resolve_expiry(
                position["expiry"] if isinstance(position, dict) else position.expiry
            ),
        )
        for position in forward_positions or ()
    ]
    scenarios = []
    # Building each spot's markets (today's and tomorrow's) is most of a
    # ladder's time: reported as its own stage.
    report = current()
    for done, ladder_spot in enumerate(ladder_spots, start=1):
        try:
            option_markets = {
                index: market_for_expiry(row["expiry"], spot_value=ladder_spot)
                for index, row in specs
            }
            next_markets = {
                index: market_for_expiry(
                    row["expiry"],
                    as_of=next_valuation,
                    allow_short_end=True,
                    spot_value=ladder_spot,
                )
                for index, row in specs
            }
            forward_markets = [
                (position, *_forward_markets(position, expiry, market_for_expiry,
                                             market_for_delivery, next_valuation,
                                             ladder_spot))
                for position, expiry in prepared_forwards
            ]
            scenarios.append(
                (ladder_spot, option_markets, next_markets, forward_markets, None)
            )
        except Exception as exc:
            scenarios.append((ladder_spot, None, None, None, exc))
        report(done, len(ladder_spots), "markets")

    def calculate_scenario(scenario):
        ladder_spot, option_markets, next_markets, forward_markets, setup_error = scenario
        if setup_error is not None:
            return None, f"Spot {ladder_spot:.6f}: {setup_error}"
        level = _empty_level(ladder_spot, extras)
        try:
            for index, prepared_row in specs:
                row = copy(prepared_row)
                market = option_markets[index]
                vol = _solve_smile_vol(
                    base_markets[index], market, row, pricing_model, atm_type,
                    delta_base, mods,
                )
                _, _, mtm = price_option_with_vol(
                    market, row, vol, atm_type, delta_base
                )
                _add_option_pnl(level, row, mtm, float(ladder_spot))
                greeks, greek_market, greek_next = _option_greeks(
                    base_markets[index], market, next_markets[index], row, pricing_model,
                    atm_type, delta_base, mods, next_valuation, mtm,
                )
                _add_option_greeks(level, greeks, greek_market, row, extras, pricing_model,
                                   atm_type, delta_base, mods, greek_next,
                                   next_valuation)
            for position, market, next_market in forward_markets:
                _add_forward_pnl(
                    level, position, market, valuation_date,
                    next_market, next_valuation,
                )
            return level, None
        except Exception as exc:
            return None, f"Spot {ladder_spot:.6f}: {exc}"

    results = []
    for level, error in _run_scenarios(
        scenarios, scenario_workers, calculate_scenario
    ):
        if error is not None:
            errors.append(error)
        else:
            results.append(level)
    return results, errors


def _calculate_optimized_ladder(
    rows, ladder_spots, valuation_date, market_for_expiry, resolve_expiry,
    pricing_model, atm_type, delta_base, mods, forward_positions=(),
    scenario_workers=None, extras=(), market_for_delivery=None,
):
    """Calculate the cached-volatility ladder with scalar fallback."""
    # Keep forward valuation on the reference path until the optimized option
    # path has matching multi-instrument support.
    if forward_positions:
        return _calculate_reference_ladder(
            rows, ladder_spots, valuation_date, market_for_expiry, resolve_expiry,
            pricing_model, atm_type, delta_base, mods, forward_positions,
            scenario_workers, extras=extras, market_for_delivery=market_for_delivery,
        )
    specs, errors = _prepare_specs(
        rows, valuation_date, resolve_expiry, market_for_expiry, atm_type, delta_base
    )
    if errors:
        return _calculate_reference_ladder(
            rows, ladder_spots, valuation_date, market_for_expiry, resolve_expiry,
            pricing_model, atm_type, delta_base, mods, scenario_workers=scenario_workers,
            extras=extras,
        )
    markets = {}
    vols = {}
    try:
        base_markets = {
            index: market_for_expiry(row["expiry"])
            for index, row in specs
        }
        scenario_pairs = [
            (ladder_spot, index, row)
            for ladder_spot in ladder_spots
            for index, row in specs
        ]
        scenario_markets = {}
        report = current()
        for done, (ladder_spot, index, row) in enumerate(scenario_pairs, start=1):
            scenario_markets[(float(ladder_spot), index)] = _cached_market(
                markets, market_for_expiry, row["expiry"], None, ladder_spot
            )
            report(done, len(scenario_pairs), "markets")

        def solve_scenario_vol(scenario):
            ladder_spot, index, row = scenario
            return (
                (float(ladder_spot), index),
                _solve_smile_vol(
                    base_markets[index],
                    scenario_markets[(float(ladder_spot), index)],
                    row,
                    pricing_model,
                    atm_type,
                    delta_base,
                    mods,
                ),
            )

        for key, vol in _run_scenarios(
            scenario_pairs, scenario_workers, solve_scenario_vol, stage="smile vols"
        ):
            vols[key] = vol
    except Exception:
        return _calculate_reference_ladder(
            rows, ladder_spots, valuation_date, market_for_expiry, resolve_expiry,
            pricing_model, atm_type, delta_base, mods, scenario_workers=scenario_workers,
            extras=extras,
        )

    next_valuation = _next_valuation(specs, valuation_date)
    scenarios = []
    report = current()
    for done, ladder_spot in enumerate(ladder_spots, start=1):
        option_markets = {
            index: scenario_markets[(float(ladder_spot), index)]
            for index, _ in specs
        }
        next_markets = {
            index: _cached_market(
                markets, market_for_expiry, row["expiry"],
                next_valuation, ladder_spot
            )
            for index, row in specs
        }
        scenarios.append((ladder_spot, option_markets, next_markets))
        report(done, len(ladder_spots), "next-day markets")

    def calculate_scenario(scenario):
        ladder_spot, option_markets, next_markets = scenario
        level = _empty_level(ladder_spot, extras)
        try:
            for index, prepared_row in specs:
                row = copy(prepared_row)
                market = option_markets[index]
                _, _, mtm = price_option_with_vol(
                    market, row, vols[(float(ladder_spot), index)],
                    atm_type, delta_base
                )
                greeks, greek_market, greek_next = _option_greeks(
                    base_markets[index], market, next_markets[index], row, pricing_model,
                    atm_type, delta_base, mods, next_valuation, mtm,
                )
                _add_option_pnl(level, row, mtm, float(ladder_spot))
                _add_option_greeks(level, greeks, greek_market, row, extras, pricing_model,
                                   atm_type, delta_base, mods, greek_next,
                                   next_valuation)
            return level, None
        except Exception as exc:
            return None, f"Spot {ladder_spot:.6f}: {exc}"

    results = []
    for level, error in _run_scenarios(
        scenarios, scenario_workers, calculate_scenario
    ):
        if error is not None:
            errors.append(error)
        else:
            results.append(level)
    return results, errors


def _compare_ladder_results(reference, candidate, atol=1e-8, rtol=1e-8):
    """Return whether two ladder result sets agree numerically."""
    if len(reference) != len(candidate):
        return False
    if not reference:
        return True
    keys = sorted(reference[0].keys())
    for ref_row, candidate_row in zip(reference, candidate):
        if ref_row.keys() != candidate_row.keys():
            return False
        for key in keys:
            ref_value = ref_row[key]
            candidate_value = candidate_row[key]
            if not np.isclose(
                float(ref_value),
                float(candidate_value),
                atol=atol,
                rtol=rtol,
                equal_nan=True,
            ):
                return False
    return True


def calculate_spot_ladder(
    rows, ladder_spots, valuation_date, market_for_expiry, resolve_expiry,
    pricing_model, atm_type, delta_base, mods, mode="shadow",
    forward_positions=(), scenario_workers=None, extras=(), market_for_delivery=None,
    progress=None,
):
    """Calculate a ladder in reference, shadow, or optimized mode.

    ``extras`` adds on-demand Greeks (``pricing.greek_catalog``) to every level.
    ``market_for_delivery`` (``MarketContext.market_for_delivery``) values
    forwards with a ``value_date`` at their own delivery date.
    ``progress(done, total, stage)`` is told as scenarios finish, per stage
    (``app.progress``).
    """
    from app.progress import reporting
    with reporting(progress):
        return _calculate_spot_ladder(
            rows, ladder_spots, valuation_date, market_for_expiry, resolve_expiry,
            pricing_model, atm_type, delta_base, mods, mode, forward_positions,
            scenario_workers, extras, market_for_delivery)


def _calculate_spot_ladder(
    rows, ladder_spots, valuation_date, market_for_expiry, resolve_expiry,
    pricing_model, atm_type, delta_base, mods, mode, forward_positions,
    scenario_workers, extras, market_for_delivery,
):
    if mode not in {"reference", "shadow", "optimized"}:
        raise ValueError("Spot-ladder mode must be reference, shadow, or optimized")

    if mode == "reference":
        return _calculate_reference_ladder(
            rows, ladder_spots, valuation_date, market_for_expiry, resolve_expiry,
            pricing_model, atm_type, delta_base, mods, forward_positions,
            scenario_workers, extras=extras, market_for_delivery=market_for_delivery,
        )

    optimized_results, optimized_errors = _calculate_optimized_ladder(
        rows, ladder_spots, valuation_date, market_for_expiry, resolve_expiry,
        pricing_model, atm_type, delta_base, mods, forward_positions,
        scenario_workers, extras=extras, market_for_delivery=market_for_delivery,
    )
    if mode == "optimized":
        return optimized_results, optimized_errors

    reference_results, reference_errors = _calculate_reference_ladder(
        rows, ladder_spots, valuation_date, market_for_expiry, resolve_expiry,
        pricing_model, atm_type, delta_base, mods, forward_positions,
        scenario_workers, extras=extras, market_for_delivery=market_for_delivery,
    )
    if reference_errors or optimized_errors or not _compare_ladder_results(
        reference_results, optimized_results
    ):
        return reference_results, reference_errors + [
            "Shadow validation failed; reference spot-ladder results returned."
        ]
    return optimized_results, optimized_errors


def spot_moves(spots, current_spot):
    """Each ladder spot's move from ``current_spot``, in percent."""
    return [(float(spot) / float(current_spot) - 1.0) * 100.0 for spot in spots]


def format_move(move):
    """A move in percent for display: signed, 0.00% at spot, more decimals
    for moves below 0.01%."""
    if abs(move) < 5e-7:
        return "0.00%"
    return f"{move:+.4f}%" if abs(move) < 0.01 else f"{move:+.2f}%"


def render_spot_ladder(tab, results, errors, current_spot, greeks=GREEK_NAMES):
    """Render calculated spot-ladder results into a Streamlit tab.

    ``greeks`` picks the Greek columns; ones the results lack are left out.
    """
    from pricing.greek_catalog import label as greek_label
    import streamlit as st

    with tab:
        for error in errors:
            st.error(error)
        if not results:
            return
        ladder_df = pd.DataFrame(results).sort_values(
            "spot", ascending=False
        ).reset_index(drop=True)
        shown = [column for column in greek_columns(greeks) if column in ladder_df]
        ladder_df["move_pct"] = spot_moves(ladder_df["spot"], current_spot)
        columns = ["spot", "move_pct", "pnl_base", *shown]
        config = {
            "spot": st.column_config.NumberColumn("Spot", format="%.6f"),
            "move_pct": st.column_config.TextColumn(
                "Move", help="Move from the current spot, in percent."),
            "pnl_base": st.column_config.NumberColumn("P&L (base)", format="%.0f"),
        }
        for column in shown:
            from pricing.greek_catalog import PREMIUM_CCY_GREEKS, greek_of
            label = greek_label(column, "prem ccy" if greek_of(column) in PREMIUM_CCY_GREEKS
                                else None)
            config[column] = st.column_config.NumberColumn(label, format="%.0f")

        def highlight_current(row):
            if np.isclose(float(row["spot"]), float(current_spot), rtol=0.0, atol=1e-12):
                return ["background-color: #fff2cc; font-weight: bold"] * len(row)
            return [""] * len(row)

        display = ladder_df[columns].style.apply(highlight_current, axis=1).format(
            {
                "spot": "{:.6f}",
                "move_pct": format_move,
                **{
                    column: "{:,.0f}"
                    for column in ("pnl_base",)
                },
                **{column: "{:,.0f}" for column in shown},
            }
        )
        st.subheader("Spot ladder results")
        from app.ui.negatives import red_negatives
        st.dataframe(
            red_negatives(display, ["pnl_base", *shown]), use_container_width=True, hide_index=True,
            column_config=config,
        )

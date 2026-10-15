"""Spot-ladder risk report calculation and rendering."""

from copy import copy
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from app.curve_roll import next_business_day
from pricing.greeks import calculate_greeks, calculate_forward_theta, forward_delta_factor
from pricing.pricing import price_option_with_vol
from pricing.fx_forward import FxForwardPosition
from helper_functions.cashflows import signed_premium_cashflow
from helper_functions.input_resolvers import parse_notional, resolve_strike
from smile_perturbed_market import PerturbedMarket


GREEK_NAMES = ["delta", "gamma", "vega", "vanna", "volga", "theta", "rega", "sega"]


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


def _empty_level(spot):
    level = {
        "spot": spot,
        "pnl_base": 0.0,
    }
    for greek_name in GREEK_NAMES:
        level[f"fixed_{greek_name}"] = 0.0
        level[greek_name] = 0.0
    return level


def _scenario_worker_count(scenario_workers, scenario_count):
    """Return a bounded thread count for independent ladder scenarios."""
    if scenario_workers is not None:
        if not isinstance(scenario_workers, int) or scenario_workers < 1:
            raise ValueError("scenario_workers must be a positive integer or None")
        return min(scenario_workers, scenario_count)
    # Smile recalibration invokes NumPy, SciPy, and Numba kernels that can
    # already use native threads. A small outer pool avoids oversubscription.
    return min(scenario_count, 2)


def _run_scenarios(scenarios, scenario_workers, calculate_scenario):
    """Run pure, independent scenario calculations in stable input order."""
    if not scenarios:
        return []
    worker_count = _scenario_worker_count(scenario_workers, len(scenarios))
    if worker_count == 1:
        return [calculate_scenario(scenario) for scenario in scenarios]
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        return list(executor.map(calculate_scenario, scenarios))


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
    delta = base_amount * forward_delta_factor(market, position.expiry, valuation_date)
    level["fixed_delta"] += delta
    level["delta"] += delta
    if next_market is not None:
        level["theta"] += calculate_forward_theta(
            position, market, next_market, valuation_date, next_valuation
        )


def _calculate_reference_ladder(
    rows, ladder_spots, valuation_date, market_for_expiry, resolve_expiry,
    pricing_model, atm_type, delta_base, mods, forward_positions=(),
    scenario_workers=None,
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
    for ladder_spot in ladder_spots:
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
                (position, market_for_expiry(expiry, spot_value=ladder_spot),
                 market_for_expiry(
                     expiry, as_of=next_valuation, allow_short_end=True,
                     spot_value=ladder_spot,
                 ))
                for position, expiry in prepared_forwards
            ]
            scenarios.append(
                (ladder_spot, option_markets, next_markets, forward_markets, None)
            )
        except Exception as exc:
            scenarios.append((ladder_spot, None, None, None, exc))

    def calculate_scenario(scenario):
        ladder_spot, option_markets, next_markets, forward_markets, setup_error = scenario
        if setup_error is not None:
            return None, f"Spot {ladder_spot:.6f}: {setup_error}"
        level = _empty_level(ladder_spot)
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
                greeks = calculate_greeks(
                    market, row, None, pricing_model, atm_type, delta_base,
                    mods, next_market=next_markets[index],
                    next_valuation=next_valuation,
                    option_value_term=mtm,
                )
                for greek_name in GREEK_NAMES:
                    level[f"fixed_{greek_name}"] += greeks[f"fixed_{greek_name}"]
                    level[greek_name] += greeks[greek_name]
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
    scenario_workers=None,
):
    """Calculate the cached-volatility ladder with scalar fallback."""
    # Keep forward valuation on the reference path until the optimized option
    # path has matching multi-instrument support.
    if forward_positions:
        return _calculate_reference_ladder(
            rows, ladder_spots, valuation_date, market_for_expiry, resolve_expiry,
            pricing_model, atm_type, delta_base, mods, forward_positions,
            scenario_workers,
        )
    specs, errors = _prepare_specs(
        rows, valuation_date, resolve_expiry, market_for_expiry, atm_type, delta_base
    )
    if errors:
        return _calculate_reference_ladder(
            rows, ladder_spots, valuation_date, market_for_expiry, resolve_expiry,
            pricing_model, atm_type, delta_base, mods, scenario_workers=scenario_workers,
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
        for ladder_spot, index, row in scenario_pairs:
            scenario_markets[(float(ladder_spot), index)] = _cached_market(
                markets, market_for_expiry, row["expiry"], None, ladder_spot
            )

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
            scenario_pairs, scenario_workers, solve_scenario_vol
        ):
            vols[key] = vol
    except Exception:
        return _calculate_reference_ladder(
            rows, ladder_spots, valuation_date, market_for_expiry, resolve_expiry,
            pricing_model, atm_type, delta_base, mods, scenario_workers=scenario_workers,
        )

    next_valuation = _next_valuation(specs, valuation_date)
    scenarios = []
    for ladder_spot in ladder_spots:
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

    def calculate_scenario(scenario):
        ladder_spot, option_markets, next_markets = scenario
        level = _empty_level(ladder_spot)
        try:
            for index, prepared_row in specs:
                row = copy(prepared_row)
                market = option_markets[index]
                _, _, mtm = price_option_with_vol(
                    market, row, vols[(float(ladder_spot), index)],
                    atm_type, delta_base
                )
                greeks = calculate_greeks(
                    market, row, None, pricing_model, atm_type, delta_base,
                    mods, next_market=next_markets[index],
                    next_valuation=next_valuation,
                    option_value_term=mtm,
                )
                _add_option_pnl(level, row, mtm, float(ladder_spot))
                for greek_name in GREEK_NAMES:
                    level[f"fixed_{greek_name}"] += greeks[f"fixed_{greek_name}"]
                    level[greek_name] += greeks[greek_name]
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
    forward_positions=(), scenario_workers=None,
):
    """Calculate a ladder in reference, shadow, or optimized mode."""
    if mode not in {"reference", "shadow", "optimized"}:
        raise ValueError("Spot-ladder mode must be reference, shadow, or optimized")

    if mode == "reference":
        return _calculate_reference_ladder(
            rows, ladder_spots, valuation_date, market_for_expiry, resolve_expiry,
            pricing_model, atm_type, delta_base, mods, forward_positions,
            scenario_workers,
        )

    optimized_results, optimized_errors = _calculate_optimized_ladder(
        rows, ladder_spots, valuation_date, market_for_expiry, resolve_expiry,
        pricing_model, atm_type, delta_base, mods, forward_positions,
        scenario_workers,
    )
    if mode == "optimized":
        return optimized_results, optimized_errors

    reference_results, reference_errors = _calculate_reference_ladder(
        rows, ladder_spots, valuation_date, market_for_expiry, resolve_expiry,
        pricing_model, atm_type, delta_base, mods, forward_positions,
        scenario_workers,
    )
    if reference_errors or optimized_errors or not _compare_ladder_results(
        reference_results, optimized_results
    ):
        return reference_results, reference_errors + [
            "Shadow validation failed; reference spot-ladder results returned."
        ]
    return optimized_results, optimized_errors


def render_spot_ladder(tab, results, errors, current_spot):
    """Render calculated spot-ladder results into a Streamlit tab."""
    import streamlit as st

    with tab:
        for error in errors:
            st.error(error)
        if not results:
            return
        ladder_df = pd.DataFrame(results).sort_values(
            "spot", ascending=False
        ).reset_index(drop=True)
        columns = ["spot", "pnl_base"]
        for greek_name in GREEK_NAMES:
            columns.extend([f"fixed_{greek_name}", greek_name])
        config = {
            "spot": st.column_config.NumberColumn("Spot", format="%.6f"),
            "pnl_base": st.column_config.NumberColumn("P&L (base)", format="%.0f"),
        }
        for greek_name in GREEK_NAMES:
            for column in (f"fixed_{greek_name}", greek_name):
                label = column.replace("_", " ").title()
                if greek_name == "theta":
                    label += " (prem ccy)"
                config[column] = st.column_config.NumberColumn(label, format="%.0f")

        def highlight_current(row):
            if np.isclose(float(row["spot"]), float(current_spot), rtol=0.0, atol=1e-12):
                return ["background-color: #fff2cc; font-weight: bold"] * len(row)
            return [""] * len(row)

        display = ladder_df[columns].style.apply(highlight_current, axis=1).format(
            {
                "spot": "{:.6f}",
                **{
                    column: "{:,.0f}"
                    for column in ("pnl_base",)
                },
                **{
                    column: "{:,.0f}"
                    for greek_name in GREEK_NAMES
                    for column in (f"fixed_{greek_name}", greek_name)
                },
            }
        )
        st.subheader("Spot ladder results")
        st.dataframe(
            display, use_container_width=True, hide_index=True,
            column_config=config,
        )

"""Price trade lines against their markets, independent of the UI.

``markets`` is either one pair's ``MarketContext`` (every line must be on
that pair) or a ``MarketSet``, which sends each line to its own pair's
market. The Pricer page, the spot ladder and the golden tests all go through
these functions. Every function isolates failures per line (or per hedge
group): a line that cannot be priced is returned as an error message naming
its line number, and the rest of the book still prices.
"""
import re

import numpy as np
import pandas as pd

from app.curve_roll import next_business_day
from app.market_set import MarketSet
from app.premium_side import premium_currency
from app.smile_service import market_diagnostics
from app.vol_override import shift_market, shifted_market, strike_vol, typed_vol
from helper_functions.input_resolvers import parse_notional, resolve_strike
from pricing import Pricable
from pricing.fx_forward import FxForwardPosition
from pricing.greeks import (
    calculate_forward_theta, calculate_greeks, convention_deltas, forward_delta_factor,
)


#: Greeks every priced line reports, fixed (``fixed_<name>``) and smile.
GREEK_NAMES = ("delta", "gamma", "vega", "vanna", "volga", "theta", "rega", "sega")


def format_notional(value):
    """Format a numeric notional with comma separators for display."""
    amount = parse_notional(value)
    if not np.isfinite(amount):
        raise ValueError("Notional must be finite")
    return f"{amount:,.0f}"


def format_strike_for_pair(currency_pair, strike):
    """Format FX strikes using JPY and non-JPY market conventions."""
    pair = str(currency_pair).strip().upper()
    decimals = 2 if "JPY" in pair else 4
    return f"{float(strike):.{decimals}f}"


def is_complete_strike_alias(row):
    """Report whether a row carries a fully specified strike alias.

    Half-typed cells must stay silent while editing, but a complete alias that
    fails to resolve is a genuine error worth showing.
    """
    value = str(row.get("strike", "")).strip().lower().replace(" ", "")
    if value not in {"atm", "atmf", "atms"} and not re.fullmatch(
        r"(\d+(?:\.\d+)?)d", value
    ):
        return False
    expiry = row.get("expiry")
    call_put = str(row.get("call_put", "")).strip().upper()
    return (
        not pd.isna(expiry)
        and bool(str(expiry).strip())
        and call_put in {"CALL", "PUT"}
    )


def _line_pair(row):
    pair = str(row.get("currency_pair", "")).strip().upper()
    if not re.fullmatch(r"[A-Z]{6}", pair):
        raise ValueError("Currency pair must contain two 3-letter ISO codes")
    return pair


def market_for_pair(markets, pair, mismatch=None):
    """The market a line on ``pair`` prices against.

    A ``MarketSet`` looks the pair up; a single market must be on ``pair``.
    """
    if isinstance(markets, MarketSet):
        return markets.market(pair)
    if pair != markets.currency_pair:
        raise ValueError(
            mismatch or f"Market data is configured for {markets.currency_pair}, not {pair}"
        )
    return markets


def price_line(markets, line_number, source_row, mods, extras=()):
    """Price one trade line.

    Returns ``(line_number, result, error, resolved_strike)``: exactly one of
    ``result`` (the line's row for the pricer grid) and ``error`` is set.
    """
    resolved_strike = None
    try:
        row = source_row.copy()
        valuation_date = markets.valuation_date
        row["valuation"] = valuation_date
        currency_pair = _line_pair(row)
        context = market_for_pair(markets, currency_pair)
        settings = context.settings
        row["currency_pair"] = currency_pair
        row["expiry"] = context.resolve_market_expiry(row["expiry"])
        row["notional"] = parse_notional(row["notional"])
        # A line the Pricer marks auto takes the premium that makes P&L zero.
        # Otherwise (typed, and every stored trade) the premium cell is used;
        # an empty cell means no premium.
        if is_auto_premium(row.get(PREMIUM_MODE)):
            premium = None
        else:
            premium_cell = row.get("premium", 0.0)
            premium = (0.0 if _blank(premium_cell)
                       else pd.to_numeric(premium_cell, errors="coerce"))
            if pd.isna(premium) or not np.isfinite(float(premium)) or premium < 0:
                raise ValueError("Premium must be a finite non-negative amount")
        # BASE/TERM (the grid) or a code (files, older lines) -> the code.
        premium_ccy = premium_currency(currency_pair, row.get("premium_ccy"))
        if premium_ccy not in {currency_pair[:3], currency_pair[3:]}:
            raise ValueError(
                f"Premium currency must be {currency_pair[:3]} or {currency_pair[3:]}"
            )
        buy_sell = str(row["buy_sell"]).strip().upper()
        if buy_sell not in {"BUY", "SELL"}:
            raise ValueError("Buy/Sell must be BUY or SELL")
        market = context.market_for_expiry(row["expiry"])
        resolved_strike = resolve_strike(market, row, settings.atm_type, settings.delta_base)
        # Pass a numeric strike into every downstream calculator.
        row["strike"] = resolved_strike
        next_valuation = next_business_day(currency_pair, valuation_date)
        next_market = context.market_for_expiry(
            row["expiry"], as_of=next_valuation, allow_short_end=True
        )
        # A typed vol prices the line on its own copy of the smile, shifted
        # so the strike's vol is the typed one: the Greeks and an auto premium
        # (the dealt price) come from it, MTM from the market, so P&L is the
        # market value against the price at the typed vol.
        line_market, vol_fields, value_args = market, {}, {}
        override = typed_vol(row)
        if override is not None:
            unit_row = row.copy()
            unit_row["notional"] = 1.0
            model_args = (settings.pricing_model, settings.atm_type, settings.delta_base)
            market_vol, _ = strike_vol(market, unit_row, *model_args)
            line_market, shift = shifted_market(market, unit_row, override, *model_args,
                                                market_vol=market_vol)
            next_market = shift_market(next_market, shift)
            value_args = {"value_market": market}
            vol_fields = {"vol_override": True, "mkt_vol_pct": market_vol, "vol_shift": shift}
        pricing_result = Pricable(
            row=row, premium=None if premium is None else float(premium),
            premium_ccy=premium_ccy,
        ).price(
            line_market, settings.pricing_model, settings.atm_type, settings.delta_base, mods,
            next_market=next_market, next_valuation=next_valuation,
            # Only when asked: a Streamlit session that reloaded app/ but not
            # pricing/ (outside the watched folder) keeps pricing without extras.
            **({"extras": tuple(extras)} if extras else {}),
            # Only for a typed vol, as for extras.
            **value_args,
        )
        if vol_fields:
            vol_fields["vol_spread"] = pricing_result["vol_pct"] - vol_fields["mkt_vol_pct"]
            # The premium at the typed vol (what an auto premium is).
            vol_fields["prem_at_vol"] = abs(pricing_result["value_at_vol"])
        forward_rate = market.s + market.fwd_pts / market.forward_divisor
        # Spot, forward and pair-convention deltas; none if they cannot be
        # converted, and the line keeps its reported (spot-equivalent) delta.
        conventions = convention_deltas(
            line_market, row, row["expiry"], valuation_date,
            pricing_result.get("fixed_delta"), pricing_result.get("delta"))
        status, reason = market_diagnostics(market, row["expiry"])
        # A booked trade keeps its id through pricing; typed lines have none.
        identity = ({"trade_id": source_row["trade_id"]}
                    if not _blank_id(source_row.get("trade_id")) else {})
        return line_number, {
            "row": line_number,
            **identity,
            **pricing_result,
            **vol_fields,
            "strike": resolved_strike,
            "call_put": row["call_put"],
            "buy_sell": row["buy_sell"],
            "notional": format_notional(row["notional"]),
            **conventions,
            "_forward_rate": forward_rate,
            "smile_status": status,
            "smile_reason": reason,
        }, None, resolved_strike
    except Exception as exc:
        return line_number, None, "Line {}: {}".format(line_number, exc), resolved_strike


#: Pricer grid column: ``"auto"`` (the premium making P&L zero, refilled on
#: pricing) or ``"typed"`` (the premium entered). Lines without it, such as
#: stored trades, use their premium cell.
PREMIUM_MODE = "premium_mode"
AUTO, TYPED = "auto", "typed"


def is_auto_premium(mode):
    """True only for a line explicitly marked auto."""
    return isinstance(mode, str) and mode.strip().lower() == AUTO


def _blank_id(value):
    return value is None or (not isinstance(value, str) and pd.isna(value)) or not str(value).strip()


def _numbered(rows):
    """``(line_number, row)`` by visible position, not by DataFrame index.

    Dynamic editor rows can have non-consecutive indices after lines are
    removed, so line numbers follow the order the user sees.
    """
    if isinstance(rows, pd.DataFrame):
        rows = [row for _, row in rows.iterrows()]
    return [(number, row if isinstance(row, pd.Series) else pd.Series(row))
            for number, row in enumerate(rows, start=1)]


def price_lines(markets, rows, mods, extras=()):
    """Price every line; returns ``(results, errors)`` in line order.

    ``extras`` adds on-demand Greeks (``pricing.greek_catalog.EXTRA_GREEKS``).
    """
    calculated = sorted(
        (price_line(markets, number, row, mods, extras)
         for number, row in _numbered(rows)),
        key=lambda item: item[0],
    )
    results = [item[1] for item in calculated if item[1] is not None]
    errors = [item[2] for item in calculated if item[2] is not None]
    return results, errors


def build_delta_hedges(markets, rows, mods):
    """Create fixed forward hedge positions from current option deltas.

    Returns ``(positions, errors)``; a line whose delta cannot be computed is
    reported and left out of the hedge.
    """
    grouped = {}
    errors = []
    for line_number, source_row in _numbered(rows):
        try:
            row = source_row.copy()
            row["valuation"] = markets.valuation_date
            pair = _line_pair(row)
            context = market_for_pair(
                markets, pair, "Market data does not match this currency pair"
            )
            settings = context.settings
            row["currency_pair"] = pair
            expiry = context.resolve_market_expiry(row["expiry"])
            row["expiry"] = expiry
            row["notional"] = parse_notional(row["notional"])
            market = context.market_for_expiry(expiry)
            row["strike"] = resolve_strike(market, row, settings.atm_type, settings.delta_base)
            greeks = calculate_greeks(
                market, row, None, settings.pricing_model, settings.atm_type,
                settings.delta_base, mods,
            )
            smile_delta = float(greeks["delta"])
            # Convert selected delta units back to forward hedge notional.
            forward_rate = float(market.s) + float(market.fwd_pts) / float(market.forward_divisor)
            if not np.isfinite(smile_delta):
                raise ValueError("Smile-adjusted delta is invalid")
            key = (pair, expiry, forward_rate)
            grouped[key] = grouped.get(key, 0.0) - smile_delta / forward_delta_factor(
                market, expiry, markets.valuation_date
            )
        except Exception as exc:
            errors.append("Line {} hedge: {}".format(line_number, exc))

    positions = []
    for (pair, expiry, rate), base_amount in grouped.items():
        if not np.isfinite(base_amount):
            errors.append("{} {} hedge amount is not finite".format(pair, expiry))
            continue
        if abs(base_amount) < 1e-12:
            continue
        positions.append({
            "currency_pair": pair,
            "expiry": expiry,
            "base_amount": float(base_amount),
            "contract_rate": float(rate),
        })
    return positions, errors


def value_forward(markets, pair, expiry, base_amount, contract_rate):
    """MTM, delta and roll theta of one FX forward, as a hedge-trade row.

    ``delta_offset`` is the forward's delta in the units the Pricer reports
    option deltas in, so it offsets them directly. Raises on failure.
    """
    valuation_date = markets.valuation_date
    if not np.isfinite(base_amount):
        raise ValueError("hedge amount is not finite")
    context = market_for_pair(markets, pair)
    market = context.market_for_expiry(expiry)
    delta_factor = forward_delta_factor(market, expiry, valuation_date)
    position = FxForwardPosition(
        currency_pair=pair, expiry=expiry, base_amount=base_amount,
        contract_rate=contract_rate,
    )
    value = position.value(market, valuation_date)
    next_valuation = next_business_day(pair, valuation_date)
    next_market = context.market_for_expiry(expiry, as_of=next_valuation, allow_short_end=True)
    theta = calculate_forward_theta(position, market, next_market, valuation_date, next_valuation)
    return {
        "currency_pair": pair,
        "expiry": expiry,
        "direction": "BUY BASE" if base_amount > 0 else "SELL BASE",
        "base_amount": base_amount,
        "forward_rate": contract_rate,
        "delta_offset": base_amount * delta_factor,
        # A forward's forward delta is its amount; its spot delta the offset.
        "spot_delta": base_amount * forward_delta_factor(
            market.copy_with(delta_reporting="spot"), expiry, valuation_date),
        "fwd_delta": base_amount,
        "mtm_term": value["mtm_term"],
        "mtm_base": value["mtm_base"],
        "pnl_term": value["mtm_term"],
        "pnl_base": value["mtm_base"],
        "theta": theta,
    }


def hedge_amounts(markets, results):
    """Forward base amount that flattens each priced line's smile delta.

    Returns ``({(pair, expiry, forward rate): [(result, base amount)]}, errors)``:
    lines are grouped by the forward that would hedge them, and each line's
    amount is its own share, so a group's amounts sum to its hedge.
    """
    valuation_date = markets.valuation_date
    groups, errors = {}, []
    for result in results:
        key = (result["currency_pair"], result["expiry"], float(result["_forward_rate"]))
        try:
            delta = float(result["delta"])
            if not np.isfinite(delta):
                raise ValueError("Smile-adjusted delta is invalid")
            market = market_for_pair(markets, key[0]).market_for_expiry(key[1])
            amount = -delta / forward_delta_factor(market, key[1], valuation_date)
            if not np.isfinite(amount):
                raise ValueError("hedge amount is not finite")
        except Exception as exc:
            errors.append("{} {} hedge: {}".format(key[0], key[1], exc))
            continue
        groups.setdefault(key, []).append((result, amount))
    return groups, errors


def hedge_trades_for(markets, results):
    """Forward trades that flatten the priced lines' smile delta, per expiry.

    Returns ``(trades, errors)``. A hedge group that cannot be valued is
    reported and skipped; it never stops the other groups or the pricing.
    """
    grouped = {}
    for result in results:
        key = (result["currency_pair"], result["expiry"], float(result["_forward_rate"]))
        grouped[key] = grouped.get(key, 0.0) + float(result["delta"])
    trades, errors = [], []
    for (pair, expiry, forward_rate), net_smile_delta in grouped.items():
        try:
            if not np.isfinite(net_smile_delta):
                raise ValueError("Smile-adjusted delta is invalid")
            market = market_for_pair(markets, pair).market_for_expiry(expiry)
            base_amount = -net_smile_delta / forward_delta_factor(
                market, expiry, markets.valuation_date)
            if np.isfinite(base_amount) and abs(base_amount) < 1e-12:
                continue
            trades.append(value_forward(markets, pair, expiry, base_amount, forward_rate))
        except Exception as exc:
            errors.append("{} {} hedge: {}".format(pair, expiry, exc))
    return trades, errors


#: How resolved expiries are written back to the grid: unambiguous, and
#: accepted by every expiry parser.
EXPIRY_FORMAT = "%Y-%m-%d"


def _blank(value):
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return True
    # Empty cells that went through str() arrive as "nan" or "None".
    return str(value).strip().lower() in {"", "nan", "none"}


def _numeric_strike(value):
    try:
        number = float(str(value).strip().replace(",", ""))
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(number) and number > 0)


def _is_booked_expiry(value):
    """Already written as a resolved date, so it needs no work."""
    text = str(value).strip()
    try:
        return pd.Timestamp(text).strftime(EXPIRY_FORMAT) == text
    except (TypeError, ValueError):
        return False


def resolve_line_inputs(markets, rows, default_pair=None):
    """Resolve each line's inputs to the values a booked trade carries.

    * a blank currency pair on a line with other inputs takes
      ``default_pair`` (the pair being edited);
    * a tenor or any accepted date format (``1m``, ``ON``, ``18sep25``)
      becomes a date in ``EXPIRY_FORMAT``;
    * a delta or ATM strike alias (``25d``, ``atmf``) becomes a numeric
      strike, solved on the line's own market at the resolved expiry.

    Each step stands alone: an expiry still resolves when the strike cannot
    yet (for example before Call/Put is chosen). Half-typed cells stay
    silent; a complete strike alias that fails is reported, as are
    unresolvable expiries on otherwise complete lines. Returns
    ``(resolved_rows, changed, failures)``.
    """
    resolved_rows = rows.copy()
    changed = False
    failures = []

    def put(index, column, value):
        nonlocal changed
        resolved_rows.at[index, column] = value
        changed = True

    for index, source_row in rows.iterrows():
        row = source_row.copy()
        has_inputs = any(not _blank(row.get(column)) for column in ("strike", "expiry", "notional"))
        if not has_inputs:
            continue
        if _blank(row.get("currency_pair")) and default_pair:
            row["currency_pair"] = default_pair
            put(index, "currency_pair", default_pair)
        try:
            # A single market resolves every row on itself, as the one-pair
            # Pricer always has; a set uses each row's own pair.
            context = (markets.market(_line_pair(row))
                       if isinstance(markets, MarketSet) else markets)
        except Exception as exc:
            if is_complete_strike_alias(row):
                failures.append("Line {}: {}".format(index + 1, exc))
            continue

        expiry = None
        if not _blank(row.get("expiry")):
            try:
                expiry = context.resolve_market_expiry(row["expiry"])
                booked = pd.Timestamp(expiry).strftime(EXPIRY_FORMAT)
                if not _is_booked_expiry(row["expiry"]):
                    put(index, "expiry", booked)
            except Exception as exc:
                expiry = None
                if is_complete_strike_alias(row) or _numeric_strike(row.get("strike")):
                    failures.append("Line {}: {}".format(index + 1, exc))

        strike = row.get("strike")
        if expiry is None or _blank(strike) or _numeric_strike(strike):
            continue
        try:
            settings = context.settings
            row["valuation"] = markets.valuation_date
            row["expiry"] = expiry
            market = context.market_for_expiry(expiry)
            value = resolve_strike(market, row, settings.atm_type, settings.delta_base)
        except Exception as exc:
            # A well-formed alias that still fails is a real problem, not a
            # half-typed cell. Staying silent makes it indistinguishable
            # from "nothing happened" and hides the cause.
            if is_complete_strike_alias(row):
                failures.append("Line {}: {}".format(index + 1, exc))
            continue
        put(index, "strike", format_strike_for_pair(row.get("currency_pair", ""), value))
    return resolved_rows, changed, failures


def resolve_strike_aliases(markets, rows, default_pair=None):
    """Former name of ``resolve_line_inputs``, which also resolves tenors."""
    return resolve_line_inputs(markets, rows, default_pair)


def aggregate_by_pair(results, hedge_trades=(), greeks=GREEK_NAMES):
    """Aggregate base-currency line results and hedge risks by currency pair.

    ``greeks`` picks the Greek columns; one the results do not carry (an
    extra not priced yet) is left out. Forward hedges add to MTM, P&L,
    delta and theta only (sticky and smile alike: a forward has no smile).
    """
    from pricing.greek_catalog import columns as greek_columns
    wanted = greek_columns(greeks)
    if results:
        carried = set().union(*(result.keys() for result in results))
        wanted = [column for column in wanted if column in carried]
    columns = ["currency_pair", "mtm_base", "pnl_base", *wanted]
    if results:
        aggregation = (
            pd.DataFrame(results)[columns]
            .groupby("currency_pair", as_index=False, sort=True)
            .sum(min_count=1)
        )
    else:
        # Forwards alone still give their pairs a row.
        aggregation = pd.DataFrame(columns=columns)
    for trade in hedge_trades:
        pair = str(trade["currency_pair"]).upper()
        matches = aggregation["currency_pair"] == pair
        if not matches.any():
            aggregation = pd.concat(
                [
                    aggregation,
                    pd.DataFrame(
                        [{column: pair if column == "currency_pair" else 0.0
                          for column in columns}]
                    ),
                ],
                ignore_index=True,
            )
            matches = aggregation["currency_pair"] == pair
        aggregation.loc[matches, "mtm_base"] += float(trade["mtm_base"])
        aggregation.loc[matches, "pnl_base"] += float(trade["pnl_base"])
        for column, amount in (("fixed_delta", trade["delta_offset"]),
                               ("delta", trade["delta_offset"]),
                               ("fixed_spot_delta", trade.get("spot_delta")),
                               ("spot_delta", trade.get("spot_delta")),
                               ("fixed_fwd_delta", trade.get("fwd_delta")),
                               ("fwd_delta", trade.get("fwd_delta")),
                               ("fixed_theta", trade.get("theta", 0.0)),
                               ("theta", trade.get("theta", 0.0))):
            if column in aggregation and amount is not None:
                aggregation.loc[matches, column] += float(amount)
    return aggregation

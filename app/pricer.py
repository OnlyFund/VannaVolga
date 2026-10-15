"""Price trade lines against a ``MarketContext``, independent of the UI.

The Pricer page, the spot ladder and the golden tests all go through these
functions. Every function isolates failures per line (or per hedge group):
a line that cannot be priced is returned as an error message naming its
line number, and the rest of the book still prices.
"""
import re

import numpy as np
import pandas as pd

from app.curve_roll import next_business_day
from app.smile_service import market_diagnostics
from helper_functions.input_resolvers import parse_notional, resolve_strike
from helper_functions.premium_quotes import PREMIUM_QUOTES, format_premium_quote
from pricing import Pricable
from pricing.fx_forward import FxForwardPosition
from pricing.greeks import calculate_forward_theta, calculate_greeks, forward_delta_factor


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


def _line_pair(row, context):
    pair = str(row.get("currency_pair", "")).strip().upper()
    if not re.fullmatch(r"[A-Z]{6}", pair):
        raise ValueError("Currency pair must contain two 3-letter ISO codes")
    if pair != context.currency_pair:
        raise ValueError(
            f"Market data is configured for {context.currency_pair}, not {pair}"
        )
    return pair


def price_line(context, line_number, source_row, mods, premium_quote="term_cash"):
    """Price one trade line.

    Returns ``(line_number, result, error, resolved_strike)``: exactly one of
    ``result`` (the line's row for the pricer grid) and ``error`` is set.
    """
    if premium_quote not in PREMIUM_QUOTES:
        premium_quote = "term_cash"
    resolved_strike = None
    try:
        settings = context.settings
        row = source_row.copy()
        valuation_date = context.valuation_date
        row["valuation"] = valuation_date
        currency_pair = _line_pair(row, context)
        row["currency_pair"] = currency_pair
        row["expiry"] = context.resolve_market_expiry(row["expiry"])
        row["notional"] = parse_notional(row["notional"])
        premium = pd.to_numeric(row.get("premium", 0.0), errors="coerce")
        if pd.isna(premium) or not np.isfinite(float(premium)) or premium < 0:
            raise ValueError("Premium must be a finite non-negative amount")
        premium_ccy = str(row.get("premium_ccy", currency_pair[3:])).strip().upper()
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
        pricing_result = Pricable(
            row=row, premium=float(premium), premium_ccy=premium_ccy,
        ).price(
            market, settings.pricing_model, settings.atm_type, settings.delta_base, mods,
            next_market=next_market, next_valuation=next_valuation,
        )
        forward_rate = market.s + market.fwd_pts / market.forward_divisor
        # Map the chosen quote onto stable column names so the display layer
        # never has to know which convention is active. Missing conversions
        # degrade to the term-cash figures.
        quoted_mtm = pricing_result.get("mtm_" + premium_quote, pricing_result.get("mtm_term"))
        quoted_pnl = pricing_result.get("pnl_" + premium_quote, pricing_result.get("pnl_term"))
        status, reason = market_diagnostics(market, row["expiry"])
        return line_number, {
            "row": line_number,
            **pricing_result,
            "strike": resolved_strike,
            "call_put": row["call_put"],
            "buy_sell": row["buy_sell"],
            "notional": format_notional(row["notional"]),
            "mtm_quote": format_premium_quote(quoted_mtm, premium_quote),
            "pnl_quote": format_premium_quote(quoted_pnl, premium_quote),
            "_forward_rate": forward_rate,
            "smile_status": status,
            "smile_reason": reason,
        }, None, resolved_strike
    except Exception as exc:
        return line_number, None, "Line {}: {}".format(line_number, exc), resolved_strike


def _numbered(rows):
    """``(line_number, row)`` by visible position, not by DataFrame index.

    Dynamic editor rows can have non-consecutive indices after lines are
    removed, so line numbers follow the order the user sees.
    """
    if isinstance(rows, pd.DataFrame):
        rows = [row for _, row in rows.iterrows()]
    return [(number, row if isinstance(row, pd.Series) else pd.Series(row))
            for number, row in enumerate(rows, start=1)]


def price_lines(context, rows, mods, premium_quote="term_cash"):
    """Price every line; returns ``(results, errors)`` in line order."""
    calculated = sorted(
        (price_line(context, number, row, mods, premium_quote) for number, row in _numbered(rows)),
        key=lambda item: item[0],
    )
    results = [item[1] for item in calculated if item[1] is not None]
    errors = [item[2] for item in calculated if item[2] is not None]
    return results, errors


def build_delta_hedges(context, rows, mods):
    """Create fixed forward hedge positions from current option deltas.

    Returns ``(positions, errors)``; a line whose delta cannot be computed is
    reported and left out of the hedge.
    """
    grouped = {}
    errors = []
    for line_number, source_row in _numbered(rows):
        try:
            settings = context.settings
            row = source_row.copy()
            row["valuation"] = context.valuation_date
            pair = str(row.get("currency_pair", "")).strip().upper()
            if not re.fullmatch(r"[A-Z]{6}", pair):
                raise ValueError("Currency pair must contain two 3-letter ISO codes")
            if pair != context.currency_pair:
                raise ValueError("Market data does not match this currency pair")
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
                market, expiry, context.valuation_date
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


def hedge_trades_for(context, results):
    """Forward trades that flatten the priced lines' smile delta, per expiry.

    Returns ``(trades, errors)``. A hedge group that cannot be valued is
    reported and skipped; it never stops the other groups or the pricing.
    """
    valuation_date = context.valuation_date
    grouped = {}
    for result in results:
        key = (result["currency_pair"], result["expiry"], float(result["_forward_rate"]))
        grouped[key] = grouped.get(key, 0.0) + float(result["delta"])
    trades, errors = [], []
    for (pair, expiry, forward_rate), net_smile_delta in grouped.items():
        try:
            if not np.isfinite(net_smile_delta):
                raise ValueError("Smile-adjusted delta is invalid")
            market = context.market_for_expiry(expiry)
            hedge_delta_factor = forward_delta_factor(market, expiry, valuation_date)
            base_amount = -net_smile_delta / hedge_delta_factor
            if not np.isfinite(base_amount):
                raise ValueError("hedge amount is not finite")
            if abs(base_amount) < 1e-12:
                continue
            position = FxForwardPosition(
                currency_pair=pair, expiry=expiry, base_amount=base_amount,
                contract_rate=forward_rate,
            )
            value = position.value(market, valuation_date)
            next_valuation = next_business_day(pair, valuation_date)
            next_market = context.market_for_expiry(
                expiry, as_of=next_valuation, allow_short_end=True
            )
            hedge_theta = calculate_forward_theta(
                position, market, next_market, valuation_date, next_valuation
            )
        except Exception as exc:
            errors.append("{} {} hedge: {}".format(pair, expiry, exc))
            continue
        trades.append({
            "currency_pair": pair,
            "expiry": expiry,
            "direction": "BUY BASE" if base_amount > 0 else "SELL BASE",
            "base_amount": base_amount,
            "forward_rate": forward_rate,
            "delta_offset": base_amount * hedge_delta_factor,
            "mtm_term": value["mtm_term"],
            "mtm_base": value["mtm_base"],
            "pnl_term": value["mtm_term"],
            "pnl_base": value["mtm_base"],
            "theta": hedge_theta,
        })
    return trades, errors


def resolve_strike_aliases(context, rows):
    """Convert resolvable delta and ATM aliases to numeric input strikes.

    Returns ``(resolved_rows, changed, failures)``. Half-typed aliases stay
    silent; a complete alias that fails is reported.
    """
    resolved_rows = rows.copy()
    changed = False
    failures = []
    for index, source_row in rows.iterrows():
        value = source_row.get("strike")
        if pd.isna(value) or not str(value).strip():
            continue
        try:
            numeric_strike = float(str(value).strip().replace(",", ""))
            if np.isfinite(numeric_strike) and numeric_strike > 0:
                continue
        except (TypeError, ValueError):
            pass
        try:
            settings = context.settings
            row = source_row.copy()
            row["valuation"] = context.valuation_date
            row["expiry"] = context.resolve_market_expiry(row["expiry"])
            market = context.market_for_expiry(row["expiry"])
            strike = resolve_strike(market, row, settings.atm_type, settings.delta_base)
        except Exception as exc:
            # A well-formed alias that still fails is a real problem, not a
            # half-typed cell. Staying silent makes it indistinguishable
            # from "nothing happened" and hides the cause.
            if is_complete_strike_alias(source_row):
                failures.append("Line {}: {}".format(index + 1, exc))
            continue
        resolved_rows.at[index, "strike"] = format_strike_for_pair(
            source_row.get("currency_pair", ""), strike
        )
        changed = True
    return resolved_rows, changed, failures

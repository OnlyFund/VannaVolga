"""A Pricer line's hedge delta: the shown delta, or the one agreed on the deal.

The hedge delta is a percent of the line's notional in the line's own delta
convention (spot or forward, premium-adjusted where the pair is: the delta
the Pricer shows). It is entered without sign: a bought call or sold put
has positive delta, a bought put or sold call negative. Only a line
explicitly marked typed uses its own; every other line, and every stored
trade, is hedged on its priced delta.
"""
import math

#: Pricer grid columns: the hedge delta (percent of notional) and its source.
HEDGE_DELTA = "hedge_delta_pct"
HEDGE_MODE = "hedge_mode"
AUTO, TYPED = "auto", "typed"
#: A typed hedge delta must lie in [0, MAX_PCT] percent of notional.
MAX_PCT = 150.0


def typed_hedge_pct(row):
    """The line's typed hedge delta in percent, or ``None`` when it is hedged
    on its priced delta. A typed line without a valid delta raises."""
    mode = row.get(HEDGE_MODE)
    if not (isinstance(mode, str) and mode.strip().lower() == TYPED):
        return None
    try:
        pct = abs(float(row.get(HEDGE_DELTA)))
    except (TypeError, ValueError):
        pct = float("nan")
    if not math.isfinite(pct) or pct > MAX_PCT:
        raise ValueError(f"Hedge delta must be a percent of notional from 0 to {MAX_PCT:g}")
    return pct


def delta_sign(call_put, buy_sell):
    """+1 for a bought call or sold put, -1 for a bought put or sold call."""
    call = str(call_put).strip().upper() == "CALL"
    buy = str(buy_sell).strip().upper() == "BUY"
    return 1.0 if call == buy else -1.0


def typed_hedge_fields(pct, result, notional, call_put, buy_sell):
    """Result fields for a typed hedge delta: ``hedge_delta`` in the units
    of the line's reported delta (what the hedge sizing reads).

    The line's own delta is in its convention (``pair_delta``); the ratio of
    the reported delta to it converts the typed delta the same way.
    """
    pair_delta, reported = result.get("pair_delta"), result.get("delta")
    try:
        ratio = float(reported) / float(pair_delta)
    except (TypeError, ValueError, ZeroDivisionError):
        ratio = float("nan")
    if not math.isfinite(ratio):
        raise ValueError("Hedge delta cannot be converted: the line has no delta "
                         "in its convention")
    typed = delta_sign(call_put, buy_sell) * pct / 100.0 * float(notional)
    return {"hedge_override": True, "hedge_delta_pct": pct, "hedge_delta": typed * ratio}


def hedge_delta(result):
    """The delta a line is hedged on: its typed hedge delta, else its priced
    (smile) delta."""
    return float(result["hedge_delta"] if result.get("hedge_override") else result["delta"])


def shown_pct(result, notional):
    """The line's delta as a percent of notional in its convention (unsigned),
    or ``None`` when it has none."""
    try:
        pct = abs(float(result["pair_delta"]) / float(notional) * 100.0)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    return pct if math.isfinite(pct) else None

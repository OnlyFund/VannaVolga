"""A Pricer line's own volatility: the smile shifted to the trader's level.

A typed vol moves the line's ATM quote by the amount that makes the smile's
vol at the line's strike equal the typed vol; RR and BF are kept, so every
Greek, the sticky/smile split and theta work as usual around that level.
The shift applies to that line only. Lines without an explicit ``typed``
mode, such as stored trades, always price at market.
"""
import math

from pricing.pricing import price_option

#: Pricer grid columns: the vol (percent) and its source.
VOL_INPUT = "vol_input"
VOL_MODE = "vol_mode"
MARKET, TYPED = "market", "typed"
#: A typed vol must lie in (0, MAX_VOL] percent.
MAX_VOL = 200.0
#: The strike's vol matches the typed vol to TOLERANCE vol points. A smile
#: recalibrated per shift (vv-sabr) carries fit noise of about 1e-3 points
#: in the wings, so the closest shift within ACCEPT is used when TOLERANCE
#: cannot be reached; beyond ACCEPT the line reports an error.
TOLERANCE = 1e-8
ACCEPT = 1e-3
MAX_ITERATIONS = 25


def typed_vol(row):
    """The line's typed vol in percent, or ``None`` when it prices at market.

    Only a line explicitly marked typed overrides; a typed line without a
    valid vol raises rather than quietly pricing at market.
    """
    mode = row.get(VOL_MODE)
    if not (isinstance(mode, str) and mode.strip().lower() == TYPED):
        return None
    try:
        vol = float(row.get(VOL_INPUT))
    except (TypeError, ValueError):
        vol = float("nan")
    if not math.isfinite(vol) or not 0.0 < vol <= MAX_VOL:
        raise ValueError(f"Vol must be a number above 0 and at most {MAX_VOL:g}%")
    return vol


def strike_vol(market, unit_row, pricing_model, atm_type, delta_base):
    """``(vol, unit price)`` of ``unit_row`` (numeric strike) on ``market``."""
    _, vol, price = price_option(market, unit_row, pricing_model, atm_type, delta_base)
    return float(vol), float(price)


def shifted_market(market, unit_row, target, pricing_model, atm_type, delta_base,
                   market_vol=None):
    """``(market copy, ATM shift)``: the copy's vol at the strike is ``target``.

    Solved by secant steps on the ATM shift (a Vanna-Volga smile moves almost
    one-for-one with ATM). Raises when the typed vol cannot be reached within
    ``ACCEPT``.
    """
    if market_vol is None:
        market_vol, _ = strike_vol(market, unit_row, pricing_model, atm_type, delta_base)

    def error(shift):
        atm = float(market.v_atm) + shift
        if not atm > 0.0:
            raise ValueError(f"Vol {target:g}% is out of reach: ATM would fall to {atm:.4f}%")
        vol, _ = strike_vol(market.copy_with(v_atm=atm), unit_row, pricing_model,
                            atm_type, delta_base)
        return vol - target

    previous, previous_error = 0.0, market_vol - target
    if previous_error == 0.0:
        return market, 0.0
    shift = -previous_error
    best = (abs(previous_error), 0.0)
    for _ in range(MAX_ITERATIONS):
        current_error = error(shift)
        best = min(best, (abs(current_error), shift))
        if abs(current_error) <= TOLERANCE:
            break
        slope = (current_error - previous_error) / (shift - previous)
        # Fit noise can make a short secant step meaningless: the smile
        # moves close to one-for-one with ATM, so fall back to that.
        if not math.isfinite(slope) or not 0.5 <= slope <= 1.5:
            slope = 1.0
        previous, previous_error = shift, current_error
        shift -= current_error / slope
    gap, shift = best
    if gap > ACCEPT:
        raise ValueError(f"Vol {target:g}% could not be matched at this strike "
                         f"(closest {gap:.4f} vol points away)")
    if shift == 0.0:
        return market, 0.0
    return market.copy_with(v_atm=float(market.v_atm) + shift), shift


def shift_market(market, shift):
    """``market`` with its ATM moved by ``shift`` (the next-day market for theta)."""
    if shift == 0.0 or market is None:
        return market
    atm = float(market.v_atm) + shift
    if not atm > 0.0:
        raise ValueError(f"ATM would fall to {atm:.4f}% on the next day")
    return market.copy_with(v_atm=atm)

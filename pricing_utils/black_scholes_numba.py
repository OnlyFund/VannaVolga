"""Numba Black-Scholes kernels with a scalar compatibility wrapper."""

from math import erf, exp, log, pi, sqrt

import numpy as np
from numba import njit

from helper_functions.helper_functions import cp_flag, pos_flag, year_fraction
from helper_functions.fx_discounting import safe_option_discount


# Keep this kernel out of Numba's on-disk cache. Its argument list is part of
# the pricer API and can change as market inputs are added; old cached
# dispatchers can otherwise be loaded by a long-lived Streamlit environment.
@njit(cache=False)
def _black_scholes_values_numba(
    spot, strike, forward_points, volatility_pct, time, notional,
    call_put, buy_sell, delta_base, forward_divisor, term_discount,
    term_discount_factor, carry_rate, delta_spot,
):
    if (
        spot <= 0.0 or strike <= 0.0 or volatility_pct <= 0.0
        or time <= 0.0 or forward_divisor <= 0.0
    ):
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    cp = 1.0 if call_put > 0 else -1.0
    side = 1.0 if buy_sell > 0 else -1.0
    forward = spot + forward_points / forward_divisor
    # A negative sentinel means "use the library's continuous default"; Numba
    # kernels cannot take an optional None.
    if term_discount_factor > 0.0:
        term_discount_factor = term_discount_factor
    else:
        term_discount_factor = exp(-term_discount * time)
        carry_rate = term_discount
    base_discount_factor = term_discount_factor * forward / spot
    vol = volatility_pct / 100.0
    sqrt_time = sqrt(time)
    vol_sqrt_time = vol * sqrt_time
    d1 = (log(forward / strike) + 0.5 * vol * vol * time) / vol_sqrt_time
    d2 = d1 - vol_sqrt_time
    inv_sqrt_two_pi = 1.0 / sqrt(2.0 * pi)
    cdf_d1 = 0.5 * (1.0 + erf(cp * d1 / sqrt(2.0)))
    cdf_d2 = 0.5 * (1.0 + erf(cp * d2 / sqrt(2.0)))
    pdf_d1 = exp(-0.5 * d1 * d1) * inv_sqrt_two_pi
    pdf_d2 = exp(-0.5 * d2 * d2) * inv_sqrt_two_pi
    price = side * cp * notional * (
        spot * base_discount_factor * cdf_d1
        - strike * term_discount_factor * cdf_d2
    )
    if delta_base:
        delta = side * cp * strike / forward * cdf_d2 * notional
        dd2_dspot = 1.0 / (forward * vol * sqrt_time)
        delta_derivative = notional * (
            -cp * strike * cdf_d2 / (forward * forward)
            + strike * pdf_d2 * dd2_dspot / forward
        )
    else:
        delta = side * cp * cdf_d1 * notional
        delta_derivative = notional * pdf_d1 / (forward * vol * sqrt_time)

    # Spot delta discounts the forward delta by the base-currency factor; see
    # BlackScholes.delta. Gamma follows so it stays the derivative of the
    # delta actually reported.
    if delta_spot:
        # base_DF = term_DF * forward / spot depends on spot with the forward
        # points held fixed, so gamma differentiates the product; see
        # BlackScholes.gamma.
        forward_delta = delta / notional / side
        d_base_df_dspot = -term_discount_factor * (
            forward_points / forward_divisor) / (spot * spot)
        delta = delta * base_discount_factor
        delta_derivative = (
            delta_derivative * base_discount_factor
            + notional * forward_delta * d_base_df_dspot
        )

    vega = side * pdf_d1 * forward * sqrt_time * 0.01 * notional * term_discount_factor
    # Gamma is quoted for a 1% *relative* spot move, so the spot derivative is
    # scaled by ``spot * 0.01``. The spot factor was previously missing here,
    # leaving this kernel disagreeing with the analytic engine by exactly the
    # level of spot.
    gamma = side * delta_derivative * spot * 0.01
    volga = vega * d1 * d2 / (vol * 100.0)
    vanna = vega * (1.0 / forward - d1 / (forward * vol * sqrt_time))
    base_rate = carry_rate - log(forward / spot) / time
    theta = side * notional * (
        -spot * base_discount_factor * pdf_d1 * vol / (2.0 * sqrt_time)
        + cp * (
            base_rate * spot * base_discount_factor * cdf_d1
            - carry_rate * strike * term_discount_factor * cdf_d2
        )
    ) / 252.0
    return price, delta, gamma, vega, vanna, volga, theta


def black_scholes_values(
    spot, strike, forward_points, volatility_pct, expiry_date, valuation_date,
    call_put, buy_sell, notional, delta_base=True, forward_divisor=10000.0,
    term_discount=0.05, term_discount_factor=None, carry_rate=None,
    currency_pair=None, delta_convention="spot",
):
    """Return price and analytic Greeks, falling back by raising on invalid input.

    Discounting runs to the option's *delivery* date (expiry plus the pair's
    spot lag on the joint calendar) at a *simple* rate on the term currency's
    day-count basis whenever ``currency_pair`` resolves against the convention
    tables. ``term_discount_factor`` overrides that with an explicit factor and
    ``carry_rate`` supplies the matching decay rate used by theta. With no pair
    and no explicit factor the legacy continuous ``exp(-r * t)`` to expiry on an
    ACT/365 clock is used.

    ``delta_convention`` selects a spot (default, market standard to about one
    year) or forward delta, and gamma follows it; see
    :meth:`pricing_utils.BlackScholes.BlackScholes.delta`.
    """
    if delta_convention not in ("spot", "forward"):
        raise ValueError(
            "delta_convention must be 'spot' or 'forward', got %r"
            % (delta_convention,)
        )
    time = year_fraction(
        _date_text(expiry_date),
        _date_text(valuation_date),
    )
    if term_discount_factor is None and currency_pair is not None:
        resolved = safe_option_discount(
            currency_pair, expiry_date, valuation_date, term_discount)
        if resolved is not None:
            term_discount_factor = resolved.discount_factor
            if carry_rate is None:
                carry_rate = resolved.carry_rate
    if term_discount_factor is None:
        # Numba kernels cannot accept ``None``; a non-positive factor is the
        # sentinel telling the kernel to use the continuous default.
        factor = -1.0
        carry = float(term_discount)
    else:
        factor = float(term_discount_factor)
        if not factor > 0.0:
            raise ValueError("term_discount_factor must be positive")
        carry = float(term_discount if carry_rate is None else carry_rate)
    values = _black_scholes_values_numba(
        float(spot), float(strike), float(forward_points), float(volatility_pct),
        float(time), float(notional), cp_flag(call_put), pos_flag(buy_sell),
        bool(delta_base), float(forward_divisor), float(term_discount),
        factor, carry, bool(delta_convention == "spot"),
    )
    if not np.all(np.isfinite(np.asarray(values, dtype=float))):
        raise ValueError("Numba Black-Scholes returned non-finite values")
    return values


def _date_text(value):
    return value.isoformat() if hasattr(value, "isoformat") else str(value)

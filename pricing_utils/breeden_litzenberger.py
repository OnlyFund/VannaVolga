"""Breeden-Litzenberger risk-neutral densities implied by a volatility smile.

Breeden and Litzenberger (1978): the discounted call price satisfies
``d2C/dK2 = D_term * q(K)``, where ``q`` is the density of the spot at expiry
under the term-currency forward measure. With the undiscounted forward call
price ``c(K) = C(K) / D_term`` the discount factor drops out and
``q(K) = d2c/dK2``.

With a flat volatility this has the closed form exposed by
:meth:`pricing_utils.BlackScholes.BlackScholes.risk_neutral_density`. A smile
makes the volatility depend on strike, which adds ``dsigma/dK`` and
``d2sigma/dK2`` terms; differentiating the smile prices numerically captures
them without having to differentiate the smile model itself.
"""
from math import log, sqrt

import numpy as np

from pricing_utils.BlackScholes import ndtr


def forward_call_price(forward: float, strike: float, vol_pct: float, t: float) -> float:
    """Return the undiscounted Black call price per unit of base notional."""
    if forward <= 0.0 or strike <= 0.0 or t <= 0.0:
        raise ValueError("Forward, strike and expiry must be positive")
    sigma_sqrt_t = vol_pct / 100.0 * sqrt(t)
    if sigma_sqrt_t <= 0.0:
        return max(forward - strike, 0.0)
    d1_ = (log(forward / strike) + 0.5 * sigma_sqrt_t ** 2) / sigma_sqrt_t
    return forward * ndtr(d1_) - strike * ndtr(d1_ - sigma_sqrt_t)


def smile_density(vol_func, strikes, forward: float, t: float, rel_step: float = 1e-3):
    """Return the Breeden-Litzenberger density implied by a smile.

    ``vol_func(K)`` returns the smile volatility in percentage points at
    strike ``K``, for example
    ``lambda k: VannaVolga(market=m, K=k, ...).get_vol_for_strike()``.

    The density is the central second difference of the undiscounted call
    price, ``(c(K+h) - 2 c(K) + c(K-h)) / h**2`` with ``h = rel_step * K``,
    where each price uses the smile vol at its own strike. The truncation
    error is O(h**2), about 2e-4 relative at the default step; smaller steps
    trade it for amplified noise from the smile model's vol solver. Negative values
    mark butterfly arbitrage in the smile and are returned as they are rather
    than clipped, because hiding them would hide the arbitrage.

    Returns a float for a scalar ``strikes`` and an array otherwise.
    """
    if not 0.0 < rel_step < 0.1:
        raise ValueError("rel_step must be in (0, 0.1)")
    strike_arr = np.asarray(strikes, dtype=float)
    scalar = strike_arr.ndim == 0
    query = np.atleast_1d(strike_arr)
    if not np.all(np.isfinite(query)) or np.any(query <= 0.0):
        raise ValueError("Strikes must be positive and finite")

    def price(k):
        return forward_call_price(forward, k, float(vol_func(k)), t)

    density = np.empty_like(query)
    for i, k in enumerate(query):
        h = rel_step * k
        density[i] = (price(k + h) - 2.0 * price(k) + price(k - h)) / (h * h)
    return float(density[0]) if scalar else density

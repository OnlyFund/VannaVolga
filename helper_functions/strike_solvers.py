from __future__ import annotations
import numpy as np
from math import log
from scipy.optimize import brentq
from pricing_utils.BlackScholes import d1
import math

def phi(x: float) -> float:
    """Standard normal CDF"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def vega_ratio_phis(f: float , k: float , k1 : float , k2 : float , k3: float,
                    sigma2_dec : float ,t:float)-> tuple[float, float,float]:
    """Compute vega ratios φ(d1) across strikes for VV weighting."""
    phi_d1_K = phi(d1(f, k, sigma2_dec, t))
    phi_d1_K1 = phi(d1(f, k1, sigma2_dec, t))
    phi_d1_K2 = phi(d1(f, k2, sigma2_dec, t))
    phi_d1_K3 = phi(d1(f, k3, sigma2_dec, t))

    safe_div = lambda a, b: a / b if b and not np.isclose(b, 0.0) else 0.0
    return (
        safe_div(phi_d1_K, phi_d1_K1),
        safe_div(phi_d1_K, phi_d1_K2),
        safe_div(phi_d1_K, phi_d1_K3),
    )


def vv_price(k:float,k1 : float,k2 : float,k3 : float,sigma1_pct : float , sigma2_pct:float, sigma3_pct:float,f: float , t:float,bs_price_func) ->float:
    """
    Compute VV price for a given strike.
    bs_price_func(vol_pct, K)` must return BS price with vol in percent units.
    """
    sigma2_dec = sigma2_pct / 100.0
    VR_K_K1 , VR_K_K2 , VR_K_K3 = vega_ratio_phis(f,k,k1,k2,k3,sigma2_dec,t)
    ln = log
    denom_k1 = ln(k2/k1) * ln(k3/k1)
    denom_k2 = ln(k2/k1) * ln(k3/k2)
    denom_k3 = ln(k3/k1) * ln(k3/k2)
    if any(np.isclose(d, 0.0) for d in (denom_k1, denom_k2, denom_k3)):
        return None

    x1 = VR_K_K1 * (ln(k2/k)*ln(k3/k)) / denom_k1
    x2 = VR_K_K2 * (ln(k/k1)*ln(k3/k)) / denom_k2
    x3 = VR_K_K3 * (ln(k/k1)*ln(k/k2)) / denom_k3

    # Market vs BS pillar prices
    CMKT_K1 = bs_price_func(sigma1_pct,k1)
    CMKT_K2 = bs_price_func(sigma2_pct,k2)
    CMKT_K3 = bs_price_func(sigma3_pct,k3)

    CBS_K1 = bs_price_func(sigma2_pct, k1)
    CBS_K2 = bs_price_func(sigma2_pct, k2)
    CBS_K3 = bs_price_func(sigma2_pct, k3)
    CBS_K = bs_price_func(sigma2_pct, k)

    return CBS_K + x1 * (CMKT_K1 - CBS_K1) + x2 * (CMKT_K2 - CBS_K2) + x3 * (CMKT_K3 - CBS_K3)

def solve_implied_vol(target_price: float, K: float, bs_price_func,
                      vol_bracket=(1e-6, 500.0), xtol=1e-8, fallback_vol=None) -> float:
    """Invert BS price back to implied vol (percent units)."""
    f_root = lambda vol_pct: bs_price_func(vol_pct, K) - target_price
    vol_min, vol_max = vol_bracket
    try:
        f_min, f_max = f_root(vol_min), f_root(vol_max)
    except Exception:
        return fallback_vol
    if f_min == 0.0: return vol_min
    if f_max == 0.0: return vol_max
    if f_min * f_max > 0:
        return fallback_vol
    try:
        return brentq(f_root, vol_min, vol_max, maxiter=200, xtol=xtol)
    except Exception:
        return fallback_vol




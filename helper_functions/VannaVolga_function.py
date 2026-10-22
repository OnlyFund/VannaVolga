import math
from math import log, exp, sqrt
from scipy.stats import norm
from scipy import optimize
from pricing_utils.BlackScholes import d1
from helper_functions.helper_functions import year_fraction, cp_flag
from functools import lru_cache


# Basic Black-Scholes helpers

def normalize_atm_type(atm_type: str) -> str:
    """Normalize ATM-convention aliases to a canonical string."""
    if atm_type is None:
        return 'dn'
    key = str(atm_type).strip().lower().replace('_', '').replace('-', '')
    aliases = {
        'spot': 'spot',
        'atmf': 'atmf',
        'forward': 'atmf',
        'fwd': 'atmf',
        'atmforward': 'atmf',
        'dn': 'dn',
        'deltaneutral': 'dn',
        'delta-neutral': 'dn',
    }
    if key not in aliases:
        raise ValueError(f"Unknown ATM type: {atm_type}")
    return aliases[key]


def delta_call(F: float, K: float, v: float, t: float) -> float:
    """Call delta using Black-Scholes formula."""
    sqrt_t = sqrt(t)
    d1_ = (log(F / K) + 0.5 * v**2 * t) / (v * sqrt_t)
    return norm.cdf(d1_)

def delta_put(F: float, K: float, v: float, t: float) -> float:
    """Put delta using Black-Scholes formula."""
    return delta_call(F, K, v, t) - 1


# ATM Strike Calculations

@lru_cache(maxsize=256)
def get_atm_strike(atm_type: str, S: float, fwd: float, v: float, expiry_date, valuation_date,
                   fwd_divisor=10000, delta_base: bool = False) -> float:
    """
    Compute ATM strike based on convention: 'spot', 'atmf', 'dn' (delta-neutral).
    The ATM convention is normalized case-insensitively so that similar aliases are
    treated consistently across the repo.

    ``delta_base`` selects the delta convention and only affects the ``'dn'``
    convention: the unadjusted delta-neutral straddle solves ``d1 = 0`` while the
    premium-adjusted one solves ``d2 = 0``, which sits lower by roughly
    ``F * sigma**2 * T``.  ``'spot'`` and ``'atmf'`` are independent of it.
    """
    atm_type = normalize_atm_type(atm_type)
    t = year_fraction(expiry_date, valuation_date)
    v_ = v / 100
    atm_fwd = S + fwd / fwd_divisor
    if atm_fwd <= 0:
        raise ValueError("Forward must be positive")

    if atm_type == 'spot':
        return S
    elif atm_type == 'atmf':
        return atm_fwd
    elif atm_type == 'dn':
        if t <= 0 or v_ <= 0:
            return atm_fwd
        # Delta-neutral forward ATM has d1 = 0 (d2 = 0 when premium-adjusted),
        # so solving a narrow numerical bracket is unnecessary and fails for
        # long/high-vol tenors.
        return atm_fwd * exp((-0.5 if delta_base else 0.5) * v_ ** 2 * t)
    else:
        raise ValueError(f"Unknown ATM type: {atm_type}")


# Strike from Delta

def get_strike_from_delta(S: float, fwd: float, v: float, expiry_date, valuation_date,
                          CallPut: str, delta: float, atm_type='dn',
                          delta_base: bool = False, fwd_divisor=10000,
                          vol_in_percent: bool = True) -> float:
    """
    Approximate strike from a target FX delta using the same forward-delta convention
    used in the pricing engine.

    `delta_base=False` => unadjusted forward delta
    `delta_base=True` => premium-adjusted forward delta
    """
    c_p = cp_flag(CallPut.upper())
    t = year_fraction(expiry_date, valuation_date)
    v_pct = v if vol_in_percent else v * 100.0
    v_ = v_pct / 100.0
    forward = S + fwd / fwd_divisor
    if (
        not all(math.isfinite(value) for value in (S, fwd, v_pct, t, forward, delta))
        or forward <= 0
        or v_ <= 0
        or t <= 0
        or not 0.0 < delta < 0.5
    ):
        raise ValueError(
            "Spot, forward, volatility, expiry, and delta must define a valid "
            "0-to-50 delta strike"
        )
    atm_strike = get_atm_strike(
        atm_type, S, fwd, v_pct, expiry_date, valuation_date, fwd_divisor, delta_base
    )

    # Use the standard BS approximation as the initial guess and then refine
    # against BlackScholes.delta(). The convention is pinned to forward here
    # rather than inherited: these strikes define the quoted smile pillars, so
    # letting the reporting default leak in would silently move the whole
    # surface. Changing the pillar quoting convention is a deliberate,
    # separately validated decision.
    K_guess = atm_strike * exp(-c_p * norm.ppf(delta) * v_ * sqrt(t) + 0.5 * v_**2 * t)
    from pricing_utils.BlackScholes import BlackScholes

    def objective(K: float) -> float:
        bs = BlackScholes(
            S=S,
            K=K,
            fwd_pts=fwd,
            v=v_pct,
            expiry_date=expiry_date,
            valuation_date=valuation_date,
            CallPut=CallPut,
            BuySell='BUY',
            Notional=1,
            deltaBase=delta_base,
            fwd_divisor=fwd_divisor,
            delta_convention="forward",
        )
        target_delta = delta if c_p > 0 else -delta
        return bs.delta() - target_delta

    # Expand around the analytic approximation until a validated root is
    # bracketed. Fixed narrow brackets fail for long-dated/high-volatility
    # inputs and must not silently return an unverified strike.
    lo = max(K_guess * 0.8, 1e-8)
    hi = K_guess * 1.2
    for _ in range(20):
        try:
            objective_lo = objective(lo)
            objective_hi = objective(hi)
        except Exception as exc:
            raise ValueError("Unable to evaluate delta-strike bracket") from exc
        if objective_lo == 0.0:
            return lo
        if objective_hi == 0.0:
            return hi
        if objective_lo * objective_hi < 0.0:
            try:
                return optimize.brentq(objective, lo, hi)
            except Exception as exc:
                raise ValueError("Delta-strike root finding failed") from exc
        lo = max(lo * 0.5, 1e-12)
        hi *= 2.0
    raise ValueError("Unable to bracket a strike for the requested delta")


# Delta-neutral Strike

def get_dn_strike(S: float, fwd: float, v: float, expiry_date: str, valuation_date: str,
                  deltaBase=True, fwd_divisor=10000) -> float:
    """
    Compute delta-neutral strike, refined using d1 and optional premium-adjusted
    forward-delta adjustment.
    """
    f = S + fwd / fwd_divisor
    t = year_fraction(expiry_date, valuation_date)
    if t <= 0 or v <= 0 or f <= 0:
        raise ValueError("Forward, volatility, and expiry must be positive")
    return get_atm_strike('dn', S, fwd, v, expiry_date, valuation_date,
                          fwd_divisor, deltaBase)




# Optional: Strike refinement with Black-Scholes PA method

def get_strike_from_delta_pa(
        S: float, fwd: float, v: float,
        expiry_date, valuation_date,
        CallPut: str, delta: float, atm_type='forward', delta_base: bool = True,
        fwd_divisor=10000, vol_in_percent: bool = True
    ) -> float:
    """
    Refine strike using the premium-adjusted forward-delta convention.
    """
    return get_strike_from_delta(
        S=S,
        fwd=fwd,
        v=v,
        expiry_date=expiry_date,
        valuation_date=valuation_date,
        CallPut=CallPut,
        delta=delta,
        atm_type=atm_type,
        delta_base=delta_base,
        fwd_divisor=fwd_divisor,
        vol_in_percent=vol_in_percent,
    )


# Backwards-compatible aliases for legacy imports used across the repo.
GetStrikeFromDelta = get_strike_from_delta
GetStrikeFromDeltaPA = get_strike_from_delta_pa
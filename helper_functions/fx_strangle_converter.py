"""
fx_strangle_converter.py

One public function, `broker_to_twovol_bf`, converts a broker (market) strangle
quote at a single delta into the two-vol (smile) strangle. Scalar in, scalar out,
no pandas -- built to sit inside a Vanna-Volga vol-for-strike solver.

Inputs use your conventions: v_atm / rr / bf_broker / depo in PERCENT (9.2 = 9.2%) and
every returned vol is in PERCENT too; fwd_pts in market pips,
dates as 'mm/dd/yyyy' strings, delta in points (25, 10).

Timing
    T       = (expiry   - valuation) / 365      option time, always ACT/365
    T_depo  = (delivery - valuation) / basis    depo accrual, per-currency basis
    df_d    = 1/(1 + r*T_depo)  (simple)  or  exp(-r*T_depo)  (continuous)
    F       = spot + fwd_pts/pip_factor         forward for delivery
    df_f    = df_d * F / spot                   covered interest parity
Delta is spot (default) or forward, regular or premium-adjusted.

Procedure
    1. sigma_MS = ATM + BF_broker, the common broker strangle volatility.
    2. Fix broker strikes at +/-delta using sigma_MS and their target premium.
    3. For each trial sigma_2v, build a three-pivot Vanna-Volga smile with
       wing vols sigma_2v +/- RR/2 and their own delta strikes.
    4. Match that smile's premium at the FIXED broker strikes.
    5. bf_2v = sigma_2v - ATM.

Conversion is model-dependent. This helper uses three-pivot, price-based
Vanna-Volga; a different smile construction must match its own broker premiums.

Note: the result depends only on market inputs, NOT on the strike you are asking
a vol for. Call it once per smile, outside your strike-solver loop.
"""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass
from datetime import date, datetime
from typing import Union

from scipy.special import ndtri
from scipy.optimize import brentq

__all__ = ["broker_to_twovol_bf", "TwoVolSmile", "ValidationError", "ConvergenceError"]


class ValidationError(ValueError):
    """Bad input, caught before any solve."""


class ConvergenceError(RuntimeError):
    """A root-find failed (includes premium-adjusted-call wrong-branch / no valid strike)."""


@dataclass(frozen=True)
class TwoVolSmile:
    # vols in PERCENT (same units as the inputs)
    bf_2v: float                  # sigma_2v - v_atm : the smile fly for Vanna-Volga
    sigma_2v: float
    sigma_c: float                # sigma_2v + RR/2
    sigma_p: float                # sigma_2v - RR/2
    strangle_adjustment: float    # bf_2v - bf_broker
    # pillar strikes consistent with the smile vols (use these as VV pivots)
    K_c: float
    K_p: float
    K_atm: float
    # broker-basis strikes and premium, for reference / diagnostics
    K_c_broker: float
    K_p_broker: float
    V_broker: float
    # market data resolved along the way
    F: float
    T: float
    df_d: float
    df_f: float


# ----------------------------------------------------------------------------
# Scalar Garman-Kohlhagen helpers (math-only, fast enough for solver inner loops)
# ----------------------------------------------------------------------------

_SQRT2 = math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _N(x):
    return 0.5 * math.erfc(-x / _SQRT2)


def _n(x):
    return _INV_SQRT_2PI * math.exp(-0.5 * x * x)


def _d1d2(F, K, vol, T):
    s = vol * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * s * s) / s
    return d1, d1 - s


def _price(F, K, vol, T, df_d, phi):
    d1, d2 = _d1d2(F, K, vol, T)
    return df_d * phi * (F * _N(phi * d1) - K * _N(phi * d2))


def _vega(F, K, vol, T, df_d):
    d1, _ = _d1d2(F, K, vol, T)
    return df_d * F * _n(d1) * math.sqrt(T)


def _delta(F, K, vol, T, df_f, phi, pa):
    d1, d2 = _d1d2(F, K, vol, T)
    if pa:
        return phi * df_f * (K / F) * _N(phi * d2)
    return phi * df_f * _N(phi * d1)


def _ddelta_dK(F, K, vol, T, df_f, phi, pa):
    d1, d2 = _d1d2(F, K, vol, T)
    s = vol * math.sqrt(T)
    if pa:
        return (df_f / F) * (phi * _N(phi * d2) - _n(d2) / s)
    return -df_f * _n(d1) / (K * s)


def _strike_from_delta(F, vol, T, df_f, target, phi, pa, tol=1e-12, max_iter=60):
    """Solve K: Delta(K, vol) = target (signed). PA calls must land on the decreasing branch."""
    z = min(max(phi * target / df_f, 1e-12), 1.0 - 1e-12)
    d1 = phi * ndtri(z)
    K = F * math.exp(0.5 * vol * vol * T - d1 * vol * math.sqrt(T))   # exact seed for regular delta

    resid = float("inf")
    for _ in range(max_iter):
        resid = _delta(F, K, vol, T, df_f, phi, pa) - target
        if abs(resid) < tol:
            break
        jac = _ddelta_dK(F, K, vol, T, df_f, phi, pa)
        if jac == 0.0:
            break
        K = max(K - resid / jac, 1e-10)
    resid = _delta(F, K, vol, T, df_f, phi, pa) - target

    if abs(resid) >= tol:
        raise ConvergenceError(f"strike solve failed: target delta {target:+.4f}, vol {vol:.4%}, "
                               f"residual {resid:.2e}")
    if pa and phi > 0 and _ddelta_dK(F, K, vol, T, df_f, phi, pa) >= 0.0:
        raise ConvergenceError(f"premium-adjusted call: target delta {target:.4f} at vol {vol:.4%} "
                               f"has no strike on the valid (decreasing-delta) branch")
    return K


# ----------------------------------------------------------------------------
# Input parsing
# ----------------------------------------------------------------------------

def _to_date(x, name, fmt):
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    try:
        return datetime.strptime(str(x).strip(), fmt).date()
    except ValueError as e:
        raise ValidationError(f"{name}: cannot parse {x!r} as {fmt}") from e


def _parse_basis(x):
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        if float(x) in (360.0, 365.0):
            return float(x)
        raise ValidationError(f"basis: unsupported day-count {x!r} (360 or 365)")
    m = re.fullmatch(r"(?:ACT/?)?(360|365)(?:F|FIXED)?", str(x).upper().replace(" ", ""))
    if not m:
        raise ValidationError(f"basis: unrecognized day-count {x!r}")
    return float(m.group(1))


def _parse_bool(x):
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in ("true", "t", "1", "yes", "y"):
        return True
    if s in ("false", "f", "0", "no", "n"):
        return False
    raise ValidationError(f"is_pa: unrecognized boolean {x!r}")


_ATM = {"dn": "dn", "deltaneutral": "dn", "dns": "dn", "atmdn": "dn",
        "f": "fwd", "fwd": "fwd", "forward": "fwd", "atmf": "fwd", "atmfwd": "fwd",
        "spot": "spot", "atms": "spot", "atmspot": "spot"}


def _parse_atm(x):
    k = re.sub(r"[\s_\-]", "", str(x).lower())
    if k not in _ATM:
        raise ValidationError(f"atm_conv: unrecognized value {x!r}")
    return _ATM[k]


def _finite(name, v, positive=False):
    if not math.isfinite(v) or (positive and v <= 0):
        raise ValidationError(f"{name}: {v!r} is not a finite{' positive' if positive else ''} number")


# ----------------------------------------------------------------------------
# The one public function
# ----------------------------------------------------------------------------

def broker_to_twovol_bf(
    spot: float,
    fwd_pts: float,
    pip_factor: float,
    v_atm: float,
    rr: float,
    bf_broker: float,
    delta: float,
    valuation: Union[str, date],
    expiry: Union[str, date],
    delivery: Union[str, date],
    basis: Union[int, float, str],
    depo_pct: float,
    is_pa: Union[bool, str] = False,
    atm_conv: str = "dn",
    compounding: str = "simple",
    date_format: str = "%m/%d/%Y",
    refine_strikes: bool = True,
    outer_tol: float = 1e-10,
    outer_max_iter: int = 100,
    delta_convention: str = "spot",
) -> TwoVolSmile:
    """
    Convert a broker strangle (v_atm, rr, bf_broker at `delta`) to the two-vol
    smile strangle. Vols / rr / bf_broker / depo_pct in percent; delta in points
    (25 -> 25-delta). Returns TwoVolSmile (vols in percent).

    The matching smile is three-pivot, price-based Vanna-Volga. Broker strikes
    stay fixed; candidate wing strikes always use their own smile volatilities.
    ``refine_strikes`` is retained for compatibility but must be True.
    ``outer_tol`` is the root tolerance in decimal volatility units.

    Raises ValidationError on bad inputs, ConvergenceError if any solve fails
    (including a premium-adjusted call delta that is unattainable).
    """
    for nm, v, pos in (("spot", spot, True), ("pip_factor", pip_factor, True), ("v_atm", v_atm, True),
                       ("fwd_pts", fwd_pts, False), ("rr", rr, False), ("bf_broker", bf_broker, False),
                       ("depo_pct", depo_pct, False), ("delta", delta, True)):
        _finite(nm, float(v), pos)
    if v_atm < 0.5:
        warnings.warn(f"v_atm={v_atm} looks like a decimal; inputs must be in PERCENT (9.2 = 9.2%)",
                      stacklevel=2)
    if not 0.0 < delta < 50.0:
        raise ValidationError(f"delta: {delta!r} must be in points, between 0 and 50 (e.g. 25, 10)")

    pa = _parse_bool(is_pa)
    if delta_convention not in ("spot", "forward"):
        raise ValidationError("delta_convention must be 'spot' or 'forward'")
    if refine_strikes is not True:
        raise ValidationError("refine_strikes must be True for a delta-consistent two-vol smile")
    _finite("outer_tol", outer_tol, positive=True)
    if isinstance(outer_max_iter, bool) or not isinstance(outer_max_iter, int) or outer_max_iter <= 0:
        raise ValidationError("outer_max_iter must be a positive integer")
    atm_kind = _parse_atm(atm_conv)
    basis_d = _parse_basis(basis)
    comp = str(compounding).strip().lower()
    if comp not in ("simple", "continuous"):
        raise ValidationError(f"compounding: {compounding!r} (use 'simple' or 'continuous')")

    d_val = _to_date(valuation, "valuation", date_format)
    d_exp = _to_date(expiry, "expiry", date_format)
    d_dlv = _to_date(delivery, "delivery", date_format)
    if d_exp <= d_val:
        raise ValidationError("expiry must be after valuation")
    if d_dlv < d_exp:
        raise ValidationError("delivery must not be before expiry")

    T = (d_exp - d_val).days / 365.0
    T_depo = (d_dlv - d_val).days / basis_d
    r = depo_pct / 100.0
    if comp == "simple" and 1.0 + r * T_depo <= 0.0:
        raise ValidationError("deposit rate and tenor imply a non-positive discount factor")
    df_d = 1.0 / (1.0 + r * T_depo) if comp == "simple" else math.exp(-r * T_depo)
    _finite("df_d", df_d, positive=True)
    F = spot + fwd_pts / pip_factor
    if F <= 0.0:
        raise ValidationError(f"forward {F} is not positive; check fwd_pts / pip_factor")
    df_f = df_d * F / spot
    delta_factor = df_f if delta_convention == "spot" else 1.0
    if delta / 100.0 >= delta_factor:
        raise ValidationError("target delta must be less than the delta discount factor")

    atm, RR, BF, dl = v_atm / 100.0, rr / 100.0, bf_broker / 100.0, delta / 100.0

    v_ms = atm + BF
    _finite("broker strangle volatility", v_ms, positive=True)
    Kc_b = _strike_from_delta(F, v_ms, T, delta_factor, +dl, +1.0, pa)
    Kp_b = _strike_from_delta(F, v_ms, T, delta_factor, -dl, -1.0, pa)
    V = _price(F, Kc_b, v_ms, T, df_d, 1.0) + _price(F, Kp_b, v_ms, T, df_d, -1.0)

    if atm_kind == "dn":
        K_atm = F * math.exp((-0.5 if pa else 0.5) * atm * atm * T)
    elif atm_kind == "fwd":
        K_atm = F
    else:
        K_atm = spot

    def candidate(v2):
        vc, vp = v2 + RR / 2.0, v2 - RR / 2.0
        if min(vc, vp) <= 0.0:
            raise ConvergenceError("candidate two-vol wing volatilities must be positive")
        kc = _strike_from_delta(F, vc, T, delta_factor, +dl, +1.0, pa)
        kp = _strike_from_delta(F, vp, T, delta_factor, -dl, -1.0, pa)
        if not kp < K_atm < kc:
            raise ConvergenceError("candidate smile requires put < ATM < call strikes")
        return [(kp, vp * 100.0), (K_atm, v_atm), (kc, vc * 100.0)]

    def residual(v2):
        pivots = candidate(v2)
        value = (
            _vv_price_for_strike(Kc_b, F, T, df_d, pivots, 1.0)
            + _vv_price_for_strike(Kp_b, F, T, df_d, pivots, -1.0)
            - V
        )
        if not math.isfinite(value):
            raise ConvergenceError("non-finite candidate broker premium residual")
        return value

    seed = max(v_ms, abs(RR) / 2.0 + atm * 0.01)
    samples = {}
    last_error = None
    bracket = None
    width = max(atm * 0.02, 1e-4)
    for iteration in range(outer_max_iter):
        trials = [seed] if iteration == 0 else [seed - width, seed + width]
        for trial in trials:
            if trial <= abs(RR) / 2.0:
                continue
            try:
                value = residual(trial)
            except ConvergenceError as exc:
                # Invalid candidate smiles are outside the admissible search domain.
                last_error = exc
                continue
            samples[trial] = value
            if abs(value) <= 1e-12 * max(1.0, abs(V)):
                bracket = (trial, trial)
                break
        if bracket is None:
            ordered = sorted(samples)
            bracket = next(
                ((lo, hi) for lo, hi in zip(ordered, ordered[1:])
                 if samples[lo] * samples[hi] < 0.0),
                None,
            )
        if bracket is not None:
            break
        width *= 1.6
        if width > max(2.0, seed * 10.0):
            break
    if bracket is None:
        raise ConvergenceError("unable to bracket a broker-premium-matching two-vol smile") from last_error
    if bracket[0] == bracket[1]:
        v2 = bracket[0]
    else:
        try:
            v2 = brentq(residual, *bracket, xtol=outer_tol, maxiter=outer_max_iter)
        except (ValueError, RuntimeError) as exc:
            raise ConvergenceError("broker-premium-matching butterfly solve failed") from exc
    pivots = candidate(v2)
    Kp, Kc = pivots[0][0], pivots[2][0]
    if abs(residual(v2)) > 1e-9 * max(1.0, abs(V)):
        raise ConvergenceError("two-vol smile does not reproduce the broker premium")

    return TwoVolSmile(
        bf_2v=(v2 - atm) * 100.0,
        sigma_2v=v2 * 100.0,
        sigma_c=(v2 + RR / 2.0) * 100.0,
        sigma_p=(v2 - RR / 2.0) * 100.0,
        strangle_adjustment=(v2 - atm - BF) * 100.0,
        K_c=Kc, K_p=Kp, K_atm=K_atm,
        K_c_broker=Kc_b, K_p_broker=Kp_b, V_broker=V,
        F=F, T=T, df_d=df_d, df_f=df_f,
    )


# ----------------------------------------------------------------------------
# USE CASE: broker fly -> two-vol fly -> Vanna-Volga vol for strike
# ----------------------------------------------------------------------------

def _vv_price_for_strike(K, F, T, df_d, pivots, phi=1.0):
    """Three-pivot Vanna-Volga price; pivot vols are in percent."""
    (K1, v1), (K2, v2), (K3, v3) = pivots
    v1, v2, v3 = v1 / 100.0, v2 / 100.0, v3 / 100.0
    va = v2                                   # base (ATM) vol
    vk = _vega(F, K, va, T, df_d)
    l = math.log
    x1 = vk / _vega(F, K1, va, T, df_d) * (l(K2 / K) * l(K3 / K)) / (l(K2 / K1) * l(K3 / K1))
    x2 = vk / _vega(F, K2, va, T, df_d) * (l(K / K1) * l(K3 / K)) / (l(K2 / K1) * l(K3 / K2))
    x3 = vk / _vega(F, K3, va, T, df_d) * (l(K / K1) * l(K / K2)) / (l(K3 / K1) * l(K3 / K2))
    cost = sum(x * (_price(F, Ki, vi, T, df_d, 1.0) - _price(F, Ki, va, T, df_d, 1.0))
               for x, Ki, vi in ((x1, K1, v1), (x2, K2, v2), (x3, K3, v3)))
    return _price(F, K, va, T, df_d, phi) + cost


def _vv_vol_for_strike(K, F, T, df_d, pivots):
    """Invert the matching three-pivot Vanna-Volga price to percent volatility."""
    target = _vv_price_for_strike(K, F, T, df_d, pivots)
    va = pivots[1][1] / 100.0
    vol = va                                  # invert VV price -> implied vol (Newton)
    for _ in range(50):
        r = _price(F, K, vol, T, df_d, 1.0) - target
        if abs(r) < 1e-12:
            return vol * 100.0
        vega = _vega(F, K, vol, T, df_d)
        if vega <= 0.0:
            raise ConvergenceError("Vanna-Volga implied-volatility solve has zero vega")
        vol = max(vol - r / vega, 1e-6)
    raise ConvergenceError("Vanna-Volga implied-volatility solve did not converge")


if __name__ == "__main__":
    # 1M USDJPY, all vols in percent
    mkt = dict(spot=150.10, fwd_pts=15, pip_factor=100,
               valuation="09/27/2026", expiry="10/27/2026", delivery="10/29/2026",
               basis=360, depo_pct=3.9, is_pa=False, atm_conv="DNS")
    v_atm, rr25, bf25_broker = 9.2, 2.1, 0.28

    # Step 1 (once per smile): broker fly -> two-vol fly and pivot strikes
    sm = broker_to_twovol_bf(v_atm=v_atm, rr=rr25, bf_broker=bf25_broker, delta=25, **mkt)
    print(f"broker BF {bf25_broker:.4f}  ->  two-vol BF {sm.bf_2v:.4f}   "
          f"(adjustment {sm.strangle_adjustment:+.4f} vol pts)")
    print(f"pivots: put {sm.K_p:.4f} @ {sm.sigma_p:.3f}% | atm {sm.K_atm:.4f} @ {v_atm:.3f}% "
          f"| call {sm.K_c:.4f} @ {sm.sigma_c:.3f}%")

    pivots = [(sm.K_p, sm.sigma_p), (sm.K_atm, v_atm), (sm.K_c, sm.sigma_c)]
    v_chk = (
        _vv_price_for_strike(sm.K_c_broker, sm.F, sm.T, sm.df_d, pivots, 1.0)
        + _vv_price_for_strike(sm.K_p_broker, sm.F, sm.T, sm.df_d, pivots, -1.0)
    )
    print(f"premium check: V_broker {sm.V_broker:.8f} vs two-vol smile {v_chk:.8f}\n")

    # Step 2: Vanna-Volga vol for strike, using the converted pivots
    # For contrast: feeding the raw broker fly straight in as if it were the smile fly
    naive = [(sm.K_p, v_atm + bf25_broker - rr25 / 2), (sm.K_atm, v_atm), (sm.K_c, v_atm + bf25_broker + rr25 / 2)]

    print(f"{'strike':>9} {'VV vol (2v BF)':>15} {'VV vol (raw BF)':>16} {'diff':>8}")
    for K in (145.0, 147.0, 149.0, sm.K_atm, 152.0, 154.0, 156.0):
        a = _vv_vol_for_strike(K, sm.F, sm.T, sm.df_d, pivots)
        b = _vv_vol_for_strike(K, sm.F, sm.T, sm.df_d, naive)
        print(f"{K:9.3f} {a:15.4f} {b:16.4f} {a - b:+8.4f}")
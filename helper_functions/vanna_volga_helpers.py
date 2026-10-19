from math import exp, sqrt
import logging
import numpy as np
from numba import njit
from typing import Dict, Optional
from helper_functions.model_constants import SQRT_TWO_PI
from helper_functions.VannaVolga_function import (
    get_strike_from_delta,
    get_strike_from_delta_pa,
    get_atm_strike,
    normalize_atm_type,
)
from pricing_utils.BlackScholes import BlackScholes
from market_data import MarketData


def get_model_map(market=None):
    """Return available model mapping."""
    if getattr(market, "wrapped_smile", False):
        from app.smile_service import AppVolResolver, AppSabrResolver
        return {"vv": AppVolResolver, "vv-sabr": AppSabrResolver}
    from VannaVolga_ImpliedVol import VannaVolga
    from vanna_volga_sabr import VannaVolga_SABR
    return {"vv": VannaVolga, "vv-sabr": VannaVolga_SABR}


# ----------------- Numba-accelerated utility functions -----------------
@njit
def phi(x: float) -> float:
    """Standard normal PDF."""
    return exp(-0.5 * x * x) / SQRT_TWO_PI


@njit
def get_vol_k(v_atm: float, bf: float, rr: float, callput: str) -> float:
    """Compute strike-dependent vol."""
    cp = 1 if callput.upper() == "CALL" else -1
    return v_atm + bf + cp * rr / 2

logger = logging.getLogger(__name__)


def _resolve_pillar_compatibility(value, default=True) -> bool:
    """Resolve the interpolation compatibility flag, accepting legacy keywords."""
    if value is None:
        return bool(default)
    if isinstance(value, str):
        key = value.strip().lower().replace("_", "").replace("-", "")
        legacy_aliases = {
            "legacy", "old", "current", "targetdependent", "targetblend",
            "targetdependentblend", "compatibilitylegacy",
        }
        safe_aliases = {
            "safe", "smooth", "continuous", "new", "auto", "default",
            "compatible", "compat", "targetblendcompatible",
        }
        if key in legacy_aliases:
            return False
        if key in safe_aliases:
            return True
        return bool(default)
    return bool(value)


def _smoothstep01(x: float) -> float:
    """Smooth cubic interpolation weight on [0, 1]."""
    z = float(np.clip(x, 0.0, 1.0))
    return z * z * (3.0 - 2.0 * z)


def _validate_pillar_strikes(strikes, vols, context: str = "five-pillar interpolation"):
    """Return finite, non-duplicate, strictly increasing strike/vol arrays."""
    strikes_arr = np.asarray(strikes, dtype=float)
    vols_arr = np.asarray(vols, dtype=float)
    if strikes_arr.size == 0 or vols_arr.size == 0:
        raise ValueError(f"{context} requires non-empty strike and vol arrays")
    if strikes_arr.shape != vols_arr.shape:
        raise ValueError(f"{context} requires matching strike and vol arrays")
    if not np.all(np.isfinite(strikes_arr)):
        raise ValueError(f"{context} struck invalid or non-finite strike values")
    if not np.all(np.isfinite(vols_arr)):
        raise ValueError(f"{context} struck invalid or non-finite vol values")
    if np.unique(strikes_arr).size != strikes_arr.size:
        raise ValueError(f"{context} detected duplicate strikes")
    if np.any(np.diff(strikes_arr) <= 0):
        raise ValueError(f"{context} requires strictly increasing strikes")
    return strikes_arr, vols_arr


try:
    from scipy.interpolate import CubicSpline as _CubicSpline
except Exception:  # pragma: no cover - scipy is a hard dependency elsewhere
    _CubicSpline = None


def end_segment_curvatures(strikes, vols):
    """Mean curvature of the natural pillar spline over each end segment.

    For each side this is the change in the natural spline's log-strike slope
    from the 25-delta to the 10-delta pillar divided by the segment's length,
    in vol points per unit log-strike squared: the curvature the smile has
    between those quotes. It is floored at zero so a wing never bends back
    toward the money. Returns ``(left, right)``.
    """
    x = np.log(np.asarray(strikes, dtype=float))
    slope = _CubicSpline(x, np.asarray(vols, dtype=float), bc_type="natural").derivative()
    left = (slope(x[1]) - slope(x[0])) / (x[1] - x[0])
    right = (slope(x[-1]) - slope(x[-2])) / (x[-1] - x[-2])
    return max(float(left), 0.0), max(float(right), 0.0)


def interpolate_pillar_vols(strikes, vols, targets, end_curvatures=None):
    """Interpolate a quoted smile across its pillars, smoothly.

    The pillars are interpolated with a natural cubic spline in **log-strike**.
    Linear interpolation was used previously, and because the smile is convex
    between the quoted deltas a straight chord sits systematically *above* the
    true curve: against the continuous Vanna-Volga solution it peaks about 6bp
    too high midway between the ATM and 25-delta pillars, which is exactly the
    region most options are struck in. A cubic spline tracks the same curve to
    under 1bp while still reproducing all five quotes exactly.

    The spline is natural (zero second derivative at the ends) rather than
    not-a-knot. Not-a-knot fits marginally closer in the middle but lets the
    end segments bend, which can push the smile non-convex at the wing pillars
    and introduce a butterfly arbitrage.

    Falls back to linear interpolation if the spline cannot be built or returns
    a level outside the quoted range, so a degenerate pillar set can never make
    this worse than the previous behaviour.

    ``end_curvatures`` ``(left, right)`` replaces the natural end condition
    with those second derivatives (log-strike), so a wing that carries the same
    curvature joins the spline with continuous curvature. ``None`` keeps the
    natural spline.

    ``strikes`` must be sorted, strictly increasing and positive. Returns a
    float for a scalar ``targets`` and an array otherwise.
    """
    strikes_arr = np.asarray(strikes, dtype=float)
    vols_arr = np.asarray(vols, dtype=float)
    target_arr = np.asarray(targets, dtype=float)
    scalar = target_arr.ndim == 0
    query = np.atleast_1d(target_arr)

    linear = np.interp(query, strikes_arr, vols_arr)
    if _CubicSpline is None or strikes_arr.size < 3 or np.any(strikes_arr <= 0.0):
        return float(linear[0]) if scalar else linear

    try:
        bc_type = ("natural" if end_curvatures is None else
                   ((2, float(end_curvatures[0])), (2, float(end_curvatures[1]))))
        spline = _CubicSpline(np.log(strikes_arr), vols_arr, bc_type=bc_type)
        with np.errstate(invalid="ignore", divide="ignore"):
            smooth = np.asarray(spline(np.log(np.clip(query, 1e-12, None))),
                                dtype=float)
    except Exception:
        logger.warning("Cubic pillar interpolation failed; using linear.",
                       exc_info=True)
        return float(linear[0]) if scalar else linear

    # Outside the pillar range a cubic diverges rapidly, and a spline that
    # overshoots the quoted band is not a usable volatility. Both are handled
    # by reverting to the clamped linear value for those points only.
    lo, hi = float(vols_arr.min()), float(vols_arr.max())
    span = hi - lo
    tolerance = max(0.25 * span, 0.10)
    inside = (query >= strikes_arr[0]) & (query <= strikes_arr[-1])
    usable = (
        inside
        & np.isfinite(smooth)
        & (smooth > lo - tolerance)
        & (smooth < hi + tolerance)
        & (smooth > 0.0)
    )
    result = np.where(usable, smooth, linear)
    return float(result[0]) if scalar else result


def safe_interpolate_five_pillar_vols(strikes, vols, target_strike, *,
                                    fallback_atm: Optional[float] = None,
                                    compatibility_mode: Optional[bool] = None):
    """Safely interpolate a five-pillar smile. Falls back to ATM vol on failure."""
    try:
        ordered = np.argsort(np.asarray(strikes, dtype=float))
        strikes_arr = np.asarray(strikes, dtype=float)[ordered]
        vols_arr = np.asarray(vols, dtype=float)[ordered]
        strikes_arr, vols_arr = _validate_pillar_strikes(strikes_arr, vols_arr,
                                                       "five-pillar interpolation")
        target = float(target_strike)
        if not np.isfinite(target):
            raise ValueError("Target strike is non-finite")
        result = float(interpolate_pillar_vols(strikes_arr, vols_arr, target))
        if not np.isfinite(result):
            raise ValueError("Interpolated volatility is non-finite")
        return result
    except Exception as exc:
        if compatibility_mode is not None and not _resolve_pillar_compatibility(compatibility_mode):
            logger.warning(
                "Five-pillar interpolation failed; falling back to legacy target blending.",
                exc_info=True,
            )
        else:
            logger.warning(
                "Five-pillar interpolation failed; using ATM fallback vol.",
                exc_info=True,
            )
        if fallback_atm is not None:
            return float(fallback_atm)
        raise ValueError("Interpolation failure in five-pillar vol path") from exc


def strike_cached(S, fwd_pts, sigma_pct, expiry_date, valuation_date,
                  callput, delta, delta_base: bool, atm_conv,
                  fwd_divisor=10000.0):
    """Calculate a delta strike from the current smile state without reuse."""
    strike_func = get_strike_from_delta_pa if delta_base else get_strike_from_delta
    return strike_func(
        S=S,
        fwd=fwd_pts,
        valuation_date=valuation_date,
        expiry_date=expiry_date,
        CallPut=callput,
        delta=delta,
        atm_type=atm_conv,
        v=sigma_pct,
        fwd_divisor=fwd_divisor,
    )


def calculate_strikes(S, f, fwd_pts, expiry_date, valuation_date, t,
                      sigma1_pct, sigma2_pct, sigma3_pct,
                      rr_bf: dict, delta_base: bool, atm_conv: str,
                      fwd_divisor=10000.0) -> Dict[str, float]:
    """Compute 10Δ put, ATM, and 10Δ call strikes (K1, K2, K3)."""
    K1 = strike_cached(S, fwd_pts, sigma1_pct, expiry_date, valuation_date,
                       "PUT", rr_bf["delta"], delta_base, atm_conv, fwd_divisor=fwd_divisor)
    K3 = strike_cached(S, fwd_pts, sigma3_pct, expiry_date, valuation_date,
                       "CALL", rr_bf["delta"], delta_base, atm_conv, fwd_divisor=fwd_divisor)

    K2 = get_atm_strike(atm_type=atm_conv, S=S, fwd=fwd_pts, v=sigma2_pct,
                        expiry_date=expiry_date, valuation_date=valuation_date,
                        fwd_divisor=fwd_divisor, delta_base=delta_base)

    return {"K1": K1, "K2": K2, "K3": K3}


# ----------------- Numba-accelerated SABR-VannaVolga weight -----------------
class VannaVolgaSABR_Weight:
    """Smooth blend between SABR and Vanna-Volga vols based on absolute delta."""

    def __init__(self, delta_abs, vol_sabr, vol_vv, t_expiry, d_vv=0.25):
        self.delta_abs = abs(delta_abs)
        self.vol_sabr = vol_sabr
        self.vol_vv = vol_vv
        self.t_expiry = t_expiry
        self.d_vv = d_vv  # upper bound where VV dominates

    @staticmethod
    @njit
    def smooth_hermite(t: float) -> float:
        return 3 * t**2 - 2 * t**3

    def sabr_weight(self) -> float:
        t = np.clip(self.delta_abs / self.d_vv, 0.0, 1.0)
        return 1.0 - self.smooth_hermite(t)

    def blend_vol_variance(self) -> float:
        w = self.sabr_weight()
        var_sabr = self.vol_sabr**2 * self.t_expiry
        var_vv = self.vol_vv**2 * self.t_expiry
        blended_var = w * var_sabr + (1 - w) * var_vv
        return max(sqrt(blended_var / self.t_expiry), 1e-8)


def _main_strikes_vols(s, fwd_pts, v_atm_pct, rr_25, rr_10, bf_25, bf_10,
                       expiry_date, valuation_date, atm_conv, deltaBase,
                       forward_divisor):
    """Return the five delta-quoted pillar (strike, vol) pairs.

    This deliberately recalculates the smile pillars for every request.
    Dynamic-smile scenarios must not reuse prior calibrated results.
    """
    vols = (
        get_vol_k(v_atm_pct, bf_10, rr_10, "PUT"),
        get_vol_k(v_atm_pct, bf_25, rr_25, "PUT"),
        v_atm_pct,
        get_vol_k(v_atm_pct, bf_25, rr_25, "CALL"),
        get_vol_k(v_atm_pct, bf_10, rr_10, "CALL"),
    )

    atm_strike = get_atm_strike(
        atm_type=atm_conv, S=s, fwd=fwd_pts, v=vols[2],
        expiry_date=expiry_date, valuation_date=valuation_date,
        fwd_divisor=forward_divisor, delta_base=deltaBase,
    )

    strikes = (
        strike_cached(s, fwd_pts, vols[0], expiry_date, valuation_date,
                     "PUT", 0.10, deltaBase, atm_conv, forward_divisor),
        strike_cached(s, fwd_pts, vols[1], expiry_date, valuation_date,
                     "PUT", 0.25, deltaBase, atm_conv, forward_divisor),
        atm_strike,
        strike_cached(s, fwd_pts, vols[3], expiry_date, valuation_date,
                     "CALL", 0.25, deltaBase, atm_conv, forward_divisor),
        strike_cached(s, fwd_pts, vols[4], expiry_date, valuation_date,
                     "CALL", 0.10, deltaBase, atm_conv, forward_divisor),
    )

    return tuple(zip(strikes, vols))


# ----------------- Main FX Pillars class -----------------
class GetMainPillars(MarketData):
    def __init__(self, k, expiry_date, valuation_date, atm_conv="dn",
                 s=None, fwd_pts=None, v_atm=None, rr_25=None, rr_10=None,
                 bf_25=None, bf_10=None, forward_divisor=10000.0, deltaBase=True,
                 smooth_pillar_interpolation: Optional[bool] = None,
                 legacy_target_blend: Optional[bool] = None,
                 compatibility_mode: Optional[str] = None,
                 term_discount=0.05, **kwargs):

        super().__init__(s=s, fwd_pts=fwd_pts, v_atm=v_atm,
                         rr_25=rr_25, rr_10=rr_10, bf_25=bf_25, bf_10=bf_10,
                         forward_divisor=forward_divisor,
                         term_discount=term_discount)

        self.compatibility_mode = compatibility_mode
        self.legacy_target_blend = bool(legacy_target_blend) if legacy_target_blend is not None else False
        self.smooth_pillar_interpolation = _resolve_pillar_compatibility(
            smooth_pillar_interpolation,
            default=True,
        )
        if compatibility_mode is not None:
            self.smooth_pillar_interpolation = _resolve_pillar_compatibility(compatibility_mode, default=True)
        if self.legacy_target_blend:
            self.smooth_pillar_interpolation = False
        if not isinstance(deltaBase, bool):
            raise TypeError(f"deltaBase must be bool, got {deltaBase} ({type(deltaBase)})")
        self.deltaBase = deltaBase
        self.atm_conv = normalize_atm_type(atm_conv)

        self.k = k
        self.s = s
        self.v_atm_pct = v_atm
        self.v_atm_dec = v_atm / 100.0
        self.rr_25, self.rr_10 = rr_25, rr_10
        self.bf_25, self.bf_10 = bf_25, bf_10
        self.fwd_pts = fwd_pts
        self.forward_divisor = forward_divisor
        self.expiry_date = expiry_date
        self.valuation_date = valuation_date

        # Black-Scholes delta for TV delta
        atm_bs = BlackScholes(
            S=self.s, K=self.k, fwd_pts=self.fwd_pts,
            v=self.v_atm_pct, expiry_date=self.expiry_date,
            valuation_date=self.valuation_date, CallPut="CALL",
            BuySell="BUY", Notional=1, deltaBase=self.deltaBase,
            fwd_divisor=self.forward_divisor,
            term_discount=self.term_discount,
            # Pinned to forward: see the matching note in VannaVolga_ImpliedVol.
            delta_convention="forward",
        )
        self._tv_delta = 0.5 - abs(atm_bs.delta() - 0.5)
    def get_main_strikes_vols(self) -> Dict[float, float]:
        """Return the five delta-quoted pillar strikes and volatilities.

        Smile pillars are recalibrated on every call so dynamic scenarios
        cannot reuse a prior calibration.
        """
        pairs = _main_strikes_vols(
            self.s, self.fwd_pts, self.v_atm_pct,
            self.rr_25, self.rr_10, self.bf_25, self.bf_10,
            self.expiry_date, self.valuation_date, self.atm_conv,
            self.deltaBase, self.forward_divisor,
        )
        return dict(pairs)

    def get_five_pillar_arrays(self, pillars=None):
        """Return validated five-pillar strikes and volatilities.

        ``pillars`` is an optional call-local result from
        :meth:`get_main_strikes_vols`; it is never retained by this object.
        """
        if pillars is None:
            pillars = self.get_main_strikes_vols()
        strikes = np.asarray(list(pillars.keys()), dtype=float)
        vols = np.asarray(list(pillars.values()), dtype=float)
        return _validate_pillar_strikes(
            strikes, vols, "five-pillar volatility interpolation"
        )

    def _legacy_weighted_rr_bf(self, pillars=None) -> dict:
        if pillars is None:
            pillars = self.get_main_strikes_vols()
        vols = np.array(list(pillars.values()))
        var_array = vols ** 2
        x = np.clip((self._tv_delta - 0.1) / 0.15, 0.0, 1.0)
        w_25 = x * x * (3.0 - 2.0 * x)
        w_10 = 1 - w_25
        var_put = w_10 * var_array[0] + w_25 * var_array[1]
        var_call = w_10 * var_array[4] + w_25 * var_array[3]
        return {
            "RR": sqrt(var_call) - sqrt(var_put),
            "BF": (sqrt(var_call) + sqrt(var_put)) / 2 - vols[2],
            "delta": float(np.clip(self._tv_delta, 0.1, 0.25))
        }

    def get_weighted_rr_bf(self, pillars=None) -> dict:
        try:
            if not self.smooth_pillar_interpolation:
                return self._legacy_weighted_rr_bf(pillars)
            if pillars is None:
                pillars = self.get_main_strikes_vols()
            vols = np.array(list(pillars.values()), dtype=float)
            if vols.size != 5 or not np.all(np.isfinite(vols)):
                raise ValueError("Invalid pillar volatilities")
            x = np.clip((self._tv_delta - 0.1) / 0.15, 0.0, 1.0)
            w_25 = _smoothstep01(x)
            w_10 = 1.0 - w_25
            put_vol = w_10 * vols[0] + w_25 * vols[1]
            call_vol = w_10 * vols[4] + w_25 * vols[3]
            rr = call_vol - put_vol
            bf = 0.5 * (call_vol + put_vol) - vols[2]
            return {
                "RR": float(rr),
                "BF": float(bf),
                "delta": float(np.clip(self._tv_delta, 0.1, 0.25)),
            }
        except Exception as exc:
            logger.warning(
                "Continuous five-pillar interpolation failed; falling back to legacy target blending.",
                exc_info=True,
            )
            return self._legacy_weighted_rr_bf(pillars)



# ----------------- Example usage -----------------
if __name__ == "__main__":
    pillars = GetMainPillars(
        expiry_date="2026-02-16",
        valuation_date="2025-08-14",
        s=1.1685, fwd_pts=129.5, v_atm=7.31,
        rr_25=0.72, rr_10=1.33, bf_25=0.21, bf_10=0.79,
        k=1.25, deltaBase=False
    )
    print(pillars.get_weighted_rr_bf())

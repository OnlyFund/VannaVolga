"""Strike-only adapters for the existing VV and VV-SABR algorithms."""
import math
from numbers import Real

import numpy as np
from scipy.special import ndtr

from helper_functions.fx_strangle_converter import _strike_from_delta
from helper_functions.vanna_volga_helpers import (
    _validate_pillar_strikes, interpolate_pillar_vols,
)
from resolved_smile_inputs import PreparationError


def validate_strike(strike):
    if isinstance(strike, bool) or not isinstance(strike, Real):
        raise ValueError("strike must be a finite positive number")
    if not math.isfinite(float(strike)) or strike <= 0:
        raise ValueError("strike must be a finite positive number")
    return float(strike)


class VannaVolgaCore:
    """Natural-cubic five-pillar smile with curvature-extended wings.

    Beyond each 10-delta pillar the smile keeps its vol and slope and carries
    the mean curvature of the 25-10 delta segment on that side, so skew and
    convexity extend into the wings instead of flattening out. Past
    ``wing_min_delta`` the wing continues linearly, and Lee's moment bound caps
    total variance at ``2|log(K/F)|``.
    """

    wing_min_delta = 0.01
    _step = 1e-4

    def __init__(self, context, pillars):
        self.context, self.pillars = context, pillars
        try:
            self.strikes, self.vols = _validate_pillar_strikes(
                pillars.strikes, pillars.vols, "resolved five-pillar interpolation")
            if len(self.strikes) != 5 or np.any(self.vols <= 0):
                raise ValueError("VV requires five positive-volatility pillars")
            if context.time <= 0 or context.forward <= 0:
                raise ValueError("VV requires positive expiry and forward")
        except (TypeError, ValueError) as exc:
            raise PreparationError("invalid resolved VV pillars: {}".format(exc)) from exc
        self._wings = {}

    def _interior_vol(self, strike):
        try:
            return float(interpolate_pillar_vols(self.strikes, self.vols, strike))
        except (TypeError, ValueError, RuntimeError, FloatingPointError) as exc:
            raise PreparationError("VV pillar interpolation failed") from exc

    def _outward_slope(self, log_strike, inward):
        # One-sided second-order slope (vol points per unit log-strike) taken
        # inside the quoted range and expressed in the outward direction.
        h = self._step
        values = [self._interior_vol(math.exp(log_strike + inward * i * h)) for i in range(3)]
        return (3.0 * values[0] - 4.0 * values[1] + values[2]) / (2.0 * h)

    def _wing(self, edge):
        """Strike-independent wing parameters, built once per side."""
        if edge in self._wings:
            return self._wings[edge]
        inner = 1 if edge == 0 else 3
        inward = 1.0 if edge == 0 else -1.0
        log_edge = math.log(self.strikes[edge])
        log_inner = math.log(self.strikes[inner])
        vol = float(self.vols[edge])
        slope = self._outward_slope(log_edge, inward)
        # Outward slope just outside the 25-delta pillar, toward the wing.
        inner_slope = -self._outward_slope(log_inner, -inward)
        length = abs(log_edge - log_inner)
        curvature = max((slope - inner_slope) / length, 0.0)
        if not all(math.isfinite(x) for x in (slope, curvature)):
            raise PreparationError("VV wing slope or curvature is not finite")
        wing = {"log_edge": log_edge, "vol": vol, "slope": slope,
                "curvature": curvature, "cap": math.inf}
        self._wings[edge] = wing
        wing["cap"] = self._min_delta_distance(edge, wing)
        return wing

    def _min_delta_distance(self, edge, wing):
        """Outward log-distance of the ``wing_min_delta`` strike on this wing."""
        ctx = self.context
        phi = -1 if edge == 0 else 1
        distance = 0.0
        for _ in range(100):
            vol = self._parabola(wing, distance)
            if vol <= 0:
                return math.inf
            try:
                strike = _strike_from_delta(
                    ctx.forward, vol / 100.0, ctx.time, ctx.delta_factor,
                    phi * self.wing_min_delta, phi, ctx.premium_adjusted)
            except (TypeError, ValueError, RuntimeError, OverflowError):
                return math.inf
            if not (math.isfinite(strike) and strike > 0):
                return math.inf
            updated = abs(math.log(strike) - wing["log_edge"])
            if abs(updated - distance) < 1e-12:
                return updated
            distance = updated
        return distance

    @staticmethod
    def _parabola(wing, distance):
        return wing["vol"] + wing["slope"] * distance + 0.5 * wing["curvature"] * distance ** 2

    def _wing_vol(self, strike):
        edge = 0 if strike < self.strikes[0] else 4
        wing = self._wing(edge)
        distance = abs(math.log(strike) - wing["log_edge"])
        cap = wing["cap"]
        if distance <= cap:
            vol = self._parabola(wing, distance)
        else:
            cap_slope = max(wing["slope"] + wing["curvature"] * cap, 0.0)
            vol = self._parabola(wing, cap) + cap_slope * (distance - cap)
        moneyness = abs(math.log(strike / self.context.forward))
        lee_bound = 100.0 * math.sqrt(2.0 * moneyness / self.context.time)
        return min(vol, lee_bound)

    def get_vol_for_strike(self, strike):
        strike = validate_strike(strike)
        if self.strikes[0] <= strike <= self.strikes[-1]:
            vol = self._interior_vol(strike)
        else:
            vol = self._wing_vol(strike)
        if not math.isfinite(vol) or vol <= 0:
            raise PreparationError("VV produced non-positive/non-finite volatility")
        return float(vol)

    def get_vol_for_strikes(self, strikes):
        return np.array([self.get_vol_for_strike(k) for k in strikes])

    def _target_delta(self, strike):
        ctx = self.context
        sigma = float(self.vols[2]) / 100.0
        d1 = (math.log(ctx.forward / strike) + 0.5 * sigma ** 2 * ctx.time) / (
            sigma * math.sqrt(ctx.time))
        if ctx.premium_adjusted:
            d2 = d1 - sigma * math.sqrt(ctx.time)
            delta = strike / ctx.forward * float(ndtr(d2))
        else:
            delta = float(ndtr(d1))
        return 0.5 - abs(delta - 0.5)


class VannaVolgaSABRCore(VannaVolgaCore):
    """Use the legacy calibration and variance blend over explicit pillars."""

    def __init__(self, context, pillars, beta=0.5):
        super().__init__(context, pillars)
        self.beta = beta
        from market_data import MarketData
        from vanna_volga_sabr import VannaVolga_SABR

        rr_25 = float(self.vols[3] - self.vols[1])
        rr_10 = float(self.vols[4] - self.vols[0])
        bf_25 = float(0.5 * (self.vols[3] + self.vols[1]) - self.vols[2])
        bf_10 = float(0.5 * (self.vols[4] + self.vols[0]) - self.vols[2])
        self._market = MarketData(
            s=context.spot,
            fwd_pts=(context.forward - context.spot) * context.forward_divisor,
            v_atm=float(self.vols[2]),
            rr_25=rr_25, rr_10=rr_10, bf_25=bf_25, bf_10=bf_10,
            forward_divisor=context.forward_divisor,
            atm_conv=context.atm_convention,
            deltaBase=context.premium_adjusted,
            term_discount=0.0)
        self._sabr_class = VannaVolga_SABR
        try:
            calibration = self._new_legacy_model(context.forward)
            calibration.smile = dict(zip(self.strikes, self.vols))
            params = calibration.calibrate_sabr()
        except (ValueError, RuntimeError, FloatingPointError, OverflowError) as exc:
            raise PreparationError("legacy SABR calibration failed") from exc
        if params is None or any(param is None for param in params):
            raise PreparationError("legacy SABR calibration did not converge")
        self.parameters = tuple(float(param) for param in params)
        self._calibration = calibration
        fitted = tuple(
            calibration._stable_single_strike_vol(strike)
            for strike in self.strikes)
        if any(vol is None or not math.isfinite(vol) or vol <= 0
               for vol in fitted):
            raise PreparationError("legacy SABR calibration produced invalid vols")
        self.calibration_relative_rms = float(
            np.sqrt(np.mean(((np.asarray(fitted) - self.vols) / self.vols) ** 2)))

    def _new_legacy_model(self, strike):
        return self._sabr_class(
            market=self._market, K=strike,
            expiry_date=self.context.expiry_date.isoformat(),
            valuation_date=self.context.valuation_date.isoformat(),
            CallPut="CALL", beta=self.beta,
            deltaBase=self.context.premium_adjusted,
            atm_conv=self.context.atm_convention,
            frozen_sabr_params=self.parameters if hasattr(self, "parameters") else None)

    def get_vol_for_strike(self, strike):
        strike = validate_strike(strike)
        vv = super().get_vol_for_strike(strike)
        try:
            sabr_value = self._calibration._stable_single_strike_vol(strike)
            if sabr_value is None:
                raise PreparationError("legacy SABR returned no volatility")
            sabr = float(sabr_value)
            if not math.isfinite(sabr) or sabr <= 0:
                raise PreparationError("legacy SABR returned invalid volatility")
            tv_delta = self._target_delta(strike)
            from helper_functions.vanna_volga_helpers import VannaVolgaSABR_Weight
            blended = VannaVolgaSABR_Weight(
                delta_abs=tv_delta, vol_sabr=sabr, vol_vv=vv,
                t_expiry=self.context.time).blend_vol_variance()
        except (ValueError, RuntimeError, FloatingPointError, OverflowError) as exc:
            if isinstance(exc, PreparationError):
                raise
            raise PreparationError("legacy VV-SABR evaluation failed") from exc
        if not math.isfinite(blended) or blended <= 0:
            raise PreparationError("legacy VV-SABR blend produced invalid volatility")
        return float(blended)


def make_core(model, context, pillars, beta=0.5):
    if model == "vv":
        return VannaVolgaCore(context, pillars)
    if model == "vv-sabr":
        return VannaVolgaSABRCore(context, pillars, beta=beta)
    raise ValueError("model must be 'vv' or 'vv-sabr'")

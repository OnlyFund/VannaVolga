"""Strike-only adapters for the existing VV and VV-SABR algorithms."""
import math
from numbers import Real

import numpy as np
from scipy.special import ndtr

from helper_functions.fx_strangle_converter import _strike_from_delta
from helper_functions.vanna_volga_helpers import (
    PillarSpline, _validate_pillar_strikes, end_segment_curvatures, interpolate_pillar_vols,
)
from resolved_smile_inputs import PreparationError


def validate_strike(strike):
    if isinstance(strike, bool) or not isinstance(strike, Real):
        raise ValueError("strike must be a finite positive number")
    if not math.isfinite(float(strike)) or strike <= 0:
        raise ValueError("strike must be a finite positive number")
    return float(strike)


class VannaVolgaCore:
    """Cubic five-pillar smile with controlled, arbitrage-checked wings.

    Outside each 10-delta pillar, the wing starts with the spline's level and
    slope, then decays exponentially toward a finite asymptote on the 10-to-1
    delta log-strike scale. A sampled Black-price check rejects extrapolations
    that violate monotonicity or convexity; a natural cubic extrapolation is
    tried next, followed by the former Lee-bounded wing as a last resort.
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
        try:
            self.end_curvatures = end_segment_curvatures(self.strikes, self.vols)
        except (TypeError, ValueError) as exc:
            raise PreparationError("VV wing curvature failed: {}".format(exc)) from exc
        if not all(math.isfinite(c) for c in self.end_curvatures):
            raise PreparationError("VV wing curvature is not finite")
        self._wings = {}
        self._wing_modes = {}
        self._wing_validation = {}
        self._natural_spline = None

    def _interior_vol(self, strike):
        try:
            # Built once per core (``PillarSpline``); lazily, so a core pickled
            # before it existed (the disk calibration cache) still loads.
            spline = self.__dict__.get("_pillar_spline")
            if spline is None:
                spline = self._pillar_spline = PillarSpline(self.strikes, self.vols,
                                                            self.end_curvatures)
            return float(spline(strike))
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
        inward = 1.0 if edge == 0 else -1.0
        log_edge = math.log(self.strikes[edge])
        vol = float(self.vols[edge])
        slope = self._outward_slope(log_edge, inward)
        # Retain the end curvature for the last-resort legacy wing.
        curvature = self.end_curvatures[0 if edge == 0 else 1]
        if not all(math.isfinite(x) for x in (slope, curvature)):
            raise PreparationError("VV wing slope or curvature is not finite")
        wing = {"log_edge": log_edge, "vol": vol, "slope": slope,
                "curvature": curvature, "cap": math.inf, "decay_scale": math.nan}
        self._wings[edge] = wing
        wing["cap"] = self._min_delta_distance(edge, wing)
        try:
            ctx = self.context
            phi = -1 if edge == 0 else 1
            one_delta_strike = _strike_from_delta(
                ctx.forward, vol / 100.0, ctx.time, ctx.delta_factor,
                phi * 0.01, phi, ctx.premium_adjusted)
            scale = abs(math.log(one_delta_strike) - log_edge)
            if math.isfinite(scale) and scale > 0:
                wing["decay_scale"] = scale
        except (TypeError, ValueError, RuntimeError, OverflowError):
            pass
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

    @staticmethod
    def _decayed_wing_vol(wing, distance):
        scale = wing["decay_scale"]
        if not math.isfinite(scale) or scale <= 0:
            return math.nan
        return wing["vol"] + wing["slope"] * scale * (-math.expm1(-distance / scale))

    def _cubic_wing_vol(self, strike):
        if self._natural_spline is None:
            from scipy.interpolate import CubicSpline
            self._natural_spline = CubicSpline(
                np.log(self.strikes), self.vols, bc_type="natural", extrapolate=True)
        return float(self._natural_spline(math.log(strike)))

    def _wing_candidate_is_valid(self, edge, target_distance, candidate):
        """Check the extrapolated wing's sampled Black prices for convexity."""
        if not math.isfinite(target_distance) or target_distance <= 0:
            return False
        inward = 1.0 if edge == 0 else -1.0
        segment = abs(math.log(self.strikes[1]) - math.log(self.strikes[0])) \
            if edge == 0 else abs(math.log(self.strikes[4]) - math.log(self.strikes[3]))
        start = -segment
        distances = np.linspace(start, target_distance, 129)
        strikes = np.exp(self._wings[edge]["log_edge"] + inward * distances)
        vols = np.asarray([
            self._interior_vol(strike) if distance <= 0 else candidate(strike, distance)
            for strike, distance in zip(strikes, distances)
        ], dtype=float)
        if (not np.all(np.isfinite(vols)) or np.any(vols <= 0)
                or np.any(vols > 500)):
            return False
        forward, time = self.context.forward, self.context.time
        prices = []
        for strike, vol in zip(strikes, vols):
            stdev = vol / 100.0 * math.sqrt(time)
            d1 = (math.log(forward / strike) + 0.5 * stdev ** 2) / stdev
            prices.append(forward * float(ndtr(d1)) - strike * float(ndtr(d1 - stdev)))
        order = np.argsort(strikes)
        prices = np.asarray(prices)[order]
        strikes = strikes[order]
        slopes = np.diff(prices) / np.diff(strikes)
        return bool(np.all(np.isfinite(slopes))
                    and np.all(slopes <= 1e-10)
                    and np.all(np.diff(slopes) >= -1e-8))

    def _wing_vol(self, strike):
        edge = 0 if strike < self.strikes[0] else 4
        wing = self._wing(edge)
        distance = abs(math.log(strike) - wing["log_edge"])
        validated = self._wing_validation.get(edge)
        if (validated is not None and distance <= validated[0]
                and validated[1] != "legacy-lee"):
            self._wing_modes[edge] = validated[1]
            candidate = (self._decayed_wing_vol(wing, distance)
                         if validated[1] == "slope-decay"
                         else self._cubic_wing_vol(strike))
            return float(candidate)

        coverage = max(distance, wing["decay_scale"] * 5.0
                       if math.isfinite(wing["decay_scale"]) else distance)
        coverage = min(coverage, 20.0)
        candidates = (
            ("slope-decay", self._decayed_wing_vol),
            ("natural-cubic", lambda _strike, _distance: self._cubic_wing_vol(_strike)),
        )
        for name, candidate_fn in candidates:
            candidate = lambda k, d: candidate_fn(wing, d) if name == "slope-decay" \
                else candidate_fn(k, d)
            if self._wing_candidate_is_valid(edge, coverage, candidate):
                self._wing_modes[edge] = name
                self._wing_validation[edge] = (coverage, name)
                return float(candidate(strike, distance))

        # Preserve the former Lee-bounded wing as a last-resort valid-vol
        # fallback if both the controlled tail and cubic extrapolation fail.
        cap = wing["cap"]
        if distance <= cap:
            vol = self._parabola(wing, distance)
        else:
            cap_slope = max(wing["slope"] + wing["curvature"] * cap, 0.0)
            vol = self._parabola(wing, cap) + cap_slope * (distance - cap)
        moneyness = abs(math.log(strike / self.context.forward))
        lee_bound = 100.0 * math.sqrt(2.0 * moneyness / self.context.time)
        self._wing_modes[edge] = "legacy-lee"
        self._wing_validation[edge] = (coverage, "legacy-lee")
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

    @property
    def wing_modes(self):
        """Methods currently selected for the put and call extrapolation wings."""
        return dict(self._wing_modes)

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
    if model == "sabr":
        from pricing_utils.sabr_smile import SABRCore
        return SABRCore(context, pillars, beta=beta)
    raise ValueError("model must be 'vv', 'vv-sabr' or 'sabr'")

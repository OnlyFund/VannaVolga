"""Calibrated Hagan-SABR smile interpolation over explicit FX pillars."""
import math

import numpy as np
from scipy.optimize import least_squares
from scipy.special import ndtr

from pricing_utils.sabr_numba import hagan_sabr_log_normal_vol_numba
from resolved_smile_inputs import PreparationError


class SABRCore:
    """Fit five SABR pillars, then use a controlled outer-wing extrapolation."""

    max_rms_error = 0.0025

    def __init__(self, context, pillars, beta=0.5):
        self.context, self.pillars = context, pillars
        if isinstance(beta, bool):
            raise PreparationError("SABR beta must be finite and between 0 and 1")
        try:
            beta = float(beta)
        except (TypeError, ValueError) as exc:
            raise PreparationError("SABR beta must be finite and between 0 and 1") from exc
        if not math.isfinite(beta) or not 0.0 <= beta <= 1.0:
            raise PreparationError("SABR beta must be finite and between 0 and 1")
        self.beta = beta
        try:
            self.strikes = np.asarray(pillars.strikes, dtype=np.float64)
            self.vols = np.asarray(pillars.vols, dtype=np.float64)
        except (AttributeError, TypeError, ValueError) as exc:
            raise PreparationError("invalid SABR pillars") from exc
        if (self.strikes.shape != (5,) or self.vols.shape != (5,)
                or not np.all(np.isfinite(self.strikes))
                or not np.all(np.isfinite(self.vols))
                or np.any(self.strikes <= 0) or np.any(np.diff(self.strikes) <= 0)
                or np.any(self.vols <= 0)):
            raise PreparationError("SABR requires five ordered positive-volatility pillars")
        if (not math.isfinite(context.forward) or context.forward <= 0
                or not math.isfinite(context.time) or context.time <= 0):
            raise PreparationError("SABR requires a positive forward and expiry")

        self.parameters, self.calibration_rms = self._calibrate()
        self.calibration_relative_rms = self.calibration_rms / max(
            float(np.mean(self.vols / 100.0)), 1e-12)
        self._check_pillar_prices()
        self._wings = {}
        self._wing_modes = {}
        self._wing_validation = {}
        self._natural_spline = None

    def _vols_decimal(self, parameters, strikes=None):
        alpha, rho, nu = parameters
        target_strikes = self.strikes if strikes is None else strikes
        try:
            return hagan_sabr_log_normal_vol_numba(
                alpha, self.beta, rho, nu, self.context.forward, self.context.time,
                np.asarray(target_strikes, dtype=np.float64))
        except (ArithmeticError, ValueError, FloatingPointError, OverflowError) as exc:
            raise PreparationError("SABR numerical evaluation failed") from exc

    def _calibrate(self):
        observed = self.vols / 100.0
        forward = self.context.forward
        alpha0 = observed[2] * forward ** (1.0 - self.beta)
        lower = np.array([1e-8, -0.999, 0.0])
        upper = np.array([5.0, 0.999, 5.0])
        best = None

        def residual(parameters):
            try:
                model = self._vols_decimal(parameters)
            except PreparationError:
                return np.full(observed.shape, 1e3)
            if not np.all(np.isfinite(model)):
                return np.full(observed.shape, 1e3)
            return model - observed

        initial_alpha = float(np.clip(alpha0, lower[0] + 1e-10, upper[0] - 1e-10))
        for rho0 in (-0.5, 0.0, 0.5):
            for nu0 in (0.2, 0.7, 1.5):
                try:
                    fit = least_squares(
                        residual, np.array([initial_alpha, rho0, nu0]),
                        bounds=(lower, upper), max_nfev=500,
                        xtol=1e-11, ftol=1e-11, gtol=1e-11)
                except (ValueError, RuntimeError, FloatingPointError, OverflowError,
                        np.linalg.LinAlgError):
                    continue
                error = residual(fit.x)
                rms = float(np.sqrt(np.mean(error ** 2)))
                if fit.success and np.all(np.isfinite(fit.x)) and math.isfinite(rms):
                    if best is None or rms < best[0]:
                        best = rms, tuple(float(value) for value in fit.x)

        if best is None:
            raise PreparationError("SABR calibration did not converge")
        rms, parameters = best
        if rms > self.max_rms_error:
            raise PreparationError(
                "SABR fit error {:.4%} exceeds {:.4%}".format(
                    rms, self.max_rms_error))
        fitted = self._vols_decimal(parameters)
        if (not np.all(np.isfinite(fitted)) or np.any(fitted <= 0)
                or np.max(np.abs(fitted - observed)) > 2.0 * self.max_rms_error):
            raise PreparationError("SABR fit failed pillar-volatility safeguards")
        return parameters, rms

    def _check_pillar_prices(self):
        vols = self._vols_decimal(self.parameters)
        forward, t = self.context.forward, self.context.time
        prices = []
        for strike, vol in zip(self.strikes, vols):
            stdev = vol * math.sqrt(t)
            d1 = (math.log(forward / strike) + 0.5 * stdev ** 2) / stdev
            prices.append(forward * ndtr(d1) - strike * ndtr(d1 - stdev))
        slopes = [(b - a) / (kb - ka) for a, b, ka, kb in
                  zip(prices, prices[1:], self.strikes, self.strikes[1:])]
        if (not all(math.isfinite(slope) and slope < 0 for slope in slopes)
                or not all(right > left for left, right in zip(slopes, slopes[1:]))):
            raise PreparationError("SABR fitted pillar prices fail the no-arbitrage check")

    def _native_vol(self, strike):
        vol = float(self._vols_decimal(self.parameters, [strike])[0]) * 100.0
        if not math.isfinite(vol) or not 0 < vol <= 500:
            raise PreparationError("SABR volatility is outside the supported finite range")
        return vol

    def _wing(self, edge):
        if edge in self._wings:
            return self._wings[edge]
        log_edge = math.log(self.strikes[edge])
        vol = self._native_vol(self.strikes[edge])
        h = 1e-5
        outward = -1.0 if edge == 0 else 1.0
        slope = (self._native_vol(math.exp(log_edge + outward * h)) - vol) / h
        wing = {"log_edge": log_edge, "vol": vol, "slope": slope,
                "decay_scale": math.nan}
        try:
            from helper_functions.fx_strangle_converter import _strike_from_delta
            phi = -1 if edge == 0 else 1
            one_delta_strike = _strike_from_delta(
                self.context.forward, vol / 100.0, self.context.time,
                self.context.delta_factor, phi * 0.01, phi,
                self.context.premium_adjusted)
            scale = abs(math.log(one_delta_strike) - log_edge)
            if math.isfinite(scale) and scale > 0:
                wing["decay_scale"] = scale
        except (AttributeError, TypeError, ValueError, RuntimeError, OverflowError):
            pass
        self._wings[edge] = wing
        return wing

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

    def _candidate_is_valid(self, edge, target_distance, candidate):
        if not math.isfinite(target_distance) or target_distance <= 0:
            return False
        inward = 1.0 if edge == 0 else -1.0
        segment = abs(math.log(self.strikes[1]) - math.log(self.strikes[0])) \
            if edge == 0 else abs(math.log(self.strikes[4]) - math.log(self.strikes[3]))
        distances = np.linspace(-segment, target_distance, 129)
        wing = self._wings[edge]
        strikes = np.exp(wing["log_edge"] + inward * distances)
        vols = np.asarray([
            self._native_vol(strike) if distance <= 0 else candidate(strike, distance)
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
            prices.append(forward * ndtr(d1) - strike * ndtr(d1 - stdev))
        order = np.argsort(strikes)
        prices, strikes = np.asarray(prices)[order], strikes[order]
        slopes = np.diff(prices) / np.diff(strikes)
        return bool(np.all(np.isfinite(slopes))
                    and np.all(slopes <= 1e-10)
                    and np.all(np.diff(slopes) >= -1e-8))

    def _wing_vol(self, strike):
        edge = 0 if strike < self.strikes[0] else 4
        wing = self._wing(edge)
        distance = abs(math.log(strike) - wing["log_edge"])
        validated = self._wing_validation.get(edge)
        if validated is not None and distance <= validated[0]:
            mode = validated[1]
            self._wing_modes[edge] = mode
            if mode == "slope-decay":
                return self._decayed_wing_vol(wing, distance)
            if mode == "natural-cubic":
                return self._cubic_wing_vol(strike)
            return self._native_vol(strike)

        coverage = min(max(
            distance,
            wing["decay_scale"] * 5.0 if math.isfinite(wing["decay_scale"]) else distance,
        ), 20.0)
        candidates = (
            ("slope-decay", lambda k, d: self._decayed_wing_vol(wing, d)),
            ("natural-cubic", lambda k, d: self._cubic_wing_vol(k)),
            ("hagan", lambda k, d: self._native_vol(k)),
        )
        for mode, candidate in candidates:
            try:
                valid = self._candidate_is_valid(edge, coverage, candidate)
            except (PreparationError, FloatingPointError, OverflowError, ValueError):
                valid = False
            if valid:
                self._wing_modes[edge] = mode
                self._wing_validation[edge] = coverage, mode
                return candidate(strike, distance)
        raise PreparationError("SABR wing candidates fail the no-arbitrage check")

    def get_vol_for_strike(self, strike):
        if isinstance(strike, bool) or not isinstance(strike, (int, float, np.number)):
            raise ValueError("strike must be a finite positive number")
        if not math.isfinite(float(strike)) or strike <= 0:
            raise ValueError("strike must be a finite positive number")
        strike = float(strike)
        if self.strikes[0] <= strike <= self.strikes[-1]:
            return self._native_vol(strike)
        return self._wing_vol(strike)

    @property
    def wing_modes(self):
        return dict(self._wing_modes)

    def get_vol_for_strikes(self, strikes):
        return np.asarray([self.get_vol_for_strike(strike) for strike in strikes],
                          dtype=np.float64)

"""Calibrated Hagan-SABR smile interpolation over explicit FX pillars."""
import math

import numpy as np
from scipy.optimize import least_squares
from scipy.special import ndtr

from pricing_utils.sabr_numba import hagan_sabr_log_normal_vol_numba
from resolved_smile_inputs import PreparationError


class SABRCore:
    """Fit one SABR surface to five resolved pillars and evaluate any strike."""

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

    def get_vol_for_strike(self, strike):
        if isinstance(strike, bool) or not isinstance(strike, (int, float, np.number)):
            raise ValueError("strike must be a finite positive number")
        if not math.isfinite(float(strike)) or strike <= 0:
            raise ValueError("strike must be a finite positive number")
        strike = float(strike)
        vol = float(self._vols_decimal(self.parameters, [strike])[0]) * 100.0
        if not math.isfinite(vol) or not 0 < vol <= 500:
            raise PreparationError("SABR volatility is outside the supported finite range")
        return vol

    def get_vol_for_strikes(self, strikes):
        return np.asarray([self.get_vol_for_strike(strike) for strike in strikes],
                          dtype=np.float64)

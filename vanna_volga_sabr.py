"""SABR-enhanced Vanna-Volga implementation."""

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.stats import norm

from helper_functions.VannaVolga_function import normalize_atm_type
from helper_functions.helper_functions import year_fraction
from helper_functions.vanna_volga_helpers import GetMainPillars, VannaVolgaSABR_Weight
from market_data import MarketData
from pricing_utils.BlackScholes import BlackScholes
from pricing_utils.sabr_numba import hagan_sabr_log_normal_vol_numba
from VannaVolga_ImpliedVol import VannaVolga


class VannaVolga_SABR(VannaVolga):
    """SABR-enhanced Vanna-Volga model with per-instance calibration."""

    #: Largest acceptable RMS fit error to the five pillars, in decimal vol
    #: (0.0025 = 0.25 vol points). Good fits are typically 0.01-0.04 vol; a
    #: rejected calibration falls back to plain Vanna-Volga.
    MAX_CALIBRATION_RMS_VOL = 0.0025

    def __init__(self, market: MarketData, *args, beta=0.5, debug=False, **kwargs):
        self._frozen_sabr_params = kwargs.pop("frozen_sabr_params", None)
        self._sabr_calibration_context = kwargs.pop("sabr_calibration_context", None)
        self.central_bump_vol = float(kwargs.pop("central_bump_vol", getattr(market, "central_bump_vol", 0.10)))
        if self._frozen_sabr_params is None:
            self._frozen_sabr_params = getattr(market, "frozen_sabr_params", None)
        if self._sabr_calibration_context is None:
            self._sabr_calibration_context = getattr(market, "sabr_calibration_context", None)

        super().__init__(market=market, *args, **kwargs)
        self.beta = beta
        self.alpha = None
        self.rho = None
        self.nu = None
        self.debug = debug
        self.forward_divisor = market.forward_divisor
        self.market = market
        self.atm_conv = normalize_atm_type(getattr(self, "atm_conv", "dn"))

        self.pillars = GetMainPillars(
            k=self.K,
            expiry_date=self.expiry_date,
            valuation_date=self.valuation_date,
            atm_conv=self.atm_conv,
            deltaBase=self.deltaBase,
            s=self.market.s,
            fwd_pts=market.fwd_pts,
            v_atm=market.v_atm,
            rr_25=market.rr_25,
            rr_10=market.rr_10,
            bf_25=market.bf_25,
            bf_10=market.bf_10,
            forward_divisor=market.forward_divisor,
            term_discount=market.term_discount,
        )
        self.smile = self.pillars.get_main_strikes_vols()
        self.weighted_rr_bf = self.pillars.get_weighted_rr_bf(self.smile)

    @staticmethod
    def _coerce_sabr_params(params: Optional[Any]) -> Optional[Tuple[float, float, float]]:
        if params is None:
            return None
        try:
            if isinstance(params, dict):
                if "params" in params and isinstance(params["params"], (list, tuple, np.ndarray)):
                    return VannaVolga_SABR._coerce_sabr_params(params["params"])
                if all(key in params for key in ("alpha", "rho", "nu")):
                    return float(params["alpha"]), float(params["rho"]), float(params["nu"])
            if isinstance(params, (tuple, list, np.ndarray)) and len(params) >= 3:
                return float(params[0]), float(params[1]), float(params[2])
            if hasattr(params, "alpha") and hasattr(params, "rho") and hasattr(params, "nu"):
                return float(params.alpha), float(params.rho), float(params.nu)
        except (TypeError, ValueError):
            return None
        return None

    @staticmethod
    def _normalise_context(source: Optional[Any]) -> Optional[Dict[str, float]]:
        if source is None:
            return None
        if isinstance(source, dict):
            if all(key in source for key in ("alpha", "rho", "nu")):
                return {"alpha": float(source["alpha"]), "rho": float(source["rho"]), "nu": float(source["nu"])}
            if "params" in source:
                params = VannaVolga_SABR._coerce_sabr_params(source["params"])
                if params is not None:
                    alpha, rho, nu = params
                    return {"alpha": alpha, "rho": rho, "nu": nu}
        params = VannaVolga_SABR._coerce_sabr_params(source)
        if params is not None:
            alpha, rho, nu = params
            return {"alpha": alpha, "rho": rho, "nu": nu}
        return None

    def _validate_sabr_params(self, alpha: Any, rho: Any, nu: Any, residual: Optional[float] = None) -> bool:
        try:
            alpha = float(alpha)
            rho = float(rho)
            nu = float(nu)
        except (TypeError, ValueError):
            return False

        if not (np.isfinite(alpha) and np.isfinite(rho) and np.isfinite(nu)):
            return False
        if not (1e-8 < alpha < 5.0 and 1e-8 < nu < 5.0 and -0.999 < rho < 0.999):
            return False
        if residual is not None:
            if not np.isfinite(residual):
                return False
            if math.sqrt(max(residual, 0.0)) > self.MAX_CALIBRATION_RMS_VOL:
                return False
        return True

    def _set_calibration(self, alpha: float, rho: float, nu: float, residual: Optional[float] = None) -> None:
        self.alpha = float(alpha)
        self.rho = float(rho)
        self.nu = float(nu)

    def calibrate_sabr(self, force=False, frozen_params=None, calibration_context=None):
        if not force and self.alpha is not None and self.rho is not None and self.nu is not None:
            return self.alpha, self.rho, self.nu
        if frozen_params is None:
            frozen_params = self._frozen_sabr_params
        if calibration_context is None:
            calibration_context = self._sabr_calibration_context

        candidate = self._coerce_sabr_params(frozen_params)
        candidate_ctx = self._normalise_context(calibration_context)
        if candidate is None and candidate_ctx is not None:
            candidate = self._coerce_sabr_params(candidate_ctx)
        if candidate is not None:
            alpha, rho, nu = candidate
            if self._validate_sabr_params(alpha, rho, nu):
                self._set_calibration(alpha, rho, nu)
                return self.alpha, self.rho, self.nu

        smile = self.smile
        strikes = np.array(sorted(smile.keys()), dtype=np.float64)
        vols = np.array([smile[k] / 100.0 for k in strikes], dtype=np.float64)
        f = self.s + self.fwd_pts / self.forward_divisor
        t = year_fraction(expiry_date=self.expiry_date, value_date=self.valuation_date)
        if t <= 0 or f <= 0:
            self.alpha = None
            self.rho = None
            self.nu = None
            return None, None, None

        def objective(params):
            alpha, rho, nu = params
            if not self._validate_sabr_params(alpha, rho, nu):
                return 1e10
            try:
                vols_model = hagan_sabr_log_normal_vol_numba(alpha, self.beta, rho, nu, f, t, strikes)
            except Exception:
                return 1e10
            if np.any(~np.isfinite(vols_model)):
                return 1e10
            error = float(np.mean((vols_model - vols) ** 2))
            if self.debug:
                print(f"alpha={alpha:.6f}, rho={rho:.6f}, nu={nu:.6f}, error={error:.8f}")
            return error

        initial_alpha = self.v_atm_pct / 100.0 * f ** (1.0 - self.beta)
        bounds = [(1e-6, 3.0), (-0.999, 0.999), (1e-6, 3.0)]

        # A single L-BFGS-B descent from a fixed seed reliably stalls in a
        # local minimum here: it converges in ~10 iterations onto a smile that
        # is far too flat, undershooting both 10-delta wings by ~0.25 vol and
        # missing ATM outright. The objective is non-convex in (rho, nu), so
        # seed the descent from a small deterministic grid and keep the best.
        best = None
        for rho0 in (-0.4, 0.0, 0.4):
            for nu0 in (0.3, 0.8, 1.4):
                try:
                    candidate = minimize(
                        objective,
                        [initial_alpha, rho0, nu0],
                        bounds=bounds,
                        method="L-BFGS-B",
                        options={"maxiter": 1000},
                    )
                except Exception:
                    continue
                if not np.isfinite(candidate.fun):
                    continue
                if not self._validate_sabr_params(*candidate.x):
                    continue
                if best is None or candidate.fun < best.fun:
                    best = candidate

        if best is None:
            return (None, None, None)

        res = best
        alpha, rho, nu = res.x
        try:
            vols_model = hagan_sabr_log_normal_vol_numba(alpha, self.beta, rho, nu, f, t, strikes)
            residual = float(np.mean((vols_model - vols) ** 2))
        except Exception:
            residual = None

        if not self._validate_sabr_params(alpha, rho, nu, residual):
            return (None, None, None)

        self._set_calibration(alpha, rho, nu, residual=residual)
        if self.debug:
            print(f"Calibration done: alpha={self.alpha:.6f}, rho={self.rho:.6f}, nu={self.nu:.6f}, residual={residual:.8f}")
        return self.alpha, self.rho, self.nu

    def _stable_single_strike_vol(self, K: Optional[float] = None) -> Optional[float]:
        strike = float(self.K if K is None else K)
        if strike <= 0.0:
            return None
        params = self.calibrate_sabr()
        if params is None or any(p is None for p in params):
            return None
        alpha, rho, nu = params
        try:
            f = self.s + self.fwd_pts / self.forward_divisor
            t = year_fraction(expiry_date=self.expiry_date, value_date=self.valuation_date)
            if t <= 0 or f <= 0:
                return None
            K_array = np.array([strike], dtype=np.float64)
            vols = hagan_sabr_log_normal_vol_numba(alpha, self.beta, rho, nu, f, t, K_array)
            if not np.all(np.isfinite(vols)):
                return None
            vol = float(vols[0] * 100.0)
            if not np.isfinite(vol) or vol < 0.0:
                return None
            return vol
        except Exception:
            return None

    def enforce_no_butterfly_arbitrage(self, strikes, vols, f, t):
        """Adjust vols to ensure butterfly price >= 0.

        This keeps the legacy local price-curve projection for explicit multi-
        strike smile construction, while the single-strike path intentionally
        avoids local per-strike arbitrage projection to prevent discontinuities.
        """
        strikes = np.asarray(strikes, dtype=float)
        vols = np.asarray(vols, dtype=float)
        if len(strikes) < 3 or np.any(np.diff(strikes) <= 0):
            raise ValueError("Strikes must be strictly increasing for arbitrage checks")

        def call_price(K, sigma):
            if sigma <= 0:
                return max(f - K, 0.0)
            d1 = (np.log(f / K) + 0.5 * sigma ** 2 * t) / (sigma * np.sqrt(t))
            return f * norm.cdf(d1) - K * norm.cdf(d1 - sigma * np.sqrt(t))

        prices = np.array([call_price(k, v) for k, v in zip(strikes, vols)])
        for _ in range(20):
            changed = False
            for i in range(1, len(strikes) - 1):
                left, right = strikes[i - 1], strikes[i + 1]
                upper = ((right - strikes[i]) * prices[i - 1] + (strikes[i] - left) * prices[i + 1]) / (right - left)
                if prices[i] > upper:
                    prices[i] = upper
                    changed = True
            if not changed:
                break

        adjusted = np.empty_like(vols)
        for i, (K, price) in enumerate(zip(strikes, prices)):
            intrinsic = max(f - K, 0.0)
            upper = f
            price = min(max(price, intrinsic), upper)
            adjusted[i] = brentq(
                lambda sigma: call_price(K, sigma) - price,
                1e-10, 5.0,
            ) if price > intrinsic + 1e-12 else 0.0
        return adjusted

    def sabr_vol(self, K):
        stable = self._stable_single_strike_vol(K)
        if stable is not None:
            return stable
        legacy = VannaVolga.get_vol_for_strike(self)
        if np.isfinite(legacy):
            return float(legacy)
        return 0.0

    def sabr_smile(self, strikes):
        K_array = np.asarray(strikes, dtype=np.float64)
        params = self.calibrate_sabr()
        if params is None or any(p is None for p in params):
            fallback = np.full(K_array.shape, float(self.v_atm_pct), dtype=np.float64)
            return fallback
        alpha, rho, nu = params
        f = self.s + self.fwd_pts / self.forward_divisor
        t = year_fraction(expiry_date=self.expiry_date, value_date=self.valuation_date)
        vols = hagan_sabr_log_normal_vol_numba(alpha, self.beta, rho, nu, f, t, K_array)
        if not np.all(np.isfinite(vols)):
            return np.full(K_array.shape, float(self.v_atm_pct), dtype=np.float64)
        return np.asarray(vols * 100.0, dtype=np.float64)

    def _get_vol_for_strike(self):
        """Return a stable single-strike SABR vol with Vanna-Volga fallback."""
        stable = self._stable_single_strike_vol(self.K)
        if stable is not None:
            return stable
        legacy = VannaVolga.get_vol_for_strike(self)
        if np.isfinite(legacy):
            return float(legacy)
        return float(self.v_atm_pct)

    def get_vv_vol_for_strike(self):
        return VannaVolga.get_vol_for_strike(self)

    def volga(self, eps_vol: Optional[float] = None):
        eps = self.central_bump_vol if eps_vol is None else float(eps_vol)
        eps = max(1e-6, abs(eps))
        eps = min(eps, 1.0)

        base_vol = self.get_vol_for_strike()
        if not np.isfinite(base_vol):
            return 0.0

        vol_up = max(base_vol + eps, 1e-6)
        vol_dn = max(base_vol - eps, 1e-6)
        try:
            vega_up = BlackScholes(
                S=self.market.s,
                K=self.K,
                fwd_pts=self.market.fwd_pts,
                v=vol_up,
                expiry_date=self.expiry_date,
                valuation_date=self.valuation_date,
                CallPut=self.CallPut,
                BuySell=self.BuySell,
                Notional=self.Notional,
                deltaBase=self.deltaBase,
                fwd_divisor=self.forward_divisor, term_discount=self.market.term_discount,
            ).vega()
            vega_dn = BlackScholes(
                S=self.market.s,
                K=self.K,
                fwd_pts=self.market.fwd_pts,
                v=vol_dn,
                expiry_date=self.expiry_date,
                valuation_date=self.valuation_date,
                CallPut=self.CallPut,
                BuySell=self.BuySell,
                Notional=self.Notional,
                deltaBase=self.deltaBase,
                fwd_divisor=self.forward_divisor, term_discount=self.market.term_discount,
            ).vega()
        except Exception:
            return 0.0

        if not (np.isfinite(vega_up) and np.isfinite(vega_dn)):
            return 0.0
        return float((vega_up - vega_dn) / (2.0 * eps))

    def get_vol_for_strike(self):
        sabr_vol = self._get_vol_for_strike()
        vv_vol = self.get_vv_vol_for_strike()
        if not np.isfinite(sabr_vol):
            sabr_vol = vv_vol
        if not np.isfinite(vv_vol):
            vv_vol = sabr_vol

        try:
            blended = VannaVolgaSABR_Weight(
                delta_abs=self._tv_delta,
                vol_sabr=sabr_vol,
                vol_vv=vv_vol,
                t_expiry=self.t,
            ).blend_vol_variance()
            if np.isfinite(blended) and blended >= 0:
                return float(blended)
        except Exception:
            pass

        if np.isfinite(sabr_vol):
            return float(sabr_vol)
        return float(vv_vol)


if __name__ == "__main__":
    from helper_functions.vol_data import VolData

    vd = VolData("vol_data.csv", valuation_date="2025-08-29")
    market = MarketData.from_vol_data(vol_data=vd, target_date="2025-11-28", spot=0.8040, fwd_pts=-80.90)

    vv = VannaVolga_SABR(
        market=market,
        K=0.7960,
        expiry_date="2025-11-28",
        valuation_date="2025-08-31",
        CallPut="CALL",
        deltaBase=True,
        atm_conv="dn",
    )

    import timeit

    runtime = timeit.timeit(lambda: vv.get_vol_for_strike(), number=100)
    print(f"Average runtime over 100 runs (parallel): {runtime / 100:.8f} seconds")
    print(vv.get_vol_for_strike())

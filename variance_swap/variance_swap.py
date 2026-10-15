from helper_functions.VannaVolga_function import get_strike_from_delta_pa, get_strike_from_delta
from helper_functions.vanna_volga_helpers import get_model_map, GetMainPillars
from helper_functions.helper_functions import year_fraction
from market_data import MarketData
from surface_models.smile_extension import SmileExtension
import numpy as np
from math import log, sqrt
from scipy.stats import norm


def _forward_otm_premium(F: float, K: float, T: float, vol: float) -> float:
    """Forward OTM option premium under Black-Scholes."""
    if vol <= 0 or T <= 0:
        return 0.0
    sig = vol / 100
    srt = sig * sqrt(T)
    d1 = (log(F / K) + 0.5 * sig * sig * T) / srt
    d2 = d1 - srt
    if K <= F:  # OTM put
        return K * norm.cdf(-d2) - F * norm.cdf(-d1)
    else:       # OTM call
        return F * norm.cdf(d1) - K * norm.cdf(d2)


class VarianceSwap:
    """Compute a smile-integrated variance-swap strike.

    The implementation integrates forward OTM option prices over a supplied
    strike grid and reports both the full integral and the approximate
    discrete result. It is a research utility and inherits the pricing and
    smile-convention assumptions of the selected volatility model.
    """
    def __init__(
        self,
        market: MarketData,
        expiry_date: str,
        valuation_date: str,
        keep_smile_ratio: bool = True,
        preserve_ratio_mode: str = "additive",
        mods=None,
        pricing_model: str = "vv",
        atm_conv: str = "dn",
        deltaBase=True,
        debug: bool = False,

    ):

        self.market = market
        self.expiry_date = expiry_date
        self.valuation_date = valuation_date
        self.keep_smile_ratio = keep_smile_ratio
        self.mods = mods
        self.pricing_model = pricing_model
        self.model_cls = get_model_map()[pricing_model]
        self.atm_conv = atm_conv

        if not isinstance(deltaBase, bool):
            raise TypeError(f"deltaBase must be bool, got {deltaBase} ({type(deltaBase)})")
        self.deltaBase = deltaBase

        self.debug = debug
        self.preserve_ratio_mode = preserve_ratio_mode


        if self.debug:
            print(f"[DEBUG] VarianceSwap initialized with deltaBase={self.deltaBase}")


    # ------------------------- helpers ------------------------- #

    def _get_mod(self, name, default=0.0):
        return self.mods[name] if isinstance(self.mods, dict) else getattr(self.mods, name, default)

    def _get_main_strikes_vols(self, k):

        return GetMainPillars(
            k=k,
            expiry_date=self.expiry_date,
            valuation_date=self.valuation_date,
            atm_conv=self.atm_conv,
            deltaBase=self.deltaBase,
            s=self.market.s,
            fwd_pts=self.market.fwd_pts,
            v_atm=self.market.v_atm,
            rr_25=self.market.rr_25,
            rr_10=self.market.rr_10,
            bf_25=self.market.bf_25,
            bf_10=self.market.bf_10,
            forward_divisor=self.market.forward_divisor,
        ).get_main_strikes_vols()


    def _compute_vol(self, strike: float) -> float:
        return self.model_cls(
            market=self.market,
            K=strike,
            expiry_date=self.expiry_date,
            valuation_date=self.valuation_date,
            CallPut="CALL",
            debug=self.debug,
            deltaBase=self.deltaBase,
            atm_conv=self.atm_conv
        ).get_vol_for_strike()

    # ------------------------- core routines ------------------------- #

    def get_strikes_vols(self):
        """Build a symmetric delta grid of strikes and vols."""
        put_deltas = np.arange(0.05, 0.50, 0.05)
        call_deltas = put_deltas[::-1]
        deltas = np.concatenate([put_deltas, call_deltas]).tolist()

        kv = []
        for i, d in enumerate(deltas):
            cp_flag = "PUT" if i < len(deltas) // 2 else "CALL"
            strike_func = get_strike_from_delta_pa if self.deltaBase else get_strike_from_delta

            K = strike_func(
                S=self.market.s,
                fwd=self.market.fwd_pts,
                valuation_date=self.valuation_date,
                expiry_date=self.expiry_date,
                CallPut=cp_flag,
                delta=d,
                atm_type=self.atm_conv,
                v=self.market.v_atm,
            )
            vol = self._compute_vol(K)
            kv.append((K, vol))

        kv.sort(key=lambda x: x[0])

        if self.debug:
            print(f"[DEBUG] Strikes/Vols: {kv}")

        return kv

    def _integrate_variance(self, kv_pairs, F, T):
        """Trapezoidal integration in log-K space."""
        otm_prices = [(K, _forward_otm_premium(F, K, T, vol)) for K, vol in kv_pairs]

        integ = 0.0
        for (K0, p0), (K1, p1) in zip(otm_prices[:-1], otm_prices[1:]):
            dx = log(K1 / K0)
            f0, f1 = p0 / K0, p1 / K1
            integ += 0.5 * (f0 + f1) * dx

        return (2.0 / T) * integ

    def compute_fair_varswap_strike(self) -> float:
        kv_pairs = self.get_strikes_vols()
        F = self.market.s + (self.market.fwd_pts / self.market.forward_divisor)
        T = year_fraction(expiry_date=self.expiry_date, value_date=self.valuation_date)
        var = self._integrate_variance(kv_pairs, F, T)
        return sqrt(max(var, 0.0)) * 100

    def approximate_varswap_strike(self, k_grid=None) -> float:
        if k_grid is None:
            k_grid = [self.market.s]

        kv_pairs = []
        for k in k_grid:
            kv_pairs.extend(self._get_main_strikes_vols(k).items())
        kv_pairs.sort(key=lambda x: x[0])

        F = self.market.s + (self.market.fwd_pts / self.market.forward_divisor)
        T = year_fraction(expiry_date=self.expiry_date, value_date=self.valuation_date, annual_factor=365)
        var = self._integrate_variance(kv_pairs, F, T)
        return sqrt(max(var, 0.0)) * 100

    def smile_correction(self) -> float:
        cm_sigma = self.compute_fair_varswap_strike()
        vs_appx = self.approximate_varswap_strike()
        avg_var = (cm_sigma**2 + vs_appx**2) / 2

        w_bf25, w_bf10 = 0.6, 0.4
        delta_var_bf = (w_bf25 * self.market.bf_25 + w_bf10 * self.market.bf_10) / \
                       year_fraction(expiry_date=self.expiry_date, value_date=self.valuation_date)

        return sqrt(avg_var) + delta_var_bf


# ------------------------- Example ------------------------- #

if __name__ == "__main__":
    market = MarketData(
        s=1.1720,
        fwd_pts=67.5,
        v_atm=7.22,
        rr_25=0.48,
        rr_10=0.88,
        bf_25=0.17,
        bf_10=0.65,
    )

    vs = VarianceSwap(
        market=market,
        expiry_date="2026-11-25",
        valuation_date="2025-08-22",
        pricing_model="vv-sabr",
        deltaBase=True,
        debug=False,
    )

    print(vs.smile_correction())

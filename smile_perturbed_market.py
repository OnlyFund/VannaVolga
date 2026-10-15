from copy import deepcopy
from surface_models.smile_extension import SmileExtension
from market_data import Modifications, MarketData

class PerturbedMarket:
    """Create perturbed smile volatilities for finite-difference Greeks.

    The 25-delta quote is the primary RR/BF bump. The 10-delta quote follows
    by a configurable ratio, preserving the observed 10D/25D shape.
    """
    RR_D10_RATIO = 1.85
    BF_D10_RATIO = 3.8

    def __init__(
        self,
        base_market: MarketData,
        mods: Modifications = None,
        K=None,
        expiry_date=None,
        valuation_date=None,
        CallPut="CALL",
        debug=False,
        deltaBase=True,
        atm_type="dn",
        keep_smile_ratio=True,
        preserve_ratio_mode="additive",
        pricing_model="vv",
    ):
        self.base = base_market
        # ensure mods is never None
        self.mods = mods if mods is not None else Modifications(shifted_spot=self.base.s)

        self.K = K
        self.expiry_date = expiry_date
        self.valuation_date = valuation_date
        self.CallPut = CallPut
        self.debug = debug
        self.deltaBase = deltaBase
        self.atm_type = atm_type
        self.keep_smile_ratio = keep_smile_ratio
        self.preserve_ratio_mode = preserve_ratio_mode
        self.pricing_model = pricing_model

    def _reprice_perturbed_market(self, perturbed_market: MarketData, mods=None):
        mods_to_use = mods if mods is not None else self.mods

        perturbed_smile = SmileExtension(
            market=perturbed_market,
            K=self.K,
            expiry_date=self.expiry_date,
            valuation_date=self.valuation_date,
            CallPut=self.CallPut,
            keep_smile_ratio=self.keep_smile_ratio,
            preserve_ratio_mode=self.preserve_ratio_mode,
            mods=mods_to_use,
            pricing_model=self.pricing_model,
            deltaBase=self.deltaBase,
            atm_conv=self.atm_type,
            reference_spot=self.base.s,
            debug=self.debug
        )
        return perturbed_smile.get_vol_for_strike()

    # --- Spot perturbation ---
    def shift_spot(self, delta_spot: float = 0.0001, new_spot: float = None):
        """Return volatility at the fixed strike after an absolute spot shift."""
        bumped_mods = deepcopy(self.mods) if self.mods is not None else Modifications(shifted_spot=self.base.s)

        if new_spot is not None:
            new_spot_value = new_spot
        else:
            new_spot_value = self.base.s * (1 + delta_spot)

        # handle dict or object
        if isinstance(bumped_mods, dict):
            bumped_mods["shifted_spot"] = new_spot_value
        else:
            setattr(bumped_mods, "shifted_spot", new_spot_value)

        return self._reprice_perturbed_market(self.base, mods=bumped_mods)

    # --- ATM bump ---
    def bump_vol(self, delta_vol: float = 0.01):
        """Bump ATM volatility by percentage points and return strike vol."""
        bumped_market = self.base.copy_with(v_atm=self.base.v_atm + delta_vol)
        return self._reprice_perturbed_market(bumped_market)

    # --- RR bumps ---
    def bump_rr(self, delta: float, which="25"):
        bumped_market = self.base.copy_with(**{f"rr_{which}": getattr(self.base, f"rr_{which}") + delta})
        return self._reprice_perturbed_market(bumped_market)

    def bump_rr_with_ratio(self, eps=0.1, d10_ratio=None):
        """Bump the 25-delta RR quote and infer the 10-delta bump.

        The 10-delta quote is moved by the market-standard relative ratio
        because the 10/25-delta smile shape is treated as stable for this
        risk bump.
        """
        if d10_ratio is None:
            d10_ratio = self.RR_D10_RATIO
        bumped_market = self.base.copy_with(
            rr_25=self.base.rr_25 + eps,
            rr_10=self.base.rr_10 + eps * d10_ratio,
        )
        return self._reprice_perturbed_market(bumped_market)

    # --- BF bumps ---
    def bump_bf(self, delta: float, which="25"):
        bumped_market = self.base.copy_with(**{f"bf_{which}": getattr(self.base, f"bf_{which}") + delta})
        return self._reprice_perturbed_market(bumped_market)

    def bump_bf_with_ratio(self, eps=0.1, d10_ratio=None):
        """Bump the 25-delta BF quote and infer the 10-delta bump."""
        if d10_ratio is None:
            d10_ratio = self.BF_D10_RATIO
        bumped_market = self.base.copy_with(
            bf_25=self.base.bf_25 + eps,
            bf_10=self.base.bf_10 + eps * d10_ratio,
        )
        return self._reprice_perturbed_market(bumped_market)

    # --- Smile scaling ---
    def scale_all_vols(self, eps: float):
        """Scale all five quoted smile vols proportionally.

        The five points are 10P, 25P, ATM, 25C, and 10C.  They are scaled
        together, then RR/BF quotes are reconstructed from the scaled points.
        ``eps`` is a volatility-point bump applied to ATM.
        """
        atm = self.base.v_atm
        if atm <= 0:
            raise ValueError("ATM volatility must be positive for proportional scaling")
        scale = (atm + eps) / atm
        v10p = (atm + self.base.bf_10 - self.base.rr_10 / 2.0) * scale
        v25p = (atm + self.base.bf_25 - self.base.rr_25 / 2.0) * scale
        v_atm = atm * scale
        v25c = (atm + self.base.bf_25 + self.base.rr_25 / 2.0) * scale
        v10c = (atm + self.base.bf_10 + self.base.rr_10 / 2.0) * scale
        bumped_market = self.base.copy_with(
            v_atm=v_atm,
            rr_25=v25c - v25p,
            rr_10=v10c - v10p,
            bf_25=(v25c + v25p) / 2.0 - v_atm,
            bf_10=(v10c + v10p) / 2.0 - v_atm,
        )
        return self._reprice_perturbed_market(bumped_market)

    def scale_vol(self, method="atm", eps=0.01):
        """Apply either an ATM-only or proportional five-point vol bump."""
        if method == "atm":
            return self.bump_vol(eps)
        elif method == "smile":
            return self.scale_all_vols(eps)
        else:
            raise ValueError(f"Unknown scaling method: {method}")

    # --- Fixed strike vol under spot shift ---
    def vol_for_strike(self, new_spot: float = None):
        """Return fixed-strike volatility under a specified perturbed spot."""
        bumped_mods = deepcopy(self.mods) if self.mods is not None else Modifications(shifted_spot=self.base.s)
        shifted_spot = new_spot if new_spot is not None else self.base.s * 1.001
        if isinstance(bumped_mods, dict):
            bumped_mods["shifted_spot"] = shifted_spot
        else:
            setattr(bumped_mods, "shifted_spot", shifted_spot)

        smile = SmileExtension(
            market=self.base,
            K=self.K,
            expiry_date=self.expiry_date,
            valuation_date=self.valuation_date,
            CallPut=self.CallPut,
            keep_smile_ratio=self.keep_smile_ratio,
            preserve_ratio_mode=self.preserve_ratio_mode,
            mods=bumped_mods,
            pricing_model=self.pricing_model,
            deltaBase=self.deltaBase,
            atm_conv=self.atm_type,
            reference_spot=self.base.s,
            debug=self.debug

        )
        return smile.get_vol_for_strike()

# # Usage
# market = MarketData(
#     s=1.170,
#     fwd_pts=67.5,
#     v_atm=7.24,
#     rr_25=0.53,
#     rr_10=0.95,
#     bf_25=0.18,
#     bf_10=0.68
# )
#
# mods = Modifications(beta_atm=0.7, beta_rr=0.5, beta_bf=0.5, shifted_spot=1.1702)
#
# pm = PerturbedMarket(
#     base_market=market,
#     mods=mods,
#     K=1.21,
#     expiry_date="2025-11-18",
#     valuation_date="2025-08-15",
#     CallPut="CALL",
#     debug=False
# )
#
# print(pm.shift_spot())
"""Helpers for constructing quote and spot perturbations of an FX smile."""

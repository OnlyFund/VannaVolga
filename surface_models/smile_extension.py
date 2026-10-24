import logging
import warnings
from math import isfinite
from numbers import Real
from market_data import MarketData, Modifications
from helper_functions.vanna_volga_helpers import (
    get_model_map, GetMainPillars, get_vol_k, _resolve_pillar_compatibility,
    interpolate_pillar_vols,
)
from typing import Union,Optional
import numpy as np

logger = logging.getLogger(__name__)


class SmileExtension:
    """Apply ATM/RR/BF spot dynamics and extract a fixed-strike volatility.

    ``beta_atm``, ``beta_rr``, and ``beta_bf`` define the spot response of the
    quoted smile. The 25-delta wings are first rebuilt, then the 10-delta
    quotes are inferred from their 10D/25D ratios. The resulting quotes are
    passed to the selected pricing model at the perturbed spot.
    """
    def __init__(
        self,
        market: MarketData,
        K: float,
        expiry_date,
        valuation_date,
        CallPut: str = "CALL",
        mods: Optional[Union[dict, Modifications]] = None,
        pricing_model: str = "vv",
        keep_smile_ratio: bool = True,
        preserve_ratio_mode: str = "additive",
        atm_conv: str = "dn",
        deltaBase: bool = True,
        reference_spot: float = None,
        debug: bool = False,
        smooth_pillar_interpolation: Optional[bool] = None,
        compatibility_mode: Optional[str] = None,
    ):
        self.market = market
        self.K = K
        self.expiry_date = expiry_date
        self.valuation_date = valuation_date
        self.CallPut = CallPut
        self.debug = debug
        self.keep_smile_ratio = keep_smile_ratio
        self.preserve_ratio_mode = preserve_ratio_mode
        self.atm_conv = atm_conv
        self.deltaBase = deltaBase
        self.reference_spot = market.s if reference_spot is None else reference_spot
        self.smooth_pillar_interpolation = (
            getattr(market, "smooth_pillar_interpolation", True)
            if smooth_pillar_interpolation is None
            else smooth_pillar_interpolation
        )
        self.compatibility_mode = (
            getattr(market, "compatibility_mode", "auto")
            if compatibility_mode is None
            else compatibility_mode
        )

        self.pricing_model = pricing_model
        self.model_cls = get_model_map(market)[pricing_model]
        self.mods = {} if mods is None else mods
        if preserve_ratio_mode not in ("additive", "multiplicative"):
            raise ValueError(f"Unknown preserve_ratio_mode '{preserve_ratio_mode}'")
        self._spot_move()
        for name in ("beta_atm", "beta_rr", "beta_bf"):
            value = self._get_mod(name)
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")

    # ---------------------------
    # Utility
    # ---------------------------
    def _get_mod(self, name: str, default: float = 0.0) -> float:
        if isinstance(self.mods, dict):
            return self.mods.get(name, default)
        return getattr(self.mods, name, default)

    def _spot_move(self):
        shifted = self._get_mod("shifted_spot", self.reference_spot)
        for name, value in (("reference_spot", self.reference_spot), ("shifted_spot", shifted)):
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        return shifted, np.log(shifted) - np.log(self.reference_spot)

    @staticmethod
    def _validate_vols(values):
        values = np.asarray(values, dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("Smile volatilities must be positive and finite")

    def _fallback_vol(self, shifted_spot, exc):
        from wrapped_vol_resolver import DegradedSmileWarning
        reason = f"Smile dynamics failed ({exc}); using unchanged quotes at shifted spot {shifted_spot}"
        logger.warning(reason)
        if self.market.wrapped_smile:
            from app.smile_service import record
            record(self.market, self.pricing_model, self.expiry_date, reason)
        warnings.warn(reason, DegradedSmileWarning, stacklevel=2)
        vol = self.model_cls(
            market=self.market.copy_with(s=shifted_spot),
            K=self.K, expiry_date=self.expiry_date,
            valuation_date=self.valuation_date, CallPut=self.CallPut,
            atm_conv=self.atm_conv, deltaBase=self.deltaBase,
        ).get_vol_for_strike()
        self._validate_vols([vol])
        return vol

    def _compute_vol(self, strike: float) -> float:
        """Compute vol at a given strike using the pricing model."""
        return self.model_cls(
            market=self.market,
            K=strike,
            expiry_date=self.expiry_date,
            valuation_date=self.valuation_date,
            CallPut=self.CallPut,
            debug=self.debug,
            atm_conv=self.atm_conv,
            deltaBase=self.deltaBase,
        ).get_vol_for_strike()

    # ---------------------------
    # Local skew estimation
    # ---------------------------
    def _cached_local_skews(self, prepared, eps):
        """The wrapped smile's local skews: they depend on the market, expiry
        and model, not on this strike, so they are computed once per market
        (in its surface cache, keyed like its prepared smile) instead of at
        every strike and Greek bump. A failure is not kept; each caller gets
        its own arrays; ``VV_MARKET_CACHE=0`` turns the cache off."""
        import os
        from app.smile_service import _cache_for, cache_key

        def compute():
            return self._wrapped_local_skews(prepared, eps)
        if os.environ.get("VV_MARKET_CACHE", "1").strip() == "0":
            return compute()
        key = (cache_key(self.market, self.expiry_date, self.valuation_date,
                         self.pricing_model, None, None),
               self.model_cls.__module__ + "." + self.model_cls.__qualname__,
               self.CallPut, self.atm_conv, self.deltaBase, eps)
        found = _cache_for(self.market).get("local_skews", key, compute)
        return {name: (value.copy() if isinstance(value, np.ndarray) else value)
                for name, value in found.items()}

    def _wrapped_local_skews(self, prepared, eps):
        strikes = np.array(prepared.pillars.strikes[1:4])
        vols = np.array(prepared.pillars.vols[1:4])
        k_up, k_dn = strikes * np.exp(eps), strikes * np.exp(-eps)
        vols_up = np.array([self._compute_vol(k) for k in k_up])
        vols_dn = np.array([self._compute_vol(k) for k in k_dn])
        self._validate_vols(np.concatenate((vols, vols_up, vols_dn)))
        local_skews = (vols_up - vols_dn) / (2 * eps)
        return {
            "local_skew": tuple(local_skews),
            "rr_skew": local_skews[2] - local_skews[0],
            "bf_skew": (local_skews[2] + local_skews[0]) / 2 - local_skews[1],
            "main_vols": vols,
            "rr_ratio": self.market.rr_10 / self.market.rr_25
            if not np.isclose(self.market.rr_25, 0) else 1.0,
            "bf_ratio": self.market.bf_10 / self.market.bf_25
            if not np.isclose(self.market.bf_25, 0) else 1.0,
        }

    def get_local_skews(self) -> dict:
        """Estimate local ATM, RR, and BF smile slopes by finite differences."""
        eps = 1e-4
        if self.market.wrapped_smile and not self.market.force_legacy_smile:
            from app.smile_service import prepared_smile
            prepared = prepared_smile(self.market, self.expiry_date,
                                      self.valuation_date, self.pricing_model)
            if prepared.pillars is not None:
                return self._cached_local_skews(prepared, eps)

        pillars = GetMainPillars(
            k=self.K,
            expiry_date=self.expiry_date,
            valuation_date=self.valuation_date,
            atm_conv=self.atm_conv,
            deltaBase=self.deltaBase,
            s=self.market.s,
            fwd_pts=self.market.fwd_pts,
            v_atm=self.market.v_atm,
            rr_25=self.market.rr_25,
            rr_10=getattr(self.market, "rr_10", 0.0),
            bf_25=self.market.bf_25,
            bf_10=getattr(self.market, "bf_10", 0.0),
            forward_divisor=getattr(self.market, "forward_divisor", 10000),
            term_discount=getattr(self.market, "term_discount", 0.05),
            smooth_pillar_interpolation=self.smooth_pillar_interpolation,
            compatibility_mode=self.compatibility_mode,
        )

        strikes_vols = pillars.get_main_strikes_vols()
        strikes = np.array(list(strikes_vols.keys())[1:4], dtype=float)  # 25P, ATM, 25C
        vols = np.array(list(strikes_vols.values())[1:4], dtype=float)

        k_up, k_dn = strikes * np.exp(eps), strikes * np.exp(-eps)
        fast_vv_interpolation = (
            self.pricing_model == "vv"
            and not self.market.wrapped_smile
            and self.smooth_pillar_interpolation
            and not getattr(self.market, "legacy_target_blend", False)
            and _resolve_pillar_compatibility(self.compatibility_mode, default=True)
        )
        if fast_vv_interpolation:
            try:
                # These are the same five pillars VannaVolga builds for each
                # target strike. Validate and sort once, then interpolate all
                # six local-skew probes in one NumPy call.
                pillar_strikes, pillar_vols = pillars.get_five_pillar_arrays(
                    strikes_vols
                )
                order = np.argsort(pillar_strikes)
                targets = np.concatenate((k_up, k_dn))
                if not np.all(np.isfinite(targets)):
                    raise ValueError("Non-finite local-skew probe strike")
                # Must use the same interpolator as the pricing path, or the
                # finite-difference skew is taken off a different curve than
                # the one being priced.
                interpolated = interpolate_pillar_vols(
                    pillar_strikes[order], pillar_vols[order], targets
                )
                vols_up, vols_dn = interpolated[:3], interpolated[3:]
            except (ValueError, RuntimeError, FloatingPointError, OverflowError) as exc:
                # Retain the pricing-model path for invalid or unusual pillars.
                logger.warning("Fast local-skew interpolation failed; using model probes: %s", exc)
                vols_up = np.array([self._compute_vol(k) for k in k_up])
                vols_dn = np.array([self._compute_vol(k) for k in k_dn])
        else:
            vols_up = np.array([self._compute_vol(k) for k in k_up])
            vols_dn = np.array([self._compute_vol(k) for k in k_dn])

        self._validate_vols(np.concatenate((vols, vols_up, vols_dn)))
        local_skews = (vols_up - vols_dn) / (2 * eps)
        rr_skew = local_skews[2] - local_skews[0]
        bf_skew = (local_skews[2] + local_skews[0]) / 2 - local_skews[1]

        if self.debug:
            print({
                "local_skews": local_skews,
                "rr_skew": rr_skew,
                "bf_skew": bf_skew,
                "atm_k_v": {strikes[1]: vols[1]},
            })

        result = {
            "local_skew": tuple(local_skews),
            "rr_skew": rr_skew,
            "bf_skew": bf_skew,
            "main_vols": vols,
            "rr_ratio": (
                pillars.rr_10 / pillars.rr_25
                if not np.isclose(pillars.rr_25, 0.0) else 1.0
            ),
            "bf_ratio": (
                pillars.bf_10 / pillars.bf_25
                if not np.isclose(pillars.bf_25, 0.0) else 1.0
            ),
        }
        return result

    # ---------------------------
    # Vol perturbation
    # ---------------------------
    def get_new_pillar_vols(self, local_skews=None) -> dict:
        """Apply the ATM-driven spot move to the principal smile pillars."""
        shifted_spot, eps = self._spot_move()

        pillars = self.get_local_skews() if local_skews is None else local_skews
        base_25p, base_atm, base_25c = pillars["main_vols"]

        local_skew = pillars["local_skew"]
        rr_skew = pillars["rr_skew"]
        bf_skew = pillars["bf_skew"]


        beta_atm = self._get_mod("beta_atm")


        v_atm_new = base_atm + local_skew[1] * eps * beta_atm

        if self.keep_smile_ratio:
            if self.preserve_ratio_mode == "additive":
                # Wings keep their vol-point spread to ATM.
                v_25p_new = v_atm_new + (base_25p - base_atm)
                v_25c_new = v_atm_new + (base_25c - base_atm)
            elif self.preserve_ratio_mode == "multiplicative":
                # Wings keep their ratio to ATM.
                p25_ratio, c25_ratio = base_25p / base_atm, base_25c / base_atm
                v_25p_new, v_25c_new = v_atm_new * p25_ratio, v_atm_new * c25_ratio
            else:
                raise ValueError(
                    f"Unknown preserve_ratio_mode '{self.preserve_ratio_mode}'"
                )
        else:
            # RR and BF scale with the relative ATM move. ``v_atm_new -
            # base_atm`` is in vol points, so it cannot be used as a factor.
            atm_scale = v_atm_new / base_atm
            rr_base = (base_25c - base_25p) * atm_scale
            bf_base = ((base_25c + base_25p) / 2 - base_atm) * atm_scale
            v_25p_new = get_vol_k(v_atm_new, bf_base, rr_base, 'put')
            v_25c_new = get_vol_k(v_atm_new, bf_base, rr_base, 'call')

        rr_25_new = v_25c_new - v_25p_new
        bf_25_new = 0.5 * (v_25c_new + v_25p_new) - v_atm_new

        rr_ratio = pillars["rr_ratio"]
        bf_ratio = pillars["bf_ratio"]
        if np.isclose(self.market.rr_25, 0.0) and not np.isclose(self.market.rr_10, 0.0):
            rr_ratio = 1.0
            logger.warning(
                "RR25 is zero while RR10 is non-zero; using a neutral RR10/25 "
                "ratio in smile extension at spot %s.",
                shifted_spot,
            )
        if np.isclose(self.market.bf_25, 0.0) and not np.isclose(self.market.bf_10, 0.0):
            bf_ratio = 1.0
            logger.warning(
                "BF25 is zero while BF10 is non-zero; using a neutral BF10/25 "
                "ratio in smile extension at spot %s.",
                shifted_spot,
            )

        rr_10_new = rr_25_new * rr_ratio
        bf_10_new = bf_25_new * bf_ratio

        # Enforce positivity, then reconstruct quote differences so the
        # stored quotes exactly reproduce the floored 25-delta wing vols.
        floor = 1e-4
        if min(v_atm_new, v_25p_new, v_25c_new) < floor:
            logger.warning(
                "Smile spot shock reached the volatility floor at spot %s; flooring principal vols to %s",
                shifted_spot, floor,
            )
        v_atm_new = max(v_atm_new, floor)
        v_25p_new = max(v_25p_new, floor)
        v_25c_new = max(v_25c_new, floor)
        rr_25_new = v_25c_new - v_25p_new
        bf_25_new = 0.5 * (v_25c_new + v_25p_new) - v_atm_new
        rr_10_new = rr_25_new * rr_ratio
        bf_10_new = bf_25_new * bf_ratio

        if self.debug:
            print({"v_atm": v_atm_new, "rr_25": rr_25_new, "bf_25": bf_25_new})

        result = {
            "v_atm": v_atm_new,
            "v_25p": v_25p_new,
            "v_25c": v_25c_new,
            "rr_25": rr_25_new,
            "bf_25": bf_25_new,
            "rr_10": rr_10_new,
            "bf_10": bf_10_new,
        }
        return result

    def extend_wings_convexity(self, pillar_vols=None, local_skews=None) -> dict:
        """Apply explicit RR/BF beta shocks and reconstruct both wings."""
        local_skew = self.get_local_skews() if local_skews is None else local_skews
        if pillar_vols is None:
            pillar_vols = self.get_new_pillar_vols(local_skew)

        shifted_spot, eps = self._spot_move()

        v_atm = pillar_vols["v_atm"]
        v_25p, v_25c = pillar_vols["v_25p"], pillar_vols["v_25c"]

        rr_raw = v_25c - v_25p
        bf_raw = 0.5 * (v_25c + v_25p) - v_atm

        rr_25 = rr_raw + self._get_mod("beta_rr") * eps * local_skew["rr_skew"]
        bf_25 = bf_raw + self._get_mod("beta_bf") * eps * local_skew["bf_skew"]


        v_25p_new = v_atm + bf_25 - rr_25 / 2
        v_25c_new = v_atm + bf_25 + rr_25 / 2

        rr_10, bf_10 = rr_25 * local_skew["rr_ratio"], bf_25 * local_skew["bf_ratio"]
        self._validate_vols([
            v_atm, v_25p_new, v_25c_new,
            v_atm + bf_10 - rr_10 / 2, v_atm + bf_10 + rr_10 / 2,
        ])

        if self.debug:
            print({"v_atm": v_atm, "rr_25": rr_25, "bf_25": bf_25})

        result = {
            "v_atm": v_atm,
            "v_25p": v_25p_new,
            "v_25c": v_25c_new,
            "rr_25": rr_25,
            "bf_25": bf_25,
            "rr_10": rr_10,
            "bf_10": bf_10,
        }
        return result

    # ---------------------------
    # Vol extraction
    # ---------------------------
    def get_vol_for_strike(self) -> float:
        """Return the modified-surface volatility at the configured strike."""
        if self.market.wrapped_smile and str(self.expiry_date)[:10] == str(self.valuation_date)[:10]:
            return float(self.market.v_atm)
        shifted_spot, eps = self._spot_move()
        if eps == 0:
            vol = self._compute_vol(self.K)
            self._validate_vols([vol])
            return vol
        try:
            local_skews = self.get_local_skews()
            pillar_vols = self.get_new_pillar_vols(local_skews)
            shifted_vols = self.extend_wings_convexity(
                pillar_vols, local_skews
            )
        except (ValueError, RuntimeError, FloatingPointError, OverflowError) as exc:
            return self._fallback_vol(shifted_spot, exc)

        try:
            shifted_market = self.market.copy_with(
                s=shifted_spot,
                v_atm=shifted_vols["v_atm"],
                rr_25=shifted_vols["rr_25"],
                rr_10=shifted_vols["rr_10"],
                bf_25=shifted_vols["bf_25"],
                bf_10=shifted_vols["bf_10"],
                atm_conv=self.atm_conv,
                deltaBase=self.deltaBase,
                smooth_pillar_interpolation=self.smooth_pillar_interpolation,
                compatibility_mode=self.compatibility_mode,
            )
            vol = self.model_cls(
                market=shifted_market,
                K=self.K,
                expiry_date=self.expiry_date,
                valuation_date=self.valuation_date,
                CallPut=self.CallPut,
                atm_conv=self.atm_conv,
                deltaBase=self.deltaBase,
            ).get_vol_for_strike()
            self._validate_vols([vol])
            return vol
        except (ValueError, RuntimeError, FloatingPointError, OverflowError) as exc:
            return self._fallback_vol(shifted_spot, exc)

    def reprice(self, market: Optional[MarketData] = None) -> MarketData:
        """Attach the extracted volatility to a market copy and return it."""
        market_to_use = market or self.market
        market_to_use.vol_surface = self.model_cls(
            market=market_to_use,
            K=self.K,
            expiry_date=self.expiry_date,
            valuation_date=self.valuation_date,
            CallPut=self.CallPut,
            atm_conv=self.atm_conv,
            deltaBase=self.deltaBase,
        ).get_vol_for_strike()
        return market_to_use
"""Spot-dependent smile reconstruction for finite-difference smile risks."""

from dataclasses import dataclass , replace
from copy import deepcopy
from math import isfinite
from numbers import Real
from typing_extensions import TypedDict
from market_data.vol_data import VolData
from typing import Optional


@dataclass
class MarketData:
    """One-expiry FX volatility quote set.

    ``rr_*`` is call volatility minus put volatility and ``bf_*`` is the
    average wing volatility minus ATM volatility. ``term_discount`` is the
    continuously compounded annual term-currency rate. The class stays mutable
    for existing UI/model-option callers. Use :meth:`copy_with` for perturbations.
    """
    s: float
    fwd_pts: float
    v_atm: float
    rr_25: float
    rr_10: float
    bf_25: float
    bf_10: float
    forward_divisor: float = 10000.0
    atm_conv: str = "dn"
    deltaBase: bool = True
    smooth_pillar_interpolation: bool = True
    legacy_target_blend: bool = False
    compatibility_mode: str = "auto"
    frozen_sabr_params: Optional[dict] = None
    sabr_calibration_context: Optional[dict] = None
    central_bump_vol: float = 0.10
    term_discount: float = 0.05
    currency_pair: Optional[str] = None
    settlement_discounting: bool = True
    wrapped_smile: bool = False
    broker_strangle: bool = True
    force_legacy_smile: bool = False
    smile_model: str = "vv"
    smile_reasons: tuple = ()
    diagnostics_token: Optional[str] = None
    delta_reporting: Optional[str] = None

    def __post_init__(self):
        """Reject malformed inputs before they reach pricing calculations."""
        numeric_fields = (
            "s", "fwd_pts", "v_atm", "rr_25", "rr_10", "bf_25", "bf_10",
            "forward_divisor", "term_discount", "central_bump_vol",
        )
        for name in numeric_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(float(value)):
                raise ValueError(f"{name} must be a finite number")
        if self.s <= 0:
            raise ValueError("s (spot) must be greater than zero")
        if self.forward_divisor <= 0:
            raise ValueError("forward_divisor must be greater than zero")
        if self.s + self.fwd_pts / self.forward_divisor <= 0:
            raise ValueError("derived forward must be greater than zero")
        if self.v_atm <= 0:
            raise ValueError("v_atm must be greater than zero (percentage points)")
        if self.central_bump_vol <= 0:
            raise ValueError("central_bump_vol must be greater than zero")
        aliases = {
            "spot": "spot", "atmf": "atmf", "forward": "atmf", "fwd": "atmf",
            "atmforward": "atmf", "dn": "dn", "deltaneutral": "dn",
        }
        key = str(self.atm_conv).strip().lower().replace("_", "").replace("-", "")
        if key not in aliases:
            raise ValueError(f"Unknown ATM convention: {self.atm_conv!r}")
        self.atm_conv = aliases[key]
        if not isinstance(self.deltaBase, bool):
            raise ValueError("deltaBase must be a bool (True=premium-adjusted)")
        if self.delta_reporting not in (None, "spot", "forward"):
            raise ValueError("delta_reporting must be None, 'spot' or 'forward'")
        if not isinstance(self.settlement_discounting, bool):
            raise ValueError("settlement_discounting must be a bool")
        for name in ("wrapped_smile", "broker_strangle", "force_legacy_smile"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError("{} must be a bool".format(name))
        if self.smile_model not in ("vv", "vv-sabr"):
            raise ValueError("smile_model must be vv or vv-sabr")
        if not isinstance(self.smile_reasons, tuple) or any(
                not isinstance(reason, str) for reason in self.smile_reasons):
            raise ValueError("smile_reasons must be a tuple of strings")
        if self.diagnostics_token is not None and not isinstance(self.diagnostics_token, str):
            raise ValueError("diagnostics_token must be a string or None")
        if self.currency_pair is not None:
            pair = str(self.currency_pair).replace("/", "").replace(" ", "").upper()
            if len(pair) != 6 or not pair.isalpha():
                raise ValueError("currency_pair must be a six-letter code like 'EURUSD'")
            self.currency_pair = pair
        for name in ("frozen_sabr_params", "sabr_calibration_context"):
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, dict):
                    raise ValueError(f"{name} must be a dict or None")
                setattr(self, name, deepcopy(value))

    def get_surface_ratios(self):
        """Return ATM, RR/BF, and 10D/25D ratios used by smile bumps.

        A zero denominator produces ``0.0`` rather than a division error.
        Callers that use these ratios must interpret zero as an unavailable
        ratio and apply their own fallback policy.
        """
        def safe_ratio(numerator, denominator):
            return numerator / denominator if denominator != 0 else 0.0

        return {
            'atm_rr_25': safe_ratio(self.v_atm, abs(self.rr_25)),
            'atm_rr_10': safe_ratio(self.v_atm, abs(self.rr_10)),
            'atm_bf_25': safe_ratio(self.v_atm, self.bf_25),
            'atm_bf_10': safe_ratio(self.v_atm, self.bf_10),
            'rr_ratio': safe_ratio(self.rr_10, self.rr_25),
            'bf_ratio': safe_ratio(self.bf_10, self.bf_25),
        }

    def copy_with(self, **changes) -> 'MarketData':
        """Return a copy with the supplied dataclass fields replaced."""
        return replace(self, **changes)

    @classmethod
    def from_vol_data(cls, vol_data: VolData, target_date: str, spot: float, fwd_pts: float) -> 'MarketData':
        """Create a MarketData instance from a VolData object."""
        interpolated = vol_data.get_interpolated_atm_rr_bf(target_date)

        return cls(
            s=spot,
            fwd_pts=fwd_pts,
            v_atm=interpolated['v_atm'],
            rr_25=interpolated['rr_25'],
            rr_10=interpolated['rr_10'],
            bf_25=interpolated['bf_25'],
            bf_10=interpolated['bf_10'],
        )


class Modifications(TypedDict, total=False):
    """Spot-dependent smile parameters.

    ``shifted_spot`` is the absolute spot used for a perturbation. The beta
    fields control ATM, RR, and BF changes as a function of log spot change.
    """
    shifted_spot: float
    beta_atm: float
    beta_rr: float
    beta_bf: float


"""Market quote containers used by the FX volatility models.

Volatilities and smile quotes are stored in percentage points. Forward points
are stored in market points and converted with ``forward_divisor`` only by
pricing/model components.
"""

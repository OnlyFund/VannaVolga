"""Public one-expiry wrapper; library consumers remain legacy unless opted in.

    prepared = prepare_smile(market, "2026-12-02", "2026-10-02", model="vv")
    resolver = WrappedVolResolver(prepared=prepared)
    vols = resolver.get_vol_for_strikes([1.05, 1.10, 1.15])
    legacy_vol = resolver.benchmark(1.10)

``broker_strangle=True`` is the default for raw MarketData. Pass False for
already two-vol BF quotes. Reuse ``prepared`` across strikes/resolvers; passing
it never reconverts. Raw inputs are detached snapshots, not modified markets.
Pair conventions govern divisor, ATM, premium adjustment and spot/forward
delta independently of legacy MarketData flags. Rates are decimal SIMPLE
deposits; quotes and returned vols are percentage points.

By default expected preparation/conversion failures warn and explicitly mark
diagnostics as degraded, returning the unchanged legacy model on ORIGINAL raw
quotes. ``fallback=False`` raises instead. Malformed dates, quotes, strikes,
model names and options always raise. Benchmark options affect only the legacy
comparison/fallback, not the explicit-pillar cores. No term-structure
interpolation or automatic consumer migration is performed here.
"""
import math
import warnings
from dataclasses import dataclass, field
from numbers import Real
from typing import Optional

import numpy as np

from apeiron.atlas.dates import to_date as _to_date
from resolved_smile_inputs import (
    MarketSnapshot, Pillars, PreparationError, ResolvedConventions,
    ResolvedQuotes, _freeze, _thaw, resolve_conventions,
)
from resolved_smile_cores import validate_strike
from smile_preparation import prepare_core

__all__ = ["prepare_smile", "WrappedVolResolver", "PreparedSmile",
           "SmileDiagnostics", "PreparationError", "DegradedSmileWarning",
           "ResolvedConventions", "ResolvedQuotes", "Pillars"]


class DegradedSmileWarning(RuntimeWarning):
    """A requested convention-aware result was replaced by the legacy result."""


@dataclass(frozen=True)
class SmileDiagnostics:
    """``status`` is "resolved", "fallback" (a documented substitute was used
    and ``reason`` says which) or "degraded" (the legacy model was used)."""
    status: str
    reason: Optional[str] = None
    stage: Optional[str] = None
    premium_matches: tuple = ()
    sabr_parameters: Optional[tuple] = None
    sabr_calibration_relative_rms: Optional[float] = None

    @property
    def degraded(self):
        return self.status == "degraded"

    @property
    def fallback(self):
        return self.status == "fallback"


@dataclass(frozen=True)
class PreparedSmile:
    raw: MarketSnapshot
    model: str
    broker_strangle: bool
    beta: float
    expiry_date: object
    valuation_date: object
    conventions: Optional[ResolvedConventions]
    resolved: Optional[ResolvedQuotes]
    pillars: Optional[Pillars]
    diagnostics: SmileDiagnostics
    fallback: bool
    _core: object = field(repr=False, compare=False)
    _legacy_params: dict = field(default_factory=dict, repr=False, compare=False)


def _validate_options(model, broker_strangle, beta, fallback):
    if model not in ("vv", "vv-sabr"):
        raise ValueError("model must be 'vv' or 'vv-sabr'")
    if not isinstance(broker_strangle, bool) or not isinstance(fallback, bool):
        raise ValueError("broker_strangle and fallback must be bool")
    if isinstance(beta, bool) or not isinstance(beta, Real) or not math.isfinite(beta) or not 0 <= beta <= 1:
        raise ValueError("beta must be finite and between 0 and 1")


def _warn(reason):
    stage = reason.split(":", 1)[0]
    if stage.startswith("evaluation"):
        stage = "evaluation"
    warnings.warn("Smile {} failed; using legacy fallback.".format(stage),
                  DegradedSmileWarning, stacklevel=3)


def prepare_smile(market, expiry_date, valuation_date, model="vv",
                  broker_strangle=True, beta=.5, fallback=True):
    """Resolve and cache a strike-independent smile, with visible fallback."""
    _validate_options(model, broker_strangle, beta, fallback)
    raw = MarketSnapshot.capture(market)
    expiry, valuation = _to_date(expiry_date), _to_date(valuation_date)
    if expiry <= valuation:
        raise ValueError("expiry must be after valuation")
    if broker_strangle:
        if min(raw.v_atm + raw.bf_25, raw.v_atm + raw.bf_10) <= 0:
            raise ValueError("broker common volatilities must be positive")
    elif min(raw.v_atm + raw.bf_25 - abs(raw.rr_25) / 2,
             raw.v_atm + raw.bf_10 - abs(raw.rr_10) / 2) <= 0:
        raise ValueError("two-vol wing volatilities must be positive")
    ctx = None
    stage = "conventions"
    try:
        ctx = resolve_conventions(raw, expiry, valuation)
        stage = "conversion" if broker_strangle else "preparation"
        quotes, pillars, core, matches, note = prepare_core(
            raw, ctx, model, broker_strangle, beta)
        diagnostics = SmileDiagnostics(
            "fallback" if note else "resolved",
            "{}: {}".format(stage, note) if note else None,
            stage if note else None,
            premium_matches=matches,
            sabr_parameters=getattr(core, "parameters", None),
            sabr_calibration_relative_rms=getattr(core, "calibration_relative_rms", None))
    except (PreparationError, FloatingPointError, OverflowError) as exc:
        if not fallback:
            raise PreparationError("{}: {}".format(stage, exc)) from exc
        reason = "{}: {}".format(stage, exc)
        _warn(reason)
        quotes, pillars, core = None, None, None
        diagnostics = SmileDiagnostics("degraded", reason, stage)
    return PreparedSmile(raw, model, broker_strangle, beta, expiry, valuation,
                         ctx, quotes, pillars, diagnostics, fallback, core)


class WrappedVolResolver:
    """Strike evaluator over a prepared smile; benchmark uses untouched classes.

    Batch evaluation equals independent scalar calls (no batch-specific
    arbitrage projection). ``diagnostics`` is persistent: once any evaluation
    degrades, that reason remains visible even after a successful later call.
    """
    def __init__(self, market=None, expiry_date=None, valuation_date=None,
                 model=None, broker_strangle=None, beta=None, fallback=None,
                 prepared=None, CallPut="CALL", benchmark_kwargs=None):
        if not isinstance(CallPut, str) or CallPut.upper() not in ("CALL", "PUT"):
            raise ValueError("CallPut must be CALL or PUT")
        self.CallPut = CallPut.upper()
        options = dict(benchmark_kwargs or {})
        supported = {"deltaBase", "atm_conv", "smooth_pillar_interpolation",
                     "legacy_target_blend", "compatibility_mode", "frozen_sabr_params",
                     "sabr_calibration_context"}
        if set(options) - supported:
            raise ValueError("unsupported benchmark option(s): {}".format(set(options) - supported))
        if "deltaBase" in options and not isinstance(options["deltaBase"], bool):
            raise ValueError("benchmark deltaBase must be bool")
        for key in ("smooth_pillar_interpolation", "legacy_target_blend"):
            if key in options and not isinstance(options[key], bool):
                raise ValueError("benchmark {} must be bool".format(key))
        for key in ("frozen_sabr_params", "sabr_calibration_context"):
            if key in options and options[key] is not None and not isinstance(options[key], dict):
                raise ValueError("benchmark {} must be dict or None".format(key))
        if "atm_conv" in options:
            from helper_functions.VannaVolga_function import normalize_atm_type
            options["atm_conv"] = normalize_atm_type(options["atm_conv"])
        self._benchmark_options = _freeze(options)
        if prepared is not None:
            if not isinstance(prepared, PreparedSmile):
                raise TypeError("prepared must be PreparedSmile")
            if any(value is not None for value in (
                    market, expiry_date, valuation_date, model, broker_strangle, beta)):
                raise ValueError("prepared cannot be combined with raw preparation arguments")
            self.prepared = prepared
            if fallback is not None and not isinstance(fallback, bool):
                raise ValueError("fallback must be bool")
            self.fallback = prepared.fallback if fallback is None else fallback
            if prepared.diagnostics.degraded and not self.fallback:
                raise PreparationError(prepared.diagnostics.reason)
        else:
            self.fallback = True if fallback is None else fallback
            self.prepared = prepare_smile(
                market, expiry_date, valuation_date, model="vv" if model is None else model,
                broker_strangle=True if broker_strangle is None else broker_strangle,
                beta=.5 if beta is None else beta, fallback=self.fallback)
        self.diagnostics = self.prepared.diagnostics

    @property
    def resolved_inputs(self):
        return self.prepared

    def benchmark_resolver(self, strike):
        strike = validate_strike(strike)
        from VannaVolga_ImpliedVol import VannaVolga
        from vanna_volga_sabr import VannaVolga_SABR
        cls = VannaVolga if self.prepared.model == "vv" else VannaVolga_SABR
        options = _thaw(self._benchmark_options)
        if self.prepared.model == "vv-sabr":
            options["beta"] = self.prepared.beta
            if options.get("frozen_sabr_params") is None:
                options["frozen_sabr_params"] = self._legacy_sabr_parameters(cls, options)
        return cls(market=self.prepared.raw.to_market(), K=strike,
                   expiry_date=self.prepared.expiry_date.isoformat(),
                   valuation_date=self.prepared.valuation_date.isoformat(),
                   CallPut=self.CallPut, **options)

    def _legacy_sabr_parameters(self, cls, options):
        # Legacy SABR calibration is strike independent; calibrate once per smile.
        key = repr(sorted(options.items()))
        cache = self.prepared._legacy_params
        if key not in cache:
            raw = self.prepared.raw
            model = cls(market=raw.to_market(), K=raw.s + raw.fwd_pts / raw.forward_divisor,
                        expiry_date=self.prepared.expiry_date.isoformat(),
                        valuation_date=self.prepared.valuation_date.isoformat(),
                        CallPut="CALL", **options)
            alpha, rho, nu = model.calibrate_sabr()
            cache[key] = None if alpha is None else {
                "alpha": float(alpha), "rho": float(rho), "nu": float(nu)}
        return cache[key]

    def benchmark(self, strike):
        """Legacy result on original raw quotes and explicit benchmark options."""
        return self.benchmark_resolver(strike).get_vol_for_strike()

    def get_vol_for_strike(self, strike):
        strike = validate_strike(strike)
        if self.prepared._core is None:
            return self.benchmark(strike)
        try:
            return self.prepared._core.get_vol_for_strike(strike)
        except (PreparationError, FloatingPointError, OverflowError) as exc:
            if not self.fallback:
                raise
            reason = "evaluation at strike {}: {}".format(strike, exc)
            _warn(reason)
            self.diagnostics = SmileDiagnostics("degraded", reason, "evaluation",
                                               self.prepared.diagnostics.premium_matches,
                                               self.prepared.diagnostics.sabr_parameters,
                                               self.prepared.diagnostics.sabr_calibration_relative_rms)
            return self.benchmark(strike)

    def get_vol_for_strikes(self, strikes):
        strikes = tuple(validate_strike(k) for k in strikes)
        return np.array([self.get_vol_for_strike(k) for k in strikes])

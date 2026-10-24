"""Opt-in application model seam and detached, bounded per-thread preparation."""
from collections import OrderedDict
from dataclasses import fields
import threading
import re
import uuid
import weakref

from wrapped_vol_resolver import prepare_smile, WrappedVolResolver
from resolved_smile_cores import validate_strike

_local = threading.local()
_collectors = weakref.WeakValueDictionary()
_registry_lock = threading.RLock()


class DiagnosticsCollector:
    """Session-owned diagnostics; workers only append immutable records."""
    def __init__(self):
        self.token = uuid.uuid4().hex
        self._records = []
        self._lock = threading.RLock()
        with _registry_lock:
            _collectors[self.token] = self

    def append(self, record):
        with self._lock:
            if record not in self._records:
                self._records.append(record)

    def records(self):
        with self._lock:
            return tuple(self._records)


#: Prefix marking a diagnostics record as a documented fallback rather than
#: a degradation to the legacy model.
FALLBACK_PREFIX = "fallback - "


def record(market, model, expiry, reason, status="degraded"):
    reason = re.sub(r"^evaluation at strike [^:]+:", "evaluation:", str(reason))
    if status == "fallback" and not reason.startswith(FALLBACK_PREFIX):
        reason = FALLBACK_PREFIX + reason
    with _registry_lock:
        collector = _collectors.get(market.diagnostics_token)
    if collector is not None:
        collector.append((model, str(expiry), str(reason)))


def cache_key(market, expiry, valuation, model, broker, beta):
    # ``resolved`` restates the expiry and valuation already in the key.
    values = tuple((f.name, repr(getattr(market, f.name))) for f in fields(market)
                   if f.name not in ("diagnostics_token", "resolved"))
    return values, str(expiry)[:10], str(valuation)[:10], model, broker, beta


class SurfaceCache:
    """Session-owned, thread-safe store of calibrated smiles.

    Entries are keyed by the full market state, so a cached surface can only be
    reused for exactly the inputs it was calibrated on; bumped and scenario
    markets get their own entries.
    """
    def __init__(self, maxsize=4096):
        self.maxsize = maxsize
        self.token = None
        self._lock = threading.RLock()
        self._stores = {}

    def get(self, store, key, factory):
        with self._lock:
            entries = self._stores.setdefault(store, OrderedDict())
            if key in entries:
                entries.move_to_end(key)
                return entries[key]
        value = factory()
        self.put(store, key, value)
        return value

    def put(self, store, key, value):
        with self._lock:
            entries = self._stores.setdefault(store, OrderedDict())
            entries[key] = value
            entries.move_to_end(key)
            while len(entries) > self.maxsize:
                entries.popitem(last=False)

    def size(self, store):
        with self._lock:
            return len(self._stores.get(store, ()))

    def entries(self):
        """``{store: [(key, value), ...]}``: a copy of every cached entry."""
        with self._lock:
            return {store: list(items.items()) for store, items in self._stores.items()}

    def merge(self, entries):
        """Add ``entries`` (as from ``entries()``); keys are full market states,
        so an entry can only ever be reused for the inputs it was built from."""
        for store, items in entries.items():
            for key, value in items:
                self.put(store, key, value)


_caches = weakref.WeakValueDictionary()


def bind_cache(token, cache):
    """Route markets carrying ``token`` to a session's surface cache."""
    with _registry_lock:
        if cache.token is not None:
            _caches.pop(cache.token, None)
        cache.token = token
        _caches[token] = cache


def _cache_for(market):
    with _registry_lock:
        cache = _caches.get(getattr(market, "diagnostics_token", None))
    if cache is None:
        cache = getattr(_local, "cache", None)
        if cache is None:
            cache = _local.cache = SurfaceCache(256)
    return cache


#: Bump when cached smile objects change shape. The app keeps its surface
#: cache in the session, and Streamlit does not reload root modules such as
#: ``wrapped_vol_resolver``, so objects built by an older build can outlive a
#: code change; a new version starts a fresh cache instead of reusing them.
SURFACE_CACHE_VERSION = 7   # 7: the pure SABR smile model ("sabr")


def is_fallback(diagnostics):
    """True for a documented substitute smile.

    Reads ``status`` rather than the ``fallback`` property, which smile
    diagnostics built by an older loaded class do not have.
    """
    return getattr(diagnostics, "status", None) == "fallback"


def prepared_smile(market, expiry, valuation, model, broker=None, beta=.5):
    broker = market.broker_strangle if broker is None else broker
    key = cache_key(market, expiry, valuation, model, broker, beta)
    prepared = _cache_for(market).get("prepared", key, lambda: prepare_smile(
        market, str(expiry)[:10], str(valuation)[:10],
        model=model, broker_strangle=broker, beta=beta))
    if prepared.diagnostics.degraded or is_fallback(prepared.diagnostics):
        record(market, model, expiry, prepared.diagnostics.reason, prepared.diagnostics.status)
    return prepared


def reuse_prepared(market, prepared):
    _cache_for(market).put("prepared", cache_key(
        market, prepared.expiry_date, prepared.valuation_date,
        prepared.model, False, prepared.beta), prepared)


_LEGACY_CALIBRATION_KEYS = ("beta", "deltaBase", "atm_conv", "smooth_pillar_interpolation",
                            "legacy_target_blend", "compatibility_mode",
                            "sabr_calibration_context")


def legacy_sabr_parameters(market, expiry, valuation, kwargs):
    """Strike-independent legacy SABR calibration, computed once per market."""
    from vanna_volga_sabr import VannaVolga_SABR
    options = {k: v for k, v in kwargs.items() if k in _LEGACY_CALIBRATION_KEYS}
    key = (cache_key(market, expiry, valuation, "legacy-sabr", None, None),
           repr(sorted(options.items())))

    def calibrate():
        model = VannaVolga_SABR(market=market, K=market.s + market.fwd_pts / market.forward_divisor,
                                expiry_date=expiry, valuation_date=valuation, CallPut="CALL",
                                **options)
        alpha, rho, nu = model.calibrate_sabr()
        if alpha is None:
            return None
        return {"alpha": float(alpha), "rho": float(rho), "nu": float(nu)}
    return _cache_for(market).get("legacy-sabr", key, calibrate)


class AppVolResolver:
    model = "vv"

    def __init__(self, market, K, expiry_date, valuation_date, CallPut="CALL", **kwargs):
        self.market, self.K = market, K
        self.expiry_date, self.valuation_date = expiry_date, valuation_date
        self.CallPut = CallPut
        self.kwargs = kwargs
        self.resolver = None

    def get_vol_for_strike(self):
        validate_strike(self.K)
        if str(self.expiry_date)[:10] == str(self.valuation_date)[:10]:
            return float(self.market.v_atm)
        if self.market.force_legacy_smile:
            from helper_functions.vanna_volga_helpers import get_model_map
            for reason in self.market.smile_reasons:
                record(self.market, self.model, self.expiry_date, reason)
            kwargs = dict(self.kwargs)
            legacy_model = self.model
            if self.model == "sabr":
                record(self.market, self.model, self.expiry_date,
                       "standalone SABR unavailable on legacy path; using VV",
                       status="fallback")
                legacy_model = "vv"
            if (self.model == "vv-sabr" and kwargs.get("frozen_sabr_params") is None
                    and getattr(self.market, "frozen_sabr_params", None) is None):
                params = legacy_sabr_parameters(self.market, self.expiry_date,
                                                self.valuation_date, kwargs)
                if params is not None:
                    kwargs["frozen_sabr_params"] = params
            return get_model_map()[legacy_model](
                market=self.market, K=self.K, expiry_date=self.expiry_date,
                valuation_date=self.valuation_date, CallPut=self.CallPut,
                **kwargs).get_vol_for_strike()
        prepared = prepared_smile(self.market, self.expiry_date, self.valuation_date,
                                  self.model, beta=self.kwargs.get("beta", .5))
        options = {key: value for key, value in self.kwargs.items() if key in {
            "deltaBase", "atm_conv", "smooth_pillar_interpolation",
            "legacy_target_blend", "compatibility_mode", "frozen_sabr_params",
            "sabr_calibration_context"}}
        self.resolver = WrappedVolResolver(prepared=prepared, CallPut=self.CallPut,
                                           benchmark_kwargs=options)
        vol = self.resolver.get_vol_for_strike(self.K)
        if self.resolver.diagnostics.degraded:
            record(self.market, self.model, self.expiry_date,
                   self.resolver.diagnostics.reason)
        return vol


def market_diagnostics(market, expiry=None):
    with _registry_lock:
        collector = _collectors.get(market.diagnostics_token)
    records = collector.records() if collector is not None else ()
    reasons = tuple(dict.fromkeys(
        r[2] for r in records
        if expiry is None or r[1] in ("curve", str(expiry)[:10])))
    if not reasons:
        status = "resolved"
    elif all(reason.startswith(FALLBACK_PREFIX) for reason in reasons):
        status = "fallback"
    else:
        status = "degraded"
    return status, "; ".join(reasons)


class AppSabrResolver(AppVolResolver):
    model = "vv-sabr"


class AppPureSabrResolver(AppVolResolver):
    model = "sabr"

"""Calibrate once per market state; reuse the surface for every strike."""
import warnings

import pandas as pd
import pytest

import vanna_volga_sabr
from app.market_service import MarketService, pair_delivery
from app.smile_service import (
    AppSabrResolver, DiagnosticsCollector, SurfaceCache, bind_cache,
)
from vanna_volga_sabr import VannaVolga_SABR
from wrapped_vol_resolver import WrappedVolResolver
from test_app_wrapped_smile import app_curve, market


@pytest.fixture
def minimize_calls(monkeypatch):
    calls = []
    original = vanna_volga_sabr.minimize

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(vanna_volga_sabr, "minimize", counted)
    return calls


def test_legacy_fallback_calibrates_once_and_matches_fresh_legacy(minimize_calls):
    build, collector = app_curve(model="vv-sabr")
    cache = SurfaceCache()
    bind_cache(collector.token, cache)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        broken = build("2026-11-17")
    assert broken.force_legacy_smile
    del minimize_calls[:]
    strikes = (1.05, 1.1, 1.15)
    actual = [AppSabrResolver(broken, k, "2026-11-17", "2026-10-02",
                              atm_conv="dn", deltaBase=True).get_vol_for_strike()
              for k in strikes]
    assert 0 < len(minimize_calls) <= 9
    del minimize_calls[:]
    expected = [VannaVolga_SABR(market=broken, K=k, expiry_date="2026-11-17",
                                valuation_date="2026-10-02", CallPut="CALL",
                                atm_conv="dn", deltaBase=True).get_vol_for_strike()
                for k in strikes]
    assert len(minimize_calls) == 9 * len(strikes)
    assert actual == expected


def test_benchmark_calibrates_once_per_prepared_smile(minimize_calls):
    resolver = WrappedVolResolver(market(), "2026-12-02", "2026-10-02",
                                  model="vv-sabr", broker_strangle=False)
    del minimize_calls[:]
    actual = [resolver.benchmark(k) for k in (1.05, 1.1, 1.15)]
    assert len(minimize_calls) == 9
    raw = resolver.prepared.raw.to_market()
    expected = [VannaVolga_SABR(market=raw, K=k, expiry_date="2026-12-02",
                                valuation_date="2026-10-02", CallPut="CALL",
                                beta=.5).get_vol_for_strike()
                for k in (1.05, 1.1, 1.15)]
    assert actual == expected


def test_curve_conversion_survives_reruns_with_new_collectors(monkeypatch):
    rows = pd.DataFrame(dict(
        tenor=["1M", "2M"], expiry=pd.to_datetime(["2026-11-02", "2026-12-02"]),
        atm=[10., 10.], rr_25=[-.5, -.5], rr_10=[-1., -1.],
        bf_25=[.2, .2], bf_10=[.6, .6]))
    service = MarketService(DiagnosticsCollector())
    progress = []
    service.progress = lambda i, n, label: progress.append((i, n, label))
    args = (rows, "2026-10-02", 1.1, "EURUSD", 10000., "vv", True,
            lambda expiry: 30., lambda expiry: .05)
    first = service.convert_curve(*args)
    assert progress == [(0, 2, "1M"), (1, 2, "2M")]
    service.collector = DiagnosticsCollector()  # Streamlit rerun
    monkeypatch.setattr("app.market_service.prepared_smile",
                        lambda *a, **k: pytest.fail("curve recalibrated on rerun"))
    second = service.convert_curve(*args)
    pd.testing.assert_frame_equal(first[0], second[0])
    assert second[2] is first[2]


def test_session_cache_routes_by_token_and_rebinds():
    cache = SurfaceCache()
    first, second = DiagnosticsCollector(), DiagnosticsCollector()
    bind_cache(first.token, cache)
    bind_cache(second.token, cache)
    from app.smile_service import _cache_for, prepared_smile
    m = market(diagnostics_token=second.token)
    prepared_smile(m, "2026-12-02", "2026-10-02", "vv")
    assert cache.size("prepared") == 1
    assert _cache_for(market(diagnostics_token=first.token)) is not cache

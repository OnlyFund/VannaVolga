import builtins
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from VannaVolga_ImpliedVol import VannaVolga
from wrapped_vol_resolver import (
    DegradedSmileWarning, PreparationError, WrappedVolResolver, prepare_smile,
)
from test_resolved_smile_inputs import market


def test_public_default_and_benchmark():
    m = market()
    resolver = WrappedVolResolver(m, "2026-12-02", "2026-10-02",
                                  model="vv", fallback=False)
    assert resolver.prepared.broker_strangle
    assert resolver.diagnostics.status == "resolved"
    assert len(resolver.diagnostics.premium_matches) == 2
    expected = VannaVolga(market=m, K=1.1, expiry_date="2026-12-02",
                   valuation_date="2026-10-02", CallPut="CALL").get_vol_for_strike()
    assert resolver.benchmark(1.1) == expected
    strikes = [1.08, 1.1, 1.12, 1.08]
    assert np.array_equal(resolver.get_vol_for_strikes(strikes),
                          [resolver.get_vol_for_strike(k) for k in strikes])


def test_unmatched_sabr_conversion_degrades_to_unchanged_legacy_model():
    m = market()
    with pytest.warns(DegradedSmileWarning, match="conversion"):
        resolver = WrappedVolResolver(
            m, "2026-12-02", "2026-10-02", model="vv-sabr")
    assert resolver.diagnostics.degraded
    assert resolver.diagnostics.stage == "conversion"
    assert resolver.get_vol_for_strike(1.1) == resolver.benchmark(1.1)


def test_prepared_reuse_no_double_conversion(monkeypatch):
    m = market()
    prepared = prepare_smile(m, "2026-12-02", "2026-10-02", fallback=False)
    m.v_atm = 20
    m.bf_25 = 5
    with pytest.raises(FrozenInstanceError):
        prepared.resolved.bf_25 = 8
    monkeypatch.setattr("wrapped_vol_resolver.prepare_core",
                        lambda *args: pytest.fail("prepared smile must not be rebuilt"))
    first = WrappedVolResolver(prepared=prepared)
    second = WrappedVolResolver(prepared=prepared)
    assert first.prepared._core is second.prepared._core
    assert first.get_vol_for_strike(1.1) == second.get_vol_for_strike(1.1)
    assert first.prepared.raw.v_atm == 10
    with pytest.raises(ValueError):
        WrappedVolResolver(prepared=prepared, broker_strangle=True)


def test_direct_two_vol_does_not_call_converter(monkeypatch):
    monkeypatch.setattr("smile_preparation.broker_to_twovol_bf",
                        lambda *args, **kwargs: pytest.fail("no broker conversion"))
    prepared = prepare_smile(market(), "2026-12-02", "2026-10-02", broker_strangle=False)
    assert prepared.resolved.bf_25 == prepared.raw.bf_25
    assert prepared.diagnostics.premium_matches == ()


@pytest.mark.parametrize("pair", ["USDEUR", "ABCDEF", None])
@pytest.mark.parametrize("model", ["vv", "vv-sabr"])
def test_convention_fallback_warning_and_exact_legacy(pair, model):
    with pytest.warns(DegradedSmileWarning, match="conventions"):
        resolver = WrappedVolResolver(market(pair), "2026-12-02", "2026-10-02", model=model)
    assert resolver.diagnostics.degraded
    assert resolver.diagnostics.reason
    assert resolver.get_vol_for_strike(1.1) == resolver.benchmark(1.1)
    with pytest.raises(PreparationError):
        prepare_smile(market(pair), "2026-12-02", "2026-10-02", model=model, fallback=False)


def test_missing_quantlib_degrades_through_convention_fallback(monkeypatch):
    original_import = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name == "QuantLib":
            raise ImportError("QuantLib unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable)
    with pytest.warns(DegradedSmileWarning, match="conventions"):
        resolver = WrappedVolResolver(market(), "2026-12-02", "2026-10-02")
    assert resolver.diagnostics.degraded
    assert resolver.diagnostics.stage == "conventions"
    assert resolver.get_vol_for_strike(1.1) == resolver.benchmark(1.1)


@pytest.mark.parametrize("stage", ["conversion", "preparation"])
def test_expected_preparation_failure(monkeypatch, stage):
    def fail(*args, **kwargs):
        raise PreparationError("test numerical failure")
    monkeypatch.setattr("wrapped_vol_resolver.prepare_core", fail)
    with pytest.warns(DegradedSmileWarning, match=stage):
        resolver = WrappedVolResolver(market(), "2026-12-02", "2026-10-02",
                                      broker_strangle=stage == "conversion")
    assert resolver.diagnostics.stage == stage
    assert resolver.get_vol_for_strike(1.1) == resolver.benchmark(1.1)


@pytest.mark.parametrize("strike", [0, -1, float("nan"), float("inf"), True, "bad"])
def test_invalid_strikes_not_swallowed(strike):
    resolver = WrappedVolResolver(market(), "2026-12-02", "2026-10-02")
    with pytest.raises(ValueError):
        resolver.get_vol_for_strike(strike)
    with pytest.raises(ValueError):
        resolver.get_vol_for_strikes([1.1, strike])
    assert not resolver.diagnostics.degraded


@pytest.mark.parametrize("options", [
    {"model": "bad"}, {"beta": float("nan")}, {"beta": 2},
    {"broker_strangle": "yes"}, {"fallback": 1}, {"CallPut": "bad"},
    {"benchmark_kwargs": {"deltaBase": 1}}, {"benchmark_kwargs": {"unknown": 1}},
    {"benchmark_kwargs": {"smooth_pillar_interpolation": 1}},
    {"benchmark_kwargs": {"frozen_sabr_params": "bad"}}])
def test_invalid_options_not_swallowed(options):
    with pytest.raises(ValueError):
        WrappedVolResolver(market(), "2026-12-02", "2026-10-02", **options)


def test_invalid_mutated_input_and_dates_not_swallowed():
    m = market()
    m.s = float("nan")
    with pytest.raises(ValueError):
        prepare_smile(m, "2026-12-02", "2026-10-02")
    for expiry in ("bad", "2026-10-02", "2026-01-01"):
        with pytest.raises(ValueError):
            prepare_smile(market(), expiry, "2026-10-02")
    m = market()
    m.bf_25 = -11
    with pytest.raises(ValueError):
        prepare_smile(m, "2026-12-02", "2026-10-02")
    m = market()
    m.term_discount = -10
    with pytest.raises(ValueError, match="non-positive discount"):
        prepare_smile(m, "2026-12-02", "2026-10-02")
    m = market("USDJPY")
    m.fwd_pts = -20000  # Legacy divisor gives F=148, pair divisor gives F=-50.
    with pytest.raises(ValueError, match="non-positive forward"):
        prepare_smile(m, "2026-12-02", "2026-10-02")


def test_evaluation_failure_is_visible_not_silent(monkeypatch):
    resolver = WrappedVolResolver(market(), "2026-12-02", "2026-10-02")
    def fail(strike):
        raise PreparationError("wing inversion")
    monkeypatch.setattr(resolver.prepared._core, "get_vol_for_strike", fail)
    with pytest.warns(DegradedSmileWarning, match="evaluation"):
        assert resolver.get_vol_for_strike(1.1) == resolver.benchmark(1.1)
    assert resolver.diagnostics.degraded
    strict = WrappedVolResolver(prepared=resolver.prepared, fallback=False)
    with pytest.raises(PreparationError):
        strict.get_vol_for_strike(1.1)

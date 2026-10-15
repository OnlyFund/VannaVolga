from dataclasses import replace

import pytest
import QuantLib as ql

from resolved_smile_cores import make_core
from resolved_smile_inputs import (
    MarketSnapshot, PreparationError, ResolvedQuotes, build_pillars,
    resolve_conventions,
)
from smile_preparation import prepare_core
from test_resolved_smile_inputs import market


@pytest.mark.parametrize("pair,expiry", [
    ("EURUSD", "2026-12-02"), ("EURUSD", "2028-10-02"),
    ("USDJPY", "2026-12-02"), ("USDMXN", "2026-12-02")])
def test_joint_vv_conversion_matches_actual_broker_premiums(pair, expiry):
    raw = MarketSnapshot.capture(market(pair))
    ctx = resolve_conventions(raw, expiry, "2026-10-02")
    quotes, pillars, core, matches = prepare_core(raw, ctx, "vv", True)
    assert len(matches) == 2
    assert all(abs(match.residual) <= match.tolerance for match in matches)
    for match in matches:
        target = match.target
        # Independent Black premium calculation at FIXED common-vol strikes,
        # not candidate delta pillars or another conversion's premiums.
        broker = sum(ql.blackFormula(phi, strike, ctx.forward,
                                    target.common_vol / 100 * ctx.time ** .5,
                                    ctx.domestic_discount)
                     for phi, strike in ((ql.Option.Put, target.put_strike),
                                         (ql.Option.Call, target.call_strike)))
        actual = sum(ql.blackFormula(phi, strike, ctx.forward,
                                    core.get_vol_for_strike(strike) / 100 * ctx.time ** .5,
                                    ctx.domestic_discount)
                     for phi, strike in ((ql.Option.Put, target.put_strike),
                                         (ql.Option.Call, target.call_strike)))
        assert abs(actual - broker) <= match.tolerance
    for strike, vol in zip(pillars.strikes, pillars.vols):
        assert core.get_vol_for_strike(strike) == pytest.approx(vol, abs=1e-9)
    assert raw.bf_25 == .2


@pytest.mark.parametrize("delta", ["spot", "forward"])
@pytest.mark.parametrize("pa", [False, True])
def test_vv_joint_conversion_supports_delta_and_adjustment_conventions(delta, pa):
    raw = MarketSnapshot.capture(market())
    ctx = replace(resolve_conventions(raw, "2026-12-02", "2026-10-02"),
                  delta_convention=delta, premium_adjusted=pa)
    assert all(abs(m.residual) <= m.tolerance for m in
               prepare_core(raw, ctx, "vv", True)[3])


def test_sabr_without_a_premium_match_fails_explicitly():
    raw = MarketSnapshot.capture(market())
    ctx = resolve_conventions(raw, "2026-12-02", "2026-10-02")
    with pytest.raises(PreparationError, match="premium residual"):
        prepare_core(raw, ctx, "vv-sabr", True)


def test_sabr_successful_conversion_matches_both_broker_premiums(monkeypatch):
    m = market()
    m.rr_25, m.rr_10 = -.1, -.2
    m.bf_25, m.bf_10 = .1, .3
    raw = MarketSnapshot.capture(m)
    ctx = resolve_conventions(raw, "2026-12-02", "2026-10-02")
    _, _, core, matches = prepare_core(raw, ctx, "vv-sabr", True)
    assert all(abs(match.residual) <= match.tolerance for match in matches)
    monkeypatch.setattr(core, "_new_legacy_model",
                        lambda *args: pytest.fail("strike queries must reuse calibration"))
    for match in matches:
        target = match.target
        premium = sum(
            ql.blackFormula(phi, strike, ctx.forward,
                            core.get_vol_for_strike(strike) / 100 * ctx.time ** .5,
                            ctx.domestic_discount)
            for phi, strike in ((ql.Option.Put, target.put_strike),
                                (ql.Option.Call, target.call_strike)))
        assert abs(premium - target.premium) <= match.tolerance


def test_zero_rr_identity_and_no_conversion():
    m = market()
    m.rr_25 = m.rr_10 = 0
    raw = MarketSnapshot.capture(m)
    ctx = resolve_conventions(raw, "2026-12-02", "2026-10-02")
    quotes, _, _, _ = prepare_core(raw, ctx, "vv", True)
    assert quotes.bf_25 == pytest.approx(raw.bf_25, abs=1e-8)
    assert quotes.bf_10 == pytest.approx(raw.bf_10, abs=1e-8)
    quotes, _, _, matches = prepare_core(raw, ctx, "vv-sabr", False)
    assert quotes.bf_25 == raw.bf_25
    assert quotes.bf_10 == raw.bf_10
    assert matches == ()


def test_converter_seed_is_not_final_result(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("scalar seed unavailable")
    monkeypatch.setattr("smile_preparation.broker_to_twovol_bf", fail)
    raw = MarketSnapshot.capture(market())
    ctx = resolve_conventions(raw, "2026-12-02", "2026-10-02")
    assert all(abs(m.residual) <= m.tolerance for m in prepare_core(raw, ctx, "vv", True)[3])


def test_non_matching_solver_result_is_rejected(monkeypatch):
    from types import SimpleNamespace
    from resolved_smile_inputs import PreparationError
    monkeypatch.setattr("smile_preparation.least_squares",
                        lambda *args, **kwargs: SimpleNamespace(success=True, x=(.2, .6)))
    raw = MarketSnapshot.capture(market())
    ctx = resolve_conventions(raw, "2026-12-02", "2026-10-02")
    with pytest.raises(PreparationError, match="residual"):
        prepare_core(raw, ctx, "vv", True)


def test_existing_models_remain_distinct_and_wings_continuous():
    raw = MarketSnapshot.capture(market())
    ctx = resolve_conventions(raw, "2026-12-02", "2026-10-02")
    quotes = ResolvedQuotes(raw.v_atm, raw.rr_25, raw.rr_10,
                            raw.bf_25, raw.bf_10)
    pillars = build_pillars(ctx, quotes, raw.s)
    vv = make_core("vv", ctx, pillars)
    sabr = make_core("vv-sabr", ctx, pillars)
    for core in (vv, sabr):
        for i in (0, 4):
            edge = core.pillars.strikes[i]
            at = core.get_vol_for_strike(edge)
            assert core.get_vol_for_strike(edge * (1 - 1e-7)) == pytest.approx(at, abs=1e-4)
            assert core.get_vol_for_strike(edge * (1 + 1e-7)) == pytest.approx(at, abs=1e-4)
    strike = vv.pillars.strikes[0] * .98
    assert abs(vv.get_vol_for_strike(strike) - sabr.get_vol_for_strike(strike)) > 1e-3

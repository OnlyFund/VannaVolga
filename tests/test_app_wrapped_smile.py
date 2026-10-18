import ast
from copy import deepcopy
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import QuantLib as ql

from market_data import MarketData
from app.market_service import MarketService, pair_delivery, pair_divisor
from app.smile_service import DiagnosticsCollector, prepared_smile, market_diagnostics
from helper_functions.input_resolvers import resolve_strike
from helper_functions.vanna_volga_helpers import get_model_map
from wrapped_vol_resolver import DegradedSmileWarning
from pricing.pricing import price_option
from pricing.greeks import calculate_greeks
from app.reports.spot_ladder import calculate_spot_ladder


def market(**changes):
    kwargs = dict(s=1.1, fwd_pts=30, v_atm=10, rr_25=-.5, rr_10=-1,
                  bf_25=.2, bf_10=.6, currency_pair="EURUSD",
                  wrapped_smile=True, broker_strangle=False)
    kwargs.update(changes)
    return MarketData(**kwargs)


def app_curve(model="vv", broker=True, pair="EURUSD"):
    """The app's market builder (``MarketContext``) on a two-pillar curve."""
    from app.market_context import MarketContext, PricingSettings
    rows = pd.DataFrame(dict(
        tenor=["1M", "2M"], expiry=pd.to_datetime(["2026-11-02", "2026-12-02"]),
        atm=[10., 10.], rr_25=[-.5, -.5], rr_10=[-1., -1.],
        bf_25=[.2, .2], bf_10=[.6, .6]))
    code = pair.replace("/", "").replace(" ", "").upper()
    forward = pd.DataFrame(dict(
        delivery_date=[pair_delivery(code, x) for x in rows.expiry],
        fwd_pts=[30., 30.], term_discount=[.05, .05]))
    collector = DiagnosticsCollector()
    context = MarketContext(
        pair, date(2026, 10, 2), 150. if code == "USDJPY" else 1.1, pair_divisor(code),
        broker, rows, forward, settings=PricingSettings(pricing_model=model, delta_reporting=None),
        market_service=MarketService(collector), diagnostics=collector,
    )
    return context.market_for_expiry, collector


def test_conversion_precedes_variance_interpolation_and_reused_once(monkeypatch):
    build, collector = app_curve()
    first, second = build("2026-11-02"), build("2026-12-02")
    assert not first.force_legacy_smile
    assert first.bf_25 != pytest.approx(.2, abs=1e-6)
    assert not first.broker_strangle
    original = prepared_smile(first, "2026-11-02", "2026-10-02", "vv")
    assert original.broker_strangle  # Exact pillar reuses its original solve.
    monkeypatch.setattr("smile_preparation.broker_to_twovol_bf",
                        lambda *a, **kw: pytest.fail("broker conversion repeated"))
    broken = build("2026-11-17")
    assert not broken.force_legacy_smile
    prepared = prepared_smile(broken, "2026-11-17", "2026-10-02", "vv")
    assert not prepared.broker_strangle
    assert prepared.resolved.bf_25 == pytest.approx(broken.bf_25)
    bumped = build("2026-11-17", spot_value=1.11)
    assert bumped.bf_25 == broken.bf_25
    assert prepared_smile(bumped, "2026-11-17", "2026-10-02", "vv").raw.s == 1.11
    # Rolling the curve converts each tenor's broker quote again at its rolled
    # expiry and the new valuation date (sticky tenor), so lift the guard.
    monkeypatch.undo()
    rolled = build("2026-11-17", as_of=date(2026, 10, 5))
    assert not rolled.force_legacy_smile
    assert prepared_smile(rolled, "2026-11-17", "2026-10-05", "vv").valuation_date == date(2026, 10, 5)
    assert collector.records() == ()


def fail_pillars(monkeypatch, dates=None):
    """Make the curve's pillar conversion fail at ``dates`` (all if None)."""
    from dataclasses import replace
    from wrapped_vol_resolver import SmileDiagnostics
    import app.market_service as service
    original = service.prepared_smile

    def failing(market, expiry, valuation, model, broker):
        p = original(market, expiry, valuation, model, broker)
        if dates is None or str(expiry)[:10] in dates:
            return replace(p, diagnostics=SmileDiagnostics(
                "degraded", "synthetic pillar failure", "conversion"),
                resolved=None, pillars=None, _core=None)
        return p
    monkeypatch.setattr(service, "prepared_smile", failing)


def test_sabr_curve_converts_with_vv_butterflies_as_visible_fallback():
    vv_build, _ = app_curve(model="vv")
    build, collector = app_curve(model="vv-sabr")
    market = build("2026-11-17")
    assert not market.force_legacy_smile and market.smile_model == "vv-sabr"
    expected = vv_build("2026-11-17")
    assert market.bf_25 == pytest.approx(expected.bf_25, abs=1e-12)
    assert market.bf_10 == pytest.approx(expected.bf_10, abs=1e-12)
    status, reason = market_diagnostics(market, "2026-11-17")
    assert status == "fallback"
    assert "conversion fallback at 1M, 2M" in reason


def test_one_failed_pillar_borrows_the_neighbouring_strangle_adjustment(monkeypatch):
    fail_pillars(monkeypatch, {"2026-12-02"})
    build, collector = app_curve()
    first, failed = build("2026-11-02"), build("2026-12-02")
    assert not first.force_legacy_smile and not failed.force_legacy_smile
    # Same broker quotes on both pillars, so the 2M pillar takes the 1M
    # pillar's adjustment exactly (flat beyond the last converted pillar).
    assert first.bf_25 != pytest.approx(.2, abs=1e-6)
    assert failed.bf_25 == pytest.approx(first.bf_25, abs=1e-9)
    assert failed.bf_10 == pytest.approx(first.bf_10, abs=1e-9)
    status, reason = market_diagnostics(failed, "2026-12-02")
    assert status == "fallback"
    assert "conversion failed at 2M" in reason


def test_failure_of_every_pillar_retains_model_and_whole_original_curve(monkeypatch):
    fail_pillars(monkeypatch)
    build, collector = app_curve(model="vv-sabr")
    with pytest.warns(DegradedSmileWarning):
        broken = build("2026-11-17")
    assert broken.force_legacy_smile and broken.smile_model == "vv-sabr"
    assert broken.bf_25 == pytest.approx(.2)
    assert broken.bf_10 == pytest.approx(.6)
    assert collector.records()
    row = dict(strike=1.1, expiry="2026-11-17", valuation="2026-10-02",
               call_put="CALL", buy_sell="BUY", notional=1)
    actual = price_option(broken, row, "vv-sabr", "dn", True)[1]
    expected = get_model_map()["vv-sabr"](
        market=broken, K=1.1, expiry_date=row["expiry"],
        valuation_date=row["valuation"], CallPut="CALL").get_vol_for_strike()
    assert actual == expected


@pytest.mark.parametrize("pair", ["EURUSD", "USDJPY", "USDCAD", "USDTRY"])
def test_pair_delivery_and_forward_scale(pair):
    from helper_functions.fx_discounting import delivery_date
    assert pair_delivery(pair, "2026-12-24").date() == delivery_date(pair, "2026-12-24")
    from helper_functions.fx_conventions import get_pair
    assert pair_divisor(pair) == get_pair(pair).fwd_divisor


def test_unknown_delivery_warns_and_preserves_legacy_divisor():
    collector = DiagnosticsCollector()
    with pytest.warns(DegradedSmileWarning):
        assert pair_delivery("USDEUR", "2026-10-02", collector).date() == date(2026, 10, 6)
    assert pair_divisor("USDEUR", 100.) == 100.
    assert collector.records()


@pytest.mark.parametrize("model", ["vv", "vv-sabr"])
@pytest.mark.parametrize("pair", ["EURUSD", "USDJPY", "USDMXN"])
@pytest.mark.parametrize("cp", ["CALL", "PUT"])
def test_alias_delta_verified_on_selected_model(model, pair, cp):
    m = market(currency_pair=pair, s=150 if pair == "USDJPY" else 1.1,
               forward_divisor=pair_divisor(pair), smile_model=model)
    prepared = prepared_smile(m, "2026-12-02", "2026-10-02", model)
    assert not prepared.diagnostics.degraded
    row = dict(strike="25d", expiry="2026-12-02", valuation="2026-10-02", call_put=cp)
    strike = resolve_strike(m, row, "spot", True)
    vol = get_model_map(m)[model](market=m, K=strike, expiry_date=row["expiry"],
                                valuation_date=row["valuation"], CallPut=cp).get_vol_for_strike()
    ctx = prepared.conventions
    kind = {"spot": ql.DeltaVolQuote.PaSpot if ctx.premium_adjusted else ql.DeltaVolQuote.Spot,
            "forward": ql.DeltaVolQuote.PaFwd if ctx.premium_adjusted else ql.DeltaVolQuote.Fwd}[ctx.delta_convention]
    calc = ql.BlackDeltaCalculator(ql.Option.Call if cp == "CALL" else ql.Option.Put,
                                  kind, m.s, ctx.domestic_discount, ctx.foreign_discount,
                                  vol / 100 * ctx.time ** .5)
    assert abs(calc.deltaFromStrike(strike)) == pytest.approx(.25, abs=1e-8)
    assert resolve_strike(m, dict(row, strike="atm"), "spot", True) == prepared.pillars.strikes[2]
    assert resolve_strike(m, dict(row, strike="atmf"), "spot", True) == ctx.forward
    assert resolve_strike(m, dict(row, strike="atms"), "spot", True) == m.s


def test_cache_invalidates_all_perturbations_and_metadata_copies():
    m = market()
    baseline = prepared_smile(m, "2026-12-02", "2026-10-02", "vv")
    assert prepared_smile(deepcopy(m), "2026-12-02", "2026-10-02", "vv") is baseline
    for changes in (dict(s=1.11), dict(bf_25=.21), dict(rr_10=-.9),
                    dict(term_discount=.06), dict(fwd_pts=31)):
        bumped = m.copy_with(**changes)
        assert bumped.wrapped_smile and not bumped.broker_strangle
        assert prepared_smile(bumped, "2026-12-02", "2026-10-02", "vv") is not baseline
    assert prepared_smile(m, "2026-12-03", "2026-10-02", "vv") is not baseline


def test_pricing_greeks_and_both_ladders_use_wrapper_without_reconversion(monkeypatch):
    build, collector = app_curve(broker=False)
    m = build("2026-11-17")
    monkeypatch.setattr("smile_preparation.broker_to_twovol_bf",
                        lambda *a, **kw: pytest.fail("two-vol scenario reconverted"))
    row = pd.Series(dict(currency_pair="EURUSD", strike=1.1, expiry=date(2026, 11, 17),
                         valuation=date(2026, 10, 2), call_put="CALL", buy_sell="BUY",
                         notional="1", premium=0., premium_ccy="USD"))
    price_option(m, row, "vv", "dn", True)
    greeks = calculate_greeks(m, row, None, "vv", "dn", True, {},
                             next_market=build(row.expiry, as_of=date(2026, 10, 5)),
                             next_valuation=date(2026, 10, 5))
    assert all(np.isfinite(x) for x in greeks.values())
    results = []
    for mode in ("reference", "optimized"):
        levels, errors = calculate_spot_ladder(
            [(0, row)], [1.099, 1.101], date(2026, 10, 2), build,
            lambda x: x, "vv", "dn", True, {}, mode=mode, scenario_workers=2)
        assert not errors
        assert len(levels) == 2
        results.append(levels)
    for a, b in zip(*results):
        for key in a:
            assert a[key] == pytest.approx(b[key], rel=1e-8, abs=1e-8)
    assert collector.records() == ()


def test_greek_only_degradation_is_collected(monkeypatch):
    from resolved_smile_inputs import PreparationError
    from resolved_smile_cores import VannaVolgaCore
    collector = DiagnosticsCollector()
    m = market(diagnostics_token=collector.token)
    original = VannaVolgaCore.get_vol_for_strike
    def failing(self, strike):
        if self.context.spot != 1.1:
            raise PreparationError("synthetic bump failure")
        return original(self, strike)
    monkeypatch.setattr(VannaVolgaCore, "get_vol_for_strike", failing)
    row = dict(strike=1.1, expiry="2026-12-02", valuation="2026-10-02",
               call_put="CALL", buy_sell="BUY", notional=1)
    price_option(m, row, "vv", "dn", True)
    assert not collector.records()
    with pytest.warns(DegradedSmileWarning):
        calculate_greeks(m, row, None, "vv", "dn", True, {})
    assert market_diagnostics(m)[0] == "degraded"
    assert "synthetic bump failure" in market_diagnostics(m)[1]


def test_save_load_defaults_and_removed_controls():
    import app.ui.app as ui
    from app.ui import market_block
    source = ast.parse(Path(ui.__file__).read_text(encoding="utf-8-sig"))
    render = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "render_app")
    load = next(n for n in render.body if isinstance(n, ast.FunctionDef) and n.name == "apply_loaded_market")
    class State(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__
    state = State()
    namespace = dict(vars(ui), st=SimpleNamespace(session_state=state))
    exec(compile(ast.Module(body=[load]), ui.__file__, "exec"), namespace)
    payload = dict(currency_pair="eur / usd", valuation_date="2026-10-02",
                   spot=1.1, event_calendar=[], vol_data=[dict(
                       tenor="1M", expiry="2026-11-02", atm=10, rr_25=-.5,
                       bf_25=.2, rr_10=-1, bf_10=.6)],
                   forward_points=[dict(tenor="1M", delivery_date="2026-11-04",
                                        fwd_pts=30)])
    namespace["apply_loaded_market"](payload)
    loaded = state["_pending_market_load"]
    assert loaded["currency_pair"] == "EURUSD"
    assert loaded["broker_strangle"] is True
    assert loaded["forward_divisor"] == 10000.
    assert loaded["forward_rows"].term_discount.iloc[0] == .05
    namespace["apply_loaded_market"](dict(payload, broker_strangle=False, forward_divisor=100.))
    assert not state["_pending_market_load"]["broker_strangle"]
    assert state["_pending_market_load"]["forward_divisor"] == 100.
    labels = [n.args[0].s for n in ast.walk(render) if isinstance(n, ast.Call)
              and isinstance(n.func, ast.Attribute) and n.func.attr in ("selectbox", "checkbox")
              and n.args and isinstance(n.args[0], ast.Str)]
    assert "ATM convention" not in labels
    # Deltas follow each pair's convention, spot-equivalent in totals; no
    # global setting.
    assert "Delta convention" not in labels
    from app.market_context import PricingSettings
    assert PricingSettings().delta_reporting == "spot"
    assert "Use smooth 5-pillar interpolation" not in labels
    assert "Pricing model" in labels and "Premium quote" not in labels
    # The BF convention is per pair, on the Markets tab, not a global setting.
    assert "BF quotes are broker strangles" not in labels
    assert state["_pending_market_load"]["_broker_strangle_pair"] == "EURUSD"
    market_source = Path(market_block.__file__).read_text(encoding="utf-8-sig")
    assert 'st.session_state.get("broker_strangle", True)' in market_source
    from app.market_store import market_payload
    frames = state["_pending_market_load"]
    saved = market_payload("EURUSD", "2026-10-02", 1.1, 1e4, False, frames["vol_rows"],
                           frames["forward_rows"], frames["event_rows"])
    assert saved["broker_strangle"] is False
    assert "Simple term deposit rate" in market_source


def test_simple_rate_conversion_is_percentage_adapted(monkeypatch):
    import smile_preparation
    original = smile_preparation.broker_to_twovol_bf
    rates = []
    def capture(*args, **kwargs):
        rates.append(args[11])
        return original(*args, **kwargs)
    monkeypatch.setattr(smile_preparation, "broker_to_twovol_bf", capture)
    m = market(broker_strangle=True, term_discount=.05123)
    prepared = prepared_smile(m, "2026-12-02", "2026-10-02", "vv")
    assert not prepared.diagnostics.degraded
    assert rates == pytest.approx([5.123, 5.123])


def test_failed_curve_never_mixes_converted_and_original_quotes(monkeypatch):
    """A failed pillar never feeds raw broker BFs into a converted curve."""
    fail_pillars(monkeypatch)
    build, collector = app_curve()
    with pytest.warns(DegradedSmileWarning):
        first = build("2026-11-02")
    broken = build("2026-11-17")
    assert first.force_legacy_smile and broken.force_legacy_smile
    assert first.bf_25 == pytest.approx(.2)
    assert broken.bf_25 == pytest.approx(.2)
    assert deepcopy(broken).smile_reasons == ("synthetic pillar failure",)
    assert market_diagnostics(broken)[0] == "degraded"


def test_expiry_theta_preserves_intrinsic_expiry_handling():
    build, collector = app_curve(broker=False)
    at_expiry = build(date(2026, 11, 2), as_of=date(2026, 11, 2),
                      allow_short_end=True)
    row = dict(strike=1.1, expiry="2026-11-02", valuation="2026-11-02",
               call_put="CALL", buy_sell="BUY", notional=1)
    strike, vol, price = price_option(at_expiry, row, "vv", "dn", True)
    assert price >= 0 and vol == at_expiry.v_atm
    assert set(calculate_greeks(at_expiry, row, None, "vv", "dn", True, {}).values()) == {0.}
    assert not collector.records()


def test_pair_normalized_ui_value_does_not_drop_market_conventions():
    build, collector = app_curve(pair="eur / usd")
    m = build("2026-11-17")
    assert m.currency_pair == "EURUSD"
    from helper_functions.fx_conventions import get_pair
    assert m.deltaBase == get_pair("EURUSD").premium_adjusted


@pytest.mark.parametrize("strike", [0, -1, float("nan"), float("inf"), True])
@pytest.mark.parametrize("legacy,expiry", [
    (True, "2026-12-02"), (False, "2026-10-02")])
def test_app_fallback_and_expiry_do_not_hide_invalid_strikes(strike, legacy, expiry):
    m = market(force_legacy_smile=legacy)
    resolver = get_model_map(m)["vv"](
        market=m, K=strike, expiry_date=expiry, valuation_date="2026-10-02")
    with pytest.raises(ValueError, match="strike"):
        resolver.get_vol_for_strike()


def test_ui_omits_verbose_diagnostic_tables():
    import app.ui.app as ui
    source = Path(ui.__file__).read_text(encoding="utf-8-sig")
    assert "st.dataframe(status_rows" not in source
    assert "legacy fallback: " not in source
    assert "Some calculations used legacy fallback." in source

"""On-demand Greeks: catalogue, values against independent calculations, isolation."""
from datetime import date

import pandas as pd
import pytest

import Greeks.cross_greeks as cross_module
from app import pricer
from app.curve_roll import next_business_day
from app.market_set import MarketSet
from Greeks.weighted_greeks import scale_t
from golden.app_harness import load_markets
from market_data import Modifications
from pricing import greek_catalog as catalog
from pricing.greeks import calculate_extra_greeks, calculate_greeks

pytestmark = pytest.mark.filterwarnings("ignore")
MODS = Modifications(beta_atm=0.7, beta_rr=1.2, beta_bf=0.0)
VALUATION = date(2026, 10, 15)
ALL_EXTRAS = list(catalog.EXTRA_GREEKS)


@pytest.fixture(scope="module")
def markets():
    return MarketSet.from_payloads(load_markets("MARKETS_20261015.json"), VALUATION)


def line(**changes):
    base = dict(currency_pair="EURUSD", strike="1.18", expiry="2026-12-17", call_put="CALL",
                buy_sell="BUY", notional="10m", premium=0.0, premium_ccy="USD")
    return dict(base, **changes)


@pytest.fixture(scope="module")
def priced(markets):
    rows = pd.DataFrame([line(), line(currency_pair="USDJPY", strike="150", call_put="PUT",
                                      buy_sell="SELL", premium_ccy="JPY")])
    plain, errors = pricer.price_lines(markets, rows, MODS)
    assert errors == []
    extra, errors = pricer.price_lines(markets, rows, MODS, extras=ALL_EXTRAS)
    assert errors == []
    return plain, extra


def unit_inputs(markets, pair="EURUSD", strike=1.18, expiry="2026-12-17"):
    context = pricer.market_for_pair(markets, pair)
    market = context.market_for_expiry(pd.Timestamp(expiry))
    nxt = next_business_day(pair, markets.valuation_date)
    next_market = context.market_for_expiry(pd.Timestamp(expiry), as_of=nxt, allow_short_end=True)
    row = pd.Series(dict(currency_pair=pair, strike=strike, expiry=expiry, call_put="CALL",
                         buy_sell="BUY", notional=1.0, premium_ccy=pair[3:],
                         valuation=markets.valuation_date))
    return context.settings, market, next_market, nxt, row


# -- catalogue -------------------------------------------------------------------

def test_catalogue_lists_every_greek_once_in_groups():
    assert list(catalog.CATALOG)[:8] == list(catalog.BASE_GREEKS)
    assert set(catalog.EXTRA_GREEKS) == {*catalog.DECAY_GREEKS, "gamma_vol",
                                         *catalog.WEIGHTED_GREEKS, *catalog.THETA_PARTS,
                                         "rho", "phi"}
    assert "theta_decay" not in catalog.CATALOG
    assert all(greek.group in catalog.GROUPS for greek in catalog.CATALOG.values())
    assert catalog.columns(["vega", "gamma_vol"]) == ["fixed_vega", "vega",
                                                      "fixed_gamma_vol", "gamma_vol"]
    assert catalog.known(["sega", "nonsense", "delta", "delta"]) == ["delta", "sega"]
    assert catalog.extras(["delta", "vega_decay"]) == ["vega_decay"]
    assert catalog.label("fixed_delta_decay") == "Delta decay (charm) (sticky)"
    assert catalog.label("vega_usd") == "Vega (smile, USD)"
    assert catalog.label("theta", "prem ccy") == "Theta (smile, prem ccy)"
    assert catalog.label("pnl_usd") == "pnl_usd"
    assert catalog.label("gamma_vol") == "Gamma per vol (smile)"
    assert catalog.label("mtm_base") == "mtm_base"


# -- pricing ---------------------------------------------------------------------

def test_without_extras_nothing_extra_is_computed(priced):
    plain, _ = priced
    for result in plain:
        assert "greek_problems" not in result
        assert not any(column in result for column in catalog.columns(ALL_EXTRAS))


def test_extras_never_change_the_base_results(priced):
    plain, extra = priced
    for before, after in zip(plain, extra):
        for column in ("mtm_base", "pnl_base", "vol_pct", *catalog.columns(catalog.BASE_GREEKS)):
            assert after[column] == before[column], column


def test_every_extra_is_computed_and_finite(priced):
    _, extra = priced
    for result in extra:
        assert result["greek_problems"] == {}
        for column in catalog.columns(ALL_EXTRAS):
            assert result[column] is not None and result[column] == result[column], column


def test_decay_is_tomorrows_greek_minus_todays(markets, priced):
    settings, market, next_market, nxt, row = unit_inputs(markets)
    args = (None, settings.pricing_model, settings.atm_type, settings.delta_base, MODS)
    today = calculate_greeks(market, row, *args)
    rolled = row.copy(); rolled["valuation"] = nxt
    tomorrow = calculate_greeks(next_market, rolled, *args)
    result = priced[1][0]                                   # EURUSD, 10m
    for greek in catalog.DECAY_OF:
        for key in (f"fixed_{greek}", greek):
            expected = (tomorrow[key] - today[key]) * 10e6
            assert result[key.replace(greek, f"{greek}_decay")] == pytest.approx(
                expected, rel=1e-9, abs=1e-6), key


def test_gamma_per_vol_is_the_reported_gamma_on_scaled_vols(markets, priced):
    settings, market, _, _, row = unit_inputs(markets)
    cross = cross_module.CrossGreeks(
        market=market, mods=MODS, k=1.18, expiry_date="2026-12-17",
        valuation_date=str(markets.valuation_date), CallPut="CALL", BuySell="BUY", Notional=1.0,
        atm_type=settings.atm_type, pricing_model=settings.pricing_model,
        deltaBase=settings.delta_base)
    up, dn = (calculate_greeks(cross._scaled_market(eps), row, None, settings.pricing_model,
                               settings.atm_type, settings.delta_base, MODS)
              for eps in (0.01, -0.01))
    for prefix in ("fixed_", ""):
        expected = (up[f"{prefix}gamma"] - dn[f"{prefix}gamma"]) / 0.02 * 10e6
        assert priced[1][0][f"{prefix}gamma_vol"] == pytest.approx(expected, rel=1e-9)


def test_weighted_greeks_are_the_greek_times_the_expiry_weight(priced):
    weight = scale_t("2026-12-17", str(VALUATION))
    result = priced[1][0]
    for greek in ("gamma", "vega"):
        for prefix in ("fixed_", ""):
            assert result[f"{prefix}weighted_{greek}"] == pytest.approx(
                result[f"{prefix}{greek}"] * weight, rel=1e-12)


def test_a_failing_greek_is_blank_and_isolated(markets, priced, monkeypatch):
    def broken(self, eps=0.01):
        raise RuntimeError("boom")
    monkeypatch.setattr(cross_module.CrossGreeks, "_scaled_market", broken)
    results, errors = pricer.price_lines(markets, pd.DataFrame([line()]), MODS,
                                         extras=["gamma_vol", "vega_decay"])
    assert errors == []
    result = results[0]
    assert result["gamma_vol"] is None and result["fixed_gamma_vol"] is None
    assert "boom" in result["greek_problems"]["gamma_vol"]
    assert result["vega_decay"] == pytest.approx(priced[1][0]["vega_decay"])
    assert result["vega"] == priced[0][0]["vega"]


def test_decay_without_a_next_day_market_is_blank_with_a_reason(markets):
    settings, market, _, _, row = unit_inputs(markets)
    base = calculate_greeks(market, row, None, settings.pricing_model, settings.atm_type,
                            settings.delta_base, MODS)
    values, problems = calculate_extra_greeks(
        market, row, ["delta_decay", "weighted_vega"], base, settings.pricing_model,
        settings.atm_type, settings.delta_base, MODS)
    assert values["delta_decay"] is None and "next-business-day" in problems["delta_decay"]
    assert values["weighted_vega"] is not None


def test_an_option_expiring_before_tomorrow_decays_to_zero(markets):
    settings, market, next_market, nxt, row = unit_inputs(markets, expiry="2026-10-16")
    base = {"fixed_vega": 5.0, "vega": 6.0}
    values, problems = calculate_extra_greeks(
        market, row, ["vega_decay"], base, settings.pricing_model, settings.atm_type,
        settings.delta_base, MODS, next_market=next_market, next_valuation="2026-10-16")
    assert problems == {} and (values["fixed_vega_decay"], values["vega_decay"]) == (-5.0, -6.0)


def test_a_stale_engine_without_extras_still_prices(markets, monkeypatch):
    """Streamlit reloads app/ but not pricing/: without extras requested the
    pricer must not pass the new argument to an engine that lacks it."""
    import pricing.pricing as engine
    current = engine.Pricable.price

    def old_price(self, market, pricing_model, atm_type, delta_base, mods=None,
                  next_market=None, next_valuation=None):
        return current(self, market, pricing_model, atm_type, delta_base, mods,
                       next_market=next_market, next_valuation=next_valuation)
    monkeypatch.setattr(engine.Pricable, "price", old_price)
    results, errors = pricer.price_lines(markets, pd.DataFrame([line()]), MODS)
    assert errors == [] and results[0]["vega"] is not None


# -- Risk ------------------------------------------------------------------------

def _ladder(markets, rows, spots, mode, extras):
    from app.reports.spot_ladder import calculate_spot_ladder
    context = pricer.market_for_pair(markets, "EURUSD")
    s = context.settings
    return calculate_spot_ladder(
        list(rows.iterrows()), spots, markets.valuation_date, context.market_for_expiry,
        context.resolve_market_expiry, s.pricing_model, s.atm_type, s.delta_base, MODS,
        mode=mode, extras=extras)


def test_ladder_extras_at_spot_match_the_pricer(markets):
    rows = pd.DataFrame([line(), line(strike="1.12", call_put="PUT", buy_sell="SELL")])
    resolved, _, _ = pricer.resolve_line_inputs(markets, rows, "EURUSD")
    extras = ["vega_decay", "gamma_vol", "weighted_vega", "rho", "phi"]
    priced, errors = pricer.price_lines(markets, resolved, MODS, extras=extras)
    assert errors == []
    spot = float(pricer.market_for_pair(markets, "EURUSD").spot)
    levels, errors = _ladder(markets, resolved, [spot * 0.99, spot, spot * 1.01], "reference",
                             extras)
    assert errors == []
    at_spot = levels[1]
    for column in catalog.columns(extras):
        assert at_spot[column] == pytest.approx(sum(r[column] for r in priced), rel=1e-6), column
    assert levels[0]["gamma_vol"] != at_spot["gamma_vol"]          # moves with spot


def test_rate_risks_include_forward_hedges_in_pair_totals_and_ladder(markets):
    from app.reports.spot_ladder import calculate_spot_ladder
    from pricing.fx_forward import FxForwardPosition
    from pricing.greeks import calculate_forward_rate_risks

    extras = ["rho", "phi"]
    rows = pd.DataFrame([line()])
    resolved, _, _ = pricer.resolve_line_inputs(markets, rows, "EURUSD")
    results, errors = pricer.price_lines(markets, resolved, MODS, extras=extras)
    assert errors == []
    hedges, errors = pricer.hedge_trades_for(markets, results)
    assert errors == [] and len(hedges) == 1
    context = pricer.market_for_pair(markets, "EURUSD")
    hedge = hedges[0]
    position = FxForwardPosition("EURUSD", hedge["expiry"], hedge["base_amount"],
                                 hedge["forward_rate"])
    mkt = context.market_for_expiry(hedge["expiry"])
    expected = calculate_forward_rate_risks(position, mkt, markets.valuation_date)
    total = pricer.aggregate_by_pair(results, hedges, greeks=extras).iloc[0]
    settings = context.settings
    levels, errors = calculate_spot_ladder(
        list(resolved.iterrows()), [float(context.spot)], markets.valuation_date,
        context.market_for_expiry, context.resolve_market_expiry,
        settings.pricing_model, settings.atm_type, settings.delta_base, MODS,
        forward_positions=[position], mode="reference", extras=extras)
    assert errors == []
    for key in catalog.columns(extras):
        assert hedge[key] == pytest.approx(expected[key])
        assert total[key] == pytest.approx(results[0][key] + expected[key])
        assert levels[0][key] == pytest.approx(total[key], rel=1e-6)


def test_ladder_modes_agree_with_extras(markets):
    rows = pd.DataFrame([line()])
    resolved, _, _ = pricer.resolve_line_inputs(markets, rows, "EURUSD")
    spot = float(pricer.market_for_pair(markets, "EURUSD").spot)
    spots = [spot * 0.995, spot, spot * 1.005]
    reference, ref_errors = _ladder(markets, resolved, spots, "reference", ["delta_decay"])
    shadow, shadow_errors = _ladder(markets, resolved, spots, "shadow", ["delta_decay"])
    assert ref_errors == [] and shadow_errors == []          # no shadow-validation fallback
    for ref, other in zip(reference, shadow):
        assert other["delta_decay"] == pytest.approx(ref["delta_decay"], rel=1e-8)


def test_a_plain_ladder_has_no_extra_columns(markets):
    rows = pd.DataFrame([line()])
    resolved, _, _ = pricer.resolve_line_inputs(markets, rows, "EURUSD")
    spot = float(pricer.market_for_pair(markets, "EURUSD").spot)
    levels, _ = _ladder(markets, resolved, [spot], "optimized", ())
    assert not set(catalog.columns(ALL_EXTRAS)) & set(levels[0])


def test_risk_lines_carry_the_chosen_greeks(markets, priced):
    from app.reports import risk_grid
    from app.trades.model import FX_OPTION, Trade
    result = dict(priced[1][0], trade_id="TR-1")
    trade = Trade(product_type=FX_OPTION, currency_pair="EURUSD", trade_date=VALUATION,
                  expiry=date(2026, 12, 17), buy_sell="BUY", notional=10e6, strike=1.18,
                  call_put="CALL", premium=0.0, premium_ccy="USD", trade_id="TR-1")
    lines, errors = risk_grid.risk_lines(markets, "EURUSD", [trade], [result], [],
                                         greeks=["vega", "vega_decay", "sega"], fixed=True)
    assert errors == []
    assert lines[0].values["vega_decay"] == result["fixed_vega_decay"]
    assert lines[0].values["sega"] == result["fixed_sega"]
    table = risk_grid.greek_buckets(lines, ["3M"], [date(2027, 1, 15)], ["vega_decay"])
    assert table.loc["vega_decay", "Total"] == pytest.approx(result["fixed_vega_decay"])
    assert "gamma_vol" in risk_grid.MEASURES and list(risk_grid.MEASURES)[1] == "vega"

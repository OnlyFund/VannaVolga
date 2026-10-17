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
                                         *catalog.WEIGHTED_GREEKS}
    assert "theta_decay" not in catalog.CATALOG
    assert all(greek.group in catalog.GROUPS for greek in catalog.CATALOG.values())
    assert catalog.columns(["vega", "gamma_vol"]) == ["fixed_vega", "vega",
                                                      "fixed_gamma_vol", "gamma_vol"]
    assert catalog.known(["sega", "nonsense", "delta", "delta"]) == ["delta", "sega"]
    assert catalog.extras(["delta", "vega_decay"]) == ["vega_decay"]
    assert catalog.label("fixed_delta_decay") == "Fixed delta decay (charm)"
    assert catalog.label("gamma_vol") == "Smile gamma per vol"
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


def test_gamma_per_vol_matches_the_cross_greeks_module(markets, priced):
    settings, market, _, _, row = unit_inputs(markets)
    cross = cross_module.CrossGreeks(
        market=market, mods=MODS, k=1.18, expiry_date="2026-12-17",
        valuation_date=str(markets.valuation_date), CallPut="CALL", BuySell="BUY", Notional=1.0,
        atm_type=settings.atm_type, pricing_model=settings.pricing_model,
        deltaBase=settings.delta_base)
    assert priced[1][0]["gamma_vol"] == pytest.approx(cross.dGammaDvol() * 10e6, rel=1e-9)


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
    monkeypatch.setattr(cross_module.CrossGreeks, "dGammaDvol", broken)
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

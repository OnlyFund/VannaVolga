"""The illustrative sample market: complete, consistent, used only without a store."""
import json
import warnings

import pandas as pd
import pytest

import market_data.sample_market as sample_market
import market_data.store as store_module
from app.market_context import MarketContext, PricingSettings
from app.market_service import MarketService
from app.market_store import read_market_file
from app.smile_service import DiagnosticsCollector, SurfaceCache, bind_cache
from app.store_markets import app_market_store, load_store_markets
from market_data.discount_curves import DiscountCurves
from market_data.market_tables import market_bundle
from market_data.sample_market import (SAMPLE_SOURCE, ensure_sample_market, rate_curve,
                                       sample_quotes)
from market_data.store import MarketDataStore
from market_data.universe import CCY_LIST, FX_UNIVERSE

pytestmark = pytest.mark.filterwarnings("ignore")
DAY = "2026-10-16"


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    store = MarketDataStore(tmp_path_factory.mktemp("sample") / "sample.sqlite")
    ensure_sample_market(store, DAY)
    return store


@pytest.fixture(scope="module")
def loaded(store):
    return load_store_markets(DAY, store=store)


def test_quotes_cover_every_pair_and_tenor():
    quotes = sample_quotes(DAY)
    assert set(quotes["spots"]) == set(CCY_LIST)
    assert set(quotes["vols"]) == set(quotes["forward_points"]) == set(FX_UNIVERSE)
    assert all(len(surface) == 9 for surface in quotes["vols"].values())
    assert all(len(points) == 8 for points in quotes["forward_points"].values())
    assert "CNY" in quotes["depo_curves"] and "CNH" not in quotes["depo_curves"]


def test_quotes_are_plausible():
    quotes = sample_quotes(DAY)
    for pair, surface in quotes["vols"].items():
        for tenor, row in surface.items():
            assert 1.0 < row["atm"] < 25.0, (pair, tenor)
            assert row["bf_10"] > row["bf_25"] > 0, (pair, tenor)
            assert abs(row["rr_25"]) < 0.4 * row["atm"], (pair, tenor)
    assert quotes["vols"]["USDTRY"]["1M"]["rr_25"] > 0          # USD calls bid
    assert quotes["vols"]["AUDJPY"]["1M"]["rr_25"] < 0          # AUD puts bid
    assert quotes["forward_points"]["USDJPY"]["1Y"] < 0         # JPY rates below USD


def test_every_pair_builds_with_no_problem(loaded):
    assert loaded.problem is None and not loaded.unavailable
    assert set(loaded.payloads) == set(FX_UNIVERSE)
    assert loaded.snapshot.market_source == SAMPLE_SOURCE


def test_forward_implied_rates_give_back_the_sample_curves(store):
    curves = DiscountCurves(store.load(date=DAY))
    for ccy in ("EUR", "GBP", "JPY", "CHF", "TRY", "HUF", "CNH", "MXN", "ZAR"):
        curve = curves.curve(ccy)
        assert curve.source == "implied"
        for tenor, rate in rate_curve(ccy).items():
            assert curve.rates[tenor] == pytest.approx(rate, abs=1e-6), (ccy, tenor)
    assert curves.curve("USD").source == "sofr"


@pytest.mark.parametrize("pair", ["EURUSD", "USDJPY", "USDTRY", "USDHKD", "EURPLN", "NOKSEK"])
def test_smile_calibrates(loaded, pair):
    diagnostics = DiagnosticsCollector()
    bind_cache(diagnostics.token, SurfaceCache())
    context = MarketContext.from_payload(
        loaded.payloads[pair], PricingSettings(pricing_model="vv"),
        market_service=MarketService(diagnostics), diagnostics=diagnostics, valuation_date=DAY)
    one_month = loaded.payloads[pair]["vol_data"][2]["atm"]
    vol = context.market_for_expiry(pd.Timestamp("2026-11-16")).v_atm
    assert vol == pytest.approx(one_month, rel=0.05)


def test_one_sample_market_per_date(store):
    first = ensure_sample_market(store, DAY)
    assert ensure_sample_market(store, DAY) == first
    other = ensure_sample_market(store, "2026-10-19")
    assert other != first and store.load(date="2026-10-19").market_id == other


def test_saved_bundle_can_be_edited_and_loaded_back(store):
    bundle, _ = market_bundle(store.load(date=DAY))
    edited = json.loads(json.dumps(bundle))           # what a user downloads
    edited["markets"]["EURUSD"]["spot"] = 1.2345
    edited["markets"]["EURUSD"]["vol_data"][0]["atm"] = 9.0
    markets, errors = read_market_file(edited)
    assert not errors and len(markets) == len(FX_UNIVERSE)
    assert markets["EURUSD"]["spot"] == pytest.approx(1.2345)


@pytest.fixture
def no_store(tmp_path, monkeypatch):
    """No market data store: nothing at the default path, no override."""
    monkeypatch.delenv("VV_MARKET_STORE", raising=False)
    monkeypatch.setattr(store_module, "DEFAULT_PATH", tmp_path / "missing" / "market.sqlite")
    monkeypatch.setattr(sample_market, "sample_store_path", lambda: tmp_path / "sample.sqlite")
    return tmp_path


def test_without_a_store_the_app_uses_the_sample(no_store):
    store = app_market_store(DAY)
    assert store.path == no_store / "sample.sqlite"
    assert not (no_store / "missing").exists()         # no empty store created
    result = load_store_markets(DAY)
    assert result.problem is None and result.snapshot.market_source == SAMPLE_SOURCE


def test_any_valuation_date_gets_its_own_sample(no_store):
    result = load_store_markets("2026-12-01")
    assert result.problem is None and result.snapshot.date == "2026-12-01"


def test_a_real_store_is_never_replaced_by_the_sample(no_store, monkeypatch):
    real = MarketDataStore(no_store / "real.sqlite")
    monkeypatch.setenv("VV_MARKET_STORE", str(real.path))
    result = load_store_markets(DAY)
    assert result.problem.startswith("no market data")  # the real (empty) store's answer
    assert not (no_store / "sample.sqlite").exists()


def test_an_existing_default_store_is_used(no_store, monkeypatch):
    path = no_store / "default.sqlite"
    MarketDataStore(path)
    monkeypatch.setattr(store_module, "DEFAULT_PATH", path)
    assert app_market_store(DAY).path == path
    assert not (no_store / "sample.sqlite").exists()


def test_sample_market_uses_no_bloomberg():
    source = open(sample_market.__file__, encoding="utf-8").read()
    assert "xbbg" not in source and "bloomberg_fetch" not in source

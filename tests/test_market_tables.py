"""Vol tables built from stored quotes: tenor dates and use by the app."""
import json
import warnings
from pathlib import Path

import pandas as pd
import pytest

from market_data.store import VOL_FIELDS, MarketDataStore
from market_data.market_tables import APP_TENORS, COLUMNS, tenor_dates, vol_table, vol_tables

FIXTURE = Path(__file__).parent / "fixtures" / "markets" / "MARKETS_20261015.json"
SURFACE = {tenor: dict(zip(VOL_FIELDS, (6.0 + i / 10, -0.3, 0.15, -0.6, 0.5)))
           for i, tenor in enumerate(("ON", *APP_TENORS))}


def test_tenor_dates_follow_spot_and_settlement_conventions():
    # EURUSD settles T+2: spot Tue 20-Oct, 1W delivers Tue 27-Oct, expires Fri 23-Oct.
    assert tenor_dates("EURUSD", "2026-10-16", "1W") == (pd.Timestamp("2026-10-23"),
                                                          pd.Timestamp("2026-10-27"))
    assert tenor_dates("EURUSD", "2026-10-16", "1M") == (pd.Timestamp("2026-11-18"),
                                                          pd.Timestamp("2026-11-20"))
    # USDCAD settles T+1.
    assert tenor_dates("USDCAD", "2026-10-16", "1W")[1] == pd.Timestamp("2026-10-26")
    # ON: next business day (Friday -> Monday), delivered on its spot date.
    assert tenor_dates("EURUSD", "2026-10-16", "ON") == (pd.Timestamp("2026-10-19"),
                                                          pd.Timestamp("2026-10-21"))


def test_vol_table_has_the_app_columns_in_tenor_order():
    table = vol_table(SURFACE, "EURUSD", "2026-10-16")
    assert list(table.columns) == list(COLUMNS)
    assert list(table["tenor"]) == list(APP_TENORS)        # ON left out by default
    assert table["expiry"].is_monotonic_increasing
    partial = vol_table({"1M": SURFACE["1M"]}, "EURUSD", "2026-10-16")
    assert list(partial["tenor"]) == ["1M"]


def _snapshot(tmp_path, spots=None, depo_curves=None):
    fixture = json.loads(FIXTURE.read_text())["markets"]["EURUSD"]
    points = {row["tenor"].upper(): row["fwd_pts"] for row in fixture["forward_points"]}
    store = MarketDataStore(tmp_path / "market_data.sqlite")
    store.save("2026-10-16", "2026-10-16T12:00:00", spots=spots or {"EURUSD": 1.1258},
               depo_curves=depo_curves or {}, vols={"EURUSD": SURFACE},
               forward_points={"EURUSD": points})
    return store.load()


def test_stored_quotes_build_a_market_the_app_loads_and_calibrates(tmp_path):
    from app.market_context import MarketContext, PricingSettings
    from app.market_store import read_market_file
    from market_data.market_tables import market, market_bundle
    usd_depo = {tenor: 0.04 for tenor in APP_TENORS if tenor != "9M"}
    snapshot = _snapshot(tmp_path, depo_curves={"USD": usd_depo})
    payload, notes = market(snapshot, "EURUSD")
    assert payload["forward_divisor"] == 10000.0 and payload["spot"] == 1.1258
    forwards = pd.DataFrame(payload["forward_points"])
    assert list(forwards["tenor"]) == list(APP_TENORS)
    assert list(forwards["term_discount"]) == pytest.approx([0.04] * 8)   # SOFR, 9M interpolated
    assert notes == ["USD: SOFR 9M interpolated"]

    bundle, _ = market_bundle(snapshot)
    markets, errors = read_market_file(bundle)               # the Markets tab's loader
    assert not errors and list(markets) == ["EURUSD"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        context = MarketContext.from_payload(payload, PricingSettings(pricing_model="vv"))
        for expiry in context.pillar_expiries():
            context.calibrate_pillar(expiry)
        one_month = context.market_for_expiry(pd.Timestamp("2026-11-18"))
    assert len(context.pillar_expiries()) == 8
    assert one_month.v_atm == pytest.approx(SURFACE["1M"]["atm"])
    assert not context.diagnostics.records()          # no legacy fallback


def test_market_needs_spot_and_tenors_with_both_vols_and_forwards(tmp_path):
    from market_data.market_tables import market
    snapshot = _snapshot(tmp_path, depo_curves={"USD": {"1M": 0.04}})
    assert market(snapshot, "USDJPY")[0] is None             # no spot, no quotes
    partial = snapshot._replace(forward_points={"EURUSD": {"1M": 14.2, "3M": 44.4}})
    payload, notes = market(partial, "EURUSD")
    assert [row["tenor"] for row in payload["vol_data"]] == ["1M", "3M"]
    assert any("left out" in note for note in notes)


def test_forward_tickers_and_divisors():
    from app.market_service import pair_divisor
    from market_data.fwd_tickers import fwd_ticker
    assert fwd_ticker("EURUSD", "1M") == "EUR1M Curncy"
    assert fwd_ticker("USDJPY", "1Y") == "JPY12M Curncy"
    assert fwd_ticker("EURGBP", "2W") == "EURGBP2W Curncy"
    # Divisors as Bloomberg quotes the points (checked against depo carry).
    assert {pair: pair_divisor(pair) for pair in ("USDTRY", "EURTRY", "USDHUF", "EURHUF",
                                                  "USDCZK", "EURCZK", "USDJPY", "EURUSD")} == {
        "USDTRY": 10000.0, "EURTRY": 10000.0, "USDHUF": 100.0, "EURHUF": 100.0,
        "USDCZK": 1000.0, "EURCZK": 1000.0, "USDJPY": 100.0, "EURUSD": 10000.0}


def test_butterfly_convention_by_pair(tmp_path):
    from market_data.market_tables import market
    from market_data.universe import EM_CROSSES, G10_CROSSES, broker_strangle
    assert all(broker_strangle(pair) for pair in G10_CROSSES)
    assert not any(broker_strangle(pair) for pair in EM_CROSSES) and len(EM_CROSSES) == 20
    snapshot = _snapshot(tmp_path, depo_curves={"USD": {"1M": 0.04}})
    assert market(snapshot, "EURUSD")[0]["broker_strangle"] is True
    assert market(snapshot, "EURUSD", broker_strangle=False)[0]["broker_strangle"] is False
    em = snapshot._replace(spots={"USDTRY": 49.5}, vols={"USDTRY": SURFACE},
                           forward_points={"USDTRY": snapshot.forward_points["EURUSD"]})
    assert market(em, "USDTRY")[0]["broker_strangle"] is False

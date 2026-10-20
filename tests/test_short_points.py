"""The short swaps around spot (ON/TN/SN) date value-dated deals: inside
spot a deal takes spot less TN (tomorrow) or less TN and ON (today), the day
after spot spot plus SN, and broken dates up to 1W interpolate from SN.
Without the points a deal inside spot is valued at spot, with a note. Options
and hedges never read them. They come from the fetch through the stored
snapshot and the market payload, and implied rates ignore them."""
import copy
import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from app.curve_roll import short_end_nodes
from app.market_set import MarketSet
from app.market_store import market_payload, validate_market
from app.trades import service
from app.trades.model import FX_FORWARD, FX_OPTION, Trade
from golden.app_harness import load_markets
from market_data import Modifications

VALUATION = date(2026, 10, 15)              # Thursday: tomorrow Fri 16, spot Mon 19
MODS = Modifications(beta_atm=0.7, beta_rr=1.2, beta_bf=0.0)
SPOT, DIVISOR = 1.1213, 10000.0
SHORT = {"ON": 0.5, "TN": 0.4, "SN": 0.45}
FIXTURE = Path(__file__).parent / "fixtures" / "markets" / "MARKETS_20261015.json"


def market_set(short_points):
    payloads = copy.deepcopy(load_markets("MARKETS_20261015.json"))
    if short_points is not None:
        payloads["EURUSD"]["short_points"] = short_points
    return MarketSet.from_payloads(payloads, VALUATION)


@pytest.fixture(scope="module")
def quoted():
    return market_set(SHORT)


def deal(value_date, trade_id="TR-1"):
    return Trade(product_type=FX_FORWARD, currency_pair="EURUSD", trade_date=date(2026, 10, 13),
                 expiry=None, value_date=value_date, buy_sell="BUY", notional=10e6,
                 contract_rate=1.12, trade_id=trade_id).normalized()


def outright(markets, value_date):
    results, forwards, errors = service.price_trades(markets, [deal(value_date)], MODS)
    assert errors == []
    return forwards[0]


@pytest.mark.parametrize("value_date, points", [
    (date(2026, 10, 16), -0.4),              # tomorrow: spot less TN
    (date(2026, 10, 19), 0.0),               # spot
    (date(2026, 10, 20), 0.45),              # spot next: spot plus SN
    # Thu 22-Oct between SN (Tue 20) and 1W (Mon 26, 3.09 points)
    (date(2026, 10, 22), 0.45 + 2 / 6 * (3.09 - 0.45)),
])
def test_deals_around_spot_take_the_short_swaps(quoted, value_date, points):
    row = outright(quoted, value_date)
    assert row["outright"] == pytest.approx(SPOT + points / DIVISOR, abs=1e-12)
    assert row["forward_note"] is None


def test_today_is_spot_less_tn_and_on(quoted):
    # A deal settling today is settled (the book reports it expired); its
    # market still dates today on the short swaps.
    market = quoted.market("EURUSD").market_for_delivery(VALUATION)
    assert market.fwd_pts == pytest.approx(-(0.5 + 0.4)) and market.forward_note is None


def test_beyond_the_first_pillar_the_short_swaps_change_nothing(quoted):
    plain = market_set(None)
    for value_date in (date(2026, 10, 26), date(2027, 3, 5)):
        assert outright(quoted, value_date)["mtm_term"] == outright(plain, value_date)["mtm_term"]


def test_without_the_points_inside_spot_is_valued_at_spot_with_a_note():
    no_on = market_set({"TN": 0.4})
    today = no_on.market("EURUSD").market_for_delivery(VALUATION)    # needs ON as well
    assert today.fwd_pts == 0.0
    assert "inside spot" in today.forward_note and "ON/TN points are not quoted" in today.forward_note
    assert outright(no_on, date(2026, 10, 16))["outright"] == pytest.approx(SPOT - 0.4 / DIVISOR)
    # Without SN the day after spot interpolates from spot to 1W, as before.
    assert outright(no_on, date(2026, 10, 20))["outright"] == pytest.approx(
        SPOT + 1 / 7 * 3.09 / DIVISOR)


def test_settling_tomorrow_still_loses_its_mtm_as_theta_and_spot_deals_roll(quoted):
    row = outright(quoted, date(2026, 10, 16))
    assert row["theta"] == pytest.approx(-row["mtm_term"])
    # A spot deal survives tomorrow and rolls down the short end instead.
    spot_deal = outright(quoted, date(2026, 10, 19))
    assert spot_deal["theta"] != pytest.approx(-spot_deal["mtm_term"])
    assert abs(spot_deal["theta"]) < 10e6 * 1e-4


def test_options_and_hedges_never_read_them(quoted):
    plain = market_set(None)
    trades = [
        Trade(product_type=FX_OPTION, currency_pair="EURUSD", trade_date=VALUATION,
              expiry=date(2026, 10, 16), buy_sell="BUY", notional=1e6, strike=1.12,
              call_put="CALL", premium=0.0, premium_ccy="USD", trade_id="O").normalized(),
        Trade(product_type=FX_FORWARD, currency_pair="EURUSD", trade_date=VALUATION,
              expiry=date(2026, 10, 16), buy_sell="SELL", notional=1e6, contract_rate=1.12,
              trade_id="H").normalized(),
    ]
    with_short = service.price_trades(quoted, trades, MODS)
    without = service.price_trades(plain, trades, MODS)
    assert with_short == without


def test_a_t_plus_1_pair_steps_back_by_on_only():
    # USDCAD settles T+1: ON ends at spot (Fri 16), so today is spot less ON;
    # its TN would start at spot and is left out. SN ends Mon 19.
    nodes = short_end_nodes("USDCAD", VALUATION, {"ON": 0.3, "TN": 0.2, "SN": 0.25})
    assert nodes == [(pd.Timestamp("2026-10-15"), -0.3, "ON"),
                     (pd.Timestamp("2026-10-19"), 0.25, "SN")]
    # T+2 without ON: tomorrow only.
    assert short_end_nodes("EURUSD", VALUATION, {"TN": 0.4}) == [
        (pd.Timestamp("2026-10-16"), -0.4, "TN")]
    assert short_end_nodes("EURUSD", VALUATION, None) == []


def test_the_payload_carries_and_checks_them():
    loaded = validate_market(load_markets("MARKETS_20261015.json")["EURUSD"])
    assert loaded["short_points"] == {}
    args = [loaded[k] for k in ("currency_pair", "valuation_date", "spot", "forward_divisor",
                                "broker_strangle", "vol_rows", "forward_rows", "event_rows")]
    assert "short_points" not in market_payload(*args)
    payload = market_payload(*args, SHORT)
    assert validate_market(payload)["short_points"] == SHORT
    for bad, message in (({"1W": 1.0}, "only hold ON, TN and SN"),
                         ({"TN": float("nan")}, "must be a finite number"),
                         ([0.4], "must map ON/TN/SN")):
        with pytest.raises(ValueError, match=message):
            validate_market(dict(payload, short_points=bad))


def test_the_fetch_tickers_and_the_stored_snapshot(tmp_path):
    from market_data.fwd_tickers import fwd_ticker
    from market_data.implied_rates import implied_rates
    from market_data.market_tables import APP_TENORS, market
    from market_data.store import VOL_FIELDS, MarketDataStore
    assert [fwd_ticker("EURUSD", t) for t in ("TN", "SN", "1Y")] == [
        "EURTN Curncy", "EURSN Curncy", "EUR12M Curncy"]
    assert fwd_ticker("EURGBP", "ON") == "EURGBPON Curncy"

    fixture = json.loads(FIXTURE.read_text())["markets"]["EURUSD"]
    points = {row["tenor"].upper(): row["fwd_pts"] for row in fixture["forward_points"]}
    surface = {tenor: dict(zip(VOL_FIELDS, (6.0, -0.3, 0.15, -0.6, 0.5))) for tenor in APP_TENORS}
    store = MarketDataStore(tmp_path / "market_data.sqlite")
    store.save("2026-10-16T12:00:00", "bloomberg", spots={"EURUSD": 1.1258},
               depo_curves={"USD": {"ON": 0.039, **{t: 0.04 for t in APP_TENORS}}},
               vols={"EURUSD": surface}, forward_points={"EURUSD": dict(points, **SHORT)})
    snapshot = store.load()
    payload, _ = market(snapshot, "EURUSD")
    assert payload["short_points"] == SHORT
    assert list(pd.DataFrame(payload["forward_points"])["tenor"]) == list(APP_TENORS)
    rates = implied_rates("EUR", "EURUSD", snapshot.date, 1.1258, snapshot.forward_points["EURUSD"],
                          snapshot.depo_curves["USD"])
    assert set(rates) == set(APP_TENORS)              # ON/TN/SN are not spot-to-tenor points

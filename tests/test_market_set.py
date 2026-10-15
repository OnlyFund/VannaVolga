"""Several pairs priced together: routing, isolation and stale-market guards."""
from copy import deepcopy
from datetime import date

import pandas as pd
import pytest

from app import pricer
from app.market_context import MarketContext, MarketContextError, UnavailableMarket
from app.market_set import MarketSet
from golden.app_harness import load_markets
from market_data import Modifications

pytestmark = pytest.mark.filterwarnings("ignore")

MODS = Modifications(beta_atm=0.7, beta_rr=1.2, beta_bf=0.0)
VALUATION = date(2026, 10, 15)


@pytest.fixture(scope="module")
def payloads():
    return load_markets("MARKETS_20261015.json")


@pytest.fixture(scope="module")
def markets(payloads):
    return MarketSet.from_payloads(payloads, VALUATION)


def trade(pair, **changes):
    line = dict(currency_pair=pair, strike="atm", expiry="1M", call_put="CALL",
                buy_sell="BUY", notional="1m", premium=0.0, premium_ccy=pair[3:])
    line.update(changes)
    return line


def test_bundle_builds_one_market_per_pair(markets):
    assert sorted(markets.pairs) == ["EURUSD", "USDJPY", "USDMXN"]
    assert sorted(markets.available()) == sorted(markets.pairs)
    assert markets.problems() == {}
    assert "eur/usd" in markets


def test_mixed_book_prices_each_line_on_its_own_pair(markets, payloads):
    book = [trade("USDJPY"), trade("EURUSD", strike="25d", call_put="PUT"),
            trade("USDMXN", expiry="3M"), trade("EURUSD", expiry="1W")]
    results, errors = pricer.price_lines(markets, book, MODS)
    assert errors == []
    assert [result["currency_pair"] for result in results] == ["USDJPY", "EURUSD", "USDMXN", "EURUSD"]
    for number, line in enumerate(book, start=1):
        alone = MarketContext.from_payload(payloads[line["currency_pair"]])
        expected = pricer.price_line(alone, number, pd.Series(line), MODS)[1]
        assert results[number - 1] == expected


def test_missing_pair_is_named_and_an_inverse_pair_gets_a_hint(markets):
    results, errors = pricer.price_lines(
        markets, [trade("GBPUSD"), trade("USDEUR"), trade("EURUSD")], MODS)
    assert len(results) == 1
    assert errors[0] == "Line 1: No market loaded for GBPUSD"
    assert errors[1] == "Line 2: No market loaded for USDEUR; EURUSD is loaded, enter the pair as EURUSD"


def test_broken_pair_fails_only_its_own_lines(payloads):
    stored = deepcopy(payloads)
    stored["USDJPY"]["spot"] = 0.0
    del stored["USDMXN"]["vol_data"]
    markets = MarketSet.from_payloads(stored, VALUATION)
    assert markets.available() == ["EURUSD"]
    assert set(markets.problems()) == {"USDJPY", "USDMXN"}
    results, errors = pricer.price_lines(
        markets, [trade("USDJPY"), trade("EURUSD"), trade("USDMXN")], MODS)
    assert [result["currency_pair"] for result in results] == ["EURUSD"]
    assert errors[0] == "Line 1: USDJPY Invalid market: Spot must be finite and positive"
    assert errors[1].startswith("Line 3: USDMXN Invalid market")


def test_market_saved_for_another_day_is_not_used(payloads):
    stored = deepcopy(payloads)
    stored["USDJPY"]["valuation_date"] = "2026-10-14"
    markets = MarketSet.from_payloads(stored, VALUATION)
    assert isinstance(markets.market("USDJPY"), UnavailableMarket)
    _, errors = pricer.price_lines(markets, [trade("USDJPY")], MODS)
    assert "valued on 2026-10-14, not on the pricing date 2026-10-15" in errors[0]


def test_market_stored_under_the_wrong_pair_is_rejected(payloads):
    markets = MarketSet.from_payloads({"USDJPY": payloads["EURUSD"]}, VALUATION)
    assert markets.problems() == {"USDJPY": "market stored under USDJPY contains EURUSD"}


def test_live_edits_replace_the_stored_market(payloads):
    live_payload = deepcopy(payloads["EURUSD"])
    live_payload["spot"] = 1.2
    live = MarketContext.from_payload(live_payload)
    markets = MarketSet.from_payloads(payloads, VALUATION, live=live)
    assert markets.market("EURUSD") is live
    assert sorted(markets.available()) == ["EURUSD", "USDJPY", "USDMXN"]


def test_empty_set_reports_every_line():
    _, errors = pricer.price_lines(MarketSet(VALUATION), [trade("EURUSD")], MODS)
    assert errors == ["Line 1: No market loaded for EURUSD"]
    with pytest.raises(MarketContextError):
        MarketSet(VALUATION).market("EURUSD")


def test_hedges_book_one_forward_per_pair_and_expiry(markets):
    book = [trade("EURUSD"), trade("USDJPY"), trade("EURUSD", strike="25d")]
    results, _ = pricer.price_lines(markets, book, MODS)
    trades, errors = pricer.hedge_trades_for(markets, results)
    assert errors == []
    assert sorted(t["currency_pair"] for t in trades) == ["EURUSD", "USDJPY"]
    positions, position_errors = pricer.build_delta_hedges(markets, pd.DataFrame(book), MODS)
    assert position_errors == []
    assert sorted(p["currency_pair"] for p in positions) == ["EURUSD", "USDJPY"]


def test_strike_aliases_resolve_on_each_rows_own_pair(markets):
    rows = pd.DataFrame([trade("USDJPY", strike="25d"), trade("EURUSD", strike="25d")])
    resolved, changed, failures = pricer.resolve_strike_aliases(markets, rows)
    assert changed and failures == []
    assert float(resolved.at[0, "strike"]) > 100 > 2 > float(resolved.at[1, "strike"])

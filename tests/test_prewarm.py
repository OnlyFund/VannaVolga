"""The pre-warm after the daily fetch (performance step 2): it fills the disk
caches for the app's opening state, and never fails the fetch."""
from datetime import date, datetime, timedelta

import pytest

from apeiron.hyperion.cache import pricing as cached_pricing
from apeiron.hyperion.cache.store import ResultCache
from app import prewarm
from app.calibration_cache import CalibrationCache
from app.market_set import MarketSet
from app.store_markets import load_store_markets
from app.trades import service
from app.trades.store import SqliteTradeStore
from market_data.store import MarketDataStore
from test_store_markets import TODAY, full_snapshot

pytestmark = pytest.mark.filterwarnings("ignore")

VALUATION = date.fromisoformat(TODAY)


@pytest.fixture
def stores(tmp_path, monkeypatch):
    """A market store with one snapshot, a book of three trades on it, and
    empty result and calibration caches."""
    markets_path = tmp_path / "market_data.sqlite"
    market_store = MarketDataStore(markets_path)
    full_snapshot(market_store)
    monkeypatch.setenv("VV_MARKET_STORE", str(markets_path))
    monkeypatch.setenv("VV_LIVE_DATES", "0")
    trades = SqliteTradeStore(tmp_path / "trades.sqlite",
                              clock=lambda: datetime(2026, 10, 16, 9), today=VALUATION)
    settings, mods, _ = prewarm.app_settings()
    loaded = load_store_markets(VALUATION, store=market_store)
    markets = MarketSet.from_payloads(loaded.payloads, VALUATION, settings)
    lines = [dict(currency_pair=pair, strike="atm", expiry=tenor, call_put="CALL",
                  buy_sell="BUY", notional="1m", premium=0.0, premium_ccy=pair[3:])
             for pair, tenor in (("EURUSD", "1M"), ("USDJPY", "3M"), ("EURUSD", "6M"))]
    booked = service.book_lines(trades, markets, lines, mods, trade_date=VALUATION)
    assert not booked.errors
    return dict(trades=tmp_path / "trades.sqlite",
                results=ResultCache(path=tmp_path / "results.sqlite"),
                calibrations=CalibrationCache(tmp_path / "calibration"),
                lock=tmp_path / "prewarm.lock")


def run(stores, **changes):
    arguments = dict(valuation_date=VALUATION, trade_store=stores["trades"],
                     result_cache=stores["results"], calibration_cache=stores["calibrations"],
                     lock=stores["lock"], out=lambda *_: None)
    arguments.update(changes)
    return prewarm.prewarm(**arguments)


def test_it_fills_both_caches_and_a_fresh_session_reads_them(stores):
    report = run(stores)
    assert report.problems == []
    assert report.pairs == 2 and report.calibrations == {"calibrated": 2}
    assert report.trades == 3 and report.cached == 3
    again = run(stores)                        # the next run finds the smiles on disk
    assert again.calibrations == {"disk": 2} and again.cached == 3


def test_the_app_finds_the_trades_it_would_price(stores):
    run(stores)
    settings, mods, extras = prewarm.app_settings()
    loaded = load_store_markets(VALUATION)
    markets = MarketSet.from_payloads(loaded.payloads, VALUATION, settings)
    live, _ = service.book_on(SqliteTradeStore(stores["trades"]).trades(status="live"), VALUATION)
    keys = cached_pricing._keys(markets, live, mods, extras, "smile")
    assert all(stores["results"].get(keys[t.trade_id]) is not None for t in live)


def test_no_market_is_a_report_not_an_exception(stores):
    report = run(stores, valuation_date=date(2020, 1, 2))
    assert report.trades == 0 and any("pre-warm failed" in p for p in report.problems)


def test_an_unreadable_book_still_calibrates(stores, tmp_path):
    report = run(stores, trade_store=tmp_path / "missing" / "trades.sqlite")
    assert report.calibrations == {"calibrated": 2}
    assert any("book not priced" in p for p in report.problems)


def test_a_running_prewarm_is_not_doubled_and_a_stale_lock_is_taken_over(stores):
    stores["lock"].write_text("1234")
    assert run(stores).problems == ["another pre-warm is running"]
    old = (datetime.now() - prewarm.LOCK_STALE - timedelta(minutes=1)).timestamp()
    import os
    os.utime(stores["lock"], (old, old))
    assert run(stores).cached == 3
    assert not stores["lock"].exists()           # released when done

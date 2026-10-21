"""The result cache: cached results are exactly fresh ones, each change reprices
only what it affects, the disk layer survives a restart, and every failure of
the cache falls back to pricing fresh."""
import time
from dataclasses import replace
from datetime import date

import pytest

from apeiron.hyperion.cache import pricing as cached
from apeiron.hyperion.cache.store import ResultCache
from app.market_set import MarketSet
from app.trades import service
from app.trades.model import FX_FORWARD, FX_OPTION, Trade
from golden.app_harness import load_markets
from market_data import Modifications

VALUATION = date(2026, 10, 15)
MODS = Modifications(beta_atm=0.7, beta_rr=1.2, beta_bf=0.0)


@pytest.fixture(scope="module")
def markets():
    return MarketSet.from_payloads(load_markets("MARKETS_20261015.json"), VALUATION)


def book():
    def option(n, pair, strike, expiry=date(2027, 1, 15), **changes):
        terms = dict(product_type=FX_OPTION, currency_pair=pair, trade_date=VALUATION,
                     expiry=expiry, buy_sell="BUY", notional=1e6, strike=strike,
                     call_put="CALL", premium=0.0, premium_ccy=pair[3:],
                     trade_id=f"TR-{n:06d}")
        terms.update(changes)
        return Trade(**terms).normalized()
    return [
        option(1, "EURUSD", 1.12), option(2, "EURUSD", 1.15, call_put="PUT"),
        option(3, "USDJPY", 160.0),
        Trade(product_type=FX_FORWARD, currency_pair="EURUSD", trade_date=VALUATION,
              expiry=date(2027, 1, 15), buy_sell="SELL", notional=1e6, contract_rate=1.13,
              trade_id="TR-000004").normalized(),
        option(5, "EURUSD", 1.12, expiry=date(2026, 10, 1),
               trade_date=date(2026, 9, 1)),                            # expired: an error
        option(6, "EURUSD", 1.12, trade_date=date(2026, 10, 20)),     # booked later: an error
    ]


@pytest.fixture
def cache(tmp_path):
    return ResultCache(tmp_path / "results.sqlite")


def counting(monkeypatch):
    calls = []
    real = service.price_trades

    def spy(markets, trades, *args, **kwargs):
        calls.append([t.trade_id for t in trades])
        return real(markets, trades, *args, **kwargs)
    monkeypatch.setattr(service, "price_trades", spy)
    return calls


def test_cached_results_are_exactly_the_fresh_ones(markets, cache, monkeypatch):
    fresh = service.price_trades(markets, book(), MODS)
    calls = counting(monkeypatch)
    first = cached.price_trades(markets, book(), MODS, cache=cache)
    second = cached.price_trades(markets, book(), MODS, cache=cache)
    assert first == fresh and second == fresh
    assert len(calls) == 1 and len(calls[0]) == 6            # the second call priced nothing
    assert len(fresh[2]) == 2                                  # the two errors, cached too


@pytest.mark.parametrize("change, repriced", [
    ("amend", {"TR-000001"}),
    ("spot", {"TR-000001", "TR-000002", "TR-000004", "TR-000005", "TR-000006"}),
    ("quote", {"TR-000001", "TR-000002", "TR-000004", "TR-000005", "TR-000006"}),
    ("mods", None), ("extras", None), ("valuation", None),
])
def test_each_change_reprices_only_what_it_affects(markets, cache, monkeypatch, change, repriced):
    trades = book()
    cached.price_trades(markets, trades, MODS, cache=cache)
    calls = counting(monkeypatch)
    m, mods, extras = markets, MODS, ()
    if change == "amend":
        trades[0] = replace(trades[0], version=trades[0].version + 1, notional=2e6)
    elif change == "spot":
        m = markets.with_spots({"EURUSD": 1.13})
    elif change == "quote":
        payloads = load_markets("MARKETS_20261015.json")
        payloads["EURUSD"]["vol_data"][3]["atm"] += 0.25
        m = MarketSet.from_payloads(payloads, VALUATION)
    elif change == "mods":
        mods = Modifications(beta_atm=0.5, beta_rr=1.2, beta_bf=0.0)
    elif change == "extras":
        extras = ("rho",)
    elif change == "valuation":
        m = MarketSet.from_payloads(load_markets("MARKETS_20261015.json"), date(2026, 10, 16))
    result = cached.price_trades(m, trades, mods, extras=extras, cache=cache)
    priced = set(calls[0]) if calls else set()
    monkeypatch.undo()
    assert result == service.price_trades(m, trades, mods, extras=extras)
    assert priced == (repriced if repriced is not None else {t.trade_id for t in trades})


def test_the_disk_layer_survives_a_restart_and_expires(markets, tmp_path, monkeypatch):
    path = tmp_path / "results.sqlite"
    cached.price_trades(markets, book(), MODS, cache=ResultCache(path))
    calls = counting(monkeypatch)
    cached.price_trades(markets, book(), MODS, cache=ResultCache(path))      # new process
    assert calls == []
    stale = ResultCache(path, ttl_hours=1e-9)
    time.sleep(0.01)
    cached.price_trades(markets, book(), MODS, cache=stale)
    assert len(calls) == 1


def test_an_unusable_cache_prices_fresh(markets, tmp_path, monkeypatch):
    broken = tmp_path / "results.sqlite"
    broken.write_bytes(b"not a database")
    cache = ResultCache(broken)
    assert cached.price_trades(markets, book(), MODS, cache=cache) == \
        service.price_trades(markets, book(), MODS)
    monkeypatch.setattr(cached, "_keys", lambda *a, **k: (_ for _ in ()).throw(RuntimeError()))
    assert cached.price_trades(markets, book(), MODS, cache=ResultCache(tmp_path / "b.sqlite")) \
        == service.price_trades(markets, book(), MODS)


def test_results_are_copies_and_memory_is_bounded(tmp_path):
    cache = ResultCache(tmp_path / "r.sqlite", max_memory=2, disk=False)
    cache.put("a", {"x": [1]})
    got = cache.get("a")
    got["x"].append(2)
    assert cache.get("a") == {"x": [1]}
    cache.put("b", 1)
    cache.put("c", 2)
    assert cache.get("a") is None and cache.get("c") == 2


def test_verify_agrees_and_catches_a_tampered_entry(markets, cache):
    trades = book()
    cached.price_trades(markets, trades, MODS, cache=cache)
    assert cached.verify(markets, trades, MODS, cache=cache, sample=6) == []
    key = cached._keys(markets, trades, MODS, (), "smile")["TR-000001"]
    kind, row = cache.get(key)
    cache.put(key, (kind, dict(row, mtm_base=row["mtm_base"] + 1.0)))
    assert ("TR-000001", "mtm_base") in {d[:2] for d in cached.verify(
        markets, trades, MODS, cache=cache, sample=6)}


def test_results_served_in_another_batch_are_still_exactly_fresh(markets, cache):
    # Cached from the whole book, served for part of it: identical to pricing
    # that part fresh, line numbers included.
    cached.price_trades(markets, book(), MODS, cache=cache)
    part = book()[1:]
    assert cached.price_trades(markets, part, MODS, cache=cache) == \
        service.price_trades(markets, part, MODS)

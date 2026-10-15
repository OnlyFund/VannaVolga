"""Booking service: Pricer lines -> booked trades and hedges -> priced book."""
import json
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from app import pricer
from app.market_set import MarketSet
from app.trades import service
from app.trades.model import FX_FORWARD, FX_OPTION, Trade
from app.trades.store import SqliteTradeStore
from golden.app_harness import load_markets
from market_data import Modifications

pytestmark = pytest.mark.filterwarnings("ignore")

MODS = Modifications(beta_atm=0.7, beta_rr=1.2, beta_bf=0.0)
VALUATION = date(2026, 10, 15)


class Clock:
    def __init__(self):
        self.now = datetime(2026, 10, 15, 9, 0, tzinfo=timezone.utc)

    def __call__(self):
        self.now += timedelta(seconds=1)
        return self.now


@pytest.fixture(scope="module")
def markets():
    return MarketSet.from_payloads(load_markets("MARKETS_20261015.json"), VALUATION)


@pytest.fixture
def store(tmp_path):
    opened = SqliteTradeStore(tmp_path / "trades.sqlite", clock=Clock())
    yield opened
    opened.close()


def line(pair="EURUSD", **changes):
    terms = dict(currency_pair=pair, strike="25d", expiry="1M", call_put="CALL",
                 buy_sell="BUY", notional="10m", premium=None, premium_ccy=pair[3:])
    terms.update(changes)
    return terms


def test_lines_book_at_the_values_the_pricer_shows(store, markets):
    rows = [line(counterparty="ACME"), line("USDJPY", strike="atm", expiry="2W", book="JPY")]
    result = service.book_lines(store, markets, rows, MODS, book="FXO", counterparty="DESK",
                                user="harel")
    assert result.ok, result.errors
    priced, _ = pricer.price_lines(markets, pricer.resolve_line_inputs(markets, pd.DataFrame(rows))[0], MODS)
    eur, jpy = result.booked
    assert (eur.trade_id, jpy.trade_id) == ("TR-000001", "TR-000002")
    assert result.lines == {1: "TR-000001", 2: "TR-000002"}
    assert eur.strike == pytest.approx(priced[0]["strike"]) and eur.expiry == priced[0]["expiry"]
    assert (eur.book, eur.counterparty) == ("FXO", "ACME")        # line column wins
    assert (jpy.book, jpy.counterparty) == ("JPY", "DESK")
    assert eur.trade_date == VALUATION and eur.premium == 0.0


def test_duplicate_lines_book_as_two_trades(store, markets):
    result = service.book_lines(store, markets, [line(), line()], MODS)
    assert [t.trade_id for t in result.booked] == ["TR-000001", "TR-000002"]


def test_one_bad_line_books_nothing(store, markets):
    result = service.book_lines(store, markets, [line(), line("GBPUSD")], MODS)
    assert not result.ok and result.booked == []
    assert result.errors == ["Line 2: No market loaded for GBPUSD"]
    assert store.trades(status=None) == []


def test_hedges_are_booked_and_linked_to_their_options(store, markets):
    rows = [line(), line(call_put="PUT", buy_sell="SELL", notional="4m"),
            line("USDJPY", expiry="2W")]
    result = service.book_lines(store, markets, rows, MODS, with_hedges=True)
    assert result.ok, result.errors
    options = [t for t in result.booked if t.product_type == FX_OPTION]
    hedges = [t for t in result.booked if t.product_type == FX_FORWARD]
    assert len(options) == 3 and len(hedges) == 2
    priced, _ = pricer.price_lines(markets, [o.to_pricer_row() for o in options], MODS)
    expected, _ = pricer.hedge_trades_for(markets, priced)
    assert sorted(h.signed_base_amount() for h in hedges) == pytest.approx(
        sorted(t["base_amount"] for t in expected))
    eur_hedge = next(h for h in hedges if h.currency_pair == "EURUSD")
    linked = {a.option for a in store.allocations(eur_hedge.trade_id)}
    assert linked == {options[0].trade_id, options[1].trade_id}
    assert [h.trade_id for h, _ in store.hedges_for(options[2].trade_id)] == [
        next(h.trade_id for h in hedges if h.currency_pair == "USDJPY")]


def test_booked_book_prices_with_trade_ids_and_hedges_flatten_delta(store, markets):
    rows = [line(), line(strike="10d", call_put="PUT"), line("USDJPY", expiry="2W")]
    service.book_lines(store, markets, rows, MODS, with_hedges=True)
    results, forwards, errors = service.price_trades(markets, service.select_trades(store), MODS)
    assert errors == []
    assert {r["trade_id"] for r in results} == {"TR-000001", "TR-000002", "TR-000003"}
    assert {f["trade_id"] for f in forwards} == {"TR-000004", "TR-000005"}
    for pair in ("EURUSD", "USDJPY"):
        option_delta = sum(r["delta"] for r in results if r["currency_pair"] == pair)
        hedge_delta = sum(f["delta_offset"] for f in forwards if f["currency_pair"] == pair)
        assert option_delta + hedge_delta == pytest.approx(0.0, abs=1e-6)
        # Booked at the forward rate: worth nothing on the booking date.
        assert sum(f["mtm_base"] for f in forwards if f["currency_pair"] == pair) == pytest.approx(0.0, abs=1e-6)


def test_pricing_skips_expired_and_names_failures_by_trade_id(store, markets):
    store.book([Trade(product_type=FX_OPTION, currency_pair="EURUSD", trade_date=date(2026, 9, 1),
                      expiry=date(2026, 10, 15), buy_sell="BUY", notional=1e6, strike=1.1,
                      call_put="CALL", premium_ccy="USD"),
                Trade(product_type=FX_OPTION, currency_pair="GBPUSD", trade_date=VALUATION,
                      expiry=date(2026, 11, 17), buy_sell="BUY", notional=1e6, strike=1.3,
                      call_put="CALL", premium_ccy="USD")])
    results, _, errors = service.price_trades(markets, store.trades(), MODS)
    assert results == []
    assert errors == ["TR-000001: expired on 2026-10-15",
                      "TR-000002: No market loaded for GBPUSD"]


def test_export_and_import_carry_trades_and_hedge_links(tmp_path, store, markets):
    service.book_lines(store, markets, [line(), line(call_put="PUT")], MODS, with_hedges=True,
                       counterparty="ACME")
    exported = json.dumps(service.export_book(store))
    target = SqliteTradeStore(tmp_path / "other.sqlite", clock=Clock())
    target.book([Trade(product_type=FX_OPTION, currency_pair="EURUSD", trade_date=VALUATION,
                       expiry=date(2026, 11, 17), buy_sell="BUY", notional=1e6, strike=1.1,
                       call_put="CALL", premium_ccy="USD")])
    mapping, notes = service.import_book(target, exported, user="harel")
    assert notes == []
    assert mapping == {"TR-000001": "TR-000002", "TR-000002": "TR-000003", "TR-000003": "TR-000004"}
    assert {a.option for a in target.allocations("TR-000004")} == {"TR-000002", "TR-000003"}
    assert {t.counterparty for t in target.trades()} == {"", "ACME"}
    target.close()


def test_import_keeps_a_hedge_whose_option_was_cancelled_unlinked(tmp_path, store, markets):
    service.book_lines(store, markets, [line()], MODS, with_hedges=True)
    store.cancel("TR-000001", 1, reason="unwind")
    target = SqliteTradeStore(tmp_path / "other.sqlite", clock=Clock())
    mapping, notes = service.import_book(target, service.export_book(store))
    assert mapping == {"TR-000002": "TR-000001"}
    assert notes == ["TR-000002 imported without its links: TR-000001 not imported"]
    target.close()


def test_import_refuses_other_files(store):
    for payload in ('{"format": "market"}', {"format": "trade_book", "version": 99, "trades": []},
                    {"format": "trade_book", "version": 1, "trades": []}):
        with pytest.raises(Exception, match="not a trade book|newer than supported|no live trades"):
            service.import_book(store, payload)


def test_csv_export_lists_every_trade(store, markets):
    service.book_lines(store, markets, [line(), line()], MODS)
    store.cancel("TR-000002", 1)
    text = service.export_csv(store)
    rows = list(pd.read_csv(pd.io.common.StringIO(text))["trade_id"])
    assert rows == ["TR-000001", "TR-000002"]

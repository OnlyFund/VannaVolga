"""No trade leaks to an earlier date: the book on a valuation date holds only
trades booked by then (trade date on or before it). A hedge booked today is
not in yesterday's delta or P&L; on its trade date and after it is."""
from datetime import date
from types import SimpleNamespace

import pytest

from app.market_set import MarketSet
from app.trades import service
from app.trades.model import FX_FORWARD, FX_OPTION, Booking, Trade
from app.trades.store import SqliteTradeStore
from app.ui import risk_block
from golden.app_harness import load_markets
from market_data import Modifications

MODS = Modifications(beta_atm=0.7, beta_rr=1.2, beta_bf=0.0)
OPTION_DAY, HEDGE_DAY = date(2026, 10, 14), date(2026, 10, 16)


def markets_on(valuation):
    return MarketSet.from_payloads(load_markets("MARKETS_20261015.json"), valuation)


@pytest.fixture
def store(tmp_path):
    store = SqliteTradeStore(tmp_path / "trades.sqlite")
    option = Trade(product_type=FX_OPTION, currency_pair="EURUSD", trade_date=OPTION_DAY,
                   expiry=date(2027, 1, 15), buy_sell="BUY", notional=10e6, strike=1.12,
                   call_put="CALL", premium=0.0, premium_ccy="USD")
    store.book(Booking((option,)))
    hedge = Trade(product_type=FX_FORWARD, currency_pair="EURUSD", trade_date=HEDGE_DAY,
                  expiry=date(2027, 1, 15), buy_sell="SELL", notional=5e6, contract_rate=1.13)
    store.book(Booking((hedge,)))
    return store


def test_the_book_on_a_date_holds_only_trades_booked_by_then(store):
    trades = store.trades()
    for day, ids in ((date(2026, 10, 13), []), (OPTION_DAY, ["TR-000001"]),
                     (date(2026, 10, 15), ["TR-000001"]),
                     (HEDGE_DAY, ["TR-000001", "TR-000002"])):
        in_book, later = service.book_on(trades, day)
        assert [t.trade_id for t in in_book] == ids, day
        assert len(in_book) + len(later) == 2
    _, later = service.book_on(trades, date(2026, 10, 15))
    assert service.not_booked_note(later, date(2026, 10, 15)) == (
        "1 trade booked after 15-Oct-2026 not in the book on that date: TR-000002 (16-Oct-2026)")
    assert service.not_booked_note([], date(2026, 10, 15)) is None


def test_pricing_refuses_a_trade_booked_after_the_valuation_date(store):
    results, forwards, errors = service.price_trades(markets_on(date(2026, 10, 15)),
                                                     store.trades(), MODS)
    assert [r["trade_id"] for r in results] == ["TR-000001"] and forwards == []
    assert errors == ["TR-000002: booked 2026-10-16, after the valuation date 2026-10-15; "
                      "not in that date's book"]


def test_a_hedge_booked_later_moves_delta_and_pnl_only_from_its_trade_date(store):
    option, hedge = store.trades()

    def book(valuation):
        in_book, _ = service.book_on(store.trades(), valuation)
        results, forwards, errors = service.price_trades(markets_on(valuation), in_book, MODS)
        assert errors == []
        rows = [*results, *forwards]
        return sum(r["spot_delta"] for r in rows), sum(r["pnl_base"] for r in rows), len(forwards)

    def alone(trade, valuation):
        results, forwards, _ = service.price_trades(markets_on(valuation), [trade], MODS)
        (row,) = results or forwards
        return row["spot_delta"], row["pnl_base"]

    before, on = date(2026, 10, 15), HEDGE_DAY
    # The day before the hedge: the option alone, exactly.
    delta, pnl, hedges = book(before)
    assert hedges == 0 and (delta, pnl) == alone(option, before)
    # From the hedge's trade date: option and hedge.
    delta, pnl, hedges = book(on)
    (od, op), (hd, hp) = alone(option, on), alone(hedge, on)
    assert hedges == 1
    assert delta == pytest.approx(od + hd, rel=1e-12) and pnl == pytest.approx(op + hp, rel=1e-12)
    assert hd < -4e6                                         # the 5m sell hedge


def test_risk_takes_the_selection_in_the_book_on_its_date(store, monkeypatch):
    monkeypatch.setattr(risk_block, "st", SimpleNamespace(
        session_state={"_portfolio_selection": ["TR-000001", "TR-000002"]}))
    trades, note = risk_block._selection(store, date(2026, 10, 15))
    assert [t.trade_id for t in trades] == ["TR-000001"] and "TR-000002" in note
    trades, note = risk_block._selection(store, HEDGE_DAY)
    assert len(trades) == 2 and note is None
    monkeypatch.setattr(risk_block, "st", SimpleNamespace(session_state={}))
    trades, note = risk_block._selection(store, date(2026, 10, 13))
    assert trades == [] and "2 trades booked after 13-Oct-2026" in note

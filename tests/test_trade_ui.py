"""Pricer grid booking: ticked lines, trade ids written back, edits unbook."""
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

import app.ui.app as ui
from app.market_set import MarketSet
from app.trades.store import SqliteTradeStore
from app.ui.portfolio_block import TRADE_ID_COLUMN, _apply_editor_state
from app.ui.trades_block import trades_frame
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


@pytest.fixture
def session(monkeypatch):
    state = SimpleNamespace()
    monkeypatch.setattr(ui, "st", SimpleNamespace(session_state=state))
    return state


@pytest.fixture
def store(tmp_path):
    opened = SqliteTradeStore(tmp_path / "trades.sqlite", clock=Clock())
    yield opened
    opened.close()


@pytest.fixture(scope="module")
def markets():
    return MarketSet.from_payloads(load_markets("MARKETS_20261015.json"), VALUATION)


def grid(*lines):
    base = dict(currency_pair="EURUSD", strike="25d", expiry="1M", call_put="CALL",
                buy_sell="BUY", notional="1m", premium=0.0, premium_ccy="USD", book="",
                counterparty="", trade_id="")
    return pd.DataFrame([dict(base, **line) for line in lines])


def test_ticked_lines_book_and_get_their_trade_ids(session, store, markets):
    rows = grid({}, {"currency_pair": "USDJPY", "premium_ccy": "JPY"}, {"strike": "1.20"})
    messages = ui.book_ticked_lines(store, markets, rows, [0, 2], MODS, "FXO", "ACME",
                                    True, "EURUSD")
    assert messages == [("success", "Booked TR-000001, TR-000002; hedges TR-000003.")]
    written = session._pending_resolved_portfolio_rows
    assert list(written[TRADE_ID_COLUMN]) == ["TR-000001", "", "TR-000002"]
    assert written.at[0, "expiry"] == "2026-11-17" and written.at[0, "strike"] != "25d"
    assert (written.at[2, "book"], written.at[2, "counterparty"]) == ("FXO", "ACME")


def test_booked_lines_are_not_booked_twice(session, store, markets):
    rows = grid({"trade_id": "TR-000009"}, {})
    messages = ui.book_ticked_lines(store, markets, rows, [0, 1], MODS, "", "", False, "EURUSD")
    assert messages[0] == ("warning", "Line 1 is already booked as TR-000009; add a new line "
                                      "to book the same terms again.")
    assert messages[1] == ("success", "Booked TR-000001.")


def test_failed_booking_names_grid_lines_and_books_nothing(session, store, markets):
    rows = grid({}, {}, {"currency_pair": "GBPUSD"})
    messages = ui.book_ticked_lines(store, markets, rows, [1, 2], MODS, "", "", False, "EURUSD")
    assert messages == [("error", "Nothing booked — Line 3: No market loaded for GBPUSD")]
    assert store.trades(status=None) == []
    assert not hasattr(session, "_pending_resolved_portfolio_rows")


def test_editing_a_booked_line_unbooks_it_but_ticks_do_not():
    rows = grid({"trade_id": "TR-000001"}, {"trade_id": "TR-000002"})
    rows["_book"] = False
    edited = _apply_editor_state(rows, {"edited_rows": {0: {"notional": "2m"},
                                                        1: {"_book": True}}})
    assert list(edited[TRADE_ID_COLUMN]) == ["", "TR-000002"]
    assert list(edited["_book"]) == [False, True]


def test_blotter_shows_amounts_and_rates_at_pair_precision(store, markets):
    from app.trades import service
    service.book_lines(store, markets, grid({}), MODS, with_hedges=True)
    frame = trades_frame(store.trades())
    assert frame.at[0, "strike"].count(".") == 1 and len(frame.at[0, "strike"].split(".")[1]) == 4
    assert frame.at[1, "contract_rate"].count(".") == 1 and len(frame.at[1, "contract_rate"].split(".")[1]) == 5
    assert "," in frame.at[1, "notional"] or float(frame.at[1, "notional"]) < 1000


def test_portfolio_defaults_to_totals_per_pair(store, markets):
    from app.trades import service
    from app.ui.trades_block import BY_PAIR, PORTFOLIO_VIEWS, pair_totals
    assert PORTFOLIO_VIEWS[0] == BY_PAIR
    service.book_lines(store, markets, grid({}, {}, {"currency_pair": "USDJPY", "premium_ccy": "JPY"}),
                       MODS, with_hedges=True)
    results, forwards, errors = service.price_trades(markets, store.trades(), MODS)
    assert errors == []
    totals = pair_totals(results, forwards)
    assert list(totals["currency_pair"]) == ["EURUSD", "USDJPY"]
    assert list(totals["options"]) == [2, 1] and list(totals["forwards"]) == [1, 1]
    assert list(totals["delta"]) == pytest.approx([0.0, 0.0], abs=1e-6)
    eur = [r for r in results if r["currency_pair"] == "EURUSD"]
    assert totals.at[0, "mtm_base"] == pytest.approx(sum(r["mtm_base"] for r in eur))

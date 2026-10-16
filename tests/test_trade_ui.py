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


def test_portfolio_reports_spot_and_usd_columns(store, markets, tmp_path):
    from app import usd_spots
    from app.trades import service
    from app.ui.trades_block import pair_totals, usd_totals
    service.book_lines(store, markets, grid({}, {}, {"currency_pair": "USDJPY", "premium_ccy": "JPY"}),
                       MODS, with_hedges=True)
    results, forwards, _ = service.price_trades(markets, store.trades(), MODS)
    from market_data.store import MarketDataStore
    spots = {"EURUSD": 1.12, "USDJPY": 158.0}
    market_data = MarketDataStore(tmp_path / "market_data.sqlite")
    market_data.save("2026-10-15T17:00:00", "bloomberg", spots={**spots, "EURUSD": 1.0})
    market_data.save("2026-10-16T09:00:00", "bloomberg", spots=spots)
    assert usd_spots.latest_snapshot("2026-10-16", market_data).spots == spots
    assert usd_spots.latest_snapshot("2026-10-15", market_data).spots["EURUSD"] == 1.0
    assert usd_spots.pair_spot("EURJPY", spots) == pytest.approx(1.12 * 158.0)
    totals = pair_totals(results, forwards)
    frame, usd = usd_totals(totals, spots)
    assert usd == ["mtm_usd", "pnl_usd", "vega_usd", "vanna_usd", "volga_usd",
                   "rega_usd", "sega_usd"]
    assert list(frame["currency_pair"]) == ["EURUSD", "USDJPY", "Total (USD)"]
    assert list(frame["spot"][:2]) == [1.12, 158.0]
    # EUR base converts at EURUSD; USD base is already USD.
    assert frame.at[0, "vega_usd"] == pytest.approx(totals.at[0, "vega"] * 1.12)
    assert frame.at[1, "mtm_usd"] == pytest.approx(totals.at[1, "mtm_base"])
    assert frame.at[2, "pnl_usd"] == pytest.approx(frame["pnl_usd"][:2].sum())
    assert usd_spots.add_usd_columns(totals, {"USDJPY": 158.0})["vega_usd"].isna()[0]


def test_portfolio_spot_reprices_without_moving_the_market(store, markets):
    from app.trades import service
    from app.ui.trades_block import apply_spot_edits, market_spots, portfolio_spots
    service.book_lines(store, markets, grid({}, {}), MODS, with_hedges=True)
    base = market_spots(markets)["EURUSD"]
    moved = markets.with_spots({"EURUSD": base * 1.01})
    assert moved.market("EURUSD").spot == pytest.approx(base * 1.01)
    assert markets.market("EURUSD").spot == base          # market data untouched
    before, before_fwd, _ = service.price_trades(markets, store.trades(), MODS)
    after, after_fwd, errors = service.price_trades(moved, store.trades(), MODS)
    assert errors == []
    assert [r["mtm_base"] for r in after] != pytest.approx([r["mtm_base"] for r in before])
    assert after_fwd[0]["mtm_base"] != pytest.approx(before_fwd[0]["mtm_base"])
    with pytest.raises(Exception):
        markets.with_spots({"EURUSD": -1.0})

    market = {"EURUSD": base, "USDJPY": 158.0}
    pairs = ["EURUSD", "USDJPY"]
    overrides = apply_spot_edits({0: {"spot": 1.2}, 2: {"spot": 9.0}}, pairs, market, {})
    assert overrides == {"EURUSD": 1.2}                    # total row edit ignored
    assert apply_spot_edits({0: {"spot": base}}, pairs, market, overrides) == {}
    assert apply_spot_edits({0: {"spot": None}}, pairs, market, overrides) == {}
    assert apply_spot_edits({1: {"spot": 160.0}}, pairs, market, overrides) == {
        "EURUSD": 1.2, "USDJPY": 160.0}
    spots = portfolio_spots(market, overrides, {"EURUSD": 1.0, "GBPUSD": 1.3})
    assert spots == {"EURUSD": 1.2, "USDJPY": 158.0, "GBPUSD": 1.3}


def test_forwards_alone_still_total_per_pair(store, markets):
    from app.trades import service
    from app.ui.trades_block import pair_totals
    service.book_lines(store, markets, grid({}, {}), MODS, with_hedges=True)
    forwards_only = [t for t in store.trades() if t.is_forward]
    results, forwards, errors = service.price_trades(markets, forwards_only, MODS)
    assert errors == [] and results == [] and len(forwards) == 1
    totals = pair_totals(results, forwards)
    assert list(totals["currency_pair"]) == ["EURUSD"]
    assert list(totals["options"]) == [0] and list(totals["forwards"]) == [1]
    assert float(totals.at[0, "mtm_base"]) == pytest.approx(forwards[0]["mtm_base"])
    assert float(totals.at[0, "delta"]) == pytest.approx(forwards[0]["delta_offset"])
    assert float(totals.at[0, "vega"]) == 0.0

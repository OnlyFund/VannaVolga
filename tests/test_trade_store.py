"""Trade model and SQLite store: ids, versions, hedge links and failsafes."""
import sqlite3
from datetime import date, datetime, timedelta, timezone

import pytest

from app.trades.model import (
    CANCELLED, FX_FORWARD, FX_OPTION, LIVE, Booking, HedgeAllocation, Trade,
    TradeValidationError,
)
from app.trades.store import (
    BACKUPS_KEPT, SqliteTradeStore, StaleVersionError, TradeNotFound, TradeStoreReadOnly,
    TradeStoreUnavailable, open_trade_store,
)

TRADE_DATE = date(2026, 10, 15)


class Clock:
    """Deterministic UTC clock, one second per call."""

    def __init__(self):
        self.now = datetime(2026, 10, 15, 9, 0, tzinfo=timezone.utc)

    def __call__(self):
        self.now += timedelta(seconds=1)
        return self.now


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def store(tmp_path, clock):
    opened = SqliteTradeStore(tmp_path / "trades.sqlite", clock=clock)
    yield opened
    opened.close()


def option(**changes):
    terms = dict(product_type=FX_OPTION, currency_pair="EURUSD", trade_date=TRADE_DATE,
                 expiry=date(2026, 11, 17), buy_sell="BUY", notional=10e6, strike=1.12,
                 call_put="CALL", premium=0.0, premium_ccy="USD", book="FXO")
    terms.update(changes)
    return Trade(**terms)


def forward(base_amount, **changes):
    terms = dict(product_type=FX_FORWARD, currency_pair="EURUSD", trade_date=TRADE_DATE,
                 expiry=date(2026, 11, 17), buy_sell="BUY" if base_amount > 0 else "SELL",
                 notional=abs(base_amount), contract_rate=1.1250, book="FXO")
    terms.update(changes)
    return Trade(**terms)


# -- model -------------------------------------------------------------------

def test_trade_holds_booked_values_only():
    with pytest.raises(TradeValidationError) as error:
        option(strike="25d", expiry="1M", call_put="C", premium_ccy="JPY").normalized()
    message = str(error.value)
    for problem in ("strike must be a number", "expiry must be a date",
                    "call/put must be CALL or PUT", "premium currency must be EUR or USD"):
        assert problem in message
    clean = option(currency_pair="eur/usd", strike="1.1200", expiry="2026-11-17",
                   buy_sell="buy", call_put="put").normalized()
    assert (clean.currency_pair, clean.strike, clean.expiry, clean.buy_sell, clean.call_put) == (
        "EURUSD", 1.12, date(2026, 11, 17), "BUY", "PUT")


def test_forward_and_option_terms_do_not_mix():
    with pytest.raises(TradeValidationError, match="a forward has no strike"):
        forward(1e6, strike=1.1).normalized()
    with pytest.raises(TradeValidationError, match="a forward has no premium"):
        forward(1e6, premium=100.0).normalized()
    with pytest.raises(TradeValidationError, match="an option has no contract rate"):
        option(contract_rate=1.1).normalized()
    with pytest.raises(TradeValidationError, match="must be after the trade date"):
        option(expiry=TRADE_DATE).normalized()


# -- booking -----------------------------------------------------------------

def test_identical_lines_book_as_separate_trades(store):
    first, second = store.book([option(), option()], user="harel")
    assert (first.trade_id, second.trade_id) == ("TR-000001", "TR-000002")
    assert first.package_id == second.package_id == "PK-000001"
    assert [t.trade_id for t in store.trades()] == ["TR-000001", "TR-000002"]
    third, = store.book([option()])
    assert third.trade_id == "TR-000003" and third.package_id == "PK-000002"


def test_hedge_is_booked_with_its_options(store):
    call, put, hedge = store.book(Booking(
        (option(), option(call_put="PUT", strike=1.10), forward(-4e6)),
        hedges={2: [HedgeAllocation(0, -5e6), HedgeAllocation(1, 1e6)]},
    ))
    assert store.allocations(hedge.trade_id) == [
        HedgeAllocation(call.trade_id, -5e6), HedgeAllocation(put.trade_id, 1e6)]
    assert store.hedges_for(call.trade_id) == [(hedge, -5e6)]
    # A later hedge can reference an already booked option by id.
    top_up, = store.book(Booking((forward(-1e6),), hedges={0: [HedgeAllocation(call.trade_id, -1e6)]}))
    assert {h.trade_id for h, _ in store.hedges_for(call.trade_id)} == {hedge.trade_id, top_up.trade_id}


@pytest.mark.parametrize("booking, message", [
    (Booking((option(), forward(-4e6)), hedges={1: [HedgeAllocation(0, -5e6)]}),
     "sum to -5,000,000.00, not the hedge's signed base amount -4,000,000.00"),
    (Booking((option(currency_pair="USDJPY", strike=150.0, premium_ccy="JPY"), forward(-1e6)),
             hedges={1: [HedgeAllocation(0, -1e6)]}), "cannot hedge"),
    (Booking((option(), option()), hedges={1: [HedgeAllocation(0, -1e6)]}), "only forwards hedge"),
    (Booking((forward(-1e6),), hedges={0: [HedgeAllocation(3, -1e6)]}), "not in the booking"),
    (Booking((option(), option(strike="bad"))), "strike must be a number"),
])
def test_failed_booking_writes_nothing(store, booking, message):
    with pytest.raises(TradeValidationError, match=message):
        store.book(booking)
    assert store.trades(status=None) == []
    assert store.book([option()])[0].trade_id == "TR-000001"   # no id was consumed


# -- versions ----------------------------------------------------------------

def test_amendment_adds_a_version_and_old_books_stay_readable(store, clock):
    booked, = store.book([option()], user="harel")
    before = clock.now
    amended = store.amend(booked.trade_id, 1, {"strike": 1.13}, user="harel", reason="strike fix")
    assert (amended.version, amended.strike) == (2, 1.13)
    assert store.get(booked.trade_id).strike == 1.13
    assert store.get(booked.trade_id, as_of=before).strike == 1.12
    history = store.history(booked.trade_id)
    assert [(v.action, v.trade.version, v.reason) for v in history] == [
        ("new", 1, ""), ("amend", 2, "strike fix")]


def test_edit_from_an_old_version_is_rejected(store):
    booked, = store.book([option()])
    store.amend(booked.trade_id, 1, {"notional": 5e6})
    with pytest.raises(StaleVersionError, match="at version 2, not 1"):
        store.amend(booked.trade_id, 1, {"notional": 7e6})
    with pytest.raises(StaleVersionError):
        store.cancel(booked.trade_id, 1)
    assert store.get(booked.trade_id).notional == 5e6


def test_amendment_cannot_change_identity_or_product(store):
    booked, = store.book([option()])
    for change in ({"trade_id": "X"}, {"product_type": FX_FORWARD}, {"version": 9}):
        with pytest.raises(TradeValidationError, match="cannot amend"):
            store.amend(booked.trade_id, 1, change)
    with pytest.raises(TradeValidationError, match="strike must be"):
        store.amend(booked.trade_id, 1, {"strike": "25d"})


def test_cancelled_option_leaves_a_visible_orphan_hedge(store):
    call, hedge = store.book(Booking((option(), forward(-5e6)),
                                     hedges={1: [HedgeAllocation(0, -5e6)]}))
    assert store.orphaned_hedges() == []
    cancelled = store.cancel(call.trade_id, 1, reason="client unwind")
    assert cancelled.status == CANCELLED
    assert [t.trade_id for t in store.trades()] == [hedge.trade_id]
    assert store.orphaned_hedges() == [(hedge, call.trade_id)]
    with pytest.raises(TradeValidationError, match="is cancelled"):
        store.amend(call.trade_id, 2, {"notional": 1e6})
    # The hedge can still be amended with its existing links...
    with pytest.raises(TradeValidationError, match="sum to"):
        store.amend(hedge.trade_id, 1, {"notional": 2e6})
    # ...and resized only together with its allocations.
    resized = store.amend(hedge.trade_id, 1, {"notional": 2e6},
                          allocations=[HedgeAllocation(store.book([option()])[0].trade_id, -2e6)])
    assert resized.version == 2


def test_filters_and_matured_trades(store):
    store.book([option(), option(currency_pair="USDJPY", strike=150.0, premium_ccy="JPY",
                                 book="RATES", expiry=date(2026, 10, 30))])
    assert [t.currency_pair for t in store.trades(currency_pair="usdjpy")] == ["USDJPY"]
    assert [t.book for t in store.trades(book="RATES")] == ["RATES"]
    assert len(store.trades(live_on=date(2026, 11, 1))) == 1
    assert store.trades(product_type=FX_FORWARD) == []
    with pytest.raises(TradeNotFound):
        store.get("TR-999999")


# -- failsafes ---------------------------------------------------------------

def test_versions_cannot_be_edited_or_deleted_in_the_database(store):
    store.book([option()])
    raw = sqlite3.connect(str(store.path))
    for sql in ("UPDATE trade_versions SET notional = 1", "DELETE FROM trade_versions"):
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            raw.execute(sql)
    raw.close()


def test_book_survives_reopening_and_ids_continue(tmp_path, clock):
    path = tmp_path / "trades.sqlite"
    first = SqliteTradeStore(path, clock=clock)
    first.book([option()])
    first.close()
    second = SqliteTradeStore(path, clock=clock)
    assert [t.trade_id for t in second.trades()] == ["TR-000001"]
    assert second.book([option()])[0].trade_id == "TR-000002"
    second.close()


def test_unreadable_file_is_never_replaced_by_an_empty_book(tmp_path):
    path = tmp_path / "trades.sqlite"
    path.write_bytes(b"not a database" * 100)
    store, problem = open_trade_store(path)
    assert store is None and "trade store unavailable" in problem
    assert path.read_bytes() == b"not a database" * 100


def test_foreign_database_is_refused(tmp_path):
    path = tmp_path / "other.sqlite"
    raw = sqlite3.connect(str(path))
    raw.execute("CREATE TABLE something (x)")
    raw.commit()
    raw.close()
    store, problem = open_trade_store(path)
    assert store is None and "not a trade store" in problem


def test_newer_schema_opens_read_only(tmp_path, clock):
    path = tmp_path / "trades.sqlite"
    writable = SqliteTradeStore(path, clock=clock)
    writable.book([option()])
    writable.close()
    raw = sqlite3.connect(str(path))
    raw.execute("UPDATE schema_info SET version = 99")
    raw.commit()
    raw.close()
    store, problem = open_trade_store(path, clock=clock)
    assert store.read_only and problem.startswith("trade store is read-only") and "newer version" in problem
    assert [t.trade_id for t in store.trades()] == ["TR-000001"]   # still viewable
    with pytest.raises(TradeStoreReadOnly):
        store.book([option()])
    store.close()
    raw = sqlite3.connect(str(path))
    raw.execute("CREATE TABLE renamed AS SELECT * FROM hedge_links")
    raw.execute("DROP TABLE hedge_links")
    raw.execute("CREATE TABLE hedge_links (hedge_id, hedge_version, option_id)")
    raw.commit()
    raw.close()
    store, problem = open_trade_store(path, clock=clock)
    assert store is None and "hedge_links lacks base_amount" in problem


def test_daily_backups_are_kept_and_pruned(tmp_path, clock):
    path = tmp_path / "trades.sqlite"
    for day in range(BACKUPS_KEPT + 3):
        SqliteTradeStore(path, clock=clock, today=date(2026, 10, 1) + timedelta(days=day)).close()
    backups = sorted((tmp_path / "backups").glob("trades_*.sqlite"))
    assert len(backups) == BACKUPS_KEPT
    assert backups[-1].name == "trades_20261017.sqlite"
    copy = SqliteTradeStore(backups[-1], clock=clock, backup_dir=tmp_path / "other")
    assert copy.trades() == []
    copy.close()


def test_booked_option_prices_like_the_grid_line():
    from app import pricer
    from app.market_set import MarketSet
    from golden.app_harness import load_markets
    from market_data import Modifications

    mods = Modifications(beta_atm=0.7, beta_rr=1.2, beta_bf=0.0)
    markets = MarketSet.from_payloads(load_markets("MARKETS_20261015.json"), TRADE_DATE)
    trade = option(strike=1.1250).normalized()
    grid_line = dict(currency_pair="EURUSD", strike="1.1250", expiry="2026-11-17",
                     call_put="CALL", buy_sell="BUY", notional="10m", premium=0.0,
                     premium_ccy="USD")
    booked, _ = pricer.price_lines(markets, [trade.to_pricer_row()], mods)
    typed, _ = pricer.price_lines(markets, [grid_line], mods)
    assert booked == typed


def test_two_sessions_cannot_overwrite_each_others_edits(tmp_path, clock):
    path = tmp_path / "trades.sqlite"
    first, second = SqliteTradeStore(path, clock=clock), SqliteTradeStore(path, clock=clock)
    booked, = first.book([option()])
    first.amend(booked.trade_id, 1, {"notional": 5e6})
    with pytest.raises(StaleVersionError):
        second.amend(booked.trade_id, 1, {"notional": 7e6})
    assert second.get(booked.trade_id).notional == 5e6
    assert second.book([option()])[0].trade_id == "TR-000002"   # shared id sequence
    first.close()
    second.close()


def test_failure_midway_through_a_booking_writes_nothing(store, monkeypatch):
    def broken_link(hedge, allocations):
        raise sqlite3.OperationalError("disk I/O error")
    monkeypatch.setattr(store, "_link", broken_link)
    with pytest.raises(sqlite3.OperationalError):
        store.book(Booking((option(), forward(-5e6)), hedges={1: [HedgeAllocation(0, -5e6)]}))
    monkeypatch.undo()
    assert store.trades(status=None) == []
    assert store.book([option()])[0].trade_id == "TR-000001"


def test_book_accepts_a_booking_from_reloaded_code(store):
    """Streamlit reloads changed modules while the session keeps its store, so
    a Booking from the new code is not an instance of the store's class."""
    from dataclasses import dataclass, field as dc_field

    @dataclass(frozen=True)
    class ReloadedBooking:                 # same shape, different class identity
        trades: tuple
        hedges: dict = dc_field(default_factory=dict)

    booked = store.book(ReloadedBooking((option(),)))
    assert len(booked) == 1 and booked[0].trade_id


# -- schema 3: standalone spot/forward deals -----------------------------------

def deal(**changes):
    """A standalone forward: a value date, no option behind it."""
    terms = dict(product_type=FX_FORWARD, currency_pair="EURUSD", trade_date=TRADE_DATE,
                 expiry=None, value_date=date(2026, 12, 17), buy_sell="BUY", notional=1e6,
                 contract_rate=1.17, book="SPOT", counterparty="BANK")
    terms.update(changes)
    return Trade(**terms)


def test_a_value_dated_forward_expires_on_its_value_date():
    clean = deal(value_date="2026-12-17", amount_ccy="usd").normalized()
    assert (clean.expiry, clean.value_date, clean.amount_ccy) == (
        date(2026, 12, 17), date(2026, 12, 17), "USD")
    assert deal(expiry=date(2026, 12, 17)).normalized().expiry == date(2026, 12, 17)
    with pytest.raises(TradeValidationError, match="must equal its value date"):
        deal(expiry=date(2026, 12, 16)).normalized()
    with pytest.raises(TradeValidationError, match="value date 2026-10-15 must be after"):
        deal(value_date=TRADE_DATE).normalized()
    with pytest.raises(TradeValidationError, match="value date must be a date"):
        deal(value_date="1M").normalized()
    with pytest.raises(TradeValidationError, match="amount currency must be EUR or USD"):
        deal(amount_ccy="JPY").normalized()
    with pytest.raises(TradeValidationError, match="only a forward has a value date"):
        option(value_date=date(2026, 12, 17)).normalized()
    with pytest.raises(TradeValidationError, match="only a forward has a value date"):
        option(amount_ccy="EUR").normalized()


@pytest.mark.parametrize("buy_sell, amount_ccy, notional, base, term", [
    ("BUY", None, 1e6, 1e6, -1.17e6),          # base by default
    ("BUY", "EUR", 1e6, 1e6, -1.17e6),
    ("SELL", "EUR", 1e6, -1e6, 1.17e6),
    ("BUY", "USD", 1.17e6, -1e6, 1.17e6),      # buying term is selling base
    ("SELL", "USD", 1.17e6, 1e6, -1.17e6),
])
def test_the_entered_leg_gives_both_signed_amounts(buy_sell, amount_ccy, notional, base, term):
    trade = deal(buy_sell=buy_sell, amount_ccy=amount_ccy, notional=notional).normalized()
    assert trade.signed_base_amount() == pytest.approx(base)
    assert trade.signed_term_amount() == pytest.approx(term)
    assert trade.signed_term_amount() == pytest.approx(-trade.signed_base_amount() * 1.17)


def test_deals_round_trip_and_amend_their_value_date(store):
    booked, term = store.book([deal(), deal(amount_ccy="USD", buy_sell="SELL", notional=2.34e6)])
    assert store.get(booked.trade_id) == booked
    assert store.get(term.trade_id) == term and term.signed_base_amount() == pytest.approx(2e6)
    moved = store.amend(booked.trade_id, 1, {"value_date": date(2027, 1, 19)})
    assert (moved.value_date, moved.expiry) == (date(2027, 1, 19), date(2027, 1, 19))
    assert store.get(booked.trade_id) == moved
    hedge, = store.book([forward(-1e6)])
    assert (hedge.value_date, hedge.amount_ccy) == (None, None)


def test_a_term_entered_hedge_sums_in_base(store):
    call, hedge = store.book(Booking(
        (option(), forward(0, buy_sell="BUY", notional=1.125e6, amount_ccy="USD")),
        hedges={1: [HedgeAllocation(0, -1e6)]}))
    assert hedge.signed_base_amount() == pytest.approx(-1e6)


def test_a_schema_2_store_migrates_with_a_backup_and_reads_as_before(tmp_path, clock):
    import app.trades.store as store_module
    path = tmp_path / "trades.sqlite"
    raw = sqlite3.connect(str(path))
    v2 = store_module._SCHEMA.replace("    value_date TEXT,\n    amount_ccy TEXT,\n", "")
    assert "value_date" not in v2
    raw.close()
    raw = sqlite3.connect(str(path))
    raw.executescript(v2 + "INSERT INTO schema_info (version) VALUES (2);"
                      "INSERT INTO sequences (name, next) VALUES ('trade', 2);")
    raw.execute("INSERT INTO trade_versions (trade_id, version, action, recorded_at, status, "
                "product_type, currency_pair, trade_date, expiry, buy_sell, notional, "
                "contract_rate) VALUES ('TR-000001', 1, 'new', '2026-10-15T09:00:00.000000Z', "
                "'live', 'fx_forward', 'EURUSD', '2026-10-15', '2026-11-17', 'SELL', 1e6, 1.125)")
    raw.commit()
    raw.close()
    store, problem = open_trade_store(path, clock=clock)
    assert problem is None and not store.read_only
    assert store._schema_version() == 3
    assert (tmp_path / "backups" / "trades_before_schema_3.sqlite").exists()
    trade = store.get("TR-000001")
    assert (trade.value_date, trade.amount_ccy, trade.signed_base_amount()) == (None, None, -1e6)
    booked, = store.book([deal()])
    assert store.get(booked.trade_id).value_date == date(2026, 12, 17)
    store.close()


def test_a_value_dated_forward_is_not_priced_on_the_option_expiry_convention():
    from app.trades import service
    from app.market_set import MarketSet
    from golden.app_harness import load_markets
    from market_data import Modifications
    markets = MarketSet.from_payloads(load_markets("MARKETS_20261015.json"), TRADE_DATE)
    trade = replace_id(deal().normalized(), "TR-000009")
    with pytest.raises(TradeValidationError, match="not valued yet"):
        trade.to_position()
    _, forwards, errors = service.price_trades(markets, [trade], Modifications())
    assert forwards == [] and errors == ["TR-000009: value-dated forwards are not valued yet"]


def replace_id(trade, trade_id):
    from dataclasses import replace
    return replace(trade, trade_id=trade_id)

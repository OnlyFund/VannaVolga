"""Event sets: CSV import, pinning per market, markets built with their events."""
import sqlite3
import warnings

import pandas as pd
import pytest

import market_data.store as store_module
from app.store_markets import load_store_markets
from market_data.events import event_table, import_csv, read_events_csv
from market_data.store import SCHEMA_VERSION, MarketDataStore, MarketDataStoreError
from test_discount_curves import SURFACE, USD, points

TODAY = "2026-10-16"
GOOD_CSV = """pair,date,event,variance_days
EURUSD,2026-11-06,NFP,1.5
EURUSD,2026-11-06,ECB speech,0.25
eur/usd,2026-10-29,CPI,0.5
USDJPY,2026-10-30,BOJ,1.0
"""


@pytest.fixture
def store(tmp_path):
    return MarketDataStore(tmp_path / "market_data.sqlite")


def write(tmp_path, text, name="calendar.csv"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def market(store, event_set_id=None, date_time=f"{TODAY}T12:00:00"):
    return store.save(date_time, "bloomberg", spots={"EURUSD": 1.12, "USDJPY": 150.0},
                      depo_curves={"USD": USD}, vols={"EURUSD": SURFACE, "USDJPY": SURFACE},
                      forward_points={"EURUSD": points("EURUSD", TODAY, 1.12, 0.02, 0.04, 10000),
                                      "USDJPY": points("USDJPY", TODAY, 150.0, 0.04, 0.01, 100)},
                      event_set_id=event_set_id)


def test_csv_imports_as_a_set_with_named_events(store, tmp_path):
    event_set_id = import_csv(write(tmp_path, GOOD_CSV), store)
    event_set = store.load_event_set(event_set_id)
    assert (event_set.source, event_set.note) == ("csv", "calendar.csv")
    assert event_set.events["EURUSD"] == [("2026-10-29", "CPI", 0.5),
                                          ("2026-11-06", "ECB speech", 0.25),
                                          ("2026-11-06", "NFP", 1.5)]
    # One named row per event; same-day events stay separate (their
    # variance days add up when the market counts them).
    table = event_table(event_set, "EURUSD")
    assert list(table["event_date"].dt.strftime("%Y-%m-%d")) == [
        "2026-10-29", "2026-11-06", "2026-11-06"]
    assert list(table["event"]) == ["CPI", "ECB speech", "NFP"]
    assert list(table["variance_days"]) == [0.5, 0.25, 1.5]
    assert table.groupby("event_date")["variance_days"].sum().tolist() == [0.5, 1.75]
    assert event_table(event_set, "GBPUSD").empty
    assert list(store.event_sets()["events"]) == [4]


def test_a_csv_missing_a_column_is_rejected(store, tmp_path):
    with pytest.raises(ValueError, match="missing column"):
        import_csv(write(tmp_path, "pair,date,weight\nEURUSD,2026-11-06,1.5\n"), store)
    assert store.event_sets().empty


def test_bad_rows_reject_the_whole_file_and_name_every_problem(store, tmp_path):
    bad = GOOD_CSV + ("EURUS,2026-11-06,NFP,1\n"            # row 5: pair
                      "EURUSD,not a date,NFP,1\n"           # row 6: date
                      "EURUSD,2026-11-07,FOMC,lots\n"       # row 7: number
                      "EURUSD,2026-11-08,,1\n"              # row 8: no name
                      "EURUSD,2026-11-06,NFP,2\n")          # row 9: duplicate of row 1
    with pytest.raises(ValueError) as error:
        import_csv(write(tmp_path, bad), store)
    message = str(error.value)
    for row in (5, 6, 7, 8, 9):
        assert f"row {row}:" in message
    assert "row 1:" not in message and store.event_sets().empty


def test_a_market_pins_the_latest_set_and_re_pinning_is_recorded(store, tmp_path):
    first = import_csv(write(tmp_path, GOOD_CSV), store)
    market_id = market(store, store.latest_event_set_id())
    assert store.load(market_id=market_id).event_set_id == first
    second = import_csv(write(tmp_path, "pair,date,event,variance_days\n"
                                        "EURUSD,2026-11-06,NFP,2.0\n", "v2.csv"), store)
    assert store.latest_event_set_id() == second
    assert store.load(market_id=market_id).event_set_id == first       # pinned: unchanged
    with pytest.raises(ValueError, match="reason"):
        store.pin_events(market_id, second, "")
    with pytest.raises(ValueError, match="does not exist"):
        store.pin_events(market_id, 99, "typo")
    with pytest.raises(ValueError, match="does not exist"):
        store.pin_events(99, second, "typo")
    store.pin_events(market_id, second, "NFP weight revised")
    assert store.load(market_id=market_id).event_set_id == second
    history = store.event_pins(market_id)
    assert list(history["event_set_id"]) == [first, second]
    assert list(history["reason"]) == ["pinned when saved", "NFP weight revised"]


def test_the_fetch_pins_the_event_set_current_at_the_fetch(store, tmp_path, monkeypatch):
    fetch = pytest.importorskip("market_data.bloomberg_fetch")
    event_set_id = import_csv(write(tmp_path, GOOD_CSV), store)
    monkeypatch.setattr(fetch, "fetch_spots", lambda date: {"EURUSD": 1.12})
    monkeypatch.setattr(fetch, "fetch_depo_curves", lambda date: {"USD": USD})
    monkeypatch.setattr(fetch, "fetch_vols", lambda date: {"EURUSD": SURFACE})
    monkeypatch.setattr(fetch, "fetch_forward_points", lambda date: {})
    market_id = fetch.fetch_and_store(TODAY, store=store)
    assert store.load(market_id=market_id).event_set_id == event_set_id


def test_markets_are_built_with_their_pinned_events(store, tmp_path):
    event_set_id = import_csv(write(tmp_path, GOOD_CSV.replace("USDJPY,2026-10-30,BOJ,1.0\n",
                                                               "")), store)
    market(store, event_set_id)
    loaded = load_store_markets(TODAY, store=store)
    assert loaded.problem is None and sorted(loaded.payloads) == ["EURUSD", "USDJPY"]
    events = pd.DataFrame(loaded.payloads["EURUSD"]["event_calendar"])
    assert list(events["event"]) == ["CPI", "ECB speech", "NFP"]
    assert list(events["variance_days"]) == [0.5, 0.25, 1.5]
    assert loaded.payloads["USDJPY"]["event_calendar"] == []
    assert f"event set {event_set_id} has no events for USDJPY; built without events" in loaded.notes


def test_a_market_without_a_pinned_set_builds_without_events(store):
    market(store)
    loaded = load_store_markets(TODAY, store=store)
    assert loaded.problem is None and loaded.payloads["EURUSD"]["event_calendar"] == []
    assert any("no event set pinned" in note for note in loaded.notes)


def test_an_unreadable_pinned_set_makes_every_pair_unavailable(store, tmp_path):
    event_set_id = import_csv(write(tmp_path, GOOD_CSV), store)
    market(store, event_set_id)
    db = sqlite3.connect(str(store.path))           # foreign keys off: break the link
    db.execute("DELETE FROM events")
    db.execute("DELETE FROM event_sets")
    db.commit()
    db.close()
    loaded = load_store_markets(TODAY, store=store)
    assert loaded.payloads == {} and "event set 1 does not exist" in loaded.problem
    assert set(loaded.unavailable) == {"EURUSD", "USDJPY"}


@pytest.mark.parametrize("event_date, direction", [("2026-10-27", 1), ("2026-10-29", -1)])
def test_events_move_the_vol_between_pillars(store, tmp_path, event_date, direction):
    """Pillar variance is fixed by the quotes; an event moves its share within
    1W-2W: before a 28 October expiry it adds vol, after it takes vol away."""
    from app.market_context import MarketContext, PricingSettings
    event_set_id = import_csv(write(tmp_path, "pair,date,event,variance_days\n"
                                              f"EURUSD,{event_date},FOMC,3.0\n"), store)
    market(store, event_set_id)
    with_events = load_store_markets(TODAY, store=store).payloads["EURUSD"]
    without = dict(with_events, event_calendar=[])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vols = [MarketContext.from_payload(payload, PricingSettings(pricing_model="vv"))
                .market_for_expiry(pd.Timestamp("2026-10-28")).v_atm
                for payload in (with_events, without)]
    assert (vols[0] - vols[1]) * direction > 0.05


# -- migration from schema 2 (markets without event sets) --------------------------------

V2_SCHEMA = """
CREATE TABLE schema_info (version INTEGER NOT NULL);
INSERT INTO schema_info VALUES (2);
CREATE TABLE markets (market_id INTEGER PRIMARY KEY AUTOINCREMENT, market_source TEXT NOT NULL,
                      date_time TEXT NOT NULL, fetched_at TEXT NOT NULL,
                      note TEXT NOT NULL DEFAULT '');
CREATE INDEX markets_date_time ON markets (date_time);
CREATE TABLE quotes (market_id INTEGER NOT NULL REFERENCES markets (market_id),
                     kind TEXT NOT NULL, name TEXT NOT NULL, tenor TEXT NOT NULL DEFAULT '',
                     value REAL NOT NULL, PRIMARY KEY (market_id, kind, name, tenor));
INSERT INTO markets VALUES (1, 'bloomberg', '2026-10-16T15:09:34', '2026-10-16T15:09:34', '');
INSERT INTO quotes VALUES (1, 'spot', 'EURUSD', '', 1.13);
"""


def _v2_store(path):
    db = sqlite3.connect(str(path))
    db.executescript(V2_SCHEMA)
    db.close()
    return path


def test_a_schema_2_store_is_backed_up_and_migrated(tmp_path):
    store = MarketDataStore(_v2_store(tmp_path / "market_data.sqlite"))
    assert store.migrated_from is not None and store.migrated_from.name.startswith(
        "market_data.v2-backup-")
    market = store.load(market_id=1)
    assert market.spots == {"EURUSD": 1.13} and market.event_set_id is None
    db = sqlite3.connect(str(store.path))
    assert db.execute("SELECT MAX(version) FROM schema_info").fetchone()[0] == SCHEMA_VERSION
    db.close()
    assert MarketDataStore(store.path).migrated_from is None


def test_a_failed_schema_2_migration_rolls_back(tmp_path, monkeypatch):
    path = _v2_store(tmp_path / "market_data.sqlite")
    monkeypatch.setattr(store_module, "_EVENT_SCHEMA", "CREATE TABLE broken (;")
    with pytest.raises(MarketDataStoreError, match="could not migrate"):
        MarketDataStore(path)
    db = sqlite3.connect(str(path))
    columns = [row[1] for row in db.execute("PRAGMA table_info(markets)")]
    assert "event_set_id" not in columns
    assert db.execute("SELECT MAX(version) FROM schema_info").fetchone()[0] == 2
    db.close()


def test_named_events_count_like_the_summed_table_and_old_files_still_load():
    # Separate same-day rows give the same variance days as one summed row,
    # and a market file from before names existed loads with blank names.
    from datetime import date
    from app.market_context import MarketContext
    from app.market_store import validate_market
    from golden.app_harness import load_markets
    payload = load_markets("MARKETS_20261015.json")["EURUSD"]
    named = dict(payload, event_calendar=[
        {"event_date": "2026-11-06", "event": "NFP", "variance_days": 1.5},
        {"event_date": "2026-11-06", "event": "ECB speech", "variance_days": 0.25}])
    summed = dict(payload, event_calendar=[{"event_date": "2026-11-06", "variance_days": 1.75}])
    days = [MarketContext.from_payload(p).variance_days_between(date(2026, 10, 15),
                                                                date(2026, 12, 1))
            for p in (named, summed)]
    assert days[0] == pytest.approx(days[1], abs=1e-12)
    rows = validate_market(summed)["event_rows"]
    assert list(rows.columns) == ["event_date", "event", "variance_days"]
    assert list(rows["event"]) == [""]

"""Per-pair event calendars: append-only versions with their save date, the
version a market uses on a valuation date, the fallbacks (pinned event set,
then none) and the failsafes (bad rows store nothing, unreadable calendars
make the pair unavailable, the schema 3 store migrates with a backup)."""
import sqlite3

import pandas as pd
import pytest

from app.store_markets import load_store_markets
from market_data.events import import_csv
from market_data.store import SCHEMA_VERSION, MarketDataStore, MarketDataStoreError
from test_event_sets import GOOD_CSV, TODAY, market, write

NFP = [("2026-11-06", "US NFP", 1.5), ("2026-11-06", "ECB speech", 0.25)]


@pytest.fixture
def store(tmp_path):
    return MarketDataStore(tmp_path / "market_data.sqlite")


def test_each_save_is_a_new_version_and_earlier_ones_never_change(store):
    assert store.load_pair_events("EURUSD") is None
    assert store.save_pair_events("eur/usd", NFP, user="harel", note="first",
                                  saved_at="2026-10-14T09:00:00") == 1
    assert store.save_pair_events("EURUSD", [("2026-11-06", "US NFP", 2.0)], note="NFP up",
                                  saved_at="2026-10-16T15:30:00") == 2
    first, latest = store.load_pair_events("EURUSD", version=1), store.load_pair_events("EURUSD")
    assert first.rows == [("2026-11-06", "ECB speech", 0.25), ("2026-11-06", "US NFP", 1.5)]
    assert (first.user, first.note, first.saved_at) == ("harel", "first", "2026-10-14T09:00:00")
    assert latest.version == 2 and latest.rows == [("2026-11-06", "US NFP", 2.0)]
    versions = store.pair_event_versions("EURUSD")
    assert list(versions["version"]) == [2, 1] and list(versions["events"]) == [1, 2]
    assert store.pair_event_versions("USDJPY").empty


def test_the_version_on_a_date_is_the_latest_saved_by_then(store):
    store.save_pair_events("EURUSD", NFP, saved_at="2026-10-14T09:00:00")
    store.save_pair_events("EURUSD", [("2026-11-06", "US NFP", 2.0)],
                           saved_at="2026-10-16T15:30:00")
    assert store.load_pair_events("EURUSD", as_of="2026-10-13") is None
    assert store.load_pair_events("EURUSD", as_of="2026-10-15").version == 1
    assert store.load_pair_events("EURUSD", as_of="2026-10-16").version == 2   # same day
    assert store.load_pair_events("EURUSD", as_of="2026-12-01").version == 2


def test_an_empty_calendar_is_a_version_and_bad_rows_store_nothing(store):
    store.save_pair_events("EURUSD", NFP)
    assert store.save_pair_events("EURUSD", [], note="all removed") == 2
    assert store.load_pair_events("EURUSD").rows == []
    for rows, message in (([("2026-11-06", "", 1.0)], "event name is empty"),
                          ([("not a date", "NFP", 1.0)], "is not a date"),
                          ([("2026-11-06", "NFP", "lots")], "variance days"),
                          ([("2026-11-06", "NFP", 1.0), ("2026-11-06", "NFP", 2.0)],
                           "duplicate")):
        with pytest.raises(ValueError, match=message):
            store.save_pair_events("EURUSD", rows)
    with pytest.raises(ValueError, match="not two 3-letter"):
        store.save_pair_events("EUR", NFP)
    assert list(store.pair_event_versions("EURUSD")["version"]) == [2, 1]   # nothing added


def test_markets_use_their_own_calendar_then_the_pinned_set_then_none(store, tmp_path):
    event_set_id = import_csv(write(tmp_path, GOOD_CSV), store)
    market(store, event_set_id)
    store.save_pair_events("EURUSD", NFP, user="harel", saved_at=f"{TODAY}T08:00:00")
    loaded = load_store_markets(TODAY, store=store)
    eurusd, usdjpy = loaded.payloads["EURUSD"], loaded.payloads["USDJPY"]
    names = [row["event"] for row in eurusd["event_calendar"]]
    assert names == ["ECB speech", "US NFP"]                       # own calendar, not the set's
    assert eurusd["event_source"].startswith("EURUSD calendar v1, saved 2026-10-16T08:00")
    assert "by harel" in eurusd["event_source"]
    assert [row["event"] for row in usdjpy["event_calendar"]] == ["BOJ"]        # pinned set
    assert usdjpy["event_source"] == f"event set {event_set_id} (pinned to market 1)"
    assert "own event calendars: EURUSD v1" in loaded.notes
    # A calendar saved after the valuation date never reaches it.
    store.save_pair_events("EURUSD", [("2026-11-06", "US NFP", 9.0)],
                           saved_at="2026-10-20T08:00:00")
    again = load_store_markets(TODAY, store=store).payloads["EURUSD"]
    assert [row["variance_days"] for row in again["event_calendar"]] == [0.25, 1.5]


def test_an_empty_own_calendar_means_no_events_not_the_pinned_set(store, tmp_path):
    market(store, import_csv(write(tmp_path, GOOD_CSV), store))
    store.save_pair_events("USDJPY", [], note="BOJ dropped", saved_at=f"{TODAY}T08:00:00")
    usdjpy = load_store_markets(TODAY, store=store).payloads["USDJPY"]
    assert usdjpy["event_calendar"] == [] and usdjpy["event_source"].startswith("USDJPY calendar v1")


def test_an_unreadable_calendar_makes_only_its_pair_unavailable(store, tmp_path, monkeypatch):
    market(store)
    real = store.load_pair_events

    def flaky(pair, **kwargs):
        if pair == "USDJPY":
            raise sqlite3.DatabaseError("disk I/O error")
        return real(pair, **kwargs)
    monkeypatch.setattr(store, "load_pair_events", flaky)
    loaded = load_store_markets(TODAY, store=store)
    assert sorted(loaded.payloads) == ["EURUSD"]
    assert "its event calendar cannot be read" in loaded.unavailable["USDJPY"]


def test_an_unreadable_pinned_set_spares_pairs_with_their_own_calendar(store, tmp_path,
                                                                       monkeypatch):
    market(store, import_csv(write(tmp_path, GOOD_CSV), store))
    store.save_pair_events("EURUSD", NFP, saved_at=f"{TODAY}T08:00:00")

    def broken(event_set_id):
        raise sqlite3.DatabaseError("malformed")
    monkeypatch.setattr(store, "load_event_set", broken)
    loaded = load_store_markets(TODAY, store=store)
    assert sorted(loaded.payloads) == ["EURUSD"] and "cannot be read" in loaded.unavailable["USDJPY"]


def test_a_schema_3_store_migrates_with_a_backup_and_keeps_its_markets(tmp_path):
    path = tmp_path / "market_data.sqlite"
    old = MarketDataStore(path)
    market_id = market(old)
    db = sqlite3.connect(str(path))
    db.executescript("DROP TABLE pair_events; DROP TABLE pair_event_versions; "
                     "DELETE FROM schema_info; INSERT INTO schema_info VALUES (3);")
    db.commit()
    db.close()
    store = MarketDataStore(path)
    assert store.migrated_from is not None and store.migrated_from.exists()
    assert ".v3-backup-" in store.migrated_from.name
    assert store.load(market_id=market_id).spots["EURUSD"] == 1.12
    assert store.save_pair_events("EURUSD", NFP) == 1
    db = sqlite3.connect(str(path))
    assert db.execute("SELECT MAX(version) FROM schema_info").fetchone()[0] == SCHEMA_VERSION == 6
    db.execute("DELETE FROM schema_info")
    db.execute("INSERT INTO schema_info VALUES (?)", (SCHEMA_VERSION + 1,))
    db.commit()
    db.close()
    with pytest.raises(MarketDataStoreError, match="newer"):
        MarketDataStore(path)


# -- the Markets tab's calendar tools (app.event_calendar) ---------------------

def editor_table(rows):
    return pd.DataFrame(rows, columns=["event_date", "event", "variance_days"])


def test_the_table_saves_without_its_blank_line_and_a_part_filled_row_is_named(store):
    from app import event_calendar
    table = editor_table([(pd.Timestamp("2026-11-06"), "US NFP", 1.5),
                          (pd.NaT, None, None)])                        # the editor's blank row
    assert event_calendar.save(store, "EURUSD", table, user="harel", note="n") == (1, 1)
    part = editor_table([(pd.Timestamp("2026-11-06"), "US NFP", 1.5),
                         (pd.Timestamp("2026-11-07"), "", 0.5)])
    with pytest.raises(ValueError, match="row 2: event name is empty"):
        event_calendar.save(store, "EURUSD", part)
    assert list(store.pair_event_versions("EURUSD")["version"]) == [1]


def test_csv_round_trip_and_other_pairs_left_out():
    from app import event_calendar
    table = editor_table([(pd.Timestamp("2026-11-06"), "US NFP", 1.5),
                          (pd.Timestamp("2026-10-29"), "EZ CPI", 0.5)])
    text = event_calendar.csv_text("EURUSD", table)
    assert text.splitlines()[0] == "pair,date,event,variance_days"
    back, notes = event_calendar.rows_from_csv(text + "USDJPY,2026-10-30,BOJ,1.0\n"
                                               ",2026-11-20,ECB,0.75\n", "EURUSD")
    assert list(back["event"]) == ["EZ CPI", "US NFP", "ECB"]
    assert notes == ["1 row(s) for other pairs left out"]


def test_a_bad_csv_loads_nothing_and_names_every_problem():
    from app import event_calendar
    with pytest.raises(ValueError, match="missing column"):
        event_calendar.rows_from_csv("pair,date,weight\nEURUSD,2026-11-06,1\n", "EURUSD")
    bad = ("pair,date,event,variance_days\nEURUSD,2026-11-06,NFP,1\n"
           "EURUSD,someday,CPI,1\nEURUSD,2026-11-07,FOMC,lots\n")
    with pytest.raises(ValueError) as caught:
        event_calendar.rows_from_csv(bad, "EURUSD")
    assert "row 2:" in str(caught.value) and "row 3:" in str(caught.value)
    assert "row 1:" not in str(caught.value)

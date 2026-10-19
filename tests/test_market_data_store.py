"""Market data store: market_id / market_source / date_time, read-back, migration."""
import json
import sqlite3

import pytest

from market_data.store import (
    DEPO, SCHEMA_VERSION, SPOT, VOL_FIELDS, MarketDataStore, MarketDataStoreError,
    close_date_time,
    data_date_time,
)


@pytest.fixture
def store(tmp_path):
    return MarketDataStore(tmp_path / "market_data.sqlite")


def test_markets_have_a_serial_id_a_source_and_a_data_date_time(store):
    assert store.load() is None
    first = store.save("2026-10-15T23:59:59", "bloomberg", spots={"EURUSD": 1.12},
                       depo_curves={"USD": {"1W": 0.0389, "1M": 0.0393}},
                       fetched_at="2026-10-16T08:00:00")
    second = store.save("2026-10-16T09:00:00", "bloomberg", spots={"EURUSD": 1.13},
                        depo_curves={"USD": {"1W": 0.0390}})
    third = store.save("2026-10-16T11:00:00", "import", spots={"EURUSD": 1.14}, note="file.json")
    assert (first, second, third) == (1, 2, 3)

    latest = store.load()
    assert (latest.market_id, latest.market_source, latest.date_time, latest.date) == (
        3, "import", "2026-10-16T11:00:00", "2026-10-16")
    assert latest.note == "file.json" and latest.spots == {"EURUSD": 1.14}
    assert store.load(market_id=1).fetched_at == "2026-10-16T08:00:00"
    assert store.load(at_or_before="2026-10-16T10:00:00").market_id == second
    assert store.load(at_or_before="2026-10-16 08:00").market_id == first
    assert store.load(at_or_before="2026-10-15T12:00:00") is None
    assert store.load(date="2026-10-15").depo_curves == {"USD": {"1W": 0.0389, "1M": 0.0393}}
    assert store.load(date="2026-10-17") is None
    assert store.load(on_or_before="2026-10-17").market_id == third
    assert store.load(on_or_before="2026-10-15").market_id == first     # never later data
    assert store.load(market_id=99) is None

    listed = store.snapshots()
    assert list(listed["market_id"]) == [3, 2, 1]
    assert list(listed["market_source"]) == ["import", "bloomberg", "bloomberg"]
    assert list(listed["quotes"]) == [1, 2, 3]
    assert list(store.snapshots("2026-10-15")["market_id"]) == [1]

    assert list(store.history(SPOT, "EURUSD")["value"]) == [1.12, 1.13, 1.14]
    assert list(store.history(DEPO, "USD", "1W")["value"]) == [0.0389, 0.0390]
    assert list(store.history(SPOT, "EURUSD", start="2026-10-16")["value"]) == [1.13, 1.14]
    with pytest.raises(ValueError):
        store.save("2026-10-16T12:00:00", "", spots={"EURUSD": 1.1})


def test_data_date_time_is_the_fetch_time_today_and_the_close_for_a_past_date():
    assert data_date_time("2026-10-16", "2026-10-16T15:09:34.6") == "2026-10-16T15:09:34"
    # A past date's close is 15:00 New York, stored in London time.
    assert data_date_time("2026-10-15", "2026-10-16T15:09:34") == "2026-10-15T20:00:00"
    assert data_date_time("2026-10-27", "2026-10-28T09:00:00") == "2026-10-27T19:00:00"


@pytest.mark.parametrize("day, close", [
    ("2026-10-16", "2026-10-16T20:00:00"),     # both on summer time
    ("2026-10-27", "2026-10-27T19:00:00"),     # UK back on GMT, US still on EDT
    ("2026-11-02", "2026-11-02T20:00:00"),     # both on winter time
    ("2027-03-20", "2027-03-20T19:00:00"),     # US on EDT, UK still on GMT
    ("2027-03-29", "2027-03-29T20:00:00")])
def test_the_close_is_15_00_new_york_in_london_time(day, close):
    assert close_date_time(day) == close


def test_json_files_import_into_one_market(store, tmp_path):
    stamp = {"valuation_date": "2026-10-16", "as_of": "2026-10-16T11:21:13"}
    spots = tmp_path / "usd_spots.json"
    spots.write_text(json.dumps({"format": "usd_spots", **stamp, "spots": {"EURUSD": 1.12}}))
    depos = tmp_path / "depo_curves.json"
    depos.write_text(json.dumps({"format": "depo_curves", **stamp,
                                 "curves": {"USD": {"1W": 0.0389}}}))
    assert store.import_json(spots) == store.import_json(depos)
    market = store.load()
    assert (market.market_source, market.date_time, market.note) == (
        "import", "2026-10-16T11:21:13", "usd_spots.json")
    assert market.spots == {"EURUSD": 1.12} and market.depo_curves == {"USD": {"1W": 0.0389}}
    bad = tmp_path / "other.json"
    bad.write_text(json.dumps({"format": "market", **stamp}))
    with pytest.raises(ValueError):
        store.import_json(bad)


def test_app_reads_stored_market_data_without_loading_bloomberg():
    # The app prices from the store; only market_data.bloomberg_fetch talks to Bloomberg.
    import subprocess
    import sys
    code = ("import sys, app.ui.app, app.ui.trades_block, app.usd_spots, market_data.store; "
            "print(sorted(n for n in sys.modules if n.split('.')[0] in "
            "('xbbg', 'blpapi') or n == 'market_data.bloomberg_fetch'))")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            timeout=300)
    assert result.stdout.strip().splitlines()[-1] == "[]", result.stderr[-2000:]


def test_vol_surfaces_round_trip(store):
    surface = {"1W": dict(zip(VOL_FIELDS, (6.4, -1.3, 0.21, -2.4, 0.7))),
               "1M": dict(zip(VOL_FIELDS, (6.1, -1.2, 0.20, -2.3, 0.69)))}
    store.save("2026-10-16T12:00:00", "bloomberg", spots={"EURUSD": 1.12},
               vols={"EURUSD": surface})
    assert store.load().vols == {"EURUSD": surface}
    assert list(store.history("vol_atm", "EURUSD", "1M")["value"]) == [6.1]
    assert store.snapshots()["quotes"].iloc[0] == 11
    with pytest.raises(ValueError):
        store.save("2026-10-16T12:00:00", "bloomberg", vols={"EURUSD": {"1M": {"atm_vol": 6.1}}})


def test_forward_points_round_trip(store):
    store.save("2026-10-16T12:00:00", "bloomberg",
               forward_points={"EURUSD": {"1M": 14.18, "1Y": 178.8}, "USDJPY": {"1M": -38.6}})
    assert store.load().forward_points == {"EURUSD": {"1M": 14.18, "1Y": 178.8},
                                           "USDJPY": {"1M": -38.6}}
    assert list(store.history("fwd_pts", "EURUSD", "1Y")["value"]) == [178.8]


# -- migration from the first schema (snapshots: id, as_of, valuation_date, source) ---------

V1_SCHEMA = """
CREATE TABLE snapshots (id INTEGER PRIMARY KEY AUTOINCREMENT, as_of TEXT NOT NULL,
                        valuation_date TEXT NOT NULL, source TEXT NOT NULL DEFAULT '');
CREATE INDEX snapshots_as_of ON snapshots (as_of);
CREATE INDEX snapshots_valuation ON snapshots (valuation_date, as_of);
CREATE TABLE quotes (snapshot_id INTEGER NOT NULL REFERENCES snapshots (id), kind TEXT NOT NULL,
                     name TEXT NOT NULL, tenor TEXT NOT NULL DEFAULT '', value REAL NOT NULL,
                     PRIMARY KEY (snapshot_id, kind, name, tenor));
CREATE INDEX quotes_series ON quotes (kind, name, tenor);
INSERT INTO snapshots VALUES (1, '2026-10-16T11:21:13', '2026-10-16', 'import:depo.json');
INSERT INTO snapshots VALUES (2, '2026-10-16T12:00:00', '2026-10-15', 'bloomberg');
INSERT INTO snapshots VALUES (3, '2026-10-16T15:09:34', '2026-10-16', 'bloomberg');
INSERT INTO quotes VALUES (1, 'depo', 'USD', '1W', 0.0389);
INSERT INTO quotes VALUES (2, 'spot', 'EURUSD', '', 1.12);
INSERT INTO quotes VALUES (3, 'spot', 'EURUSD', '', 1.13);
INSERT INTO quotes VALUES (3, 'vol_atm', 'EURUSD', '1M', 6.13);
"""


def _v1_store(path):
    db = sqlite3.connect(str(path))
    db.executescript(V1_SCHEMA)
    db.close()
    return path


def test_a_first_schema_store_is_backed_up_and_migrated_keeping_every_id(tmp_path):
    path = _v1_store(tmp_path / "market_data.sqlite")
    store = MarketDataStore(path)
    assert store.migrated_from is not None and store.migrated_from.exists()
    listed = store.snapshots().set_index("market_id")
    assert list(listed.index) == [3, 1, 2]                      # newest data first
    assert listed.loc[1, "market_source"] == "import" and listed.loc[1, "note"] == "depo.json"
    assert listed.loc[2, "date_time"] == "2026-10-15T23:59:59"   # fetched later for a past date
    assert listed.loc[2, "fetched_at"] == "2026-10-16T12:00:00"
    assert listed.loc[3, "date_time"] == "2026-10-16T15:09:34"
    assert store.load(market_id=3).vols == {"EURUSD": {"1M": {"atm": 6.13}}}
    assert int(listed["quotes"].sum()) == 4
    assert MarketDataStore(path).migrated_from is None           # migrates once
    assert store.save("2026-10-16T16:00:00", "bloomberg", spots={"EURUSD": 1.1}) == 4


def test_a_failed_migration_rolls_back_and_leaves_the_file_unchanged(tmp_path):
    path = _v1_store(tmp_path / "market_data.sqlite")
    db = sqlite3.connect(str(path))
    # A quote whose market does not exist breaks the copy's foreign key.
    db.execute("INSERT INTO quotes VALUES (42, 'spot', 'USDJPY', '', 150.0)")
    db.commit()
    db.close()
    with pytest.raises(MarketDataStoreError, match="could not migrate"):
        MarketDataStore(path)
    db = sqlite3.connect(str(path))
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"snapshots", "quotes"} <= tables and "markets" not in tables
    assert db.execute("SELECT COUNT(*) FROM quotes").fetchone()[0] == 5
    db.close()


def test_a_store_from_a_newer_schema_is_refused(tmp_path):
    path = tmp_path / "market_data.sqlite"
    MarketDataStore(path)
    db = sqlite3.connect(str(path))
    db.execute("UPDATE schema_info SET version = ?", (SCHEMA_VERSION + 1,))
    db.commit()
    db.close()
    with pytest.raises(MarketDataStoreError, match="newer than this code"):
        MarketDataStore(path)

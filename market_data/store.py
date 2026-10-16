"""Market data snapshots: every quote fetched, by date-time, in one SQLite file.

Each fetch (``market_data.bloomberg_fetch``) saves one market snapshot:

* ``market_id``: a running serial number (1, 2, 3 ...);
* ``market_source``: where the data came from (``bloomberg``, ``import`` ...);
* ``date_time``: when the data is from. For a live fetch, the fetch time;
  for a fetch of a past date, that date's close (23:59:59). A market can
  value any date on or after its ``date_time``'s date, never an earlier one;
* ``fetched_at``: when it was pulled (audit trail).

and its quotes, one row ``(kind, name, tenor, value)`` each:

* ``spot``: ``name`` is the pair (``EURUSD``), ``tenor`` is empty;
* ``depo``: ``name`` is the currency (``USD``), ``tenor`` is ``1W`` ... ``1Y``,
  ``value`` a decimal rate;
* ``vol_atm``, ``vol_rr_25``, ``vol_rr_10``, ``vol_bf_25``, ``vol_bf_10``:
  ``name`` is the pair, ``tenor`` is ``ON`` ... ``1Y``, ``value`` in vol
  percentage points;
* ``fwd_pts``: ``name`` is the pair, ``tenor`` is ``1W`` ... ``1Y``, ``value``
  in the pair's pips (forward = spot + points / forward divisor).

New kinds can be stored without a schema change. Markets are only ever
added, so any past state can be read back.

Failsafes: a store written before this schema (version 1: ``snapshots``
with ``id``, ``as_of``, ``valuation_date``, ``source``) is copied to a
timestamped backup and migrated in one transaction, all or nothing; a store
written by a newer schema is refused rather than misread.
"""
import json
import shutil
import sqlite3
from collections import namedtuple
from contextlib import closing
from datetime import datetime
from pathlib import Path

import pandas as pd

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "local" / "market_data.sqlite"
SCHEMA_VERSION = 2
SPOT, DEPO, FWD_PTS = "spot", "depo", "fwd_pts"
VOL_FIELDS = ("atm", "rr_25", "bf_25", "rr_10", "bf_10")
VOL_PREFIX = "vol_"
#: ``date_time`` of a fetch for a past date: that date's close.
CLOSE = "23:59:59"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_info (version INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS markets (
    market_id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_source TEXT NOT NULL,
    date_time TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS markets_date_time ON markets (date_time);
CREATE TABLE IF NOT EXISTS quotes (
    market_id INTEGER NOT NULL REFERENCES markets (market_id),
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    tenor TEXT NOT NULL DEFAULT '',
    value REAL NOT NULL,
    PRIMARY KEY (market_id, kind, name, tenor)
);
CREATE INDEX IF NOT EXISTS quotes_series ON quotes (kind, name, tenor);
"""

_SNAPSHOT_FIELDS = ("market_id market_source date_time fetched_at note "
                    "spots depo_curves vols forward_points")


class Snapshot(namedtuple("Snapshot", _SNAPSHOT_FIELDS)):
    """One stored market: its record and its quotes by kind."""
    __slots__ = ()

    @property
    def date(self):
        """The date the data is from (``date_time``'s date), ``YYYY-MM-DD``."""
        return self.date_time[:10]


Snapshot.__new__.__defaults__ = ("", {}, {}, {}, {})


class MarketDataStoreError(RuntimeError):
    """The store cannot be used (newer schema, failed migration)."""


def _iso(value):
    """``YYYY-MM-DDTHH:MM:SS``, so stored date-times sort and compare as text."""
    return pd.Timestamp(value).to_pydatetime().replace(microsecond=0, tzinfo=None).isoformat()


def _day(value):
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def data_date_time(valuation_date, now=None):
    """The ``date_time`` of data fetched for ``valuation_date``: now when it is
    today, else that date's close."""
    now = pd.Timestamp(now or datetime.now())
    if _day(valuation_date) == _day(now):
        return _iso(now)
    return f"{_day(valuation_date)}T{CLOSE}"


def _rows(spots=None, depo_curves=None, vols=None, forward_points=None):
    """Quote rows ``(kind, name, tenor, value)`` for one market."""
    rows = [(SPOT, pair, "", float(value)) for pair, value in (spots or {}).items()]
    rows += [(DEPO, ccy, tenor, float(rate))
             for ccy, curve in (depo_curves or {}).items() for tenor, rate in curve.items()]
    for pair, by_tenor in (vols or {}).items():
        for tenor, quotes in by_tenor.items():
            for field, value in quotes.items():
                if field not in VOL_FIELDS:
                    raise ValueError(f"unknown vol field {field!r} for {pair} {tenor}")
                rows.append((VOL_PREFIX + field, pair, tenor, float(value)))
    rows += [(FWD_PTS, pair, tenor, float(points))
             for pair, by_tenor in (forward_points or {}).items()
             for tenor, points in by_tenor.items()]
    return rows


class MarketDataStore:
    def __init__(self, path=DEFAULT_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.migrated_from = None
        with closing(self._connect()) as db:
            tables = {row[0] for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'")}
            if "snapshots" in tables and "markets" not in tables:
                self._migrate_v1(db)
            with db:
                db.executescript(_SCHEMA)
                version = db.execute("SELECT MAX(version) FROM schema_info").fetchone()[0]
                if version is None:
                    db.execute("INSERT INTO schema_info VALUES (?)", (SCHEMA_VERSION,))
                elif version > SCHEMA_VERSION:
                    raise MarketDataStoreError(
                        f"market data store {self.path} has schema version {version}, newer "
                        f"than this code ({SCHEMA_VERSION}); update the code")

    def _connect(self):
        db = sqlite3.connect(str(self.path))
        db.execute("PRAGMA foreign_keys = ON")
        return db

    def _migrate_v1(self, db):
        """``snapshots`` (id, as_of, valuation_date, source) -> ``markets``.

        The file is copied to a backup first; the migration is one explicit
        transaction, rolled back entirely on any error.
        """
        backup = self.path.with_name(
            f"{self.path.stem}.v1-backup-{datetime.now():%Y%m%d_%H%M%S}{self.path.suffix}")
        shutil.copy2(str(self.path), str(backup))
        db.isolation_level = None                      # explicit BEGIN / COMMIT below
        try:
            db.execute("BEGIN")
            db.execute("DROP INDEX IF EXISTS quotes_series")
            db.execute("DROP INDEX IF EXISTS snapshots_as_of")
            db.execute("DROP INDEX IF EXISTS snapshots_valuation")
            db.execute("ALTER TABLE quotes RENAME TO v1_quotes")
            db.execute("ALTER TABLE snapshots RENAME TO v1_snapshots")
            for statement in filter(str.strip, _SCHEMA.split(";")):
                db.execute(statement)
            db.execute("""
                INSERT INTO markets (market_id, market_source, date_time, fetched_at, note)
                SELECT id,
                       CASE WHEN source LIKE 'import:%' THEN 'import'
                            WHEN source = '' THEN 'unknown' ELSE source END,
                       CASE WHEN substr(as_of, 1, 10) = valuation_date THEN as_of
                            ELSE valuation_date || 'T' || ? END,
                       as_of,
                       CASE WHEN source LIKE 'import:%' THEN substr(source, 8) ELSE '' END
                FROM v1_snapshots""", (CLOSE,))
            db.execute("INSERT INTO quotes (market_id, kind, name, tenor, value) "
                       "SELECT snapshot_id, kind, name, tenor, value FROM v1_quotes")
            moved = db.execute("SELECT (SELECT COUNT(*) FROM quotes), "
                               "(SELECT COUNT(*) FROM v1_quotes)").fetchone()
            if moved[0] != moved[1]:
                raise MarketDataStoreError(f"migration copied {moved[0]} of {moved[1]} quotes")
            db.execute("DROP TABLE v1_quotes")
            db.execute("DROP TABLE v1_snapshots")
            db.execute("INSERT INTO schema_info VALUES (?)", (SCHEMA_VERSION,))
            db.execute("COMMIT")
        except Exception as exc:
            db.execute("ROLLBACK")
            raise MarketDataStoreError(
                f"could not migrate {self.path} (unchanged; backup at {backup}): {exc}") from exc
        finally:
            db.isolation_level = ""
        self.migrated_from = backup

    def save(self, date_time, market_source, spots=None, depo_curves=None, vols=None,
             forward_points=None, fetched_at=None, note=""):
        """Store one market in a single transaction and return its ``market_id``.

        ``vols`` is ``{pair: {tenor: {field: value}}}`` with fields from
        ``VOL_FIELDS``; ``forward_points`` is ``{pair: {tenor: points}}``.
        """
        if not market_source:
            raise ValueError("market_source is required")
        rows = _rows(spots, depo_curves, vols, forward_points)
        with closing(self._connect()) as db, db:
            cursor = db.execute(
                "INSERT INTO markets (market_source, date_time, fetched_at, note) "
                "VALUES (?, ?, ?, ?)",
                (market_source, _iso(date_time), _iso(fetched_at or datetime.now()), note))
            market_id = cursor.lastrowid
            db.executemany(
                "INSERT INTO quotes (market_id, kind, name, tenor, value) VALUES (?, ?, ?, ?, ?)",
                [(market_id, *row) for row in rows])
        return market_id

    def snapshots(self, date=None):
        """Markets, newest first: ``market_id, market_source, date_time,
        fetched_at, note, quotes``; ``date`` limits them to data of that date."""
        query = ("SELECT m.market_id, m.market_source, m.date_time, m.fetched_at, m.note, "
                 "COUNT(q.value) AS quotes FROM markets m "
                 "LEFT JOIN quotes q ON q.market_id = m.market_id")
        args = ()
        if date is not None:
            query += " WHERE substr(m.date_time, 1, 10) = ?"
            args = (_day(date),)
        query += " GROUP BY m.market_id ORDER BY m.date_time DESC, m.market_id DESC"
        with closing(self._connect()) as db:
            return pd.read_sql_query(query, db, params=args)

    def load(self, market_id=None, date=None, on_or_before=None, at_or_before=None):
        """One market with its quotes, or ``None``.

        ``market_id``: that market. Otherwise the latest market (by
        ``date_time``) with data of ``date``, or of ``on_or_before`` or
        earlier (a date: any time that day), or taken ``at_or_before`` a
        date-time; with none of them, the latest market.
        """
        query, args = ("SELECT market_id, market_source, date_time, fetched_at, note "
                       "FROM markets WHERE 1 = 1"), []
        if market_id is not None:
            query += " AND market_id = ?"
            args.append(int(market_id))
        if date is not None:
            query += " AND substr(date_time, 1, 10) = ?"
            args.append(_day(date))
        if on_or_before is not None:
            query += " AND substr(date_time, 1, 10) <= ?"
            args.append(_day(on_or_before))
        if at_or_before is not None:
            query += " AND date_time <= ?"
            args.append(_iso(at_or_before))
        query += " ORDER BY date_time DESC, market_id DESC LIMIT 1"
        with closing(self._connect()) as db:
            head = db.execute(query, args).fetchone()
            if head is None:
                return None
            quotes = db.execute("SELECT kind, name, tenor, value FROM quotes "
                                "WHERE market_id = ? ORDER BY rowid", (head[0],)).fetchall()
        spots = {name: value for kind, name, _, value in quotes if kind == SPOT}
        curves, vols, forwards = {}, {}, {}
        for kind, name, tenor, value in quotes:
            if kind == DEPO:
                curves.setdefault(name, {})[tenor] = value
            elif kind == FWD_PTS:
                forwards.setdefault(name, {})[tenor] = value
            elif kind.startswith(VOL_PREFIX):
                vols.setdefault(name, {}).setdefault(tenor, {})[kind[len(VOL_PREFIX):]] = value
        return Snapshot(*head, spots, curves, vols, forwards)

    def earlier(self, snapshot, since):
        """Markets with data from ``since`` up to (not including) ``snapshot``'s
        date, fetched no later than it; newest first."""
        with closing(self._connect()) as db:
            ids = [row[0] for row in db.execute(
                "SELECT market_id FROM markets WHERE substr(date_time, 1, 10) >= ? "
                "AND substr(date_time, 1, 10) < ? AND fetched_at <= ? "
                "ORDER BY date_time DESC, market_id DESC",
                (_day(since), snapshot.date, snapshot.fetched_at))]
        return [self.load(market_id=market_id) for market_id in ids]

    def previous(self, snapshot):
        """The market just before ``snapshot`` by ``date_time``, or ``None``."""
        with closing(self._connect()) as db:
            row = db.execute(
                "SELECT market_id FROM markets WHERE date_time < ? "
                "OR (date_time = ? AND market_id < ?) ORDER BY date_time DESC, market_id DESC "
                "LIMIT 1", (snapshot.date_time, snapshot.date_time, snapshot.market_id)).fetchone()
        return None if row is None else self.load(market_id=row[0])

    def history(self, kind, name, tenor="", start=None, end=None):
        """One quote over time: ``market_id, date_time, value``, oldest first."""
        query = ("SELECT m.market_id, m.date_time, q.value FROM quotes q "
                 "JOIN markets m ON m.market_id = q.market_id "
                 "WHERE q.kind = ? AND q.name = ? AND q.tenor = ?")
        args = [kind, name, tenor]
        if start is not None:
            query += " AND m.date_time >= ?"
            args.append(_iso(start))
        if end is not None:
            query += " AND m.date_time <= ?"
            args.append(_iso(end))
        query += " ORDER BY m.date_time, m.market_id"
        with closing(self._connect()) as db:
            return pd.read_sql_query(query, db, params=args)

    def import_json(self, path):
        """Store a ``usd_spots`` or ``depo_curves`` JSON file written before this store.

        A file whose ``as_of`` is already stored as an import is merged into
        that market instead of adding another. Returns the ``market_id``.
        """
        payload = json.loads(Path(path).read_text())
        date_time = _iso(payload["as_of"])
        if _day(date_time) != _day(payload["valuation_date"]):
            date_time = f"{_day(payload['valuation_date'])}T{CLOSE}"
        spots = payload.get("spots") if payload.get("format") == "usd_spots" else None
        curves = payload.get("curves") if payload.get("format") == "depo_curves" else None
        if spots is None and curves is None:
            raise ValueError(f"{path} is not a usd_spots or depo_curves file")
        with closing(self._connect()) as db:
            found = db.execute("SELECT market_id FROM markets WHERE date_time = ? "
                               "AND market_source = 'import'", (date_time,)).fetchone()
        if found is None:
            return self.save(date_time, "import", spots, curves, fetched_at=payload["as_of"],
                             note=Path(path).name)
        rows = _rows(spots, curves)
        with closing(self._connect()) as db, db:
            db.executemany("INSERT OR REPLACE INTO quotes (market_id, kind, name, tenor, value) "
                           "VALUES (?, ?, ?, ?, ?)", [(found[0], *row) for row in rows])
        return found[0]


def open_market_data_store(path=None):
    """The store at ``path``, ``$VV_MARKET_STORE`` or the default location."""
    import os
    return MarketDataStore(path or os.environ.get("VV_MARKET_STORE", str(DEFAULT_PATH)))

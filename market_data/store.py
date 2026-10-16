"""Market data snapshots: every quote fetched, by datetime, in one SQLite file.

Each fetch (``market_data.bloomberg_fetch``) saves one snapshot: when it was taken
(``as_of``), the valuation date it is for, and its quotes. A quote is one
row ``(kind, name, tenor, value)``:

* ``spot``: ``name`` is the pair (``EURUSD``), ``tenor`` is empty;
* ``depo``: ``name`` is the currency (``USD``), ``tenor`` is ``1W`` ... ``1Y``,
  ``value`` a decimal rate.

New kinds can be stored without a schema change. Snapshots are only ever
added, so any past state can be read back: the latest snapshot taken at or
before a datetime, the latest for a valuation date, or one quote's history.
"""
import json
import sqlite3
from collections import namedtuple
from contextlib import closing
from datetime import datetime
from pathlib import Path

import pandas as pd

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "local" / "market_data.sqlite"
SPOT, DEPO = "spot", "depo"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    as_of TEXT NOT NULL,
    valuation_date TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS snapshots_as_of ON snapshots (as_of);
CREATE INDEX IF NOT EXISTS snapshots_valuation ON snapshots (valuation_date, as_of);
CREATE TABLE IF NOT EXISTS quotes (
    snapshot_id INTEGER NOT NULL REFERENCES snapshots (id),
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    tenor TEXT NOT NULL DEFAULT '',
    value REAL NOT NULL,
    PRIMARY KEY (snapshot_id, kind, name, tenor)
);
CREATE INDEX IF NOT EXISTS quotes_series ON quotes (kind, name, tenor);
"""

Snapshot = namedtuple("Snapshot", "id as_of valuation_date source spots depo_curves")


def _iso(value):
    """``YYYY-MM-DDTHH:MM:SS``, so stored datetimes sort and compare as text."""
    return pd.Timestamp(value).to_pydatetime().replace(microsecond=0, tzinfo=None).isoformat()


def _day(value):
    return pd.Timestamp(value).strftime("%Y-%m-%d")


class MarketDataStore:
    def __init__(self, path=DEFAULT_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            db.executescript(_SCHEMA)

    def _connect(self):
        db = sqlite3.connect(str(self.path))
        db.execute("PRAGMA foreign_keys = ON")
        return db

    def save(self, valuation_date, as_of=None, spots=None, depo_curves=None, source=""):
        """Store one snapshot in a single transaction and return its id."""
        rows = [(SPOT, pair, "", float(value)) for pair, value in (spots or {}).items()]
        rows += [(DEPO, ccy, tenor, float(rate))
                 for ccy, curve in (depo_curves or {}).items() for tenor, rate in curve.items()]
        with closing(self._connect()) as db, db:
            cursor = db.execute(
                "INSERT INTO snapshots (as_of, valuation_date, source) VALUES (?, ?, ?)",
                (_iso(as_of or datetime.now()), _day(valuation_date), source))
            snapshot_id = cursor.lastrowid
            db.executemany(
                "INSERT INTO quotes (snapshot_id, kind, name, tenor, value) VALUES (?, ?, ?, ?, ?)",
                [(snapshot_id, *row) for row in rows])
        return snapshot_id

    def snapshots(self, valuation_date=None):
        """Snapshots, newest first: ``id, as_of, valuation_date, source, quotes``."""
        query = ("SELECT s.id, s.as_of, s.valuation_date, s.source, COUNT(q.value) AS quotes "
                 "FROM snapshots s LEFT JOIN quotes q ON q.snapshot_id = s.id")
        args = ()
        if valuation_date is not None:
            query += " WHERE s.valuation_date = ?"
            args = (_day(valuation_date),)
        query += " GROUP BY s.id ORDER BY s.as_of DESC, s.id DESC"
        with closing(self._connect()) as db:
            return pd.read_sql_query(query, db, params=args)

    def load(self, as_of=None, valuation_date=None, on_or_before=False):
        """The latest snapshot taken at or before ``as_of`` (default: now), or ``None``.

        ``valuation_date`` limits it to snapshots for that date, or with
        ``on_or_before`` to that date or earlier.
        """
        query, args = "SELECT id, as_of, valuation_date, source FROM snapshots WHERE 1 = 1", []
        if as_of is not None:
            query += " AND as_of <= ?"
            args.append(_iso(as_of))
        if valuation_date is not None:
            query += " AND valuation_date " + ("<= ?" if on_or_before else "= ?")
            args.append(_day(valuation_date))
        query += " ORDER BY valuation_date DESC, as_of DESC, id DESC LIMIT 1" if on_or_before \
            else " ORDER BY as_of DESC, id DESC LIMIT 1"
        with closing(self._connect()) as db:
            head = db.execute(query, args).fetchone()
            if head is None:
                return None
            quotes = db.execute("SELECT kind, name, tenor, value FROM quotes "
                                "WHERE snapshot_id = ? ORDER BY rowid", (head[0],)).fetchall()
        spots = {name: value for kind, name, _, value in quotes if kind == SPOT}
        curves = {}
        for kind, name, tenor, value in quotes:
            if kind == DEPO:
                curves.setdefault(name, {})[tenor] = value
        return Snapshot(*head, spots, curves)

    def history(self, kind, name, tenor="", start=None, end=None):
        """One quote over time: ``as_of, valuation_date, value``, oldest first."""
        query = ("SELECT s.as_of, s.valuation_date, q.value FROM quotes q "
                 "JOIN snapshots s ON s.id = q.snapshot_id "
                 "WHERE q.kind = ? AND q.name = ? AND q.tenor = ?")
        args = [kind, name, tenor]
        if start is not None:
            query += " AND s.as_of >= ?"
            args.append(_iso(start))
        if end is not None:
            query += " AND s.as_of <= ?"
            args.append(_iso(end))
        query += " ORDER BY s.as_of, s.id"
        with closing(self._connect()) as db:
            return pd.read_sql_query(query, db, params=args)

    def import_json(self, path):
        """Store a ``usd_spots`` or ``depo_curves`` JSON file written before this store.

        A file whose ``as_of`` is already stored is merged into that snapshot
        instead of adding another. Returns the snapshot id.
        """
        payload = json.loads(Path(path).read_text())
        as_of, valuation = _iso(payload["as_of"]), _day(payload["valuation_date"])
        spots = payload.get("spots") if payload.get("format") == "usd_spots" else None
        curves = payload.get("curves") if payload.get("format") == "depo_curves" else None
        if spots is None and curves is None:
            raise ValueError(f"{path} is not a usd_spots or depo_curves file")
        with closing(self._connect()) as db:
            found = db.execute("SELECT id FROM snapshots WHERE as_of = ? AND valuation_date = ?",
                               (as_of, valuation)).fetchone()
        if found is None:
            return self.save(valuation, as_of, spots, curves, source=f"import:{Path(path).name}")
        rows = [(SPOT, pair, "", float(v)) for pair, v in (spots or {}).items()]
        rows += [(DEPO, ccy, tenor, float(r))
                 for ccy, curve in (curves or {}).items() for tenor, r in curve.items()]
        with closing(self._connect()) as db, db:
            db.executemany("INSERT OR REPLACE INTO quotes (snapshot_id, kind, name, tenor, value) "
                           "VALUES (?, ?, ?, ?, ?)", [(found[0], *row) for row in rows])
        return found[0]


def open_market_data_store(path=None):
    """The store at ``path``, ``$VV_MARKET_STORE`` or the default location."""
    import os
    return MarketDataStore(path or os.environ.get("VV_MARKET_STORE", str(DEFAULT_PATH)))

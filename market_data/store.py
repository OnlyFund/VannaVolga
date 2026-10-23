"""Market data snapshots: every quote fetched, by date-time, in one SQLite file.

Each fetch (``market_data.bloomberg_fetch``) saves one market snapshot:

* ``market_id``: a running serial number (1, 2, 3 ...);
* ``market_source``: where the data came from (``bloomberg``, ``import`` ...);
* ``date_time``: when the data is from, in the fetch machine's local time;
* ``timezone``: that machine's time zone (``STORE_TIMEZONE``, an IANA name;
  schema 5 - markets stored before it are in ``Europe/London``). The FX date
  of a snapshot follows from both (17:00 New York rollover,
  ``apeiron.gaia.market_dates.rollover``). For a live fetch, the fetch time; for a fetch of a
  past date, that date's close, 15:00 New York (``close_date_time``; stores
  from before this used 23:59:59). A market can value any date on or after
  its ``date_time``'s date, never an earlier one;
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

Event calendars are kept apart from the quotes, as versioned **event sets**
(``event_set_id``, ``source``, ``date_time``) of per-pair rows
``(pair, event_date, event, variance_days)``; several events on one date
are separate rows whose variance days add up. A market is **pinned** to one
event set (the latest at its fetch), so it always prices with the same
events; ``pin_events`` re-pins it explicitly and every pin is recorded.

Each currency pair also keeps its own **event calendar versions**
(``pair_event_versions``: ``pair``, ``version``, ``saved_at``, ``user``,
``note``; rows ``pair_events``): every save is a new version, never an
overwrite, so changes in event weights can be followed over time. A market
on a valuation date uses the pair's latest version saved on or before that
date (``load_pair_events(as_of=...)``); a pair without one falls back to the
market's pinned event set. Rows carry a ``currency`` (empty for now) so the
calendars can later move to per-currency events without a redesign.

**Pillar dates** (schema 6): each live fetch of the day's market dates
(``market_data.live_dates``: Bloomberg's expiry and delivery for each pair and
tenor) is kept as a ``date_fetches`` record (``fx_date``, ``fetched_at``,
``timezone``, ``source``, and ``refused`` with the reason when none may be
used), its ``pillar_dates`` (``pair``, ``tenor``, ``expiry``, ``delivery``,
``source``, and the solver's ``solver_expiry`` / ``solver_delivery`` beside
them, so disputes can be found later, and since schema 7 Bloomberg's vol
ticker expiry ``quoted_expiry``, a cross-check) and its ``pillar_date_fallbacks``
(what fell back to the solver, and why). Fetches are only ever added;
``load_pillar_dates`` reads the latest usable one for an FX date.

Failsafes: a store written by an older schema (1: ``snapshots``; 2: markets
without event sets; 3: without per-pair calendars; 4: without time zones;
5: without pillar dates; 6: without ``quoted_expiry``, whose fetches took month
expiries from the vol tickers and are marked refused, superseded) is copied to a
timestamped backup and migrated in one transaction, all or nothing; a store written by a newer schema is refused
rather than misread; event rows are validated as a whole and a set with any
bad row is not stored.
"""
import json
import os
import re
import shutil
import sqlite3
from collections import namedtuple
from contextlib import closing
from datetime import datetime
from pathlib import Path

import pandas as pd

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "local" / "market_data.sqlite"
SCHEMA_VERSION = 7
#: Why schema 6's fetches are not used (their month expiries were the vol tickers').
SUPERSEDED = ("superseded: month expiries were OPT_EXPIRE_DT, not derived from the "
              "forward date (schema 7, 2026-10-23)")
SPOT, DEPO, FWD_PTS = "spot", "depo", "fwd_pts"
VOL_FIELDS = ("atm", "rr_25", "bf_25", "rr_10", "bf_10")
VOL_PREFIX = "vol_"
#: Stored date-times are naive, in the fetch machine's local time zone, which
#: each market records (``VV_TIMEZONE``, default ``Europe/London``).
STORE_TIMEZONE = os.environ.get("VV_TIMEZONE", "Europe/London")
#: The zone of markets stored before schema 5 (the fetch machine's).
LEGACY_TIMEZONE = "Europe/London"
if not re.fullmatch(r"[A-Za-z_]+(/[A-Za-z_+-]+)*", STORE_TIMEZONE):
    raise ValueError(f"VV_TIMEZONE {STORE_TIMEZONE!r} is not a time zone name")
#: The close of business: 15:00 New York.
CLOSE_TIMEZONE, CLOSE_TIME = "America/New_York", "15:00:00"
#: The close stamped by older code (and still by the schema-1 migration).
CLOSE = LEGACY_CLOSE = "23:59:59"

_EVENT_SCHEMA = """
CREATE TABLE IF NOT EXISTS event_sets (
    event_set_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    date_time TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS events (
    event_set_id INTEGER NOT NULL REFERENCES event_sets (event_set_id),
    pair TEXT NOT NULL,
    event_date TEXT NOT NULL,
    event TEXT NOT NULL,
    variance_days REAL NOT NULL,
    PRIMARY KEY (event_set_id, pair, event_date, event)
);
CREATE TABLE IF NOT EXISTS event_pins (
    market_id INTEGER NOT NULL REFERENCES markets (market_id),
    event_set_id INTEGER NOT NULL REFERENCES event_sets (event_set_id),
    pinned_at TEXT NOT NULL,
    reason TEXT NOT NULL DEFAULT ''
);
"""

_PAIR_EVENT_SCHEMA = """
CREATE TABLE IF NOT EXISTS pair_event_versions (
    pair TEXT NOT NULL,
    version INTEGER NOT NULL,
    saved_at TEXT NOT NULL,
    user TEXT NOT NULL DEFAULT '',
    note TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (pair, version)
);
CREATE TABLE IF NOT EXISTS pair_events (
    pair TEXT NOT NULL,
    version INTEGER NOT NULL,
    event_date TEXT NOT NULL,
    event TEXT NOT NULL,
    variance_days REAL NOT NULL,
    currency TEXT,
    PRIMARY KEY (pair, version, event_date, event),
    FOREIGN KEY (pair, version) REFERENCES pair_event_versions (pair, version)
);
"""

_PILLAR_DATE_SCHEMA = """
CREATE TABLE IF NOT EXISTS date_fetches (
    fetch_id INTEGER PRIMARY KEY AUTOINCREMENT,
    fx_date TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    timezone TEXT NOT NULL,
    source TEXT NOT NULL,
    refused TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS date_fetches_fx_date ON date_fetches (fx_date);
CREATE TABLE IF NOT EXISTS pillar_dates (
    fetch_id INTEGER NOT NULL REFERENCES date_fetches (fetch_id),
    pair TEXT NOT NULL,
    tenor TEXT NOT NULL,
    expiry TEXT NOT NULL,
    delivery TEXT NOT NULL,
    source TEXT NOT NULL,
    solver_expiry TEXT NOT NULL DEFAULT '',
    solver_delivery TEXT NOT NULL DEFAULT '',
    quoted_expiry TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (fetch_id, pair, tenor),
    CHECK (expiry < delivery)
);
CREATE TABLE IF NOT EXISTS pillar_date_fallbacks (
    fetch_id INTEGER NOT NULL REFERENCES date_fetches (fetch_id),
    pair TEXT NOT NULL,
    tenor TEXT NOT NULL,
    reason TEXT NOT NULL
);
"""

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_info (version INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS markets (
    market_id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_source TEXT NOT NULL,
    date_time TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    event_set_id INTEGER REFERENCES event_sets (event_set_id),
    timezone TEXT NOT NULL DEFAULT 'Europe/London'
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
""" + _EVENT_SCHEMA + _PAIR_EVENT_SCHEMA + _PILLAR_DATE_SCHEMA

#: One stored day of pillar dates: ``dates`` is ``{(pair, tenor): (expiry,
#: delivery, source)}`` with ISO dates; ``fallbacks`` ``[(pair, tenor, reason)]``.
PillarDateFetch = namedtuple("PillarDateFetch", "fetch_id fx_date fetched_at timezone source "
                                                "refused dates fallbacks")

_SNAPSHOT_FIELDS = ("market_id market_source date_time fetched_at note event_set_id timezone "
                    "spots depo_curves vols forward_points fx_date")


class Snapshot(namedtuple("Snapshot", _SNAPSHOT_FIELDS)):
    """One stored market: its record and its quotes by kind.

    ``fx_date`` is attached by whoever builds markets from it (Hyperion,
    ``app.store_markets``): the FX date of ``date_time`` in ``timezone``
    under the 17:00 New York rollover. The store keeps data only, so it
    leaves it ``None``.
    """
    __slots__ = ()

    @property
    def date(self):
        """The date the data is from, ``YYYY-MM-DD``: the FX date when attached,
        else ``date_time``'s local date."""
        return self.fx_date or self.date_time[:10]


Snapshot.__new__.__defaults__ = ("", None, LEGACY_TIMEZONE, {}, {}, {}, {}, None)

#: ``events`` is ``{pair: [(event_date, event, variance_days), ...]}`` in date order.
EventSet = namedtuple("EventSet", "event_set_id source date_time note events")
#: One saved version of a pair's calendar; ``rows`` is ``[(date, event,
#: variance_days), ...]`` by date then name (empty: the pair has no events).
PairEvents = namedtuple("PairEvents", "pair version saved_at user note rows")


class MarketDataStoreError(RuntimeError):
    """The store cannot be used (newer schema, failed migration)."""


def _iso(value):
    """``YYYY-MM-DDTHH:MM:SS``, so stored date-times sort and compare as text."""
    return pd.Timestamp(value).to_pydatetime().replace(microsecond=0, tzinfo=None).isoformat()


def _day(value):
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def close_date_time(day):
    """``day``'s close of business (15:00 New York) as a stored date-time.

    In London that is 20:00, or 19:00 in the weeks the UK and US change their
    clocks on different dates.
    """
    close = pd.Timestamp(f"{_day(day)} {CLOSE_TIME}").tz_localize(CLOSE_TIMEZONE)
    return _iso(close.tz_convert(STORE_TIMEZONE).tz_localize(None))


def data_date_time(valuation_date, now=None):
    """The ``date_time`` of data fetched for ``valuation_date``: now when it is
    today, else that date's close (``close_date_time``)."""
    now = pd.Timestamp(now or datetime.now())
    if _day(valuation_date) == _day(now):
        return _iso(now)
    return close_date_time(valuation_date)


def validate_events(rows, what="event set not stored"):
    """``[(PAIR, YYYY-MM-DD, event, variance_days)]`` from ``rows`` of
    ``(pair, date, event, variance_days)``; raises ``ValueError`` listing
    every bad row (1-based), so a set is stored whole or not at all."""
    import math
    import re
    clean, errors, seen = [], [], set()
    for number, row in enumerate(rows, start=1):
        try:
            pair, date, event, variance_days = row
            pair = str(pair or "").replace("/", "").replace(" ", "").upper()
            if not re.fullmatch(r"[A-Z]{6}", pair):
                raise ValueError(f"pair {row[0]!r} is not two 3-letter currency codes")
            try:
                stamp = pd.Timestamp(date)
            except (TypeError, ValueError):
                stamp = pd.NaT
            if stamp is pd.NaT or pd.isna(stamp):
                raise ValueError(f"date {date!r} is not a date")
            day = stamp.strftime("%Y-%m-%d")
            event = str(event or "").strip()
            if not event:
                raise ValueError("event name is empty")
            try:
                days = float(variance_days)
            except (TypeError, ValueError):
                days = math.nan
            if not math.isfinite(days):
                raise ValueError(f"variance days {variance_days!r} is not a finite number")
            if (pair, day, event) in seen:
                raise ValueError(f"duplicate of an earlier row ({pair} {day} {event})")
            seen.add((pair, day, event))
            clean.append((pair, day, event, days))
        except (TypeError, ValueError) as exc:
            errors.append(f"row {number}: {exc}")
    if errors:
        raise ValueError(f"{what}: " + "; ".join(errors))
    if not clean:
        raise ValueError(f"{what}: no events")
    return clean


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
            elif "markets" in tables and "event_sets" not in tables:
                self._migrate_v2(db)
            elif "markets" in tables and "pair_event_versions" not in tables:
                self._migrate_v3(db)
            if "markets" in tables and "timezone" not in self._columns(db, "markets"):
                self._migrate_v4(db)
            if "markets" in tables and "date_fetches" not in tables:
                self._migrate_v5(db)
            elif "pillar_dates" in tables and "quoted_expiry" not in self._columns(db, "pillar_dates"):
                self._migrate_v6(db)
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

    def _backup(self, version):
        backup = self.path.with_name(f"{self.path.stem}.v{version}-backup-"
                                     f"{datetime.now():%Y%m%d_%H%M%S}{self.path.suffix}")
        shutil.copy2(str(self.path), str(backup))
        return backup

    @staticmethod
    def _columns(db, table):
        return {row[1] for row in db.execute(f"PRAGMA table_info({table})")}

    def _migrate_v6(self, db):
        """Schema 6 -> 7: ``pillar_dates.quoted_expiry``; schema 6's fetches are
        marked refused (``SUPERSEDED``), so none is used again.

        Backed up first; one explicit transaction, rolled back on any error.
        """
        version = db.execute("SELECT MAX(version) FROM schema_info").fetchone()[0]
        if version is not None and version >= SCHEMA_VERSION:
            return
        backup = self._backup(6)
        db.isolation_level = None
        try:
            db.execute("BEGIN")
            db.execute("ALTER TABLE pillar_dates ADD COLUMN quoted_expiry TEXT NOT NULL DEFAULT ''")
            db.execute("UPDATE date_fetches SET refused = ? WHERE refused = ''", (SUPERSEDED,))
            db.execute("DELETE FROM schema_info")
            db.execute("INSERT INTO schema_info VALUES (?)", (SCHEMA_VERSION,))
            db.execute("COMMIT")
        except Exception as exc:
            db.execute("ROLLBACK")
            raise MarketDataStoreError(
                f"could not migrate {self.path} (unchanged; backup at {backup}): {exc}") from exc
        finally:
            db.isolation_level = ""
        self.migrated_from = self.migrated_from or backup

    def _migrate_v5(self, db):
        """Schema 5 -> 6: the pillar date tables (empty).

        Backed up first; one explicit transaction, rolled back on any error.
        Skipped when an older step of this opening has just migrated.
        """
        version = db.execute("SELECT MAX(version) FROM schema_info").fetchone()[0]
        if version is not None and version >= SCHEMA_VERSION:
            return               # newer: refused below; current: an older step migrated it
        backup = self._backup(5)
        db.isolation_level = None
        try:
            db.execute("BEGIN")
            for statement in filter(str.strip, _PILLAR_DATE_SCHEMA.split(";")):
                db.execute(statement)
            db.execute("DELETE FROM schema_info")
            db.execute("INSERT INTO schema_info VALUES (?)", (SCHEMA_VERSION,))
            db.execute("COMMIT")
        except Exception as exc:
            db.execute("ROLLBACK")
            raise MarketDataStoreError(
                f"could not migrate {self.path} (unchanged; backup at {backup}): {exc}") from exc
        finally:
            db.isolation_level = ""
        self.migrated_from = self.migrated_from or backup

    def _migrate_v4(self, db):
        """Schema 4 -> 5: ``markets.timezone``; every market stored so far gets
        ``LEGACY_TIMEZONE`` (the fetch machine's zone).

        Backed up first; one explicit transaction, rolled back on any error.
        """
        version = db.execute("SELECT MAX(version) FROM schema_info").fetchone()[0]
        if version is not None and version > SCHEMA_VERSION:
            return                               # refused below as newer, unchanged
        backup = self._backup(4)
        db.isolation_level = None
        try:
            db.execute("BEGIN")
            db.execute("ALTER TABLE markets ADD COLUMN timezone TEXT NOT NULL "
                       f"DEFAULT '{LEGACY_TIMEZONE}'")
            db.execute("DELETE FROM schema_info")
            db.execute("INSERT INTO schema_info VALUES (?)", (SCHEMA_VERSION,))
            db.execute("COMMIT")
        except Exception as exc:
            db.execute("ROLLBACK")
            raise MarketDataStoreError(
                f"could not migrate {self.path} (unchanged; backup at {backup}): {exc}") from exc
        finally:
            db.isolation_level = ""
        self.migrated_from = self.migrated_from or backup

    def _migrate_v3(self, db):
        """Schema 3 -> 4: per-pair event calendar versions (empty).

        Backed up first; one explicit transaction, rolled back on any error.
        Markets and event sets are untouched.
        """
        version = db.execute("SELECT MAX(version) FROM schema_info").fetchone()[0]
        if version is not None and version > SCHEMA_VERSION:
            return                               # refused below as newer, unchanged
        backup = self._backup(3)
        db.isolation_level = None
        try:
            db.execute("BEGIN")
            for statement in filter(str.strip, _PAIR_EVENT_SCHEMA.split(";")):
                db.execute(statement)
            db.execute("DELETE FROM schema_info")
            db.execute("INSERT INTO schema_info VALUES (?)", (SCHEMA_VERSION,))
            db.execute("COMMIT")
        except Exception as exc:
            db.execute("ROLLBACK")
            raise MarketDataStoreError(
                f"could not migrate {self.path} (unchanged; backup at {backup}): {exc}") from exc
        finally:
            db.isolation_level = ""
        self.migrated_from = backup

    def _migrate_v2(self, db):
        """Schema 2 -> 3: event sets, and ``markets.event_set_id`` (unpinned).

        Backed up first; one explicit transaction, rolled back on any error.
        """
        backup = self._backup(2)
        db.isolation_level = None
        try:
            db.execute("BEGIN")
            for statement in filter(str.strip, _EVENT_SCHEMA.split(";")):
                db.execute(statement)
            db.execute("ALTER TABLE markets ADD COLUMN event_set_id INTEGER "
                       "REFERENCES event_sets (event_set_id)")
            for statement in filter(str.strip, _PAIR_EVENT_SCHEMA.split(";")):
                db.execute(statement)
            db.execute("DELETE FROM schema_info")
            db.execute("INSERT INTO schema_info VALUES (?)", (SCHEMA_VERSION,))
            db.execute("COMMIT")
        except Exception as exc:
            db.execute("ROLLBACK")
            raise MarketDataStoreError(
                f"could not migrate {self.path} (unchanged; backup at {backup}): {exc}") from exc
        finally:
            db.isolation_level = ""
        self.migrated_from = backup

    def _migrate_v1(self, db):
        """``snapshots`` (id, as_of, valuation_date, source) -> ``markets``.

        The file is copied to a backup first; the migration is one explicit
        transaction, rolled back entirely on any error.
        """
        backup = self._backup(1)
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
             forward_points=None, fetched_at=None, note="", event_set_id=None, timezone=None):
        """Store one market in a single transaction and return its ``market_id``.

        ``vols`` is ``{pair: {tenor: {field: value}}}`` with fields from
        ``VOL_FIELDS``; ``forward_points`` is ``{pair: {tenor: points}}``;
        ``event_set_id`` pins the market's events (recorded as a pin);
        ``timezone`` is ``date_time``'s zone (default ``STORE_TIMEZONE``).
        """
        if not market_source:
            raise ValueError("market_source is required")
        rows = _rows(spots, depo_curves, vols, forward_points)
        fetched_at = _iso(fetched_at or datetime.now())
        with closing(self._connect()) as db, db:
            cursor = db.execute(
                "INSERT INTO markets (market_source, date_time, fetched_at, note, event_set_id, "
                "timezone) VALUES (?, ?, ?, ?, ?, ?)",
                (market_source, _iso(date_time), fetched_at, note, event_set_id,
                 timezone or STORE_TIMEZONE))
            market_id = cursor.lastrowid
            db.executemany(
                "INSERT INTO quotes (market_id, kind, name, tenor, value) VALUES (?, ?, ?, ?, ?)",
                [(market_id, *row) for row in rows])
            if event_set_id is not None:
                db.execute("INSERT INTO event_pins VALUES (?, ?, ?, ?)",
                           (market_id, event_set_id, fetched_at, "pinned when saved"))
        return market_id

    # -- pillar dates --------------------------------------------------------

    def save_pillar_dates(self, fx_date, fetched_at, timezone, source, dates=(), fallbacks=(),
                          refused=""):
        """Store one fetch of the day's pillar dates; returns its ``fetch_id``.

        ``dates`` is ``[(pair, tenor, expiry, delivery, source, solver_expiry,
        solver_delivery[, quoted_expiry])]``, ``fallbacks`` ``[(pair, tenor, reason)]``. Checked
        as a whole (pairs, tenors, dates, expiry before delivery, no duplicate)
        and stored whole or not at all. A refused fetch is stored too, with its
        reason, for the record; it is never loaded as usable.
        """
        if not source:
            raise ValueError("pillar date source is required")
        clean, errors, seen = [], [], set()
        for number, row in enumerate(dates, start=1):
            try:
                pair, tenor, expiry, delivery, kind, solver_expiry, solver_delivery, *rest = row
                if len(rest) > 1:
                    raise ValueError(f"{len(row)} fields")
                quoted = rest[0] if rest else ""
                pair = str(pair).replace("/", "").upper()
                if not re.fullmatch(r"[A-Z]{6}", pair):
                    raise ValueError(f"pair {row[0]!r}")
                tenor = str(tenor).strip().upper()
                if not re.fullmatch(r"ON|\d+[DWMY]", tenor):
                    raise ValueError(f"tenor {row[1]!r}")
                expiry, delivery = _day(expiry), _day(delivery)
                if not expiry < delivery:
                    raise ValueError(f"expiry {expiry} is not before delivery {delivery}")
                if (pair, tenor) in seen:
                    raise ValueError(f"duplicate {pair} {tenor}")
                seen.add((pair, tenor))
                clean.append((pair, tenor, expiry, delivery, str(kind),
                              _day(solver_expiry) if solver_expiry else "",
                              _day(solver_delivery) if solver_delivery else "",
                              _day(quoted) if quoted else ""))
            except (TypeError, ValueError) as exc:
                errors.append(f"row {number}: {exc}")
        if errors:
            raise ValueError("pillar dates not stored: " + "; ".join(errors))
        with closing(self._connect()) as db, db:
            fetch_id = db.execute(
                "INSERT INTO date_fetches (fx_date, fetched_at, timezone, source, refused) "
                "VALUES (?, ?, ?, ?, ?)",
                (_day(fx_date), _iso(fetched_at), timezone or STORE_TIMEZONE, source,
                 refused or "")).lastrowid
            db.executemany("INSERT INTO pillar_dates (fetch_id, pair, tenor, expiry, delivery, source, "
                           "solver_expiry, solver_delivery, quoted_expiry) "
                           "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                           [(fetch_id, *row) for row in clean])
            db.executemany("INSERT INTO pillar_date_fallbacks VALUES (?, ?, ?, ?)",
                           [(fetch_id, str(p), str(t), str(r)) for p, t, r in fallbacks])
        return fetch_id

    def load_pillar_dates(self, fx_date=None, fetch_id=None):
        """The latest usable (not refused) fetch for ``fx_date``, or ``None``.

        ``fetch_id`` reads that fetch instead (refused ones included).
        """
        with closing(self._connect()) as db:
            if fetch_id is not None:
                head = db.execute("SELECT * FROM date_fetches WHERE fetch_id = ?",
                                  (int(fetch_id),)).fetchone()
            else:
                head = db.execute(
                    "SELECT * FROM date_fetches WHERE fx_date = ? AND refused = '' "
                    "ORDER BY fetched_at DESC, fetch_id DESC LIMIT 1", (_day(fx_date),)).fetchone()
            if head is None:
                return None
            dates = {(pair, tenor): (expiry, delivery, kind) for pair, tenor, expiry, delivery, kind in
                     db.execute("SELECT pair, tenor, expiry, delivery, source FROM pillar_dates "
                                "WHERE fetch_id = ?", (head[0],))}
            fallbacks = db.execute("SELECT pair, tenor, reason FROM pillar_date_fallbacks "
                                   "WHERE fetch_id = ? ORDER BY rowid", (head[0],)).fetchall()
        return PillarDateFetch(*head, dates, [tuple(row) for row in fallbacks])

    def pillar_date_disputes(self, fx_date=None):
        """Stored dates that differ from the solver's: ``fx_date, fetch_id, pair, tenor,
        expiry, delivery, solver_expiry, solver_delivery`` (usable fetches, newest first)."""
        query = ("SELECT f.fx_date, f.fetch_id, d.pair, d.tenor, d.expiry, d.delivery, "
                 "d.solver_expiry, d.solver_delivery FROM pillar_dates d "
                 "JOIN date_fetches f ON f.fetch_id = d.fetch_id "
                 "WHERE f.refused = '' AND d.solver_expiry != '' "
                 "AND (d.expiry != d.solver_expiry OR d.delivery != d.solver_delivery)")
        args = ()
        if fx_date is not None:
            query += " AND f.fx_date = ?"
            args = (_day(fx_date),)
        with closing(self._connect()) as db:
            return pd.read_sql_query(
                query + " ORDER BY f.fx_date DESC, f.fetch_id DESC, d.pair, d.tenor", db, params=args)

    def pillar_expiry_mismatches(self, fx_date=None):
        """Where Bloomberg's vol ticker expiry (``quoted_expiry``) differs from the
        expiry used: ``fx_date, fetch_id, pair, tenor, quoted_expiry, expiry``."""
        query = ("SELECT f.fx_date, f.fetch_id, d.pair, d.tenor, d.quoted_expiry, d.expiry "
                 "FROM pillar_dates d JOIN date_fetches f ON f.fetch_id = d.fetch_id "
                 "WHERE f.refused = '' AND d.quoted_expiry != '' AND d.quoted_expiry != d.expiry")
        args = ()
        if fx_date is not None:
            query += " AND f.fx_date = ?"
            args = (_day(fx_date),)
        with closing(self._connect()) as db:
            return pd.read_sql_query(
                query + " ORDER BY f.fx_date DESC, f.fetch_id DESC, d.pair, d.tenor", db, params=args)

    # -- event sets ----------------------------------------------------------

    def save_event_set(self, rows, source, date_time=None, note=""):
        """Store a new event set from ``(pair, date, event, variance_days)``
        rows, validated as a whole (``validate_events``); returns its id."""
        if not source:
            raise ValueError("event set source is required")
        clean = validate_events(rows)
        with closing(self._connect()) as db, db:
            event_set_id = db.execute(
                "INSERT INTO event_sets (source, date_time, note) VALUES (?, ?, ?)",
                (source, _iso(date_time or datetime.now()), note)).lastrowid
            db.executemany("INSERT INTO events VALUES (?, ?, ?, ?, ?)",
                           [(event_set_id, *row) for row in clean])
        return event_set_id

    def load_event_set(self, event_set_id):
        """The event set, or ``None`` when it does not exist."""
        with closing(self._connect()) as db:
            head = db.execute("SELECT event_set_id, source, date_time, note FROM event_sets "
                              "WHERE event_set_id = ?", (int(event_set_id),)).fetchone()
            if head is None:
                return None
            rows = db.execute("SELECT pair, event_date, event, variance_days FROM events "
                              "WHERE event_set_id = ? ORDER BY pair, event_date, event",
                              (head[0],)).fetchall()
        events = {}
        for pair, day, event, days in rows:
            events.setdefault(pair, []).append((day, event, days))
        return EventSet(*head, events)

    def latest_event_set_id(self, at_or_before=None):
        """The id of the latest event set (by ``date_time``), or ``None``."""
        query, args = "SELECT event_set_id FROM event_sets", ()
        if at_or_before is not None:
            query += " WHERE date_time <= ?"
            args = (_iso(at_or_before),)
        with closing(self._connect()) as db:
            row = db.execute(query + " ORDER BY date_time DESC, event_set_id DESC LIMIT 1",
                             args).fetchone()
        return None if row is None else row[0]

    def event_sets(self):
        """Event sets, newest first: ``event_set_id, source, date_time, note, events``."""
        with closing(self._connect()) as db:
            return pd.read_sql_query(
                "SELECT s.event_set_id, s.source, s.date_time, s.note, COUNT(e.pair) AS events "
                "FROM event_sets s LEFT JOIN events e ON e.event_set_id = s.event_set_id "
                "GROUP BY s.event_set_id ORDER BY s.date_time DESC, s.event_set_id DESC", db)

    def pin_events(self, market_id, event_set_id, reason):
        """Re-pin a market to an event set; the change is recorded."""
        if not str(reason or "").strip():
            raise ValueError("a reason is required to re-pin a market's events")
        with closing(self._connect()) as db, db:
            if db.execute("SELECT 1 FROM markets WHERE market_id = ?",
                          (int(market_id),)).fetchone() is None:
                raise ValueError(f"market {market_id} does not exist")
            if db.execute("SELECT 1 FROM event_sets WHERE event_set_id = ?",
                          (int(event_set_id),)).fetchone() is None:
                raise ValueError(f"event set {event_set_id} does not exist")
            db.execute("UPDATE markets SET event_set_id = ? WHERE market_id = ?",
                       (int(event_set_id), int(market_id)))
            db.execute("INSERT INTO event_pins VALUES (?, ?, ?, ?)",
                       (int(market_id), int(event_set_id), _iso(datetime.now()), reason))

    # -- per-pair event calendars -----------------------------------------------

    def save_pair_events(self, pair, rows, user="", note="", saved_at=None):
        """Save ``rows`` (``(date, event, variance_days)``) as the pair's next
        calendar version and return the version number.

        Validated as a whole first (``validate_events``: a valid date, a
        name, finite variance days, no duplicate); any bad row stores
        nothing. An empty calendar is a version too (the pair's events were
        all removed). Earlier versions are never changed.
        """
        import re
        pair = str(pair or "").replace("/", "").replace(" ", "").upper()
        if not re.fullmatch(r"[A-Z]{6}", pair):
            raise ValueError(f"pair {pair!r} is not two 3-letter currency codes")
        rows = list(rows)
        clean = (validate_events([(pair, *row) for row in rows],
                                 f"{pair} event calendar not saved") if rows else [])
        with closing(self._connect()) as db, db:
            version = db.execute("SELECT COALESCE(MAX(version), 0) + 1 FROM "
                                 "pair_event_versions WHERE pair = ?", (pair,)).fetchone()[0]
            db.execute("INSERT INTO pair_event_versions VALUES (?, ?, ?, ?, ?)",
                       (pair, version, _iso(saved_at or datetime.now()), str(user or ""),
                        str(note or "")))
            db.executemany("INSERT INTO pair_events VALUES (?, ?, ?, ?, ?, NULL)",
                           [(pair, version, day, event, days)
                            for _, day, event, days in clean])
        return version

    def load_pair_events(self, pair, version=None, as_of=None):
        """A saved version of the pair's calendar (``PairEvents``), or ``None``.

        ``version`` picks one; else the latest, or with ``as_of`` the latest
        saved on or before that date (a calendar saved later never reaches
        an earlier valuation date).
        """
        pair = str(pair or "").replace("/", "").replace(" ", "").upper()
        query, args = ("SELECT pair, version, saved_at, user, note FROM pair_event_versions "
                       "WHERE pair = ?"), [pair]
        if version is not None:
            query += " AND version = ?"
            args.append(int(version))
        elif as_of is not None:
            query += " AND substr(saved_at, 1, 10) <= ?"
            args.append(_day(as_of))
        with closing(self._connect()) as db:
            head = db.execute(query + " ORDER BY version DESC LIMIT 1", args).fetchone()
            if head is None:
                return None
            rows = db.execute("SELECT event_date, event, variance_days FROM pair_events "
                              "WHERE pair = ? AND version = ? ORDER BY event_date, event",
                              (head[0], head[1])).fetchall()
        return PairEvents(*head, [tuple(row) for row in rows])

    def pair_event_versions(self, pair):
        """The pair's calendar versions, newest first: ``version, saved_at,
        user, note, events``."""
        pair = str(pair or "").replace("/", "").replace(" ", "").upper()
        with closing(self._connect()) as db:
            return pd.read_sql_query(
                "SELECT v.version, v.saved_at, v.user, v.note, COUNT(e.event) AS events "
                "FROM pair_event_versions v LEFT JOIN pair_events e "
                "ON e.pair = v.pair AND e.version = v.version WHERE v.pair = ? "
                "GROUP BY v.version ORDER BY v.version DESC", db, params=(pair,))

    def event_pins(self, market_id):
        """A market's pin history, oldest first."""
        with closing(self._connect()) as db:
            return pd.read_sql_query("SELECT event_set_id, pinned_at, reason FROM event_pins "
                                     "WHERE market_id = ? ORDER BY rowid", db,
                                     params=(int(market_id),))

    def snapshots(self, date=None):
        """Markets, newest first: ``market_id, market_source, date_time,
        fetched_at, note, timezone, quotes, vols, forwards`` (``vols`` and ``forwards``
        count the vol and forward point quotes); ``date`` limits them to data
        of that date."""
        query = ("SELECT m.market_id, m.market_source, m.date_time, m.fetched_at, m.note, "
                 "m.timezone, COUNT(q.value) AS quotes, "
                 f"COALESCE(SUM(q.kind LIKE '{VOL_PREFIX}%'), 0) AS vols, "
                 f"COALESCE(SUM(q.kind = '{FWD_PTS}'), 0) AS forwards FROM markets m "
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
        query, args = ("SELECT market_id, market_source, date_time, fetched_at, note, "
                       "event_set_id, timezone FROM markets WHERE 1 = 1"), []
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
            date_time = close_date_time(payload["valuation_date"])
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

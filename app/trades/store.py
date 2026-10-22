"""Trade storage: an append-only SQLite store behind a narrow interface.

Every booking, amendment and cancellation adds a version row; nothing is
updated or deleted (database triggers enforce it). The current book is the
latest version of each trade, and any past book can be read back with
``as_of``. Hedge forwards link to the options they hedge per hedge version.

Failsafes:

* each booking call is one transaction: all of its trades and hedge links
  are written, or none are;
* trade ids come from a sequence inside that transaction and are never
  reused, so identical trades booked twice get two ids;
* an amendment or cancellation names the version it was made from and is
  rejected if the trade has moved on (``StaleVersionError``), so two edits
  can never silently overwrite each other;
* a hedge's allocations must reference options of the same pair and sum to
  the hedge's signed base amount;
* opening the store runs SQLite's integrity check and refuses a database
  written by a newer schema; a daily backup copy is kept, and another is
  taken before any schema migration.

Fallback: ``open_trade_store`` returns a read-only store when the database
cannot be opened for writing (failed check, newer schema, locked file), so
the book can still be viewed and exported; it never starts an empty book in
place of an unreadable one.

In memory (``MEMORY``, ``":memory:"``): the same store on an in-memory SQLite
database, for the online app, where there is no durable disk. It books,
amends and cancels exactly like the file store (same schema, triggers and
checks), but the book lasts only as long as the browser session, and it
takes no backups.

Two sessions opening a new store at once: the first creates (or migrates)
the schema; the second, finding it done when its own attempt fails, carries
on instead of falling back to read-only.
"""
import math
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.trades.model import (
    AMEND, CANCEL, CANCELLED, FX_FORWARD, FX_OPTION, LIVE, NEW, Booking, HedgeAllocation,
    Trade, TradeValidationError, as_date,
)

SCHEMA_VERSION = 3
#: The path of an in-memory store (the online app's: kept for the session only).
MEMORY = ":memory:"
#: What the app says about an in-memory store.
MEMORY_NOTICE = ("Trades are kept for this browser session only (the online app has no "
                 "trade database): they are not saved.")
TRADE_PREFIX, PACKAGE_PREFIX = "TR", "PK"
BACKUPS_KEPT = 14

_SCHEMA = """
CREATE TABLE schema_info (version INTEGER NOT NULL);
CREATE TABLE sequences (name TEXT PRIMARY KEY, next INTEGER NOT NULL);
CREATE TABLE trade_versions (
    trade_id TEXT NOT NULL,
    version INTEGER NOT NULL CHECK (version >= 1),
    action TEXT NOT NULL CHECK (action IN ('new', 'amend', 'cancel')),
    recorded_at TEXT NOT NULL,
    user TEXT NOT NULL DEFAULT '',
    reason TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL CHECK (status IN ('live', 'cancelled')),
    product_type TEXT NOT NULL CHECK (product_type IN ('fx_option', 'fx_forward')),
    currency_pair TEXT NOT NULL,
    trade_date TEXT NOT NULL,
    expiry TEXT NOT NULL,
    buy_sell TEXT NOT NULL CHECK (buy_sell IN ('BUY', 'SELL')),
    notional REAL NOT NULL CHECK (notional > 0),
    book TEXT NOT NULL DEFAULT '',
    counterparty TEXT NOT NULL DEFAULT '',
    strike REAL,
    call_put TEXT,
    premium REAL NOT NULL DEFAULT 0,
    premium_ccy TEXT,
    contract_rate REAL,
    package_id TEXT,
    option_type TEXT,
    payout_ccy TEXT,
    value_date TEXT,
    amount_ccy TEXT,
    PRIMARY KEY (trade_id, version)
);
CREATE INDEX trade_versions_pair ON trade_versions (currency_pair);
CREATE INDEX trade_versions_book ON trade_versions (book);
CREATE TABLE hedge_links (
    hedge_id TEXT NOT NULL,
    hedge_version INTEGER NOT NULL,
    option_id TEXT NOT NULL,
    base_amount REAL NOT NULL,
    PRIMARY KEY (hedge_id, hedge_version, option_id),
    FOREIGN KEY (hedge_id, hedge_version) REFERENCES trade_versions (trade_id, version)
);
CREATE INDEX hedge_links_option ON hedge_links (option_id);
CREATE TRIGGER trade_versions_append_only_update BEFORE UPDATE ON trade_versions
BEGIN SELECT RAISE(ABORT, 'trade versions are append-only'); END;
CREATE TRIGGER trade_versions_append_only_delete BEFORE DELETE ON trade_versions
BEGIN SELECT RAISE(ABORT, 'trade versions are append-only'); END;
CREATE TRIGGER hedge_links_append_only_update BEFORE UPDATE ON hedge_links
BEGIN SELECT RAISE(ABORT, 'hedge links are append-only'); END;
CREATE TRIGGER hedge_links_append_only_delete BEFORE DELETE ON hedge_links
BEGIN SELECT RAISE(ABORT, 'hedge links are append-only'); END;
"""

#: Schema migrations: version -> SQL taking the previous version to it.
#: 2: option types. Existing rows get NULL, which reads as a vanilla.
#: 3: standalone spot/forward deals. Existing rows get NULL: a forward
#: without a value date is a hedge on the option-expiry convention, and an
#: amount without a currency is in base.
MIGRATIONS = {
    2: ("ALTER TABLE trade_versions ADD COLUMN option_type TEXT;"
        "ALTER TABLE trade_versions ADD COLUMN payout_ccy TEXT;"),
    3: ("ALTER TABLE trade_versions ADD COLUMN value_date TEXT;"
        "ALTER TABLE trade_versions ADD COLUMN amount_ccy TEXT;"),
}

_TERMS = (
    "status", "product_type", "currency_pair", "trade_date", "expiry", "buy_sell",
    "notional", "book", "counterparty", "strike", "call_put", "premium", "premium_ccy",
    "contract_rate", "package_id", "option_type", "payout_ccy", "value_date", "amount_ccy",
)


class TradeStoreError(RuntimeError):
    """Base class for trade store failures."""


class TradeStoreUnavailable(TradeStoreError):
    """The database cannot be opened for writing."""


class TradeStoreReadOnly(TradeStoreError):
    """A write was attempted on a read-only store."""


class StaleVersionError(TradeStoreError):
    """The trade changed since the version an edit was made from."""


class TradeNotFound(TradeStoreError, KeyError):
    """No trade with that id (as of that time)."""


@dataclass(frozen=True)
class TradeVersion:
    """A trade version with its audit fields."""

    trade: Trade
    action: str
    recorded_at: datetime
    user: str
    reason: str


def _utc_now():
    return datetime.now(timezone.utc)


def _stamp(moment):
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _parse_stamp(text):
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)


def _relative_gap(a, b):
    return abs(a - b) / max(1.0, abs(a), abs(b))


class SqliteTradeStore:
    """The trade store on one SQLite file. Use ``open_trade_store`` to open."""

    def __init__(self, path, read_only=False, clock=_utc_now, backup_dir=None, today=None):
        self.in_memory = str(path).strip().lower() in (MEMORY, "memory")
        self.path = Path(MEMORY if self.in_memory else path)
        self.read_only = read_only and not self.in_memory
        self.clock = clock
        self.backup_dir = Path(backup_dir) if backup_dir else self.path.parent / "backups"
        self.problem = None
        try:
            if self.in_memory:
                self._db = sqlite3.connect(MEMORY, isolation_level=None, check_same_thread=False)
            elif read_only:
                if not self.path.exists():
                    raise TradeStoreUnavailable(f"{self.path} does not exist")
                self._db = sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True,
                                           isolation_level=None, check_same_thread=False)
            else:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._db = sqlite3.connect(str(self.path), isolation_level=None,
                                           check_same_thread=False, timeout=10)
        except sqlite3.Error as exc:
            raise TradeStoreUnavailable(f"cannot open {self.path}: {exc}") from exc
        self._db.row_factory = sqlite3.Row
        try:
            self._db.execute("PRAGMA foreign_keys = ON")
            self._check_integrity()
            if self.in_memory:
                self._prepare_schema()
            elif not read_only:
                self._prepare_schema()
                self._daily_backup(today)
            else:
                version = self._schema_version()
                if version == 0:
                    raise TradeStoreUnavailable(f"{self.path} is not a trade store")
                if version > SCHEMA_VERSION:
                    # Viewable if every column this version reads is still there.
                    self._check_columns()
                    self.problem = (f"{self.path} was written by a newer version (schema "
                                    f"{version}); opened read-only")
        except TradeStoreError:
            self._db.close()
            raise
        except sqlite3.Error as exc:
            self._db.close()
            raise TradeStoreUnavailable(f"{self.path}: {exc}") from exc

    # -- opening ----------------------------------------------------------

    def _check_integrity(self):
        result = self._db.execute("PRAGMA quick_check").fetchone()[0]
        if result != "ok":
            raise TradeStoreUnavailable(f"{self.path} failed its integrity check: {result}")

    def _check_columns(self):
        needed = {"trade_versions": {"trade_id", "version", "action", "recorded_at", "user",
                                     "reason", *_TERMS},
                  "hedge_links": {"hedge_id", "hedge_version", "option_id", "base_amount"}}
        for table, columns in needed.items():
            present = {row[1] for row in self._db.execute(f"PRAGMA table_info({table})")}
            missing = columns - present
            if missing:
                raise TradeStoreUnavailable(
                    f"{self.path} table {table} lacks {', '.join(sorted(missing))}")

    def _schema_version(self):
        exists = self._db.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_info'"
        ).fetchone()
        if not exists:
            return 0
        row = self._db.execute("SELECT version FROM schema_info").fetchone()
        return int(row[0]) if row else 0

    def _check_version(self, version):
        if version > SCHEMA_VERSION:
            raise TradeStoreUnavailable(
                f"{self.path} was written by a newer version (schema {version}, "
                f"this app reads up to {SCHEMA_VERSION})")

    def _prepare_schema(self):
        version = self._schema_version()
        self._check_version(version)
        if version == SCHEMA_VERSION:
            return
        if version == 0:
            if self._table_count():
                raise TradeStoreUnavailable(f"{self.path} is not a trade store")
            self._script_unless_done(
                _SCHEMA + f"INSERT INTO schema_info (version) VALUES ({SCHEMA_VERSION});")
            return
        self.backup(self.backup_dir / f"{self.path.stem}_before_schema_{SCHEMA_VERSION}.sqlite")
        self._script_unless_done(
            "".join(MIGRATIONS[step] for step in range(version + 1, SCHEMA_VERSION + 1))
            + f"UPDATE schema_info SET version = {SCHEMA_VERSION};")

    def _table_count(self):
        return self._db.execute("SELECT count(*) FROM sqlite_master WHERE type = 'table'").fetchone()[0]

    def _script_unless_done(self, sql):
        """Create or migrate the schema, unless another session just did.

        Two sessions can find a store new (or old) at once; the second one's
        script then fails (``table ... already exists``) after the first has
        committed. If the schema is now current, that is success.
        """
        try:
            self._script(sql)
        except sqlite3.OperationalError:
            if self._schema_version() == SCHEMA_VERSION:
                return
            raise

    def _script(self, sql):
        """Run a multi-statement script as one transaction.

        ``executescript`` commits any open transaction before it runs, so the
        script carries its own BEGIN/COMMIT and is rolled back on failure.
        """
        try:
            self._db.executescript("BEGIN IMMEDIATE;" + sql + "COMMIT;")
        except sqlite3.Error:
            if self._db.in_transaction:
                self._db.execute("ROLLBACK")
            raise

    def backup(self, target):
        """Copy the database to ``target`` (consistent even while in use)."""
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        copy = sqlite3.connect(str(target))
        try:
            self._db.backup(copy)
        finally:
            copy.close()
        return target

    def _daily_backup(self, today=None):
        """One backup per day, keeping the most recent ``BACKUPS_KEPT``."""
        day = (today or self.clock().date()).strftime("%Y%m%d")
        target = self.backup_dir / f"{self.path.stem}_{day}.sqlite"
        try:
            if not target.exists():
                self.backup(target)
            daily = sorted(self.backup_dir.glob(f"{self.path.stem}_[0-9]*.sqlite"))
            for old in daily[:-BACKUPS_KEPT]:
                old.unlink()
        except (OSError, sqlite3.Error) as exc:
            # A failed backup must not block trading, but it is reported.
            self.problem = f"daily backup failed: {exc}"

    def close(self):
        self._db.close()

    # -- writing ----------------------------------------------------------

    @contextmanager
    def _transaction(self):
        self._db.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self._db.execute("ROLLBACK")
            raise
        self._db.execute("COMMIT")

    def _writable(self):
        if self.read_only:
            raise TradeStoreReadOnly(f"{self.path} is open read-only")

    def _next_id(self, name, prefix):
        row = self._db.execute("SELECT next FROM sequences WHERE name = ?", (name,)).fetchone()
        number = row[0] if row else 1
        self._db.execute("INSERT OR REPLACE INTO sequences (name, next) VALUES (?, ?)",
                         (name, number + 1))
        return f"{prefix}-{number:06d}"

    def _recorded_at(self):
        """Now, strictly after every earlier record so as-of reads are exact."""
        now = self.clock()
        last = self._db.execute("SELECT max(recorded_at) FROM trade_versions").fetchone()[0]
        if last is not None and _stamp(now) <= last:
            now = _parse_stamp(last) + timedelta(microseconds=1)
        return _stamp(now)

    def _insert(self, trade, action, recorded_at, user, reason):
        values = trade.terms()
        row = {name: values[name] for name in _TERMS}
        row["trade_date"] = values["trade_date"].isoformat()
        row["expiry"] = values["expiry"].isoformat()
        if values["value_date"] is not None:
            row["value_date"] = values["value_date"].isoformat()
        columns = ("trade_id", "version", "action", "recorded_at", "user", "reason") + _TERMS
        self._db.execute(
            f"INSERT INTO trade_versions ({', '.join(columns)}) "
            f"VALUES ({', '.join('?' for _ in columns)})",
            (trade.trade_id, trade.version, action, recorded_at, user or "", reason or "",
             *(row[name] for name in _TERMS)),
        )

    def _link(self, hedge, allocations):
        for allocation in allocations:
            self._db.execute(
                "INSERT INTO hedge_links (hedge_id, hedge_version, option_id, base_amount) "
                "VALUES (?, ?, ?, ?)",
                (hedge.trade_id, hedge.version, allocation.option, allocation.base_amount),
            )

    def _check_allocations(self, hedge, allocations, options, require_live=True):
        """``options`` maps option id -> current Trade for every reference.

        ``require_live=False`` accepts options cancelled since they were
        linked, so a hedge keeping its existing links can still be amended.
        """
        if not hedge.is_forward:
            raise TradeValidationError(f"{hedge.trade_id or 'new trade'}: only forwards hedge options")
        seen = set()
        for allocation in allocations:
            option = options[allocation.option]
            if option.trade_id in seen:
                raise TradeValidationError(f"{option.trade_id} is allocated twice to one hedge")
            seen.add(option.trade_id)
            if not option.is_option:
                raise TradeValidationError(f"{option.trade_id} is not an option")
            if require_live and option.status != LIVE:
                raise TradeValidationError(f"{option.trade_id} is cancelled")
            if option.currency_pair != hedge.currency_pair:
                raise TradeValidationError(
                    f"hedge on {hedge.currency_pair} cannot hedge {option.trade_id} on "
                    f"{option.currency_pair}")
        total = sum(allocation.base_amount for allocation in allocations)
        if allocations and _relative_gap(total, hedge.signed_base_amount()) > 1e-9:
            raise TradeValidationError(
                f"hedge allocations sum to {total:,.2f}, not the hedge's signed base amount "
                f"{hedge.signed_base_amount():,.2f}")

    def book(self, booking, user="", reason=""):
        """Book every trade in ``booking`` (a ``Booking`` or trades); all or none.

        Returns the booked trades, in order, with their new ids.
        """
        self._writable()
        # Duck-typed: after Streamlit reloads a changed module, a Booking made by
        # the new code is not an instance of this store's (older) Booking class.
        if not hasattr(booking, "trades"):
            booking = Booking(tuple(booking))
        trades = [trade.normalized() for trade in booking.trades]
        if not trades:
            raise TradeValidationError("nothing to book")
        hedges = {}
        for index, allocations in (booking.hedges or {}).items():
            if not 0 <= int(index) < len(trades):
                raise TradeValidationError(f"hedge {index} is not in the booking")
            hedges[int(index)] = [allocation.checked() for allocation in allocations]
        with self._transaction():
            package = self._next_id("package", PACKAGE_PREFIX)
            recorded_at = self._recorded_at()
            booked = []
            for trade in trades:
                trade = replace(trade, trade_id=self._next_id("trade", TRADE_PREFIX),
                                version=1, status=LIVE, package_id=package)
                booked.append(trade)
            for index, allocations in hedges.items():
                resolved, options = [], {}
                for allocation in allocations:
                    ref = allocation.option
                    if isinstance(ref, int):
                        if not 0 <= ref < len(booked) or ref == index:
                            raise TradeValidationError(f"hedge {index} references option {ref}, "
                                                       "which is not in the booking")
                        option = booked[ref]
                    else:
                        option = self._current(ref)
                    options[option.trade_id] = option
                    resolved.append(replace(allocation, option=option.trade_id))
                self._check_allocations(booked[index], resolved, options)
                hedges[index] = resolved
            for trade in booked:
                self._insert(trade, NEW, recorded_at, user, reason)
            for index, allocations in hedges.items():
                self._link(booked[index], allocations)
        return booked

    def _current(self, trade_id, as_of=None):
        trade = self._latest(trade_id, as_of)
        if trade is None:
            raise TradeNotFound(f"no trade {trade_id}")
        return trade

    def amend(self, trade_id, expected_version, changes, user="", reason="", allocations=None):
        """Write the next version of ``trade_id`` with ``changes`` (a dict) applied.

        A hedge keeps its option allocations unless new ones are given; they
        must still sum to its (possibly amended) signed base amount.
        """
        self._writable()
        with self._transaction():
            current = self._current_for_edit(trade_id, expected_version)
            trade = replace(current.amended(**dict(changes or {})), version=current.version + 1)
            links = self._allocations(trade_id, current.version) if allocations is None else [
                allocation.checked() for allocation in allocations]
            if trade.is_forward and links:
                options = {link.option: self._current(link.option) for link in links}
                # Kept links may point at options cancelled since; new links must be live.
                self._check_allocations(trade, links, options, require_live=allocations is not None)
            elif links:
                raise TradeValidationError(f"{trade_id}: only forwards hedge options")
            self._insert(trade, AMEND, self._recorded_at(), user, reason)
            self._link(trade, links)
        return trade

    def cancel(self, trade_id, expected_version, user="", reason=""):
        """Write a cancelled version of ``trade_id``; a hedge keeps its links."""
        self._writable()
        with self._transaction():
            current = self._current_for_edit(trade_id, expected_version)
            trade = replace(current, version=current.version + 1, status=CANCELLED)
            self._insert(trade, CANCEL, self._recorded_at(), user, reason)
            self._link(trade, self._allocations(trade_id, current.version))
        return trade

    def _current_for_edit(self, trade_id, expected_version):
        current = self._current(trade_id)
        if current.version != int(expected_version):
            raise StaleVersionError(
                f"{trade_id} is at version {current.version}, not {expected_version}; "
                "reload it and edit again")
        if current.status == CANCELLED:
            raise TradeValidationError(f"{trade_id} is cancelled")
        return current

    # -- reading ----------------------------------------------------------

    @staticmethod
    def _trade(row):
        return Trade(
            trade_id=row["trade_id"], version=row["version"], status=row["status"],
            product_type=row["product_type"], currency_pair=row["currency_pair"],
            trade_date=as_date(row["trade_date"]), expiry=as_date(row["expiry"]),
            buy_sell=row["buy_sell"], notional=row["notional"], book=row["book"],
            counterparty=row["counterparty"], strike=row["strike"], call_put=row["call_put"],
            premium=row["premium"], premium_ccy=row["premium_ccy"],
            contract_rate=row["contract_rate"], package_id=row["package_id"],
            # A store opened read-only before its migration lacks the later
            # columns: its options are vanillas, its forwards hedges in base.
            **{name: row[name] for name in ("option_type", "payout_ccy", "amount_ccy")
               if name in row.keys()},
            value_date=(as_date(row["value_date"]) if "value_date" in row.keys()
                        and row["value_date"] else None),
        )

    def _latest_rows(self, as_of=None, where="", params=()):
        cutoff = "AND recorded_at <= ?" if as_of is not None else ""
        args = ((_stamp(as_of),) if as_of is not None else ()) + tuple(params)
        return self._db.execute(
            f"""SELECT v.* FROM trade_versions v
                JOIN (SELECT trade_id, max(version) AS version FROM trade_versions
                      WHERE 1 = 1 {cutoff} GROUP BY trade_id) latest
                  ON latest.trade_id = v.trade_id AND latest.version = v.version
                WHERE 1 = 1 {where}
                ORDER BY v.trade_id""",
            args,
        ).fetchall()

    def _latest(self, trade_id, as_of=None):
        rows = self._latest_rows(as_of, "AND v.trade_id = ?", (trade_id,))
        return self._trade(rows[0]) if rows else None

    def get(self, trade_id, as_of=None):
        """The trade's latest version (as of ``as_of``, a UTC datetime)."""
        return self._current(trade_id, as_of)

    def trades(self, status=LIVE, currency_pair=None, book=None, product_type=None,
               counterparty=None, as_of=None, live_on=None):
        """Latest versions matching every filter given.

        ``status=None`` includes cancelled trades; ``live_on`` keeps only
        trades that have not expired by that valuation date.
        """
        clauses, params = [], []
        for column, value in (("status", status), ("currency_pair", currency_pair),
                              ("book", book), ("product_type", product_type),
                              ("counterparty", counterparty)):
            if value is not None:
                clauses.append(f"AND v.{column} = ?")
                params.append(str(value).upper() if column == "currency_pair" else value)
        if live_on is not None:
            clauses.append("AND v.expiry > ?")
            params.append(as_date(live_on).isoformat())
        rows = self._latest_rows(as_of, " ".join(clauses), params)
        return [self._trade(row) for row in rows]

    def history(self, trade_id):
        """Every version of ``trade_id``, oldest first."""
        rows = self._db.execute(
            "SELECT * FROM trade_versions WHERE trade_id = ? ORDER BY version", (trade_id,)
        ).fetchall()
        if not rows:
            raise TradeNotFound(f"no trade {trade_id}")
        return [TradeVersion(self._trade(row), row["action"], _parse_stamp(row["recorded_at"]),
                             row["user"], row["reason"]) for row in rows]

    def _allocations(self, hedge_id, version):
        rows = self._db.execute(
            "SELECT option_id, base_amount FROM hedge_links "
            "WHERE hedge_id = ? AND hedge_version = ? ORDER BY option_id", (hedge_id, version)
        ).fetchall()
        return [HedgeAllocation(row["option_id"], row["base_amount"]) for row in rows]

    def allocations(self, hedge_id, as_of=None):
        """The options a hedge's current version hedges, with base amounts."""
        hedge = self._current(hedge_id, as_of)
        return self._allocations(hedge_id, hedge.version)

    def hedges_for(self, option_id, as_of=None, status=LIVE):
        """``[(hedge trade, base amount for this option)]`` for one option."""
        result = []
        for hedge in self.trades(status=status, product_type=FX_FORWARD, as_of=as_of):
            for allocation in self._allocations(hedge.trade_id, hedge.version):
                if allocation.option == option_id:
                    result.append((hedge, allocation.base_amount))
        return result

    def orphaned_hedges(self, as_of=None):
        """Live hedges with an allocation to an option that is no longer live."""
        orphans = []
        for hedge in self.trades(product_type=FX_FORWARD, as_of=as_of):
            for allocation in self._allocations(hedge.trade_id, hedge.version):
                option = self._latest(allocation.option, as_of)
                if option is None or option.status != LIVE:
                    orphans.append((hedge, allocation.option))
        return orphans


def open_trade_store(path, **kwargs):
    """``(store, problem)``: writable if possible, else read-only, else ``None``.

    ``problem`` explains a read-only or missing store, or a failed daily
    backup; it is ``None`` when everything is fine. ``path`` ``MEMORY`` opens
    an in-memory store (``problem`` says the book is not saved).
    """
    try:
        store = SqliteTradeStore(path, **kwargs)
        if store.in_memory:
            return store, MEMORY_NOTICE
        return store, store.problem
    except TradeStoreUnavailable as exc:
        reason = str(exc)
    try:
        return SqliteTradeStore(path, read_only=True, **kwargs), (
            f"trade store is read-only: {reason}")
    except TradeStoreUnavailable as exc:
        return None, f"trade store unavailable: {reason}; {exc}"

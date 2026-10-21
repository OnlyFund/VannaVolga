"""The result store: a bounded in-memory LRU over a SQLite file, 24-hour life.

``get(key)`` / ``put(key, value)``. Values are deep-copied in and out, so a
caller changing a result never changes the cache. Every disk operation is
guarded: a store that cannot be read or written simply misses (the caller
computes fresh), never fails. Entries older than ``ttl_hours`` are misses and
are purged; keys carry the code version, so an old code's results are never
read (``keys.CACHE_VERSION``).
"""
import copy
import os
import pickle
import sqlite3
import threading
import time
from collections import OrderedDict
from contextlib import closing
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[3] / "data" / "local" / "result_cache.sqlite"
_SCHEMA = ("CREATE TABLE IF NOT EXISTS results (key TEXT PRIMARY KEY, created REAL NOT NULL, "
           "value BLOB NOT NULL)")


class ResultCache:
    def __init__(self, path=None, max_memory=5000, ttl_hours=24.0, disk=True):
        self.path = Path(path or os.environ.get("VV_RESULT_CACHE") or DEFAULT_PATH)
        self.max_memory, self.ttl = int(max_memory), float(ttl_hours) * 3600.0
        self.disk = disk
        self._memory = OrderedDict()                 # key -> (created, value)
        self._lock = threading.Lock()
        self.hits = self.misses = self.disk_errors = 0
        if disk:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with closing(sqlite3.connect(str(self.path))) as db, db:
                    db.execute(_SCHEMA)
                    db.execute("DELETE FROM results WHERE created < ?", (time.time() - self.ttl,))
            except Exception:
                self.disk, self.disk_errors = False, self.disk_errors + 1

    def get(self, key):
        """The cached value (a copy), or ``None``."""
        if key is None:
            return None
        now = time.time()
        with self._lock:
            found = self._memory.get(key)
            if found is not None and now - found[0] <= self.ttl:
                self._memory.move_to_end(key)
                self.hits += 1
                return copy.deepcopy(found[1])
            if found is not None:
                self._memory.pop(key, None)
        value = self._disk_get(key, now)
        with self._lock:
            if value is None:
                self.misses += 1
                return None
            self.hits += 1
            self._remember(key, value[0], value[1])
        return copy.deepcopy(value[1])

    def put(self, key, value):
        if key is None:
            return
        created, value = time.time(), copy.deepcopy(value)
        with self._lock:
            self._remember(key, created, value)
        if self.disk:
            try:
                with closing(sqlite3.connect(str(self.path), timeout=5)) as db, db:
                    db.execute("INSERT OR REPLACE INTO results VALUES (?, ?, ?)",
                               (key, created, pickle.dumps(value, protocol=4)))
            except Exception:
                self.disk_errors += 1

    def clear(self):
        with self._lock:
            self._memory.clear()
        if self.disk:
            try:
                with closing(sqlite3.connect(str(self.path))) as db, db:
                    db.execute("DELETE FROM results")
            except Exception:
                self.disk_errors += 1

    def _remember(self, key, created, value):
        self._memory[key] = (created, value)
        self._memory.move_to_end(key)
        while len(self._memory) > self.max_memory:
            self._memory.popitem(last=False)

    def _disk_get(self, key, now):
        if not self.disk:
            return None
        try:
            with closing(sqlite3.connect(str(self.path), timeout=5)) as db:
                row = db.execute("SELECT created, value FROM results WHERE key = ?",
                                 (key,)).fetchone()
            if row is None or now - row[0] > self.ttl:
                return None
            return row[0], pickle.loads(row[1])
        except Exception:                      # unreadable store or entry: a miss
            self.disk_errors += 1
            return None


_DEFAULT = None
_DEFAULT_LOCK = threading.Lock()


def default_cache():
    """The process-wide cache (memory shared by every session, one disk file)."""
    global _DEFAULT
    with _DEFAULT_LOCK:
        if _DEFAULT is None:
            _DEFAULT = ResultCache()
        return _DEFAULT

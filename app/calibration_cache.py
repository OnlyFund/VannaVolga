"""Calibrated smiles kept on disk, so each stored market is calibrated once.

``ensure_calibrated`` calibrates one pair of one stored market and keeps the
result for ``TTL`` (24 hours) in ``data/local/calibration_cache/``, one file
per (market_id, pair, valuation date, pricing settings, code cache version).
The next session, or the app after a restart, restores the pair from that
file instead of calibrating it again.

The pair is calibrated against a private cache, so the file holds exactly
that pair's smiles; they are then merged into the session's caches. Cache
keys are the full market state, so a restored smile can only be reused for
exactly the inputs it was calibrated on: a wrong file can cost a
recalibration, never a wrong price.

Fallbacks, so the cache can never stop pricing:

* no file, an expired one (older than ``TTL``), one written by other code
  (``SURFACE_CACHE_VERSION`` or ``FORMAT``), or one for other inputs -> the
  pair is calibrated and the file rewritten;
* a file that cannot be read (corrupt, truncated, locked) -> moved aside,
  calibrated again;
* the cache cannot be written (read-only folder, full disk) -> the pair is
  still calibrated for the session; only the saving is skipped, noted;
* files are written to a temporary name and renamed into place, so another
  session never reads a half-written file.
"""
import hashlib
import os
import pickle
import tempfile
from collections import namedtuple
from datetime import datetime, timedelta
from pathlib import Path

DEFAULT_DIR = Path(__file__).resolve().parents[1] / "data" / "local" / "calibration_cache"
TTL = timedelta(hours=24)
#: Bump when the file layout changes.
FORMAT = 1

#: ``source``: ``disk`` (restored), ``calibrated`` (and saved) or ``session``
#: (calibrated, not saved); ``notes`` say why a file was not used or written.
Calibration = namedtuple("Calibration", "pair source notes")


def _code_version():
    from app.smile_service import SURFACE_CACHE_VERSION
    return SURFACE_CACHE_VERSION


class CalibrationCache:
    def __init__(self, directory=DEFAULT_DIR, ttl=TTL, now=datetime.now):
        self.directory, self.ttl, self.now = Path(directory), ttl, now

    @staticmethod
    def identity(market_id, pair, valuation_date, settings):
        import pandas as pd
        return (FORMAT, _code_version(), int(market_id), str(pair),
                pd.Timestamp(valuation_date).strftime("%Y-%m-%d"), repr(settings))

    def path(self, identity):
        digest = hashlib.sha1(repr(identity).encode()).hexdigest()[:16]
        _, _, market_id, pair, valuation, _ = identity
        return self.directory / f"m{market_id}_{pair}_{valuation}_{digest}.pkl"

    def load(self, identity):
        """``(entries, note)``: the cached entries, or ``None`` with why not."""
        path = self.path(identity)
        if not path.exists():
            return None, None
        try:
            with open(path, "rb") as handle:
                record = pickle.load(handle)
            created = datetime.fromisoformat(record["created"])
            stored_identity = record["identity"]
            entries = record["surface"], record["service"]
        except Exception as exc:                # corrupt, truncated, locked, foreign
            self._discard(path)
            return None, f"{path.name} unreadable ({type(exc).__name__}); recalibrated"
        if stored_identity != identity:
            return None, f"{path.name} is for other inputs; recalibrated"
        if self.now() - created > self.ttl:
            return None, f"{path.name} older than {self.ttl}; recalibrated"
        return entries, None

    def save(self, identity, surface_entries, service_entries):
        """Write atomically; returns ``None`` or a note when it could not."""
        path = self.path(identity)
        record = {"identity": identity, "created": self.now().isoformat(timespec="seconds"),
                  "surface": surface_entries, "service": service_entries}
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            handle, temporary = tempfile.mkstemp(dir=str(self.directory), suffix=".tmp")
            try:
                with os.fdopen(handle, "wb") as out:
                    pickle.dump(record, out, protocol=4)
                os.replace(temporary, str(path))
            finally:
                if os.path.exists(temporary):
                    os.remove(temporary)
        except Exception as exc:
            return f"calibration cache not saved ({type(exc).__name__}: {exc})"
        self.purge()
        return None

    def purge(self):
        """Remove expired files; never raises."""
        try:
            limit = (self.now() - self.ttl).timestamp()
            for path in self.directory.glob("*.pkl"):
                if path.stat().st_mtime < limit:
                    path.unlink()
        except Exception:
            pass

    @staticmethod
    def _discard(path):
        try:
            path.replace(path.with_suffix(".bad"))
        except Exception:
            pass


def ensure_calibrated(payload, market_id, valuation_date, settings, surface_cache,
                      market_service, cache=None):
    """Make ``payload``'s pair calibrated in the session caches, from disk when
    possible; returns a ``Calibration``. Raises only what calibrating the pair
    itself raises (the caller reports that pair as unavailable)."""
    from app.market_context import MarketContext
    from app.market_service import MarketService
    from app.smile_service import DiagnosticsCollector, SurfaceCache, bind_cache

    cache = cache or CalibrationCache()
    pair = payload["currency_pair"]
    identity = cache.identity(market_id, pair, valuation_date, settings)
    entries, note = cache.load(identity)
    notes = [note] if note else []
    if entries is not None:
        surface_entries, service_entries = entries
        surface_cache.merge(surface_entries)
        market_service.cache.update(service_entries)
        return Calibration(pair, "disk", notes)

    diagnostics = DiagnosticsCollector()
    private = SurfaceCache()
    bind_cache(diagnostics.token, private)
    service = MarketService(diagnostics)
    context = MarketContext.from_payload(payload, settings, market_service=service,
                                         diagnostics=diagnostics,
                                         valuation_date=valuation_date)
    for expiry in context.pillar_expiries():
        context.calibrate_pillar(expiry)
    surface_entries, service_entries = private.entries(), list(service.cache.items())
    surface_cache.merge(surface_entries)
    market_service.cache.update(service_entries)
    failed = cache.save(identity, surface_entries, service_entries)
    if failed:
        return Calibration(pair, "session", notes + [failed])
    return Calibration(pair, "calibrated", notes)

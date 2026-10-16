"""Calibration cache on disk: restore instead of recalibrating, and every fallback."""
import os
import pickle
import warnings
from datetime import datetime, timedelta

import pandas as pd
import pytest

import app.smile_service as smile_service
from app.calibration_cache import CalibrationCache, ensure_calibrated
from app.market_context import MarketContext, PricingSettings
from app.market_service import MarketService
from app.smile_service import DiagnosticsCollector, SurfaceCache, bind_cache
from app.store_markets import load_store_markets
from market_data.store import MarketDataStore
from test_discount_curves import SURFACE, USD, points

pytestmark = pytest.mark.filterwarnings("ignore")
TODAY = "2026-10-16"
SETTINGS = PricingSettings(pricing_model="vv")


@pytest.fixture(scope="module")
def payload(tmp_path_factory):
    store = MarketDataStore(tmp_path_factory.mktemp("store") / "market_data.sqlite")
    store.save(f"{TODAY}T12:00:00", "bloomberg", spots={"EURUSD": 1.12},
               depo_curves={"USD": USD}, vols={"EURUSD": SURFACE},
               forward_points={"EURUSD": points("EURUSD", TODAY, 1.12, 0.02, 0.04, 10000)})
    return load_store_markets(TODAY, store=store).payloads["EURUSD"]


class Session:
    """A fresh app session's caches."""
    def __init__(self):
        self.diagnostics = DiagnosticsCollector()
        self.surface = SurfaceCache()
        bind_cache(self.diagnostics.token, self.surface)
        self.service = MarketService(self.diagnostics)

    def vol(self, payload, expiry="2026-12-01"):
        context = MarketContext.from_payload(payload, SETTINGS, market_service=self.service,
                                             diagnostics=self.diagnostics, valuation_date=TODAY)
        return context.market_for_expiry(pd.Timestamp(expiry)).v_atm


@pytest.fixture
def calibrations(monkeypatch):
    """Counts actual smile calibrations."""
    calls = []
    real = smile_service.prepare_smile

    def counting(*args, **kwargs):
        calls.append(args[1])
        return real(*args, **kwargs)
    monkeypatch.setattr(smile_service, "prepare_smile", counting)
    return calls


def run(payload, cache, session=None):
    session = session or Session()
    result = ensure_calibrated(payload, 7, TODAY, SETTINGS, session.surface, session.service,
                               cache)
    return result, session


def test_first_session_calibrates_and_saves_the_next_restores(payload, tmp_path, calibrations):
    cache = CalibrationCache(tmp_path)
    first, session = run(payload, cache)
    assert first.source == "calibrated" and calibrations
    files = list(tmp_path.glob("*.pkl"))
    assert len(files) == 1 and files[0].name.startswith("m7_EURUSD_2026-10-16_")
    vol = session.vol(payload)

    calibrations.clear()
    second, fresh = run(payload, cache)
    assert second.source == "disk" and second.notes == []
    assert fresh.vol(payload) == vol                  # the same smile, restored
    assert calibrations == []                         # and nothing calibrated again
    assert not list(tmp_path.glob("*.tmp"))           # written atomically


def test_other_inputs_never_share_a_file(payload, tmp_path):
    cache = CalibrationCache(tmp_path)
    base = cache.identity(7, "EURUSD", TODAY, SETTINGS)
    for other in (cache.identity(8, "EURUSD", TODAY, SETTINGS),
                  cache.identity(7, "EURUSD", "2026-10-19", SETTINGS),
                  cache.identity(7, "EURUSD", TODAY, PricingSettings(pricing_model="vv-sabr"))):
        assert cache.path(other) != cache.path(base)


def test_an_expired_file_is_recalibrated(payload, tmp_path, calibrations):
    run(payload, CalibrationCache(tmp_path))
    later = CalibrationCache(tmp_path, now=lambda: datetime.now() + timedelta(hours=25))
    calibrations.clear()
    result, _ = run(payload, later)
    assert result.source == "calibrated" and calibrations
    assert "older than 1 day" in result.notes[0]


def test_a_file_from_other_code_is_recalibrated(payload, tmp_path, monkeypatch, calibrations):
    run(payload, CalibrationCache(tmp_path))
    monkeypatch.setattr(smile_service, "SURFACE_CACHE_VERSION", 99)
    calibrations.clear()
    result, _ = run(payload, CalibrationCache(tmp_path))
    assert result.source == "calibrated" and calibrations      # new version, new file
    assert len(list(tmp_path.glob("*.pkl"))) == 2


def test_a_corrupt_file_is_set_aside_and_recalibrated(payload, tmp_path, calibrations):
    cache = CalibrationCache(tmp_path)
    run(payload, cache)
    path = next(tmp_path.glob("*.pkl"))
    path.write_bytes(b"\x80\x04not a pickle at all")
    calibrations.clear()
    result, session = run(payload, cache)
    assert result.source == "calibrated" and calibrations
    assert "unreadable" in result.notes[0]
    assert path.with_suffix(".bad").exists() and path.exists()      # replaced by a good one
    assert run(payload, cache)[0].source == "disk"


def test_a_file_for_other_inputs_is_not_used(payload, tmp_path, calibrations):
    cache = CalibrationCache(tmp_path)
    run(payload, cache)
    path = next(tmp_path.glob("*.pkl"))
    record = pickle.loads(path.read_bytes())
    record["identity"] = record["identity"][:2] + (99,) + record["identity"][3:]
    path.write_bytes(pickle.dumps(record))
    calibrations.clear()
    result, _ = run(payload, cache)
    assert result.source == "calibrated" and calibrations and "other inputs" in result.notes[0]


def test_an_unwritable_cache_still_calibrates_for_the_session(payload, tmp_path):
    blocked = tmp_path / "not_a_folder"
    blocked.write_text("a file where the cache folder should be")
    result, session = run(payload, CalibrationCache(blocked))
    assert result.source == "session" and "not saved" in result.notes[0]
    assert session.vol(payload) > 0                    # pricing unaffected


def test_expired_files_are_purged(payload, tmp_path):
    stale = tmp_path / "m1_EURUSD_2026-10-01_0000.pkl"
    stale.write_bytes(b"old")
    old = (datetime.now() - timedelta(days=3)).timestamp()
    os.utime(str(stale), (old, old))
    run(payload, CalibrationCache(tmp_path))
    assert not stale.exists()

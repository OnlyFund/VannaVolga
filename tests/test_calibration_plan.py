"""Up-front calibration: only the pairs in use, only what changed, failures isolated."""
from types import SimpleNamespace

import pandas as pd
import pytest

import app.calibration_plan as plan
import app.smile_service as smile_service
from app.calibration_cache import CalibrationCache
from app.calibration_plan import calibrate_pairs, pairs_in_use
from app.market_context import PricingSettings
from app.market_service import MarketService
from app.market_set import MarketSet
from app.smile_service import DiagnosticsCollector, SurfaceCache, bind_cache
from golden.app_harness import load_markets

pytestmark = pytest.mark.filterwarnings("ignore")
VALUATION = "2026-10-15"
SETTINGS = PricingSettings(pricing_model="vv")
PAYLOADS = load_markets("MARKETS_20261015.json")          # EURUSD, USDJPY, USDMXN


class Session:
    def __init__(self):
        self.diagnostics = DiagnosticsCollector()
        self.surface = SurfaceCache()
        bind_cache(self.diagnostics.token, self.surface)
        self.service = MarketService(self.diagnostics)
        self.markets = MarketSet.from_payloads(PAYLOADS, VALUATION, SETTINGS,
                                               market_service=self.service,
                                               diagnostics=self.diagnostics)
        self.done = {}

    def calibrate(self, pairs, fingerprints=None, **kwargs):
        fingerprints = fingerprints or {pair: "v1" for pair in pairs}
        return calibrate_pairs(self.markets, pairs, fingerprints, self.done, SETTINGS,
                               VALUATION, self.surface, self.service, **kwargs)


@pytest.fixture
def calibrated(monkeypatch):
    """The pairs each actual smile calibration was for."""
    pairs = []
    real = smile_service.prepare_smile

    def counting(market, *args, **kwargs):
        pairs.append(market.currency_pair)
        return real(market, *args, **kwargs)
    monkeypatch.setattr(smile_service, "prepare_smile", counting)
    return pairs


def test_pairs_in_use_are_the_active_pair_the_pricer_lines_and_the_selected_trades():
    rows = pd.DataFrame({"currency_pair": ["usdjpy", "", None, "GBPUSD"]})
    trades = [SimpleNamespace(currency_pair="USDMXN")]
    available = ["EURUSD", "USDJPY", "USDMXN"]
    assert pairs_in_use(available, "eur/usd", rows, trades) == ["EURUSD", "USDJPY", "USDMXN"]
    assert pairs_in_use(available, "EURUSD") == ["EURUSD"]          # one pair: one pair
    assert pairs_in_use(available, "GBPUSD") == []                   # not loaded: skipped


def test_only_the_pairs_in_use_are_calibrated(calibrated):
    session = Session()
    outcomes = session.calibrate(["EURUSD"])
    assert [(o.pair, o.source) for o in outcomes] == [("EURUSD", "session")]
    assert set(calibrated) == {"EURUSD"}                    # USDJPY, USDMXN untouched


def test_adding_a_pair_calibrates_only_that_pair(calibrated):
    session = Session()
    session.calibrate(["EURUSD"])
    calibrated.clear()
    outcomes = session.calibrate(["EURUSD", "USDJPY"])
    assert [o.pair for o in outcomes] == ["USDJPY"] and set(calibrated) == {"USDJPY"}


def test_a_changed_market_recalibrates_only_itself(calibrated):
    session = Session()
    session.calibrate(["EURUSD", "USDJPY"])
    calibrated.clear()
    outcomes = session.calibrate(["EURUSD", "USDJPY"], {"EURUSD": "v1", "USDJPY": "v2"})
    assert [o.pair for o in outcomes] == ["USDJPY"]


def test_a_failing_pair_is_reported_and_the_others_still_calibrate(calibrated, monkeypatch):
    session = Session()
    jpy = session.markets.market("USDJPY")
    monkeypatch.setattr(jpy, "calibrate_pillar",
                        lambda expiry: (_ for _ in ()).throw(ZeroDivisionError("float division by zero")))
    outcomes = {o.pair: o for o in session.calibrate(["EURUSD", "USDJPY", "USDMXN"])}
    assert outcomes["USDJPY"].source == "failed"
    assert "float division by zero" in outcomes["USDJPY"].notes[0]
    assert outcomes["EURUSD"].source == outcomes["USDMXN"].source == "session"
    assert session.done["USDJPY"] == "v1"            # not retried on every rerun
    assert session.calibrate(["USDJPY"]) == []


def test_store_markets_go_through_the_disk_cache(calibrated, tmp_path):
    cache = CalibrationCache(tmp_path)
    store_routing = dict(origins={"EURUSD": 7}, payloads=PAYLOADS, cache=cache)
    first = Session().calibrate(["EURUSD", "USDJPY"], **store_routing)
    assert {o.pair: o.source for o in first} == {"EURUSD": "calibrated", "USDJPY": "session"}
    calibrated.clear()
    second = Session().calibrate(["EURUSD"], **store_routing)     # a new session
    assert [o.source for o in second] == ["disk"] and calibrated == []


def test_a_broken_disk_cache_falls_back_to_the_session(calibrated, monkeypatch, tmp_path):
    import app.calibration_cache as calibration_cache
    monkeypatch.setattr(calibration_cache, "ensure_calibrated",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("disk gone")))
    outcomes = Session().calibrate(["EURUSD"], origins={"EURUSD": 7}, payloads=PAYLOADS,
                                   cache=CalibrationCache(tmp_path))
    assert outcomes[0].source == "session" and "disk cache bypassed" in outcomes[0].notes[0]
    assert set(calibrated) == {"EURUSD"}

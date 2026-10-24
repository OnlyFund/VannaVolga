"""A fetch that cannot price (a weekend's: depo curves only) is not saved, and
"Latest" skips one that is stored anyway; every junction falls back safely."""
import pytest

from app.store_markets import load_store_markets
from app.usd_spots import latest_snapshot
from market_data.store import MarketDataStore
from test_discount_curves import USD
from test_store_markets import TODAY, full_snapshot

SATURDAY = "2026-10-17"


@pytest.fixture
def store(tmp_path):
    return MarketDataStore(tmp_path / "market_data.sqlite")


def weekend_snapshot(store, as_of=f"{SATURDAY}T10:05:59"):
    """What Bloomberg gives on a Saturday: depo curves, nothing else."""
    return store.save(as_of, "bloomberg", depo_curves={"USD": USD})


# -- the store ---------------------------------------------------------------

def test_priceable_loads_skip_markets_without_vols_or_forward_points(store):
    good = full_snapshot(store)
    weekend_snapshot(store)
    full_snapshot(store, as_of=f"{SATURDAY}T11:00:00", forward_points={})    # vols, no points
    assert store.load().market_id != good                    # the default is unchanged
    assert store.load(priceable=True).market_id == good
    assert store.load(on_or_before=SATURDAY, priceable=True).market_id == good


def test_priceable_finds_nothing_when_no_market_can_price(store):
    weekend_snapshot(store)
    assert store.load(priceable=True) is None and store.load() is not None


# -- "Latest" in the app -----------------------------------------------------

def test_latest_skips_newer_markets_that_cannot_price_and_says_so(store):
    good = full_snapshot(store)
    first, second = weekend_snapshot(store), weekend_snapshot(store, f"{SATURDAY}T10:06:09")
    loaded = load_store_markets("2026-10-19", store=store)
    assert loaded.problem is None and loaded.snapshot.market_id == good
    assert sorted(loaded.payloads) == ["EURUSD", "USDJPY"]
    assert loaded.notes[0] == (
        f"markets {second}, {first} skipped: no vol surfaces or forward points; using the "
        f"latest market that can price, market {good} (bloomberg, {TODAY}T12:00:00)")


def test_latest_with_no_priceable_market_reports_the_newest_as_before(store):
    weekend = weekend_snapshot(store)
    loaded = load_store_markets("2026-10-19", store=store)
    assert loaded.payloads == {} and loaded.snapshot.market_id == weekend
    assert "has no vol surfaces or forward points" in loaded.problem


def test_a_chosen_market_is_never_swapped_for_another(store):
    full_snapshot(store)
    weekend = weekend_snapshot(store)
    loaded = load_store_markets("2026-10-19", market_id=weekend, store=store)
    assert loaded.snapshot.market_id == weekend and loaded.payloads == {}
    assert "has no vol surfaces or forward points" in loaded.problem


def test_a_latest_that_can_price_has_no_skip_note(store):
    full_snapshot(store)
    assert not any("skipped" in note for note in load_store_markets(TODAY, store=store).notes)


def test_usd_spots_come_from_a_market_that_can_price(store):
    good = full_snapshot(store)
    weekend_snapshot(store)
    assert latest_snapshot(SATURDAY, store=store).market_id == good
    alone = MarketDataStore(store.path.parent / "alone.sqlite")
    weekend = weekend_snapshot(alone)
    assert latest_snapshot(SATURDAY, store=alone).market_id == weekend    # fallback


# -- the fetch ---------------------------------------------------------------

@pytest.fixture
def fetch(monkeypatch):
    module = pytest.importorskip("market_data.bloomberg_fetch")

    def returns(spots=None, curves=None, surfaces=None, forwards=None):
        monkeypatch.setattr(module, "fetch_spots", lambda date: spots or {})
        monkeypatch.setattr(module, "fetch_depo_curves", lambda date: curves or {})
        monkeypatch.setattr(module, "fetch_vols", lambda date: surfaces or {})
        monkeypatch.setattr(module, "fetch_forward_points", lambda date: forwards or {})
    module.returns = returns
    return module


def test_what_makes_a_fetch_unusable(fetch):
    assert fetch.unusable_fetch({}, {}, {}) == "no spots and no vol surfaces and no forward points"
    assert fetch.unusable_fetch({"EURUSD": 1.1}, {"EURUSD": {}}, {}) == "no forward points"
    assert fetch.unusable_fetch({"EURUSD": 1.1}, {"EURUSD": {}}, {"EURUSD": {}}) is None
    assert fetch.unusable_fetch({"EURUSD": 1.1}, {}, {}, vols=False) is None   # --no-vols
    assert fetch.unusable_fetch({}, {}, {}, vols=False) == "no spots"


def test_a_weekend_fetch_is_not_saved(fetch, store, capsys):
    fetch.returns(curves={"USD": USD})
    assert fetch.fetch_and_store(SATURDAY, store=store) is None
    assert store.snapshots().empty
    assert "Not saved: the fetch for 2026-10-17 has no spots" in capsys.readouterr().out


def test_force_saves_it_anyway_and_says_so_on_the_market(fetch, store):
    fetch.returns(curves={"USD": USD})
    market_id = fetch.fetch_and_store(SATURDAY, store=store, force=True)
    assert store.load(market_id=market_id).note.startswith("saved with --force: no spots")


def test_a_good_fetch_is_saved_with_no_note(fetch, store):
    from test_discount_curves import SURFACE
    fetch.returns(spots={"EURUSD": 1.12}, curves={"USD": USD}, surfaces={"EURUSD": SURFACE},
                  forwards={"EURUSD": {"1M": 14.0}})
    market_id = fetch.fetch_and_store(TODAY, store=store)
    assert market_id is not None and store.load(market_id=market_id).note == ""


def test_nothing_saved_skips_the_dates_and_the_prewarm_and_fails(fetch):
    calls = []
    code = fetch.main([], fetch=lambda day, vols, force: calls.append(("market", force)),
                      store_dates=lambda: calls.append(("dates",)), today="2026-10-24",
                      warm=lambda: calls.append(("warm",)))
    assert code == 1 and calls == [("market", False)]
    fetch.main(["--force"], fetch=lambda day, vols, force: calls.append(("market", force)) or 1,
               store_dates=lambda: None, today="2026-10-24", warm=lambda: None)
    assert calls[-1] == ("market", True)

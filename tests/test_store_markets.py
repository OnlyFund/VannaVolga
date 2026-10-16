"""Markets built from the store: every junction fails safe, with a reason."""
import pytest

from app.store_markets import load_store_markets
from market_data.store import MarketDataStore
from test_discount_curves import SURFACE, USD, points

TODAY = "2026-10-16"
EURUSD_POINTS = points("EURUSD", TODAY, 1.12, 0.02, 0.04, 10000)
USDJPY_POINTS = points("USDJPY", TODAY, 150.0, 0.04, 0.01, 100)


@pytest.fixture
def store(tmp_path):
    return MarketDataStore(tmp_path / "market_data.sqlite")


def full_snapshot(store, valuation=TODAY, as_of=None, **changes):
    data = dict(spots={"EURUSD": 1.12, "USDJPY": 150.0}, depo_curves={"USD": USD},
                vols={"EURUSD": SURFACE, "USDJPY": SURFACE},
                forward_points={"EURUSD": EURUSD_POINTS, "USDJPY": USDJPY_POINTS})
    data.update(changes)
    return store.save(as_of or f"{valuation}T12:00:00", "bloomberg", **data)


def test_builds_every_buildable_pair_and_gives_a_reason_for_the_rest(store):
    market_id = full_snapshot(store, spots={"EURUSD": 1.12})          # no USDJPY spot
    loaded = load_store_markets(TODAY, store=store)
    assert loaded.problem is None and loaded.snapshot.market_id == market_id
    assert list(loaded.payloads) == ["EURUSD"]
    assert loaded.unavailable == {"USDJPY": "USDJPY: no spot in market %d" % market_id}


def test_markets_load_into_the_apps_market_set(store):
    from app.market_set import MarketSet
    full_snapshot(store)
    loaded = load_store_markets(TODAY, store=store)
    markets = MarketSet.from_payloads(loaded.payloads, TODAY)
    assert sorted(markets.available()) == ["EURUSD", "USDJPY"] and not markets.problems()


def test_an_unreadable_store_is_a_problem_not_a_crash(tmp_path, monkeypatch):
    broken = tmp_path / "broken.sqlite"
    broken.write_bytes(b"this is not a database" * 100)
    monkeypatch.setenv("VV_MARKET_STORE", str(broken))
    loaded = load_store_markets(TODAY)
    assert loaded.payloads == {} and loaded.problem.startswith("market data store unavailable")


def test_no_snapshot_for_the_date(store):
    loaded = load_store_markets(TODAY, store=store)
    assert loaded.payloads == {} and "no market data on or before 2026-10-16" in loaded.problem


def test_latest_snapshot_too_old_is_refused_but_can_be_chosen(store):
    old_id = full_snapshot(store, valuation="2026-10-05")
    refused = load_store_markets(TODAY, store=store)
    assert refused.payloads == {} and "more than 5 business days" in refused.problem
    recent = load_store_markets("2026-10-08", store=store)     # 3 business days: fine
    assert recent.problem is None and "valued as of 2026-10-08" in recent.notes[0]
    chosen = load_store_markets(TODAY, market_id=old_id, store=store)
    assert chosen.problem is None and list(chosen.payloads) == ["EURUSD", "USDJPY"]
    assert chosen.notes[0] == (f"market {old_id} (bloomberg, 2026-10-05T12:00:00) "
                               "valued as of 2026-10-16")


def test_chosen_snapshot_missing_or_for_a_later_date(store):
    later_id = full_snapshot(store, valuation="2026-10-19")
    assert load_store_markets(TODAY, market_id=999, store=store).problem == (
        "market 999 does not exist")
    future = load_store_markets(TODAY, market_id=later_id, store=store)
    assert future.payloads == {} and "after the valuation date" in future.problem
    assert "future market data cannot be used" in future.problem


def test_a_snapshot_without_vols_or_forwards(store):
    full_snapshot(store, vols={}, forward_points={})
    loaded = load_store_markets(TODAY, store=store)
    assert loaded.payloads == {} and "no vol surfaces or forward points" in loaded.problem


def test_one_pair_raising_does_not_stop_the_others(store, monkeypatch):
    import market_data.market_tables as tables
    real = tables.market

    def flaky(snapshot, pair, **kwargs):
        if pair == "USDJPY":
            raise ZeroDivisionError("float division by zero")
        return real(snapshot, pair, **kwargs)
    monkeypatch.setattr(tables, "market", flaky)
    full_snapshot(store)
    loaded = load_store_markets(TODAY, store=store)
    assert list(loaded.payloads) == ["EURUSD"]
    assert "could not be built" in loaded.unavailable["USDJPY"]
    assert "float division by zero" in loaded.unavailable["USDJPY"]


def test_a_built_market_failing_validation_is_unavailable(store, monkeypatch):
    import market_data.market_tables as tables
    real = tables.market

    def broken(snapshot, pair, **kwargs):
        payload, notes = real(snapshot, pair, **kwargs)
        if pair == "USDJPY":
            payload = dict(payload, spot=-1.0)
        return payload, notes
    monkeypatch.setattr(tables, "market", broken)
    full_snapshot(store)
    loaded = load_store_markets(TODAY, store=store)
    assert list(loaded.payloads) == ["EURUSD"]
    assert loaded.unavailable["USDJPY"].startswith("USDJPY failed market validation")


def test_no_buildable_pair_is_a_problem(store):
    full_snapshot(store, depo_curves={})                   # no SOFR: no discount rates
    loaded = load_store_markets(TODAY, store=store)
    assert loaded.payloads == {} and loaded.problem.startswith("no pair could be built")
    assert set(loaded.unavailable) == {"EURUSD", "USDJPY"}

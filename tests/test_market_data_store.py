"""Market data snapshots: save, read back by datetime and valuation date, history."""
import json

import pytest

from market_data.store import DEPO, SPOT, MarketDataStore


@pytest.fixture
def store(tmp_path):
    return MarketDataStore(tmp_path / "market_data.sqlite")


def test_snapshots_read_back_by_datetime_and_valuation_date(store):
    assert store.load() is None
    first = store.save("2026-10-15", "2026-10-15T17:30:00", spots={"EURUSD": 1.12},
                       depo_curves={"USD": {"1W": 0.0389, "1M": 0.0393}}, source="bloomberg")
    second = store.save("2026-10-16", "2026-10-16T09:00:00", spots={"EURUSD": 1.13},
                        depo_curves={"USD": {"1W": 0.0390}})
    third = store.save("2026-10-16", "2026-10-16T11:00:00", spots={"EURUSD": 1.14})

    latest = store.load()
    assert (latest.id, latest.spots, latest.depo_curves) == (third, {"EURUSD": 1.14}, {})
    assert store.load(as_of="2026-10-16T10:00:00").id == second
    assert store.load(as_of="2026-10-16 08:00").id == first
    assert store.load(as_of="2026-10-15T00:00:00") is None
    assert store.load(valuation_date="2026-10-15").depo_curves == {
        "USD": {"1W": 0.0389, "1M": 0.0393}}
    assert store.load(valuation_date="2026-10-17") is None
    assert store.load(valuation_date="2026-10-17", on_or_before=True).id == third
    assert store.load(valuation_date="2026-10-16", as_of="2026-10-16T10:00:00").id == second

    listed = store.snapshots()
    assert list(listed["id"]) == [third, second, first]
    assert list(listed["quotes"]) == [1, 2, 3]
    assert list(store.snapshots("2026-10-15")["id"]) == [first]

    assert list(store.history(SPOT, "EURUSD")["value"]) == [1.12, 1.13, 1.14]
    assert list(store.history(DEPO, "USD", "1W")["value"]) == [0.0389, 0.0390]
    assert list(store.history(SPOT, "EURUSD", start="2026-10-16")["value"]) == [1.13, 1.14]


def test_json_files_import_into_one_snapshot(store, tmp_path):
    stamp = {"valuation_date": "2026-10-16", "as_of": "2026-10-16T11:21:13"}
    spots = tmp_path / "usd_spots.json"
    spots.write_text(json.dumps({"format": "usd_spots", **stamp, "spots": {"EURUSD": 1.12}}))
    depos = tmp_path / "depo_curves.json"
    depos.write_text(json.dumps({"format": "depo_curves", **stamp,
                                 "curves": {"USD": {"1W": 0.0389}}}))
    assert store.import_json(spots) == store.import_json(depos)
    snapshot = store.load()
    assert snapshot.spots == {"EURUSD": 1.12}
    assert snapshot.depo_curves == {"USD": {"1W": 0.0389}}
    bad = tmp_path / "other.json"
    bad.write_text(json.dumps({"format": "market", **stamp}))
    with pytest.raises(ValueError):
        store.import_json(bad)


def test_app_reads_stored_market_data_without_loading_bloomberg():
    # The app prices from the store; only market_data.bloomberg_fetch talks to Bloomberg.
    import subprocess
    import sys
    code = ("import sys, app.ui.app, app.ui.trades_block, app.usd_spots, market_data.store; "
            "print(sorted(n for n in sys.modules if n.split('.')[0] in "
            "('xbbg', 'blpapi') or n == 'market_data.bloomberg_fetch'))")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            timeout=300)
    assert result.stdout.strip().splitlines()[-1] == "[]", result.stderr[-2000:]


def test_vol_surfaces_round_trip(store):
    from market_data.store import VOL_FIELDS
    surface = {"1W": dict(zip(VOL_FIELDS, (6.4, -1.3, 0.21, -2.4, 0.7))),
               "1M": dict(zip(VOL_FIELDS, (6.1, -1.2, 0.20, -2.3, 0.69)))}
    store.save("2026-10-16", "2026-10-16T12:00:00", spots={"EURUSD": 1.12},
               vols={"EURUSD": surface})
    snapshot = store.load()
    assert snapshot.vols == {"EURUSD": surface}
    assert list(store.history("vol_atm", "EURUSD", "1M")["value"]) == [6.1]
    assert store.snapshots()["quotes"].iloc[0] == 11
    with pytest.raises(ValueError):
        store.save("2026-10-16", vols={"EURUSD": {"1M": {"atm_vol": 6.1}}})


def test_forward_points_round_trip(store):
    store.save("2026-10-16", "2026-10-16T12:00:00",
               forward_points={"EURUSD": {"1M": 14.18, "1Y": 178.8}, "USDJPY": {"1M": -38.6}})
    snapshot = store.load()
    assert snapshot.forward_points == {"EURUSD": {"1M": 14.18, "1Y": 178.8},
                                       "USDJPY": {"1M": -38.6}}
    assert list(store.history("fwd_pts", "EURUSD", "1Y")["value"]) == [178.8]

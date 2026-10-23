"""Markets built from the store use the day's live Bloomberg dates, with the solver as fallback."""
from datetime import date

import pytest

from apeiron.gaia.market_dates import pillars
from app.store_markets import live_pillar_dates, load_store_markets
from helper_functions.fx_conventions import settlement
from market_data.store import MarketDataStore
from test_store_markets import TODAY, full_snapshot      # TODAY: Fri 16 Oct 2026

# Bloomberg's dates for the day: 1M a day before the solver's 18 Nov, 1W as solved.
BBG = [("EURUSD", "1M", "2026-11-17", "2026-11-20", "bloomberg", "2026-11-18", "2026-11-20"),
       ("EURUSD", "1W", "2026-10-23", "2026-10-27", "bloomberg", "2026-10-23", "2026-10-27")]


@pytest.fixture
def store(tmp_path):
    store = MarketDataStore(tmp_path / "market_data.sqlite")
    full_snapshot(store)
    return store


def _expiries(loaded, pair="EURUSD"):
    return {row["tenor"]: row["expiry"][:10] for row in loaded.payloads[pair]["vol_data"]}


def _save(store, rows=BBG, fx_date=TODAY, refused=""):
    return store.save_pillar_dates(fx_date, f"{fx_date}T08:00:00", "Europe/London", "bloomberg",
                                   rows, refused=refused)


def test_without_live_dates_every_tenor_is_solved(store):
    loaded = load_store_markets(TODAY, store=store)
    assert _expiries(loaded)["1M"] == "2026-11-18"
    assert not any("live pillar dates" in note for note in loaded.notes)


def test_the_days_live_expiries_win_and_the_rest_are_solved(store):
    fetch_id = _save(store)
    loaded = load_store_markets(TODAY, store=store)
    expiries = _expiries(loaded)
    assert expiries["1M"] == "2026-11-17" and expiries["1W"] == "2026-10-23"
    assert expiries["3M"] == "2027-01-18"                                    # solved
    assert f"Bloomberg fetch {fetch_id}" in " ".join(loaded.notes)
    assert any("live expiries for 1W, 1M" in note and "solved for" in note for note in loaded.notes)
    assert _expiries(loaded, "USDJPY")["1M"] == str(
        pillars.tenor_dates(settlement("USDJPY"), TODAY, "1M").expiry)              # no live dates


def test_a_trade_booked_on_the_day_resolves_its_tenor_to_the_live_expiry(store):
    from app.market_set import MarketSet
    _save(store)
    markets = MarketSet.from_payloads(load_store_markets(TODAY, store=store).payloads, TODAY)
    assert markets.market("EURUSD").resolve_market_expiry("1m") == date(2026, 11, 17)


def test_another_days_or_a_refused_fetch_is_not_used(store):
    _save(store, fx_date="2026-10-15")
    _save(store, refused="over 30% disagree")
    assert _expiries(load_store_markets(TODAY, store=store))["1M"] == "2026-11-18"


def test_an_expiry_override_wins_over_bloomberg(store, monkeypatch):
    _save(store)
    monkeypatch.setattr(pillars, "EXPIRY_OVERRIDES", {("EURUSD", date(2026, 11, 17)): False})
    loaded = load_store_markets(TODAY, store=store)
    assert _expiries(loaded)["1M"] == "2026-11-18"
    assert any("EURUSD 1M (expiry override)" in note for note in loaded.notes)


def test_the_switch_turns_them_off(store, monkeypatch):
    _save(store)
    monkeypatch.setenv("VV_LIVE_DATES", "0")
    loaded = load_store_markets(TODAY, store=store)
    assert _expiries(loaded)["1M"] == "2026-11-18"
    assert any("VV_LIVE_DATES=0" in note for note in loaded.notes)


def test_dates_failing_the_recheck_are_solved(store):
    snapshot = load_store_markets(TODAY, store=store).snapshot

    class Store:
        def load_pillar_dates(self, fx_date):
            from market_data.store import PillarDateFetch
            return PillarDateFetch(7, TODAY, f"{TODAY}T08:00:00", "Europe/London", "bloomberg", "", {
                ("EURUSD", "1M"): ("2026-11-21", "2026-11-24", "bloomberg"),     # a Saturday
                ("EURUSD", "2M"): ("2026-10-16", "2026-12-18", "bloomberg"),     # not after the FX date
                ("ABCXYZ", "1M"): ("2026-11-17", "2026-11-20", "bloomberg"),     # unknown pair
                ("EURUSD", "1W"): ("2026-10-23", "2026-10-27", "bloomberg")}, [])
    usable, notes = live_pillar_dates(Store(), snapshot)
    assert list(usable) == [("EURUSD", "1W")]
    assert "1 of 4 tenors used" in notes[0]


def test_an_unreadable_store_falls_back_to_the_solver(store):
    snapshot = load_store_markets(TODAY, store=store).snapshot

    class Broken:
        def load_pillar_dates(self, fx_date):
            raise RuntimeError("no such table: date_fetches")
    usable, notes = live_pillar_dates(Broken(), snapshot)
    assert usable == {} and "every tenor solved" in notes[0]

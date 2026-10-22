"""The FX date rollover: 17:00 New York (apeiron.gaia.market_dates.rollover)."""
from datetime import date

import pandas as pd
import pytest

from apeiron.gaia.market_dates.rollover import cutoff, fx_date, fx_today


@pytest.mark.parametrize("moment, expected", [
    ("2026-10-21 20:21:27", date(2026, 10, 21)),   # 15:21 New York (market 41)
    ("2026-10-21 21:59:59", date(2026, 10, 21)),   # 16:59:59 New York
    ("2026-10-21 22:00:00", date(2026, 10, 22)),   # 17:00 New York: the next day
    ("2026-10-23 22:30:00", date(2026, 10, 26)),   # Friday evening: Monday
    ("2026-10-24 12:00:00", date(2026, 10, 26)),   # Saturday: Monday
    ("2026-10-27 20:59:59", date(2026, 10, 27)),   # clock-change gap: 4 hours
    ("2026-10-27 21:00:00", date(2026, 10, 28)),   #   (London GMT, New York EDT)
    ("2026-03-10 21:00:00", date(2026, 3, 11)),    #   (New York EDT, London GMT)
    ("2026-03-31 21:00:00", date(2026, 3, 31)),   # gap over: 5 hours again
])
def test_a_london_moment_rolls_at_17_new_york(moment, expected):
    assert fx_date(moment, "Europe/London") == expected


def test_aware_moments_and_other_zones():
    assert fx_date(pd.Timestamp("2026-10-21 16:59", tz="America/New_York")) == date(2026, 10, 21)
    assert fx_date(pd.Timestamp("2026-10-21 17:00", tz="America/New_York")) == date(2026, 10, 22)
    assert fx_date("2026-10-22 06:30", "Asia/Tokyo") == date(2026, 10, 22)        # 17:30 NY the day before
    assert fx_date("2026-10-22 05:59", "Asia/Tokyo") == date(2026, 10, 21)
    assert fx_date("2026-10-22 06:00", "Asia/Tokyo") == date(2026, 10, 22)


def test_a_clock_change_hour_does_not_fail():
    assert fx_date("2026-10-25 01:30:00", "Europe/London") == date(2026, 10, 26)   # ambiguous, Sunday
    assert fx_date("2026-03-29 01:30:00", "Europe/London") == date(2026, 3, 30)    # skipped, Sunday


def test_cutoff_is_the_last_moment_of_an_fx_day():
    assert cutoff("2026-10-21", "Europe/London") == pd.Timestamp("2026-10-21 21:59:59")
    assert cutoff("2026-10-28", "Europe/London") == pd.Timestamp("2026-10-28 20:59:59")  # gap week
    assert cutoff("2026-10-24", "Europe/London") == pd.Timestamp("2026-10-23 21:59:59")  # Saturday: Friday's
    for day in ("2026-10-21", "2026-10-28", "2026-03-10"):
        last = cutoff(day, "Europe/London")
        assert fx_date(last, "Europe/London") == pd.Timestamp(day).date()
        assert fx_date(last + pd.Timedelta(seconds=1), "Europe/London") > pd.Timestamp(day).date()


def test_fx_today_uses_the_clock_given():
    assert fx_today(pd.Timestamp("2026-10-22 23:00"), "Europe/London") == date(2026, 10, 23)


def test_the_loader_picks_snapshots_by_fx_date(tmp_path):
    from app.store_markets import fx_cutoff, with_fx_date
    from market_data.store import MarketDataStore
    store = MarketDataStore(tmp_path / "market_data.sqlite")
    before = store.save("2026-10-21T21:30:00", "bloomberg", spots={"EURUSD": 1.16})
    after = store.save("2026-10-21T22:30:00", "bloomberg", spots={"EURUSD": 1.17})   # 17:30 NY
    picked = with_fx_date(store.load(at_or_before=fx_cutoff("2026-10-21")))
    assert (picked.market_id, picked.date) == (before, "2026-10-21")
    late = with_fx_date(store.load(market_id=after))
    assert late.date == "2026-10-22" and late.date_time[:10] == "2026-10-21"


def test_the_picker_lists_snapshots_by_fx_date():
    from app.ui.market_picker import with_fx_dates
    from market_data.snapshot_filter import usable_snapshots
    frame = pd.DataFrame({"market_id": [1, 2], "market_source": ["bloomberg"] * 2,
                          "date_time": ["2026-10-21T21:30:00", "2026-10-21T22:30:00"],
                          "timezone": ["Europe/London"] * 2, "vols": [1, 1], "forwards": [1, 1]})
    listed = with_fx_dates(frame)
    assert list(listed["fx_date"]) == ["2026-10-21", "2026-10-22"]
    assert list(usable_snapshots(listed, "2026-10-21")["market_id"]) == [1]

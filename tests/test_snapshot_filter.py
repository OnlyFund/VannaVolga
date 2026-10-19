"""Choosing a stored snapshot: usable ones only, the 15:00 New York close of
business per source, and the quick filter with its failsafes."""
import pytest

from market_data.snapshot_filter import (
    apply_filter, close_minutes_away, close_of_business, parse_day, usable_snapshots,
)
from market_data.store import MarketDataStore

VOLS = {"EURUSD": {"1M": {"atm": 7.0}}}
FWDS = {"EURUSD": {"1M": 12.0}}


@pytest.fixture
def snapshots(tmp_path):
    store = MarketDataStore(tmp_path / "market_data.sqlite")

    def save(date_time, source="bloomberg", full=True):
        return store.save(date_time, source, spots={"EURUSD": 1.12},
                          vols=VOLS if full else None, forward_points=FWDS if full else None)
    save("2026-10-15T20:00:00")                        # 1: 15 Oct close
    save("2026-10-16T12:28:32")                        # 2
    save("2026-10-16T19:06:32")                        # 3: nearest the 20:00 close
    save("2026-10-16T21:30:00")                        # 4: later, but further from it
    save("2026-10-16T19:50:00", full=False)            # 5: spots only, never usable
    save("2026-10-16T15:00:00", source="import")       # 6
    save("2026-10-19T09:00:00")                        # 7: after a 16 Oct valuation
    return store.snapshots()


def ids(frame):
    return list(frame["market_id"])


def test_only_usable_snapshots_on_or_before_the_valuation_date_newest_first(snapshots):
    assert ids(usable_snapshots(snapshots, "2026-10-16")) == [4, 3, 6, 2, 1]
    assert ids(usable_snapshots(snapshots, "2026-10-19")) == [7, 4, 3, 6, 2, 1]
    assert ids(usable_snapshots(snapshots, "2026-10-14")) == []


def test_the_close_is_the_snapshot_nearest_15_00_new_york_per_source(snapshots):
    usable = usable_snapshots(snapshots, "2026-10-19")
    assert close_of_business(usable, "2026-10-16")["market_id"] == 3
    assert close_of_business(usable, "2026-10-16", "import")["market_id"] == 6
    assert close_of_business(usable, "2026-10-17") is None
    assert close_minutes_away("2026-10-16T20:00:00") == 0
    assert close_minutes_away("2026-10-27T19:00:00") == 0      # UK/US clocks out of step
    assert close_minutes_away("2026-10-15T23:59:59") == 0      # the old close stamp


@pytest.mark.parametrize("text, listed, selected", [
    ("", [7, 4, 3, 6, 2, 1], None),
    ("bloomberg", [7, 4, 3, 2, 1], None),
    ("bbg", [7, 4, 3, 2, 1], None),
    ("imp", [6], None),
    ("16oct", [4, 3, 6, 2], 3),
    ("2026-10-16", [4, 3, 6, 2], 3),
    ("10/16/2026", [4, 3, 6, 2], 3),
    ("bloomberg 16oct", [4, 3, 2], 3),
    ("16-Oct-26 import", [6], 6),
])
def test_quick_filter(snapshots, text, listed, selected):
    result = apply_filter(snapshots, text, "2026-10-19")
    assert ids(result.snapshots) == listed and result.selected == selected
    assert result.message is None


@pytest.mark.parametrize("text, valuation, message", [
    ("19oct", "2026-10-16", "after the valuation date"),
    ("17oct", "2026-10-19", "latest before it: market 4"),
    ("import 15oct", "2026-10-19", "no usable import market on 15-Oct-2026"),
    ("xyz", "2026-10-19", "'xyz' is not a source or a date (sources: bloomberg, import)"),
    ("bloomberg import", "2026-10-19", "one source at a time"),
    ("15oct 16oct", "2026-10-19", "one date at a time"),
])
def test_a_filter_that_fails_keeps_every_usable_snapshot_and_says_why(snapshots, text, valuation,
                                                                     message):
    result = apply_filter(snapshots, text, valuation)
    assert result.selected is None and message in result.message
    assert ids(result.snapshots) == ids(usable_snapshots(snapshots, valuation))


def test_undated_days_take_the_valuation_year_unless_months_ahead():
    assert str(parse_day("16oct", "2026-10-19").date()) == "2026-10-16"
    assert str(parse_day("19oct", "2026-10-16").date()) == "2026-10-19"     # refused later
    assert str(parse_day("28dec", "2027-01-05").date()) == "2026-12-28"
    assert parse_day("bloomberg", "2026-10-19") is None


def test_picker_labels_are_id_source_code_date_and_time():
    from app.ui.market_picker import label, source_code
    row = {"market_id": 24, "market_source": "bloomberg", "date_time": "2026-10-16T19:06:32"}
    assert label(row) == "[24] BBG 16-Oct 19:06"
    assert label(row, frozenset({24})) == "[24] BBG 16-Oct 19:06 · close"
    assert [source_code(s) for s in ("import", "sample", "refinitiv")] == ["IMP", "SMP", "REF"]

"""Variance days: Bloomberg's weights (Saturday 0, Sunday 0.1, holidays 0.2,
Christmas and New Year 0), holidays of both currencies, USD for a currency the
calendar lacks, and weekends-only when the calendar cannot be read."""
import pytest

from market_data.variance_calendar import (
    CALENDAR, DEFAULT_WEIGHTS, day_weights, load_holidays, pair_holidays, weighted_days,
)


@pytest.fixture(scope="module")
def audusd():
    return pair_holidays("AUDUSD")[0]


def test_a_friday_to_the_tuesday_after_mlk_day_is_1_3_days(audusd):
    # Sat 0 + Sun 0.1 + Mon 18 Jan (MLK, USD) 0.2 + Tue 1.
    assert list(day_weights("2027-01-15", "2027-01-19", audusd)) == [0.0, 0.1, 0.2, 1.0]
    assert weighted_days("2027-01-15", "2027-01-19", audusd, CALENDAR) == 4.0


def test_year_end_days_weigh_zero_and_weekends_win_over_holidays(audusd):
    # Thu 24 -> Tue 29 Dec 2026: Fri 25 (Christmas) 0, Sat 26 (AUD Boxing Day) 0,
    # Sun 0.1, Mon 28 (AUD Boxing Day observed) 0.2, Tue 1.
    assert list(day_weights("2026-12-24", "2026-12-29", audusd)) == [0.0, 0.0, 0.1, 0.2, 1.0]
    assert list(day_weights("2026-12-31", "2027-01-01", audusd)) == [0.0]


def test_both_currencies_holidays_count(audusd):
    assert weighted_days("2027-01-25", "2027-01-26", audusd) == 0.2    # Australia Day
    assert weighted_days("2027-01-17", "2027-01-18", audusd) == 0.2    # MLK (USD)
    assert weighted_days("2027-01-25", "2027-01-26", pair_holidays("EURUSD")[0]) == 1.0


def test_a_currency_without_holidays_uses_usd_and_says_so():
    holidays, notes = pair_holidays("USDCZK")
    assert notes == ["no CZK holidays in the calendar; using USD holidays"]
    assert holidays == load_holidays()["USD"]


def test_an_unreadable_calendar_leaves_weekends_and_year_end_only(tmp_path):
    holidays, notes = pair_holidays("AUDUSD", path=tmp_path / "missing.csv")
    assert holidays == frozenset() and "holiday calendar unavailable" in notes[0]
    assert weighted_days("2027-01-15", "2027-01-19", holidays) == pytest.approx(2.1)


def test_empty_and_default_weights():
    assert weighted_days("2027-01-15", "2027-01-15") == 0.0
    assert DEFAULT_WEIGHTS == (0.0, 0.1, 0.2, 0.0)

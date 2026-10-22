"""The pillar date solver (apeiron.gaia.time.pillars) against Bloomberg.

The reference is Bloomberg's historical settlement dates
(``tests/fixtures/bbg_settlement_dates_20261022.csv``, fetched with
``market_data.swap_dates``): spot, TN/SN and every 1W .. 1Y forward and option
delivery for 21 pairs on 34 edge-case dates. The solver must give every one.

Also: every caller gives the solver's dates; the old rules
(``legacy_dates``) still give HEAD's dates; the fail-safes (fallback, check
mode, coverage log) and the calendar fallback when the holiday list is
missing.
"""
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from apeiron.gaia.time import calendars, legacy_dates, pillars
from helper_functions import fx_conventions as conv

REFERENCE = Path(__file__).parent / "fixtures" / "bbg_settlement_dates_20261022.csv"
PAIRS = sorted(conv.PAIRS)
TENORS = ("ON", "1W", "2W", "3W", "1M", "2M", "3M", "6M", "9M", "1Y", "18M", "2Y")
#: 2027-12-01 .. 2028-03-31: month and year ends, Christmas, New Year, 29 Feb 2028.
DAYS = [date(2027, 12, 1) + timedelta(days=n) for n in range(122)]


def _solver_date(row):
    """The solver's answer to one Bloomberg reference row."""
    rules = conv.settlement(row.pair)
    spot = pillars.spot_date(rules, row.reference_date)
    if row.kind == "spot":
        return spot
    if row.tenor == "TN":       # a T+1 pair's spot is tomorrow: its TN ends the day after
        return spot if rules.spot_lag == 2 else pillars.next_business_day(rules, spot)
    if row.tenor == "SN":
        return pillars.next_business_day(rules, spot)
    if row.tenor == "ON":       # Bloomberg's O/N settles tomorrow (the swap's far date)
        return pillars.next_business_day(rules, row.reference_date)
    return pillars.tenor_dates(rules, row.reference_date, row.tenor).delivery


def test_every_bloomberg_settlement_date():
    reference = pd.read_csv(REFERENCE, dtype=str)
    assert len(reference) == 14178
    answers = [(row, str(_solver_date(row))) for row in reference.itertuples()]
    wrong = [(row.reference_date, row.pair, row.kind, row.tenor, row.settle_dt, got)
             for row, got in answers if got != row.settle_dt]
    assert wrong == [], f"{len(wrong)} dates differ from Bloomberg, e.g. {wrong[:5]}"


@pytest.mark.parametrize("pair, trade, spot", [
    ("EURUSD", date(2024, 7, 3), date(2024, 7, 5)),    # 4 Jul is a good intermediate day
    ("EURGBP", date(2024, 7, 3), date(2024, 7, 5)),    # crosses too
    ("EURUSD", date(2025, 1, 17), date(2025, 1, 21)),  # MLK Monday counts
    ("USDMXN", date(2024, 7, 3), date(2024, 7, 8)),    # MXN: US holidays block
    ("USDBRL", date(2024, 7, 3), date(2024, 7, 8)),
    ("EURUSD", date(2024, 11, 26), date(2024, 11, 29)),  # spot itself never on a US holiday
])
def test_usd_intermediate_day_rule(pair, trade, spot):
    assert pillars.spot_date(conv.settlement(pair), trade) == spot


def test_week_tenors_roll_following_months_modified_following():
    cnh = conv.settlement("USDCNH")
    # 1W from spot 21 Jan 2025 lands in Chinese New Year: rolls forward into February.
    assert pillars.tenor_dates(cnh, date(2025, 1, 17), "1W").delivery == date(2025, 2, 5)
    eur = conv.settlement("EURUSD")
    # Spot on the last business day of October -> 1M is the last of November.
    dates = pillars.tenor_dates(eur, date(2026, 10, 28), "1M")
    assert (dates.spot, dates.delivery) == (date(2026, 10, 30), date(2026, 11, 30))


@pytest.mark.parametrize("pair", PAIRS)
def test_callers_give_the_solver_dates(pair):
    from helper_functions.fx_discounting import delivery_date
    from market_data.market_tables import tenor_dates
    from app import curve_roll
    import QuantLib as ql
    rules = conv.settlement(pair)
    for day in DAYS[::3]:
        assert curve_roll.spot_date(pair, day) == pillars.spot_date(rules, day)
        assert curve_roll.overnight_expiry(pair, day) == pillars.next_business_day(rules, day)
        assert delivery_date(pair, day) == pillars.delivery_for_expiry(rules, day)
        assert conv.spot_date(pair, ql.Date(day.day, day.month, day.year)) == ql.Date(
            *reversed(pillars.spot_date(rules, day).timetuple()[:3]))
        for tenor in TENORS:
            new = pillars.tenor_dates(rules, day, tenor)
            expiry, delivery = tenor_dates(pair, day, tenor)
            assert (expiry.date(), delivery.date()) == (new.expiry, new.delivery), (day, tenor)


def test_value_date_tokens_use_the_solver():
    from app.spot_forwards import value_date_for
    for pair in ("EURUSD", "USDCAD", "USDJPY", "EURJPY", "USDTRY", "GBPNOK", "USDMXN"):
        rules = conv.settlement(pair)
        for day in DAYS[::5]:
            if not pillars.is_business_day(rules, day):
                continue
            spot = pillars.spot_date(rules, day)
            tomorrow = pillars.next_business_day(rules, day)
            assert value_date_for(pair, "SPOT", day) == spot
            assert value_date_for(pair, "TOM", day) == (tomorrow if tomorrow < spot else spot)
            assert value_date_for(pair, "SN", day) == pillars.next_business_day(rules, spot)
            for tenor in ("1W", "10D", "3M", "18M", "1Y"):
                assert value_date_for(pair, tenor, day) == pillars.tenor_delivery(rules, day, tenor)


def test_fx_conventions_keeps_its_quantlib_signatures():
    import QuantLib as ql
    trade = ql.Date(28, 10, 2026)
    dates = conv.option_dates("EURUSD", trade, "1M")
    assert (dates.spot, dates.delivery) == (ql.Date(30, 10, 2026), ql.Date(30, 11, 2026))
    assert conv.spot_date("EURUSD", trade) == ql.Date(30, 10, 2026)
    by_expiry = conv.option_dates("EURUSD", trade, expiry=ql.Date(28, 11, 2026))   # a Saturday
    assert (by_expiry.expiry, by_expiry.delivery) == (ql.Date(27, 11, 2026), ql.Date(1, 12, 2026))
    with pytest.raises(ValueError):
        conv.option_dates("EURUSD", trade, "ON")


def test_legacy_rules_keep_the_old_dates():
    # The old joint-calendar rule on QuantLib's calendars (HEAD before this change).
    eur = conv.settlement("EURUSD")
    assert legacy_dates.spot_date(eur, date(2024, 7, 3)) == date(2024, 7, 8)
    assert legacy_dates.spot_date(eur, date(2025, 1, 17)) == date(2025, 1, 22)
    assert legacy_dates.tenor_dates(conv.settlement("USDCNH"), date(2025, 1, 17), "1W").delivery \
        == date(2025, 1, 29)        # spot 22 Jan + 1W


def test_inputs_are_coerced_and_bad_tenors_refused():
    rules = conv.settlement("EUR/USD")
    assert pillars.spot_date(rules, pd.Timestamp("2026-10-22")) == date(2026, 10, 26)
    assert pillars.spot_date(rules, "2026-10-22") == date(2026, 10, 26)
    with pytest.raises(ValueError):
        pillars.tenor_dates(rules, date(2026, 10, 22), "1Q")


def test_a_failing_rule_falls_back_to_the_legacy_rule_and_is_recorded():
    rules = conv.settlement("EURUSD")
    before = len(pillars.FALLBACKS)
    # The solver reads single-unit tenors; the legacy rule (QuantLib) also reads "1Y6M".
    dates = pillars.tenor_dates(rules, date(2026, 10, 22), "1Y6M")
    assert dates == legacy_dates.tenor_dates(rules, date(2026, 10, 22), "1Y6M")
    assert len(pillars.FALLBACKS) == before + 1
    assert pillars.FALLBACKS[-1][0] == "tenor_delivery"     # recorded where it failed


def test_an_error_in_both_rules_is_raised_not_hidden():
    before = len(pillars.FALLBACKS)
    with pytest.raises(ValueError):
        pillars.tenor_dates(conv.settlement("EURUSD"), date(2026, 10, 22), "1Q")
    assert len(pillars.FALLBACKS) == before


def test_check_mode_records_what_the_new_rules_move(monkeypatch):
    rules = conv.settlement("EURUSD")
    monkeypatch.setattr(pillars, "CHECK", True)
    before = len(pillars.MISMATCHES)
    pillars.spot_date(rules, date(2026, 10, 22))
    assert len(pillars.MISMATCHES) == before               # same under both rules
    assert pillars.spot_date(rules, date(2024, 7, 3)) == date(2024, 7, 5)   # the new answer is kept
    assert pillars.MISMATCHES[-1][0] == "spot_date"
    assert pillars.MISMATCHES[-1][2:] == (date(2024, 7, 5), date(2024, 7, 8))


def test_days_outside_the_bloomberg_coverage_are_recorded_once(monkeypatch):
    monkeypatch.setattr(pillars, "UNCOVERED", [])
    pillars.spot_date(conv.settlement("EURUSD"), date(2026, 10, 22))
    assert pillars.UNCOVERED == []
    vnd = conv.settlement(next(pair for pair in conv.PAIRS if "VND" in pair))
    late = calendars.COVERAGE["VND"][1] + timedelta(days=30)
    pillars.spot_date(vnd, late)
    pillars.spot_date(vnd, late)
    assert pillars.UNCOVERED and all("VND" in missing for missing, _ in pillars.UNCOVERED)
    assert len(pillars.UNCOVERED) == len(set(pillars.UNCOVERED))


def test_calendars_come_from_the_bloomberg_list():
    import QuantLib as ql
    assert calendars.SOURCE.startswith("bloomberg")
    assert len(calendars.COVERAGE) == 33
    assert calendars.CALENDARS["CNH"].isHoliday(ql.Date(1, 10, 2026))       # Golden Week
    assert calendars.CALENDARS["PHP"].isHoliday(ql.Date(28, 3, 2024))       # Maundy Thursday
    assert not calendars.QL_CALENDARS["PHP"].isHoliday(ql.Date(28, 3, 2024))
    # Outside the coverage a day is QuantLib's.
    assert calendars.CALENDARS["USD"].isHoliday(ql.Date(4, 7, 2050))


def test_an_unreadable_holiday_list_falls_back_to_quantlib(tmp_path):
    found, coverage, source = calendars.load(tmp_path / "missing.csv")
    assert source.startswith("quantlib") and coverage == {}
    assert found["EUR"] is calendars.QL_CALENDARS["EUR"]


def test_recorded_events_are_bounded(monkeypatch):
    monkeypatch.setattr(pillars, "_KEEP", 3)
    log = []
    for n in range(10):
        pillars._record(log, n)
    assert log == [7, 8, 9]

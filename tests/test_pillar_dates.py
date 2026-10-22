"""The pillar date solver (apeiron.gaia.time.pillars) equals the old date rules.

Step 1 of item 6 moves the rules without changing them. The old rules are
kept verbatim in ``apeiron.gaia.time.legacy_dates``; every date the solver
gives, and every caller that now uses it, is checked against them for every
pair, on every day of a range that crosses month ends, year ends, a leap
February and the main holidays. The fail-safes (fallback to the old rule,
check mode) are tested too.
"""
from datetime import date, timedelta

import pandas as pd
import pytest

from apeiron.gaia.time import legacy_dates, pillars
from helper_functions import fx_conventions as conv

PAIRS = sorted(conv.PAIRS)
TENORS = ("ON", "1W", "2W", "3W", "1M", "2M", "3M", "6M", "9M", "1Y", "18M", "2Y")
#: 2027-12-01 .. 2028-03-31: month and year ends, Christmas, New Year, 29 Feb 2028.
DAYS = [date(2027, 12, 1) + timedelta(days=n) for n in range(122)]


@pytest.mark.parametrize("pair", PAIRS)
def test_every_rule_equals_the_legacy_rule(pair):
    rules = conv.settlement(pair)
    for day in DAYS:
        assert pillars.spot_date(rules, day) == legacy_dates.spot_date(rules, day), day
        assert pillars.next_business_day(rules, day) == legacy_dates.next_business_day(rules, day)
        assert (pillars.delivery_for_expiry(rules, day)
                == legacy_dates.delivery_for_expiry(rules, day)), day
        for tenor in TENORS:
            assert (pillars.tenor_dates(rules, day, tenor)
                    == legacy_dates.tenor_dates(rules, day, tenor)), (day, tenor)


@pytest.mark.parametrize("pair", PAIRS)
def test_callers_give_the_legacy_dates(pair):
    from helper_functions.fx_discounting import delivery_date
    from market_data.market_tables import tenor_dates
    from app import curve_roll
    rules = conv.settlement(pair)
    for day in DAYS[::3]:
        assert curve_roll.spot_date(pair, day) == legacy_dates.spot_date(rules, day)
        assert curve_roll.overnight_expiry(pair, day) == legacy_dates.next_business_day(rules, day)
        assert delivery_date(pair, day) == legacy_dates.delivery_for_expiry(rules, day)
        for tenor in TENORS:
            old = legacy_dates.tenor_dates(rules, day, tenor)
            expiry, delivery = tenor_dates(pair, day, tenor)
            assert (expiry.date(), delivery.date()) == (old.expiry, old.delivery), (day, tenor)


def test_value_date_tokens_give_the_legacy_dates():
    from app.spot_forwards import value_date_for
    for pair in ("EURUSD", "USDCAD", "USDJPY", "EURJPY", "USDTRY", "GBPNOK"):
        rules = conv.settlement(pair)
        for day in DAYS[::5]:
            if not pillars.is_business_day(rules, day):
                continue
            spot = legacy_dates.spot_date(rules, day)
            tomorrow = legacy_dates.next_business_day(rules, day)
            assert value_date_for(pair, "SPOT", day) == spot
            assert value_date_for(pair, "TOM", day) == (tomorrow if tomorrow < spot else spot)
            assert value_date_for(pair, "SN", day) == legacy_dates.next_business_day(rules, spot)
            for tenor in ("1W", "10D", "3M", "18M", "1Y"):
                assert value_date_for(pair, tenor, day) == legacy_dates.tenor_delivery(rules, day, tenor)


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


def test_known_end_of_month_case():
    # Spot on the last business day of October -> 1M is the last of November.
    dates = pillars.tenor_dates(conv.settlement("EURUSD"), date(2026, 10, 28), "1M")
    assert (dates.spot, dates.delivery) == (date(2026, 10, 30), date(2026, 11, 30))


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


def test_check_mode_records_a_difference(monkeypatch):
    rules = conv.settlement("EURUSD")
    monkeypatch.setattr(pillars, "CHECK", True)
    before = len(pillars.MISMATCHES)
    pillars.spot_date(rules, date(2026, 10, 22))
    assert len(pillars.MISMATCHES) == before               # equal: nothing recorded
    monkeypatch.setattr(legacy_dates, "spot_date", lambda settlement, trade: date(2000, 1, 1))
    result = pillars.spot_date(rules, date(2026, 10, 22))
    assert result == date(2026, 10, 26)                     # the solver's answer is kept
    assert pillars.MISMATCHES[-1][0] == "spot_date"
    assert pillars.MISMATCHES[-1][3] == date(2000, 1, 1)


def test_recorded_events_are_bounded(monkeypatch):
    monkeypatch.setattr(pillars, "_KEEP", 3)
    log = []
    for n in range(10):
        pillars._record(log, n)
    assert log == [7, 8, 9]

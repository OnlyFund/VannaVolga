"""Expiry overrides: a pair and day the market treats differently from the rule."""
from datetime import date

import pytest

from apeiron.gaia.market_dates import pillars
from helper_functions.fx_conventions import settlement


@pytest.fixture(autouse=True)
def no_overrides(monkeypatch):
    monkeypatch.setattr(pillars, "EXPIRY_OVERRIDES", {})


def test_both_closed_moves_to_the_next_good_day():
    dates = pillars.tenor_dates(settlement("GBPUSD"), "2024-05-20", "1W")   # 27 May: UK and US closed
    assert (dates.expiry, dates.delivery) == (date(2024, 5, 28), date(2024, 5, 30))


def test_an_override_applies_at_once_and_clears():
    rules = settlement("GBPUSD")
    pillars.set_expiry_override("GBP/USD", "2024-05-27", True)
    dates = pillars.tenor_dates(rules, "2024-05-20", "1W")
    assert (dates.expiry, dates.delivery) == (date(2024, 5, 27), date(2024, 5, 29))
    assert ("EURUSD", date(2024, 5, 27)) not in pillars.EXPIRY_OVERRIDES
    pillars.clear_expiry_override("GBPUSD", "2024-05-27")
    assert pillars.tenor_dates(rules, "2024-05-20", "1W").expiry == date(2024, 5, 28)


def test_an_override_can_close_an_open_day():
    rules = settlement("EURUSD")
    assert pillars.is_expiry_day(rules, "2024-05-27")          # TARGET open on Memorial Day
    pillars.set_expiry_override("EURUSD", "2024-05-27", False)
    assert pillars.tenor_dates(rules, "2024-05-20", "1W").expiry == date(2024, 5, 28)


def test_the_override_file(tmp_path):
    path = tmp_path / "expiry_overrides.csv"
    path.write_text("pair,date,expiry,reason,added_by,added_on\n"
                    "gbp/usd,2024-05-27,yes,market expires,me,2026-10-22\n"
                    "EURUSD,2024-05-27,No,,me,2026-10-22\n", encoding="utf-8")
    assert pillars.load_expiry_overrides(path) == {
        ("GBPUSD", date(2024, 5, 27)): True, ("EURUSD", date(2024, 5, 27)): False}
    assert pillars.load_expiry_overrides(tmp_path / "missing.csv") == {}


@pytest.mark.parametrize("line", ["EURUSD,2024-05-27,maybe,,,", "EURUSD,2024-05-25,yes,,,"])
def test_a_bad_override_raises(tmp_path, line):
    path = tmp_path / "expiry_overrides.csv"
    path.write_text("pair,date,expiry,reason,added_by,added_on\n" + line + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        pillars.load_expiry_overrides(path)
    with pytest.raises(ValueError):
        pillars.set_expiry_override("EURUSD", "2024-05-25", True)


def test_the_shipped_file_loads():
    assert pillars.load_expiry_overrides() is not None

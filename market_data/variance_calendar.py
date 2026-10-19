"""Variance days: how much of a normal day's variance each calendar day carries.

The vol time interpolation runs on this clock (``app.market_context``): the
quotes fix each pillar's total variance, and between pillars it is spread by
weighted days, so a Friday expiry before a Monday pillar loses only the
weekend's small weight, not three full days. The defaults are Bloomberg's:

* Saturday 0, Sunday 0.1;
* a holiday of either currency of the pair 0.2;
* Christmas Day and New Year's Day 0;
* every other day 1.

Event weights (``variance_days`` in the event calendar) are added on top by
the caller. A holiday that falls on a weekend takes the weekend's weight.
``CALENDAR`` weights every day 1: the calendar-day clock used before.

Holidays come from ``market_data/calendars/holidays.csv`` (``ccy``,
``holidayDate``, ``name``; dates ISO or MM/DD/YYYY). A currency the file does
not cover uses the USD holidays (noted).

Fail-safes: an unreadable file leaves no holidays, so only weekends and the
year-end days are weighted, and the problem is returned as a note; pricing
never stops on the calendar.
"""
import csv
from collections import namedtuple
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

HOLIDAYS_PATH = Path(__file__).resolve().parent / "calendars" / "holidays.csv"
FALLBACK_CCY = "USD"

VarianceWeights = namedtuple("VarianceWeights", "saturday sunday holiday year_end")
VarianceWeights.__doc__ = "Weights of a Saturday, a Sunday, a holiday and Dec 25 / Jan 1."
#: Bloomberg's weights, the default.
DEFAULT_WEIGHTS = VarianceWeights(saturday=0.0, sunday=0.1, holiday=0.2, year_end=0.0)
#: Every day a full day: the calendar-day clock.
CALENDAR = VarianceWeights(1.0, 1.0, 1.0, 1.0)
_YEAR_END = ((12, 25), (1, 1))


def _parse(text):
    text = text.strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"unreadable holiday date {text!r}")


@lru_cache(maxsize=4)
def load_holidays(path=HOLIDAYS_PATH):
    """``{CCY: frozenset(dates)}`` from the holiday file; raises if unreadable."""
    holidays = {}
    with open(path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            holidays.setdefault(row["ccy"].strip().upper(), set()).add(_parse(row["holidayDate"]))
    if not holidays:
        raise ValueError(f"no holidays in {path}")
    return {ccy: frozenset(days) for ccy, days in holidays.items()}


def pair_holidays(pair, path=HOLIDAYS_PATH):
    """``(holidays, notes)``: the union of both currencies' holidays.

    A currency the file does not cover uses the USD holidays; an unreadable
    file gives no holidays. Either way a note says so.
    """
    try:
        table = load_holidays(path)
    except Exception as exc:
        return frozenset(), [f"holiday calendar unavailable ({exc}); variance days weight "
                             "weekends and year-end only"]
    days, notes = set(), []
    for ccy in (pair[:3], pair[3:]):
        if ccy in table:
            days |= table[ccy]
        else:
            days |= table.get(FALLBACK_CCY, frozenset())
            notes.append(f"no {ccy} holidays in the calendar; using {FALLBACK_CCY} holidays")
    return frozenset(days), notes


def day_weights(start, end, holidays=frozenset(), weights=DEFAULT_WEIGHTS):
    """The weight of every day in ``(start, end]``, as an array."""
    days = pd.date_range(pd.Timestamp(start).normalize() + pd.Timedelta(days=1),
                         pd.Timestamp(end).normalize())
    if len(days) == 0:
        return np.zeros(0)
    weekday = days.weekday
    result = np.ones(len(days))
    if holidays:
        result[np.isin(days.date, list(holidays))] = weights.holiday
    result[weekday == 5] = weights.saturday
    result[weekday == 6] = weights.sunday
    year_end = np.zeros(len(days), dtype=bool)
    for month, day in _YEAR_END:
        year_end |= (days.month == month) & (days.day == day)
    result[year_end] = weights.year_end
    return result


def weighted_days(start, end, holidays=frozenset(), weights=DEFAULT_WEIGHTS):
    """Variance days in ``(start, end]`` (events not included)."""
    return float(day_weights(start, end, holidays, weights).sum())

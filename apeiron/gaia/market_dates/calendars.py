"""Gaia time's shared pieces: currency calendars, date coercion, date records.

Used by the pillar date solver (:mod:`apeiron.gaia.market_dates.pillars`) and its
legacy reference (:mod:`apeiron.gaia.market_dates.legacy_dates`); imports nothing
else in the suite.

Settlement calendars (``CALENDARS``) are Bloomberg's, from the static list
``market_data/calendars/settlement_holidays.csv`` (refreshed by hand with
``python -m market_data.settlement_holidays``): weekends plus each currency's
non-settlement dates within its coverage (``COVERAGE``). Outside the coverage,
or for a currency the list lacks, a day's holiday status is QuantLib's
(``QL_CALENDARS``, the table the app used before). ``SOURCE`` names where the
calendars came from; if the list cannot be read they are QuantLib's whole.
"""
import csv
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import QuantLib as ql

#: Tenors that mean an overnight option.
OVERNIGHT = {"ON", "O/N", "1D"}

#: The QuantLib calendar per currency (the app's table before the Bloomberg
#: list): the legacy reference, and the fallback outside the list's coverage.
#: COP, MYR, PHP and VND have no QuantLib calendar (weekends only).
QL_CALENDARS = {
    "EUR": ql.TARGET(),
    "GBP": ql.UnitedKingdom(ql.UnitedKingdom.Exchange),
    "AUD": ql.Australia(),
    "NZD": ql.NewZealand(),
    "USD": ql.UnitedStates(ql.UnitedStates.FederalReserve),
    "CAD": ql.Canada(),
    "CHF": ql.Switzerland(),
    "NOK": ql.Norway(),
    "SEK": ql.Sweden(),
    "DKK": ql.Denmark(),
    "JPY": ql.Japan(),
    "SGD": ql.Singapore(),
    "HKD": ql.HongKong(),
    "MXN": ql.Mexico(),
    "BRL": ql.Brazil(),
    "CLP": ql.Chile(),
    "COP": ql.WeekendsOnly(),
    "ARS": ql.Argentina(),
    "ZAR": ql.SouthAfrica(),
    "PLN": ql.Poland(),
    "CZK": ql.CzechRepublic(),
    "HUF": ql.Hungary(),
    "TRY": ql.Turkey(),
    "ILS": ql.Israel(),
    "CNH": ql.China(ql.China.SSE),   # approximation; offshore CNH trades HK + CN
    "KRW": ql.SouthKorea(ql.SouthKorea.Settlement),
    "INR": ql.India(),
    "TWD": ql.Taiwan(),
    "MYR": ql.WeekendsOnly(),
    "THB": ql.Thailand(),
    "IDR": ql.Indonesia(),
    "PHP": ql.WeekendsOnly(),
    "VND": ql.WeekendsOnly(),
}

HOLIDAY_LIST = Path(__file__).resolve().parents[3] / "market_data" / "calendars" / "settlement_holidays.csv"
#: QuantLib fills the years outside a currency's coverage within this span.
FALLBACK_SPAN = (1990, 2080)


def _bespoke(ccy, holidays, coverage):
    """Weekends + ``holidays`` inside ``coverage``, QuantLib's holidays outside it."""
    cal = ql.BespokeCalendar(f"Bloomberg settlement {ccy}")
    cal.addWeekend(ql.Saturday)
    cal.addWeekend(ql.Sunday)
    first, last = coverage
    fallback = QL_CALENDARS[ccy]
    spans = [(date(FALLBACK_SPAN[0], 1, 1), date.fromordinal(first.toordinal() - 1)),
             (date.fromordinal(last.toordinal() + 1), date(FALLBACK_SPAN[1], 12, 31))]
    for start, end in spans:
        if start <= end:
            for day in fallback.holidayList(_ql(start), _ql(end), False):
                cal.addHoliday(day)
    for day in holidays:
        if first <= day <= last:
            cal.addHoliday(_ql(day))
    return cal


def load(path=HOLIDAY_LIST):
    """``(calendars, coverage, source)`` from the holiday list, QuantLib's if unreadable."""
    try:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        by_ccy = {}
        for row in rows:
            entry = by_ccy.setdefault(row["ccy"], {"dates": [], "coverage": (
                as_date(row["start"]), as_date(row["end"]))})
            entry["dates"].append(as_date(row["date"]))
        if not by_ccy:
            raise ValueError("empty holiday list")
    except Exception as error:   # no list: QuantLib's calendars, said so
        return dict(QL_CALENDARS), {}, f"quantlib (holiday list unreadable: {error})"
    calendars, coverage = dict(QL_CALENDARS), {}
    for ccy, entry in by_ccy.items():
        if ccy in QL_CALENDARS:
            calendars[ccy] = _bespoke(ccy, entry["dates"], entry["coverage"])
            coverage[ccy] = entry["coverage"]
    return calendars, coverage, f"bloomberg ({Path(path).name})"


def covered(ccy, day) -> bool:
    """True when ``day`` is inside ``ccy``'s Bloomberg coverage."""
    span = COVERAGE.get(ccy)
    return bool(span) and span[0] <= as_date(day) <= span[1]

@dataclass(frozen=True)
class Settlement:
    """What the date rules need to know about a pair."""
    base: str
    quote: str
    spot_lag: int


@dataclass(frozen=True)
class PillarDates:
    """The dates of one tenor traded on ``trade``."""
    tenor: str
    trade: date
    spot: date
    expiry: date
    delivery: date


def as_date(value) -> date:
    """``date``, ``datetime``, pandas ``Timestamp``, QuantLib ``Date`` or ISO text."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, ql.Date):
        return date(value.year(), value.month(), value.dayOfMonth())
    return datetime.strptime(str(value).strip()[:10], "%Y-%m-%d").date()


def _ql(value) -> ql.Date:
    day = as_date(value)
    return ql.Date(day.day, day.month, day.year)


def _py(day: ql.Date) -> date:
    return date(day.year(), day.month(), day.dayOfMonth())


CALENDARS, COVERAGE, SOURCE = load()

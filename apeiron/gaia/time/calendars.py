"""Gaia time's shared pieces: currency calendars, date coercion, date records.

Used by the pillar date solver (:mod:`apeiron.gaia.time.pillars`) and its
legacy reference (:mod:`apeiron.gaia.time.legacy_dates`); imports nothing
else in the suite.
"""
from dataclasses import dataclass
from datetime import date, datetime

import QuantLib as ql

#: Tenors that mean an overnight option.
OVERNIGHT = {"ON", "O/N", "1D"}

#: Each currency's settlement calendar. Currencies without a QuantLib calendar
#: (COP, MYR, PHP, VND) are weekends only until a holiday list covers them.
CALENDARS = {
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

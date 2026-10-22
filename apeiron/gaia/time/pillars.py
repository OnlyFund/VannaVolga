"""The pillar date solver: one place for every FX date rule.

Spot dates, tenor (pillar) expiries and deliveries, the delivery of a given
expiry, overnight and value-date tokens, all from any valuation date. Dates go
in and come out as ``datetime.date``; QuantLib stays inside this module.

A pair arrives as a :class:`Settlement` (its two currencies and settlement
lag), so time imports no pair conventions; ``helper_functions.fx_conventions``
builds it from its tables (``fx_conventions.settlement(pair)``).

Step 1 of item 6: these are exactly the rules the app used before, gathered
here so later rule changes happen in one place:

* the pair's calendar is its two currencies' joint calendar, plus USD for a
  cross;
* spot is the trade date plus the settlement lag, Following;
* a tenor's delivery is spot plus the tenor, Modified Following with the
  end-of-month rule; its expiry is that delivery less the settlement lag,
  Preceding (week tenors too, for now);
* O/N expires on the next good day and delivers on that day's spot;
* a given expiry is rolled back to a good day (Preceding) and delivers after
  the settlement lag (Following).

Fail-safes: the previous rules are kept verbatim in
:mod:`apeiron.gaia.time.legacy_dates`. If a rule here raises where the old one
does not, the old answer is used and the case is recorded in ``FALLBACKS``.
With ``VV_PILLAR_CHECK=1`` (or ``CHECK = True``) every answer is also compared
with the old rule and any difference is recorded in ``MISMATCHES``.
"""
import os
import re
from datetime import date
import QuantLib as ql

from apeiron.gaia.time.calendars import (  # noqa: F401  (re-exported)
    CALENDARS, OVERNIGHT, PillarDates, Settlement, _py, _ql, as_date,
)

_TENOR = re.compile(r"(\d+)([DWMY])")
_UNITS = {"D": ql.Days, "W": ql.Weeks, "M": ql.Months, "Y": ql.Years}

CHECK = os.environ.get("VV_PILLAR_CHECK") == "1"
#: Recorded fail-safe events, newest last, at most ``_KEEP`` each.
FALLBACKS, MISMATCHES = [], []
_KEEP = 1000


def _record(log, entry):
    log.append(entry)
    del log[:-_KEEP]


def _guarded(name):
    """Run the rule; fall back to the legacy rule if only the new one fails."""
    def wrap(rule):
        def guarded(*args):
            from apeiron.gaia.time import legacy_dates
            legacy = getattr(legacy_dates, name)
            try:
                result = rule(*args)
            except Exception as error:
                try:
                    result = legacy(*args)
                except Exception:
                    raise error
                _record(FALLBACKS, (name, args, repr(error)))
                return result
            if CHECK:
                try:
                    reference = legacy(*args)
                except Exception as error:
                    reference = repr(error)
                if reference != result:
                    _record(MISMATCHES, (name, args, result, reference))
            return result
        guarded.__name__, guarded.__doc__, guarded.rule = rule.__name__, rule.__doc__, rule
        return guarded
    return wrap


def calendar(settlement: Settlement) -> ql.Calendar:
    """The pair's settlement calendar: both currencies, plus USD for a cross."""
    joint = ql.JointCalendar(CALENDARS[settlement.base], CALENDARS[settlement.quote])
    if "USD" not in (settlement.base, settlement.quote):
        joint = ql.JointCalendar(joint, CALENDARS["USD"])
    return joint


def is_business_day(settlement, day) -> bool:
    return calendar(settlement).isBusinessDay(_ql(day))


def roll_back(settlement, day) -> date:
    """``day`` if it is a good day, else the previous good day."""
    return _py(calendar(settlement).adjust(_ql(day), ql.Preceding))


def roll_forward(settlement, day) -> date:
    """``day`` if it is a good day, else the next good day."""
    return _py(calendar(settlement).adjust(_ql(day), ql.Following))


@_guarded("next_business_day")
def next_business_day(settlement, day) -> date:
    """The pair's next good day after ``day``."""
    return _py(calendar(settlement).advance(_ql(day), ql.Period(1, ql.Days), ql.Following))


@_guarded("spot_date")
def spot_date(settlement, trade) -> date:
    """Trade date plus the settlement lag on the pair's calendar, Following."""
    return _py(calendar(settlement).advance(
        _ql(trade), ql.Period(settlement.spot_lag, ql.Days), ql.Following))


@_guarded("delivery_for_expiry")
def delivery_for_expiry(settlement, expiry) -> date:
    """A given expiry, rolled back to a good day, plus the settlement lag."""
    cal = calendar(settlement)
    adjusted = cal.adjust(_ql(expiry), ql.Preceding)
    return _py(cal.advance(adjusted, ql.Period(settlement.spot_lag, ql.Days), ql.Following))


@_guarded("tenor_delivery")
def tenor_delivery(settlement, trade, tenor) -> date:
    """Spot plus ``tenor`` (D, W, M, Y), Modified Following, end of month."""
    count, unit = _parse(tenor)
    return _py(calendar(settlement).advance(
        _ql(spot_date(settlement, trade)), ql.Period(count, _UNITS[unit]),
        ql.ModifiedFollowing, True))


@_guarded("tenor_dates")
def tenor_dates(settlement, trade, tenor) -> PillarDates:
    """Expiry and delivery of ``tenor`` traded on ``trade`` (O/N included)."""
    key = str(tenor).strip().upper()
    trade_day = as_date(trade)
    spot = spot_date(settlement, trade_day)
    if key in OVERNIGHT:
        expiry = next_business_day(settlement, trade_day)
        return PillarDates(key, trade_day, spot, expiry, spot_date(settlement, expiry))
    delivery = tenor_delivery(settlement, trade_day, key)
    expiry = _py(calendar(settlement).advance(
        _ql(delivery), ql.Period(-settlement.spot_lag, ql.Days), ql.Preceding))
    return PillarDates(key, trade_day, spot, expiry, delivery)


def _parse(tenor):
    match = _TENOR.fullmatch(str(tenor).strip().upper())
    if not match:
        raise ValueError(f"unrecognised tenor {tenor!r}")
    return int(match.group(1)), match.group(2)

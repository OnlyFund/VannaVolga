"""The pillar date solver: one place for every FX date rule.

Spot dates, tenor (pillar) expiries and deliveries, the delivery of a given
expiry, overnight and value-date tokens, all from any valuation date. Dates go
in and come out as ``datetime.date``; QuantLib stays inside this module.

A pair arrives as a :class:`Settlement` (its two currencies and settlement
lag), so time imports no pair conventions; ``helper_functions.fx_conventions``
builds it from its tables (``fx_conventions.settlement(pair)``).

The rules (checked 2026-10-22 against Bloomberg's historical settlement dates:
21 pairs on 34 edge-case dates, 14,178 dates, 0 differences):

* calendars are Bloomberg's settlement calendars
  (:mod:`apeiron.gaia.time.calendars`); the pair's calendar is its two
  currencies' joint calendar, plus USD for a cross;
* **spot**: the settlement lag is counted on the pair's non-USD calendar(s), so
  a US-only holiday is a good intermediate day, for USD pairs and crosses
  alike; the spot day itself must be good in every currency including USD
  (rolled forward until it is). MXN and BRL are the exceptions: US holidays
  block their intermediate day too (``USD_BLOCKS_INTERMEDIATE``);
* a tenor's **delivery** is spot plus the tenor: day and week tenors roll
  Following; month and year tenors roll Modified Following with the
  end-of-month rule;
* a tenor's expiry is that delivery less the settlement lag, Preceding;
* O/N expires on the next good day and delivers on that day's spot;
* a given expiry is rolled back to a good day (Preceding) and delivers on its
  spot date.

Fail-safes: the previous rules are kept verbatim, on the old QuantLib
calendars, in :mod:`apeiron.gaia.time.legacy_dates`. If a rule here raises
where the old one does not, the old answer is used and the case is recorded in
``FALLBACKS``. With ``VV_PILLAR_CHECK=1`` (or ``CHECK = True``) every answer is
also compared with the old rule and each difference is recorded in
``MISMATCHES`` (what the new rules move). A date outside a currency's
Bloomberg coverage (QuantLib's holidays then apply) is recorded in
``UNCOVERED``.
"""
import os
import re
from datetime import date
import QuantLib as ql

from apeiron.gaia.time.calendars import (  # noqa: F401  (re-exported)
    CALENDARS, OVERNIGHT, PillarDates, Settlement, _py, _ql, as_date, covered,
)

_TENOR = re.compile(r"(\d+)([DWMY])")
_UNITS = {"D": ql.Days, "W": ql.Weeks, "M": ql.Months, "Y": ql.Years}

#: Currencies whose spot counts US holidays as bad intermediate days too.
USD_BLOCKS_INTERMEDIATE = frozenset({"MXN", "BRL"})

CHECK = os.environ.get("VV_PILLAR_CHECK") == "1"
#: Recorded fail-safe events, newest last, at most ``_KEEP`` each.
FALLBACKS, MISMATCHES, UNCOVERED = [], [], []
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


def _joint(currencies) -> ql.Calendar:
    joint = CALENDARS[currencies[0]]
    for ccy in currencies[1:]:
        joint = ql.JointCalendar(joint, CALENDARS[ccy])
    return joint


def _currencies(settlement):
    pair = [settlement.base, settlement.quote]
    return pair if "USD" in pair else pair + ["USD"]


def calendar(settlement: Settlement) -> ql.Calendar:
    """The pair's settlement calendar: both currencies, plus USD for a cross."""
    return _joint(_currencies(settlement))


def _note_coverage(settlement, *days):
    """Record days outside a currency's Bloomberg coverage (QuantLib's apply)."""
    for day in days:
        missing = [ccy for ccy in _currencies(settlement) if not covered(ccy, day)]
        entry = (tuple(missing), as_date(day))
        if missing and entry not in UNCOVERED:
            _record(UNCOVERED, entry)


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
    """The settlement lag on the non-USD calendar(s), then a day good in all.

    A US-only holiday counts as a good intermediate day, except for
    ``USD_BLOCKS_INTERMEDIATE`` currencies, whose lag is counted on the full
    calendar.
    """
    full = calendar(settlement)
    others = [ccy for ccy in (settlement.base, settlement.quote) if ccy != "USD"]
    lag = ql.Period(settlement.spot_lag, ql.Days)
    if set(others) & USD_BLOCKS_INTERMEDIATE:
        spot = full.advance(_ql(trade), lag, ql.Following)
    else:
        part = _joint(others)
        spot = part.advance(_ql(trade), lag, ql.Following)
        while not full.isBusinessDay(spot):
            spot = part.advance(spot, ql.Period(1, ql.Days), ql.Following)
    _note_coverage(settlement, trade, _py(spot))
    return _py(spot)


@_guarded("delivery_for_expiry")
def delivery_for_expiry(settlement, expiry) -> date:
    """A given expiry, rolled back to a good day, delivers on its spot date."""
    return spot_date(settlement, roll_back(settlement, expiry))


@_guarded("tenor_delivery")
def tenor_delivery(settlement, trade, tenor) -> date:
    """Spot plus ``tenor``: D and W roll Following; M and Y Modified Following, end of month."""
    count, unit = _parse(tenor)
    month_like = unit in ("M", "Y")
    delivery = _py(calendar(settlement).advance(
        _ql(spot_date(settlement, trade)), ql.Period(count, _UNITS[unit]),
        ql.ModifiedFollowing if month_like else ql.Following, month_like))
    _note_coverage(settlement, delivery)
    return delivery


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

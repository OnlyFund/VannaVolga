"""The FX date rules as the app had them before the pillar date solver.

A verbatim reference, on the old QuantLib calendars, written the way the old code was (QuantLib tenor
strings, each caller's own steps), for the fail-safes in
:mod:`apeiron.gaia.market_dates.pillars` and for reporting what each later rule change
moves. Same signatures and results as the solver's rules. Not for new code.

Sources: ``fx_conventions.joint_calendar`` / ``spot_date`` / ``option_dates``,
``market_tables.tenor_dates`` (O/N), ``fx_discounting.delivery_date``,
``curve_roll.overnight_expiry`` and ``spot_forwards.value_date_for`` (tenors).
"""
import QuantLib as ql

from apeiron.gaia.market_dates.calendars import OVERNIGHT, PillarDates, _py, _ql, as_date
from apeiron.gaia.market_dates.calendars import QL_CALENDARS as CALENDARS   # the old calendars


def joint_calendar(settlement):
    cal = ql.JointCalendar(CALENDARS[settlement.base], CALENDARS[settlement.quote])
    if "USD" not in (settlement.base, settlement.quote):
        # USD holidays still block settlement on crosses
        cal = ql.JointCalendar(cal, CALENDARS["USD"])
    return cal


def next_business_day(settlement, day):
    return _py(joint_calendar(settlement).advance(_ql(day), ql.Period(1, ql.Days), ql.Following))


def spot_date(settlement, trade):
    return _py(joint_calendar(settlement).advance(
        _ql(trade), ql.Period(settlement.spot_lag, ql.Days), ql.Following))


def delivery_for_expiry(settlement, expiry):
    calendar = joint_calendar(settlement)
    adjusted = calendar.adjust(_ql(expiry), ql.Preceding)
    return _py(calendar.advance(adjusted, ql.Period(settlement.spot_lag, ql.Days), ql.Following))


def tenor_delivery(settlement, trade, tenor):
    spot = _ql(spot_date(settlement, trade))
    return _py(joint_calendar(settlement).advance(
        spot, ql.Period(str(tenor).strip().upper()), ql.ModifiedFollowing, True))


def tenor_dates(settlement, trade, tenor):
    key = str(tenor).strip().upper()
    trade_day = as_date(trade)
    cal = joint_calendar(settlement)
    spot = _ql(spot_date(settlement, trade_day))
    if key in OVERNIGHT:
        expiry = cal.advance(_ql(trade_day), ql.Period(1, ql.Days), ql.Following)
        return PillarDates(key, trade_day, _py(spot), _py(expiry),
                           spot_date(settlement, _py(expiry)))
    delivery = cal.advance(spot, ql.Period(key), ql.ModifiedFollowing, True)   # end-of-month
    expiry = cal.advance(delivery, ql.Period(-settlement.spot_lag, ql.Days), ql.Preceding)
    return PillarDates(key, trade_day, _py(spot), _py(expiry), _py(delivery))


def expiry_for_delivery(settlement, delivery):
    """Delivery less the settlement lag, Preceding (the old month-tenor expiry)."""
    return _py(joint_calendar(settlement).advance(
        _ql(delivery), ql.Period(-settlement.spot_lag, ql.Days), ql.Preceding))

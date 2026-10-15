"""Roll the quoted market curves forward for a sticky-tenor theta.

Rolling one business day keeps every tenor's quotes (vols, forward points,
term rates) and moves its dates: the pillar now starts from the new valuation
date. An option with a fixed expiry and delivery then sits at a shorter point
on unchanged tenor curves, so theta carries the vol and forward-point
roll-down. Each pillar's total variance is fixed by its quote; events only
distribute it between pillar dates, so an event that passes leaves the
variance clock and its share is redistributed over the remaining days.

Risk-roll nodes move by elapsed calendar days, without business-day adjustment.
They are interpolation coordinates, not new contractual settlement dates.
Stored expiry/delivery offsets are preserved, including broker date conventions.
A zero-day roll leaves the market untouched.
"""
from datetime import date

import pandas as pd
from pandas.tseries.offsets import BDay


def _to_date(value) -> date:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp.date()


def _ql_date(value):
    import QuantLib as ql
    value = _to_date(value)
    return ql.Date(value.day, value.month, value.year)


def _from_ql(value) -> date:
    return date(value.year(), value.month(), value.dayOfMonth())


#: Days the global FX market does not trade, as (month, day).
FX_TRADING_CLOSURES = {(1, 1), (12, 25)}


def next_business_day(pair, valuation) -> date:
    """Return the next FX trading day, the theta horizon.

    This is a trading calendar, not the pair's settlement calendar: a
    settlement-only holiday such as US Columbus Day is still a trading day on
    which spot moves and time decays. Settlement calendars still govern actual
    contractual delivery, but do not suppress calendar-time risk-curve roll.
    ``pair`` is accepted for a future pair-specific trading calendar.
    """
    day = pd.Timestamp(_to_date(valuation)) + BDay(1)
    while (day.month, day.day) in FX_TRADING_CLOSURES:
        day += BDay(1)
    return day.date()


#: Expiry inputs that mean an overnight option.
OVERNIGHT_TENORS = {"ON", "O/N", "1D"}


def overnight_expiry(pair, valuation) -> date:
    """Return the O/N option expiry: the pair's next good business day.

    Falls back to the next FX trading day when the pair has no conventions.
    """
    try:
        import QuantLib as ql
        from helper_functions.fx_conventions import joint_calendar
        return _from_ql(joint_calendar(pair).advance(
            _ql_date(valuation), ql.Period(1, ql.Days), ql.Following))
    except Exception:
        return next_business_day(pair, valuation)


def spot_date(pair, valuation) -> date:
    """Return the pair's spot date, falling back to two weekdays."""
    try:
        from helper_functions.fx_conventions import spot_date as pair_spot_date
        return _from_ql(pair_spot_date(pair, _ql_date(valuation)))
    except Exception:
        return (pd.Timestamp(_to_date(valuation)) + BDay(2)).date()


def roll_tenor_dates(pair, tenors, dates, valuation, as_of, kind="expiry"):
    """Translate risk-curve coordinates by elapsed calendar days."""
    valuation, as_of = _to_date(valuation), _to_date(as_of)
    shift = pd.Timedelta(days=(as_of - valuation).days)
    return [pd.Timestamp(_to_date(stored)) + shift for stored in dates]


def roll_vol_rows(vol_rows, pair, valuation, as_of):
    """Return the vol table with every pillar expiry rolled to ``as_of``."""
    rolled = vol_rows.copy()
    if _to_date(as_of) == _to_date(valuation):
        return rolled
    rolled["expiry"] = roll_tenor_dates(
        pair, rolled["tenor"], rolled["expiry"], valuation, as_of, "expiry"
    )
    return rolled


def roll_forward_rows(forward_rows, rolled_vol_rows, pair, valuation, as_of, delivery_for_expiry):
    """Translate stored delivery coordinates without regenerating settlement."""
    rolled = forward_rows.copy()
    if _to_date(as_of) == _to_date(valuation):
        return rolled
    # A table without tenors rolls each delivery by the calendar-day roll.
    tenors = (
        [str(tenor).strip().upper() for tenor in rolled["tenor"]]
        if "tenor" in rolled else [None] * len(rolled)
    )
    rolled["delivery_date"] = roll_tenor_dates(
        pair, tenors, rolled["delivery_date"], valuation, as_of, "delivery"
    )
    return rolled


def anchor_forward_rows(forward_rows, pair, curve_date):
    """Prepend the spot-date point so short deliveries interpolate.

    The forward to the spot date is spot itself, so its points are exactly
    zero. The term rate is held flat at the first quoted tenor.
    """
    rows = forward_rows.copy()
    dates = pd.to_datetime(rows["delivery_date"], errors="coerce")
    anchor = pd.Timestamp(spot_date(pair, curve_date))
    if dates.notna().any() and anchor >= dates.min():
        return rows
    first = rows.loc[dates.idxmin()]
    anchor_row = {column: first[column] for column in rows.columns}
    anchor_row.update({"tenor": "SPOT", "delivery_date": anchor, "fwd_pts": 0.0})
    return pd.concat([pd.DataFrame([anchor_row]), rows], ignore_index=True)

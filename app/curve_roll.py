"""Roll the quoted market curves forward for a sticky-tenor theta.

Rolling one business day keeps every tenor's quotes (vols, forward points,
term rates) and moves its dates: the pillar now starts from the new valuation
date. An option with a fixed expiry and delivery then sits at a shorter point
on unchanged tenor curves, so theta carries the vol and forward-point
roll-down. Each pillar's total variance is fixed by its quote; events only
distribute it between pillar dates, so an event that passes leaves the
variance clock and its share is redistributed over the remaining days.

Vol pillars move by the trading days elapsed (weekdays other than Dec 25 /
Jan 1): rolling to the next trading day moves every pillar to its next
trading day, so the surface rolls one trading day forward and each option is
repriced at its fixed strike and expiry, with total variance from the quotes
spread by variance days. (A calendar-day shift would put pillars on weekends,
which the vol clock weights near zero.) Forward and delivery nodes move by
the calendar days elapsed: carry accrues over weekends. All are interpolation
coordinates, not new contractual settlement dates.
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
        from apeiron.gaia.market_dates.pillars import next_business_day as pair_next_day
        from helper_functions.fx_conventions import settlement
        return pair_next_day(settlement(pair), _to_date(valuation))
    except Exception:
        return next_business_day(pair, valuation)


def spot_date(pair, valuation) -> date:
    """Return the pair's spot date, falling back to two weekdays."""
    try:
        from apeiron.gaia.market_dates.pillars import spot_date as pair_spot_date
        from helper_functions.fx_conventions import settlement
        return pair_spot_date(settlement(pair), _to_date(valuation))
    except Exception:
        return (pd.Timestamp(_to_date(valuation)) + BDay(2)).date()


def _is_trading_day(day):
    return day.weekday() < 5 and (day.month, day.day) not in ((12, 25), (1, 1))


def trading_days_between(start, end):
    """Trading days in ``(start, end]``: weekdays other than Dec 25 / Jan 1."""
    start, end = pd.Timestamp(_to_date(start)), pd.Timestamp(_to_date(end))
    return sum(_is_trading_day(day) for day in pd.date_range(start + pd.Timedelta(days=1), end))


def _add_trading_days(day, count):
    day = pd.Timestamp(day)
    for _ in range(count):
        day += pd.Timedelta(days=1)
        while not _is_trading_day(day):
            day += pd.Timedelta(days=1)
    return day


def roll_tenor_dates(pair, tenors, dates, valuation, as_of, kind="expiry"):
    """Translate risk-curve coordinates to ``as_of``: vol pillars (``expiry``)
    by the trading days elapsed, delivery nodes by the calendar days."""
    valuation, as_of = _to_date(valuation), _to_date(as_of)
    if kind == "expiry":
        steps = trading_days_between(valuation, as_of)
        return [_add_trading_days(_to_date(stored), steps) for stored in dates]
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


def short_end_nodes(pair, curve_date, short_points):
    """``[(delivery, fwd_pts, tenor)]``: the quoted short swaps as forward nodes.

    Swap points are far minus near, and spot has zero points, so walking back
    from spot the day before it is spot less TN (less ON for a T+1 pair,
    whose ON ends at spot) and today is spot less TN and ON; the day after
    spot is spot plus SN. A swap whose dates do not meet spot on the pair's
    calendar is left out, as is a T+1 pair's TN (it would start at spot).
    """
    from apeiron.gaia.market_dates import pillars
    from helper_functions.fx_conventions import settlement

    points = short_points or {}
    rules = settlement(pair)
    today = _to_date(curve_date)
    tomorrow = pillars.next_business_day(rules, today)
    day_after = pillars.next_business_day(rules, tomorrow)
    spot = pillars.spot_date(rules, today)
    stamp = pd.Timestamp

    nodes = []
    on, tn, sn = points.get("ON"), points.get("TN"), points.get("SN")
    if tomorrow == spot:
        if on is not None:
            nodes.append((stamp(today), -on, "ON"))
    elif day_after == spot and tn is not None:
        nodes.append((stamp(tomorrow), -tn, "TN"))
        if on is not None:
            nodes.append((stamp(today), -(on + tn), "ON"))
    if sn is not None:
        nodes.append((stamp(pillars.next_business_day(rules, spot)), sn, "SN"))
    return nodes


def with_short_end(curve_rows, pair, curve_date, short_points):
    """``curve_rows`` (anchored at spot) with the short swaps as dated rows.

    Their term rate is the curve's first, as the spot anchor's. Value-dated
    deals only: option and hedge forwards keep the spot-to-pillar curve.
    """
    nodes = short_end_nodes(pair, curve_date, short_points)
    if not nodes:
        return curve_rows
    dates = pd.to_datetime(curve_rows["delivery_date"], errors="coerce")
    ordered = sorted(dates.dropna())
    template = curve_rows.loc[dates.idxmin()].to_dict()
    # Short nodes sit before the first quoted pillar (the row after spot).
    limit = ordered[1] if len(ordered) > 1 else ordered[0]
    extra = [dict(template, tenor=tenor, delivery_date=day, fwd_pts=float(value))
             for day, value, tenor in nodes if day < limit and day not in set(ordered)]
    if not extra:
        return curve_rows
    return pd.concat([pd.DataFrame(extra), curve_rows], ignore_index=True)


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

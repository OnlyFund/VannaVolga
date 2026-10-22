"""The FX date rollover: 17:00 New York.

The FX market's date rolls at 17:00 New York (EST/EDT, so DST-aware): a moment
at or after 17:00 New York belongs to the next day, and a weekend to Monday.
The spot, expiry and delivery dates of a trade, a booking or a market snapshot
follow from that FX date (``pillars``).

A moment is a date-time with its time zone. A naive one is read in
``MACHINE_TIMEZONE``, the zone of the machine that recorded it (``VV_TIMEZONE``,
default ``Europe/London``). Market snapshots carry their own zone (the store's
``timezone`` column).

Examples, from London (the gap to New York is 5 hours, 4 in the weeks the UK
and the US change their clocks on different dates):

* Wed 21 Oct 2026 20:21 London = 15:21 New York -> Wed 21 Oct;
* Wed 21 Oct 2026 22:00 London = 17:00 New York -> Thu 22 Oct;
* Fri 23 Oct 2026 22:30 London -> Mon 26 Oct; any Saturday -> Monday;
* Sun 25 Oct 2026 22:00 London = 17:00 New York -> Mon 26 Oct (also Monday).

Holidays do not move the FX date: a 25 Dec date is a date whose spot and
expiry rules then apply.
"""
import os
from datetime import date, datetime, time, timedelta

import pandas as pd

ROLLOVER_TIMEZONE = "America/New_York"
ROLLOVER_TIME = time(17, 0)
MACHINE_TIMEZONE = os.environ.get("VV_TIMEZONE", "Europe/London")


def _moment(value, timezone=None) -> pd.Timestamp:
    """``value`` as an aware Timestamp; a naive one is read in ``timezone``."""
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        # A clock-change hour: an ambiguous time is read as summer time, a
        # skipped one moved forward. Neither can cross 17:00 New York.
        stamp = stamp.tz_localize(timezone or MACHINE_TIMEZONE, ambiguous=True,
                                  nonexistent="shift_forward")
    return stamp


def _weekday_on_or_after(day: date) -> date:
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return day


def fx_date(moment, timezone=None) -> date:
    """The FX date of ``moment``: its New York date, the next day from 17:00, weekends to Monday."""
    new_york = _moment(moment, timezone).tz_convert(ROLLOVER_TIMEZONE)
    day = new_york.date()
    if new_york.time() >= ROLLOVER_TIME:
        day += timedelta(days=1)
    return _weekday_on_or_after(day)


def fx_today(now=None, timezone=None) -> date:
    """Today's FX date (``now`` defaults to this machine's clock)."""
    return fx_date(now if now is not None else datetime.now(), timezone)


def cutoff(fx_day, timezone=None) -> pd.Timestamp:
    """The last moment (naive, in ``timezone``) whose FX date is on or before ``fx_day``.

    Moments before 17:00 New York on ``fx_day`` (or on the Friday before, for a
    weekend day) belong to it or an earlier FX date.
    """
    day = pd.Timestamp(fx_day).date()
    while day.weekday() >= 5:                 # a weekend FX day is the Friday's
        day -= timedelta(days=1)
    roll = pd.Timestamp(datetime.combine(day, ROLLOVER_TIME)).tz_localize(ROLLOVER_TIMEZONE)
    last = roll - pd.Timedelta(seconds=1)
    return last.tz_convert(timezone or MACHINE_TIMEZONE).tz_localize(None)

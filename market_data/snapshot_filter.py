"""Choosing a stored market snapshot: the list, the close of business and a
quick filter.

* ``usable_snapshots``: the snapshots that can price (vols and forward points;
  spot/depo-only ones are left out) on or before the valuation date, newest
  first, so a choice can never be future data.
* ``close_of_business``: a day's close, the usable snapshot closest to 15:00
  New York, per source when one is given. A snapshot stamped with the old
  close (23:59:59) counts as exactly the close.
* ``apply_filter``: a quick filter of a source and/or a date, in any order:
  ``bloomberg`` lists that source's snapshots; ``16oct`` gives that day's
  close; ``bloomberg 16oct`` the source's close that day. A filter that
  resolves to one snapshot proposes it; it is never loaded here.

Failsafes: a date after the valuation date is refused (future data); an
unknown word is reported and the filter is ignored rather than guessed; no
match says so and names the nearest earlier snapshot. In every case the list
falls back to all usable snapshots, so a filter can never leave nothing to
choose from.
"""
import re
from collections import namedtuple
from datetime import date, datetime

import pandas as pd

from market_data.store import CLOSE_TIME, CLOSE_TIMEZONE, LEGACY_CLOSE, STORE_TIMEZONE

#: An undated day (16oct) further than this past the valuation date is last year's.
SAME_YEAR_DAYS = 180
#: Other names for a source.
SOURCE_ALIASES = {"bbg": "bloomberg", "blp": "bloomberg"}

_DATED_FORMATS = ("%Y-%m-%d", "%m/%d/%Y", "%Y%m%d", "%d%b%Y", "%d-%b-%Y", "%d%b%y",
                  "%d-%b-%y")
_UNDATED_FORMATS = ("%d%b", "%d-%b")

FilterResult = namedtuple("FilterResult", "snapshots selected message source day")
FilterResult.__doc__ = """``snapshots``: the list to choose from; ``selected``: the
proposed ``market_id`` or ``None``; ``message``: what went wrong or ``None``;
``source``/``day``: what the filter was read as."""


def _day(value):
    return pd.Timestamp(value).normalize()


def usable_snapshots(snapshots, valuation_date):
    """Snapshots with vols and forward points, dated on or before
    ``valuation_date``, newest first (``MarketDataStore.snapshots`` rows)."""
    if snapshots is None or snapshots.empty:
        return pd.DataFrame(columns=getattr(snapshots, "columns", []))
    frame = snapshots.copy()
    dates = pd.to_datetime(frame["date_time"].str[:10], errors="coerce")
    keep = (frame["vols"] > 0) & (frame["forwards"] > 0) & (dates <= _day(valuation_date))
    frame = frame[keep]
    return frame.sort_values(["date_time", "market_id"], ascending=False).reset_index(drop=True)


def close_minutes_away(date_time):
    """Minutes between a stored date-time and that day's 15:00 New York close."""
    text = str(date_time)
    if text[11:19] == LEGACY_CLOSE:
        return 0.0
    stamp = pd.Timestamp(text).tz_localize(STORE_TIMEZONE, ambiguous=True,
                                           nonexistent="shift_forward")
    close = pd.Timestamp(f"{text[:10]} {CLOSE_TIME}").tz_localize(CLOSE_TIMEZONE)
    return abs((stamp.tz_convert("UTC") - close.tz_convert("UTC")).total_seconds()) / 60.0


def close_of_business(snapshots, day, source=None):
    """The usable snapshot closest to ``day``'s close, or ``None``.

    ``snapshots`` are usable ones (``usable_snapshots``). A tie goes to the
    later snapshot.
    """
    target = _day(day).strftime("%Y-%m-%d")
    rows = snapshots[snapshots["date_time"].str[:10] == target]
    if source is not None:
        rows = rows[rows["market_source"] == source]
    if rows.empty:
        return None
    away = rows["date_time"].map(close_minutes_away)
    best = away[away == away.min()].index
    return rows.loc[best].sort_values(["date_time", "market_id"]).iloc[-1]


def parse_day(token, valuation_date):
    """A date typed as 2026-10-16, 10/16/2026, 20261016, 16oct26, 16-Oct-2026 or
    16oct (in the valuation date's year, or the year before when that is more
    than ``SAME_YEAR_DAYS`` ahead); else ``None``."""
    text = token.strip()
    for fmt in _DATED_FORMATS:
        try:
            return pd.Timestamp(datetime.strptime(text, fmt))
        except ValueError:
            continue
    valuation = _day(valuation_date)
    for fmt in _UNDATED_FORMATS:
        try:
            parsed = pd.Timestamp(datetime.strptime(f"{text}{valuation.year}", fmt + "%Y"))
        except ValueError:
            continue
        # The valuation date's year, unless that is months ahead (28dec typed
        # in January means last December); a day just ahead stays in this year
        # and is refused as future data.
        if parsed > valuation + pd.Timedelta(days=SAME_YEAR_DAYS):
            parsed = parsed.replace(year=valuation.year - 1)
        return parsed
    return None


def parse_source(token, sources):
    """``(source, problem)`` for a typed source: exact, an alias, or a unique prefix."""
    word = token.strip().lower()
    names = sorted({str(s) for s in sources})
    if word in names:
        return word, None
    if SOURCE_ALIASES.get(word) in names:
        return SOURCE_ALIASES[word], None
    matches = [name for name in names if name.startswith(word)]
    if len(matches) == 1:
        return matches[0], None
    if len(matches) > 1:
        return None, f"'{token}' could be {', '.join(matches)}"
    return None, None


def _label(row):
    return f"market {row['market_id']} ({row['market_source']}, {row['date_time'].replace('T', ' ')[:16]})"


def apply_filter(snapshots, text, valuation_date):
    """Filter usable snapshots by ``text``; see the module docstring."""
    usable = usable_snapshots(snapshots, valuation_date)
    tokens = (text or "").replace(",", " ").split()
    if not tokens:
        return FilterResult(usable, None, None, None, None)
    sources = usable["market_source"].unique() if not usable.empty else ()
    all_sources = snapshots["market_source"].unique() if snapshots is not None else ()
    source = day = None
    problems = []
    for token in tokens:
        parsed_day = parse_day(token, valuation_date)
        if parsed_day is not None and day is None:
            day = parsed_day
            continue
        if parsed_day is not None:
            problems.append(f"one date at a time ('{token}')")
            continue
        parsed_source, problem = parse_source(token, all_sources)
        if parsed_source is not None and source is None:
            source = parsed_source
            continue
        if parsed_source is not None:
            problems.append(f"one source at a time ('{token}')")
            continue
        problems.append(problem or f"'{token}' is not a source or a date")
    if problems:
        known = ", ".join(sorted({str(s) for s in all_sources})) or "none stored"
        hint = f" (sources: {known})" if any("not a source" in p for p in problems) else ""
        return FilterResult(usable, None, "; ".join(problems) + hint + "; filter ignored",
                            None, None)
    valuation = _day(valuation_date)
    if day is not None and day > valuation:
        return FilterResult(usable, None, f"{day:%d-%b-%Y} is after the valuation date "
                            f"{valuation:%d-%b-%Y}; future market data cannot be used",
                            source, day)
    if day is None:
        rows = usable[usable["market_source"] == source].reset_index(drop=True)
        if rows.empty:
            return FilterResult(usable, None, f"no usable {source} markets on or before "
                                f"{valuation:%d-%b-%Y}", source, None)
        return FilterResult(rows, None, None, source, None)
    best = close_of_business(usable, day, source)
    if best is None:
        earlier = usable[usable["date_time"].str[:10] < day.strftime("%Y-%m-%d")]
        if source is not None:
            earlier = earlier[earlier["market_source"] == source]
        what = f"{source} " if source else ""
        hint = f"; latest {what}before it: {_label(earlier.iloc[0])}" if not earlier.empty else ""
        return FilterResult(usable, None, f"no usable {what}market on {day:%d-%b-%Y}{hint}",
                            source, day)
    rows = usable[usable["date_time"].str[:10] == day.strftime("%Y-%m-%d")]
    if source is not None:
        rows = rows[rows["market_source"] == source]
    return FilterResult(rows.reset_index(drop=True), int(best["market_id"]), None, source, day)

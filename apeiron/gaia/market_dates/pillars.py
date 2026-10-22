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
  (:mod:`apeiron.gaia.market_dates.calendars`); the pair's calendar is its two
  currencies' joint calendar, plus USD for a cross;
* **spot**: the settlement lag is counted on the pair's non-USD calendar(s), so
  a US-only holiday is a good intermediate day, for USD pairs and crosses
  alike; the spot day itself must be good in every currency including USD
  (rolled forward until it is). MXN and BRL are the exceptions: US holidays
  block their intermediate day too (``USD_BLOCKS_INTERMEDIATE``). ILS is the
  other way: its own holidays do not block the intermediate day either; the
  lag is counted on weekdays (``LAG_ON_WEEKDAYS``);
* a **forward** (swap) tenor delivers on spot plus the tenor
  (``forward_delivery``): day and week tenors roll Following; month and year
  tenors roll Modified Following with the end-of-month rule;
* an **option** expires on an *expiry day* of its pair: a weekday on which at
  least one of the pair's two currencies is open, so the market is not closed.
  A US-only, Japanese or UK holiday is a valid expiry; 25 Dec and 1 Jan are no
  expiry days for EURUSD (both closed) but 25 Dec is for USDJPY (Tokyo is
  open). An option always delivers on its expiry's spot date:

  - O/N, day and week tenors expire on the trade date plus the tenor (O/N:
    one day), moved to the next expiry day if it is none;
  - month and year tenors deliver on the forward delivery, and expire on the
    latest expiry day whose spot date is that delivery
    (``expiry_for_delivery``);
  - a given (typed) expiry on a weekend moves to the next weekday; a weekday
    is taken as given (``typed_expiry``).

  Checked 2026-10-22 against Bloomberg's option pricing page: 26 tenor cases
  (holiday expiries, 25 Dec / 1 Jan, end of month, T+1, crosses, MXN, CNH),
  typed expiries over Christmas 2026, and the USDJPY 3M whose delivery
  (29 Dec 2026) only a 25 Dec expiry reaches.
  Bloomberg's vol tickers' ``SETTLE_DT`` for week tenors follows the forward
  convention instead, so the historical reference compares them with
  ``forward_delivery``.

Fail-safes: the previous rules are kept verbatim, on the old QuantLib
calendars, in :mod:`apeiron.gaia.market_dates.legacy_dates`. If a rule here raises
where the old one does not, the old answer is used and the case is recorded in
``FALLBACKS``. With ``VV_PILLAR_CHECK=1`` (or ``CHECK = True``) every answer is
also compared with the old rule and each difference is recorded in
``MISMATCHES`` (what the new rules move). A date outside a currency's
Bloomberg coverage (QuantLib's holidays then apply) is recorded in
``UNCOVERED``.
"""
import os
import re
from datetime import date, timedelta
import QuantLib as ql

from apeiron.gaia.market_dates.calendars import (  # noqa: F401  (re-exported)
    CALENDARS, OVERNIGHT, PillarDates, Settlement, _py, _ql, as_date, covered,
)

_TENOR = re.compile(r"(\d+)([DWMY])")
_UNITS = {"D": ql.Days, "W": ql.Weeks, "M": ql.Months, "Y": ql.Years}

#: How far back an expiry is searched for a month tenor's delivery.
_EXPIRY_SEARCH_DAYS = 31

#: Currencies whose spot counts US holidays as bad intermediate days too.
USD_BLOCKS_INTERMEDIATE = frozenset({"MXN", "BRL"})
#: Currencies whose spot lag is counted on weekdays: no holiday, theirs or
#: USD's, blocks the intermediate day; the spot day must still be good in
#: both. USDILS: Bloomberg's spot matched on 708 of 708 days around ILS and US
#: holidays, 2015 - 2026 (checked 2026-10-22).
LAG_ON_WEEKDAYS = frozenset({"ILS"})

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
            from apeiron.gaia.market_dates import legacy_dates
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


def is_expiry_day(settlement, day) -> bool:
    """A weekday on which at least one of the pair's currencies is open."""
    day = as_date(day)
    if day.weekday() >= 5:
        return False
    when = _ql(day)
    return any(CALENDARS[ccy].isBusinessDay(when) for ccy in (settlement.base, settlement.quote))


def next_expiry_day(settlement, day) -> date:
    """``day`` if it is an expiry day of the pair, else the next one."""
    day = as_date(day)
    while not is_expiry_day(settlement, day):
        day += timedelta(days=1)
    return day


def previous_expiry_day(settlement, day) -> date:
    """``day`` if it is an expiry day of the pair, else the previous one."""
    day = as_date(day)
    while not is_expiry_day(settlement, day):
        day -= timedelta(days=1)
    return day


def typed_expiry(day) -> date:
    """A given expiry: a weekend moves to the next weekday; a weekday stays."""
    day = as_date(day)
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return day


@_guarded("next_business_day")
def next_business_day(settlement, day) -> date:
    """The pair's next good day after ``day``."""
    return _py(calendar(settlement).advance(_ql(day), ql.Period(1, ql.Days), ql.Following))


@_guarded("spot_date")
def spot_date(settlement, trade) -> date:
    """The settlement lag on the non-USD calendar(s), then a day good in all.

    A US-only holiday counts as a good intermediate day, except for
    ``USD_BLOCKS_INTERMEDIATE`` currencies, whose lag is counted on the full
    calendar, and ``LAG_ON_WEEKDAYS`` currencies, whose lag is counted on
    weekdays.
    """
    full = calendar(settlement)
    others = [ccy for ccy in (settlement.base, settlement.quote) if ccy != "USD"]
    lag = ql.Period(settlement.spot_lag, ql.Days)
    if set(others) & USD_BLOCKS_INTERMEDIATE:
        spot = full.advance(_ql(trade), lag, ql.Following)
    else:
        part = ql.WeekendsOnly() if set(others) & LAG_ON_WEEKDAYS else _joint(others)
        spot = part.advance(_ql(trade), lag, ql.Following)
        while not full.isBusinessDay(spot):
            spot = part.advance(spot, ql.Period(1, ql.Days), ql.Following)
    _note_coverage(settlement, trade, _py(spot))
    return _py(spot)


@_guarded("delivery_for_expiry")
def delivery_for_expiry(settlement, expiry) -> date:
    """A given expiry (weekends to the next weekday) delivers on its spot date."""
    return spot_date(settlement, typed_expiry(expiry))


@_guarded("expiry_for_delivery")
def expiry_for_delivery(settlement, delivery) -> date:
    """The latest expiry day whose spot date is ``delivery``.

    Raises if no expiry day spots onto it (a delivery no expiry can reach).
    """
    delivery = as_date(delivery)
    day = previous_expiry_day(settlement, delivery - timedelta(days=1))
    for _ in range(_EXPIRY_SEARCH_DAYS):
        spot = spot_date(settlement, day)
        if spot == delivery:
            return day
        if spot < delivery:      # spot dates only fall going back: no expiry reaches it
            break
        day = previous_expiry_day(settlement, day - timedelta(days=1))
    raise ValueError(f"no expiry day delivers on {delivery}")


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


#: The forward (swap) convention: spot plus the tenor.
forward_delivery = tenor_delivery


@_guarded("tenor_dates")
def tenor_dates(settlement, trade, tenor) -> PillarDates:
    """An option's expiry and delivery for ``tenor`` traded on ``trade`` (O/N included)."""
    key = str(tenor).strip().upper()
    trade_day = as_date(trade)
    spot = spot_date(settlement, trade_day)
    count, unit = (1, "D") if key in OVERNIGHT else _parse(key)
    if unit in ("D", "W"):
        expiry = next_expiry_day(settlement, trade_day + timedelta(days=count * (7 if unit == "W" else 1)))
        return PillarDates(key, trade_day, spot, expiry, spot_date(settlement, expiry))
    delivery = forward_delivery(settlement, trade_day, key)
    return PillarDates(key, trade_day, spot, expiry_for_delivery(settlement, delivery), delivery)


def _parse(tenor):
    match = _TENOR.fullmatch(str(tenor).strip().upper())
    if not match:
        raise ValueError(f"unrecognised tenor {tenor!r}")
    return int(match.group(1)), match.group(2)

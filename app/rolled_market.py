"""The rolled market: one day's quotes rolled to a later valuation date.

A market quoted on ``T0`` and valued on a later ``T1`` (a stored snapshot
valued on a later day) is rolled, not reused with its old dates. For each of
its tenors (ROADMAP item 6):

1. **pillar dates** are the solver's tenor dates traded on ``T1``
   (``pillars.tenor_dates``): the new expiry, and for the forward curve the
   tenor's forward delivery (spot plus tenor, from ``T1``);
2. **vols keep forward variance**: each of the five smile vols (10D/25D put,
   ATM, 25D/10D call) at the new expiry ``E`` is today's forward vol between
   ``T1`` and ``E`` (``MarketContext._forward_vol_quotes``, theta's rule): the
   total variance to ``E`` less the variance of the days that pass. Only the
   passing days' own variance drops out, so weekends and event days follow
   the variance-day weighting; events on or before ``T1`` are gone;
3. **forward points are sticky tenor**: each tenor keeps its points and term
   rate (and the short ON/TN/SN swaps theirs), at its new forward delivery, so
   a fixed-delivery forward slides down the curve;
4. the portfolio is then priced on that market, as on any other.

Quotes come out as smile (two-vol) butterflies, converted with the market's
own settings, so the rolled market is ``broker_strangle=False``.

Fail-safes (each says why in the returned notes):

* a pillar whose new expiry is beyond today's last quoted expiry (the last
  tenor, by the days rolled) holds the last quoted segment's forward vol over
  those days (``_extended_quotes``: total variance runs on at the forward rate
  between the last two pillars, on the variance-day clock the surface
  interpolates on), so it keeps forward variance like the rest;
* a pillar whose forward variance cannot be formed (none left, a failure, no
  segment to extend) keeps its tenor's smile from ``T0`` (sticky tenor);
* a pair that cannot be rolled at all (no conventions, a vol or forward row
  without a tenor, dates that do not rise, a rolled market that does not
  build) raises ``RollError``: the caller values the ``T0`` quotes as of
  ``T1`` instead, as before, and says so;
* ``VV_ROLL_MARKET=0`` turns rolling off (``rolling_enabled``).
"""
import os

import pandas as pd

from app.market_context import (
    FORWARD_COLUMNS, MarketContext, VOL_COLUMNS, naive_date, quotes_from_wings, wing_vols,
)

#: ``VV_ROLL_MARKET=0``: markets are not rolled (their quotes are valued as of later dates).
ROLL_ENV = "VV_ROLL_MARKET"
_QUOTES = ("atm", "rr_25", "bf_25", "rr_10", "bf_10")
_FROM_WINGS = {"atm": "v_atm", "rr_25": "rr_25", "bf_25": "bf_25", "rr_10": "rr_10", "bf_10": "bf_10"}


class RollError(ValueError):
    """The market cannot be rolled; value its quotes as of the later date instead."""


def rolling_enabled(environ=None):
    environ = os.environ if environ is None else environ
    return environ.get(ROLL_ENV, "1").strip() != "0"


def _tenor(value, what):
    from apeiron.gaia.market_dates import pillars
    token = str(value if value is not None else "").strip().upper()
    if token in pillars.OVERNIGHT:
        return token
    try:
        pillars._parse(token)
    except ValueError:
        raise RollError(f"a {what} row has no tenor ({value!r}), so it cannot be rolled") from None
    return token


def _smile_quotes(market):
    """A market's five smile vols as ATM / RR / smile BF quotes (``VOL_COLUMNS`` names)."""
    quotes = quotes_from_wings(wing_vols(market))
    return {name: float(quotes[key]) for name, key in _FROM_WINGS.items()}


def _variance(market, start, expiry):
    """``{wing: total variance}`` (vol in points squared times calendar days)."""
    days = (naive_date(expiry) - naive_date(start)).days
    return {wing: vol * vol * days for wing, vol in wing_vols(market).items()}


def _extended_quotes(context, expiry, to_date, last, previous):
    """Quotes at ``expiry`` (after the last quoted expiry ``last``) seen from
    ``to_date``, holding the forward vol of the segment ``previous`` -> ``last``.

    Raises when a forward rate or the remaining variance is not positive.
    """
    start = naive_date(context.valuation_date)
    at_last = _variance(context._market_for_expiry(last), start, last)
    at_previous = _variance(context._market_for_expiry(previous), start, previous)
    passed = _variance(context._market_for_expiry(to_date, allow_short_end=True), start, to_date)
    # Total variance runs linearly on the variance-day clock between pillars
    # (``MarketContext._market_for_expiry``), so it is extended on that clock.
    segment = context.variance_days_between(previous, last)
    beyond = context.variance_days_between(last, expiry)
    remaining = (naive_date(expiry) - naive_date(to_date)).days
    wings = {}
    for wing in at_last:
        rate = (at_last[wing] - at_previous[wing]) / segment
        variance = at_last[wing] + rate * beyond - passed[wing]
        if not (rate > 0 and variance > 0):
            raise ValueError(f"no forward variance to extend in the {wing} vol")
        wings[wing] = (variance / remaining) ** 0.5
    return quotes_from_wings(wings)


def roll_context(context, to_date):
    """``(rolled MarketContext, notes)``: ``context`` rolled to ``to_date``.

    Returns ``context`` itself (and no notes) when ``to_date`` is not later.
    Raises ``RollError`` when the pair cannot be rolled.
    """
    from apeiron.gaia.market_dates import pillars
    from helper_functions.fx_conventions import settlement

    start, end = naive_date(context.valuation_date), naive_date(to_date)
    if end <= start:
        return context, []
    pair = context.currency_pair
    try:
        rules = settlement(pair)
    except Exception as exc:
        raise RollError(f"{pair} has no conventions to roll with: {exc}") from exc
    vols = context.vol_rows.copy()
    if vols.empty:
        raise RollError(f"{pair} has no vol pillars to roll")
    old_expiries = pd.to_datetime(vols["expiry"], errors="coerce", utc=True).dt.tz_localize(None)
    if old_expiries.isna().any():
        raise RollError(f"{pair} has a vol pillar without a valid expiry")
    last_quoted = naive_date(old_expiries.max())
    quoted = sorted({naive_date(day) for day in old_expiries})
    previous_quoted = quoted[-2] if len(quoted) > 1 else None
    trade = end.date()

    rows, forward_vol, kept, extended = [], [], [], []
    for (_, row), old_expiry in zip(vols.iterrows(), old_expiries):
        tenor = _tenor(row["tenor"], "vol")
        try:
            expiry = naive_date(pillars.tenor_dates(rules, trade, tenor).expiry)
        except Exception as exc:
            raise RollError(f"{pair} {tenor}: no expiry from {trade}: {exc}") from exc
        quotes, reason = None, None
        try:
            if expiry <= last_quoted:
                rolled = context._forward_vol_quotes(expiry, end)
            elif previous_quoted is None:
                raise ValueError("beyond the only quoted expiry")
            else:
                rolled = _extended_quotes(context, expiry, end, last_quoted, previous_quoted)
                extended.append(tenor)
            quotes = {name: float(rolled[key]) for name, key in _FROM_WINGS.items()}
        except Exception as exc:              # no forward variance left, a failure
            reason = str(exc)
        if quotes is None:
            try:
                quotes = _smile_quotes(context._market_for_expiry(naive_date(old_expiry)))
            except Exception as exc:
                raise RollError(f"{pair} {tenor}: cannot keep its smile either: {exc}") from exc
            kept.append(f"{tenor} ({reason})")
        else:
            forward_vol.append(tenor)
        rows.append({"tenor": tenor, "expiry": pd.Timestamp(expiry), **quotes})
    vol_rows = pd.DataFrame(rows, columns=list(VOL_COLUMNS))
    if not vol_rows["expiry"].is_monotonic_increasing or vol_rows["expiry"].duplicated().any():
        raise RollError(f"{pair}: the rolled pillar expiries do not rise with the tenor")

    forwards = context.forward_rows.copy()
    if "tenor" not in forwards.columns:
        raise RollError(f"{pair}: the forward points have no tenors, so they cannot be rolled")
    deliveries = []
    for value in forwards["tenor"]:
        tenor = _tenor(value, "forward")
        try:
            deliveries.append(pd.Timestamp(pillars.forward_delivery(rules, trade, tenor)))
        except Exception as exc:
            raise RollError(f"{pair} {tenor}: no forward delivery from {trade}: {exc}") from exc
    forwards["delivery_date"] = deliveries
    forwards = forwards.sort_values("delivery_date").reset_index(drop=True)
    if forwards["delivery_date"].duplicated().any():
        raise RollError(f"{pair}: two forward tenors deliver on the same rolled date")
    missing = [column for column in FORWARD_COLUMNS if column not in forwards.columns]
    if missing:
        raise RollError(f"{pair}: the forward points lack {', '.join(missing)}")

    events = context.event_rows.copy()
    if not events.empty:
        dates = pd.to_datetime(events["event_date"], errors="coerce", utc=True).dt.tz_localize(None)
        events = events.loc[dates > pd.Timestamp(end)].reset_index(drop=True)

    try:
        rolled = MarketContext(
            pair, pd.Timestamp(end), context.spot, context.forward_divisor, False, vol_rows,
            forwards, events, settings=context.settings, market_service=context.market_service,
            diagnostics=context.diagnostics, short_points=context.short_points,
        )
        for expiry in vol_rows["expiry"]:           # it must price at every pillar
            rolled.market_for_expiry(expiry)
    except Exception as exc:
        raise RollError(f"{pair}: the rolled market does not build: {exc}") from exc
    rolled.rolled_from = start
    notes = [f"{pair} quotes of {start:%Y-%m-%d} rolled to {end:%Y-%m-%d}: pillars at "
             f"{end:%Y-%m-%d}'s tenor dates, forward vols held for "
             f"{', '.join(forward_vol) or 'no tenor'}, forward points sticky tenor"]
    if extended:
        notes.append(f"{pair}: {', '.join(extended)} beyond the last quote: the last "
                     f"segment's forward vol held over the days rolled")
    if kept:
        notes.append(f"{pair}: tenor smile kept for {', '.join(kept)}")
    return rolled, notes

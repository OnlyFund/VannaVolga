"""The app's market tables, and whole markets, built from a stored snapshot.

The store keeps quotes by tenor (``snapshot.vols[pair][tenor]``,
``snapshot.forward_points[pair][tenor]``); a tenor's expiry and delivery
follow from the valuation date and the pair's conventions, so they are
worked out here rather than stored:

* ``1W`` ... ``1Y``: spot date plus the tenor (end-of-month aware), back by
  the settlement lag, on the pair's joint calendar
  (``helper_functions.fx_conventions.option_dates``);
* ``ON``: the next business day, delivering on its spot date.

``vol_table`` has the Markets tab's vol columns (vols in percentage points)
plus ``delivery_date``; ``forward_table`` its forward columns, with
``term_discount`` from the term currency's discount curve
(``market_data.discount_curves``: SOFR or forward-implied, with fallbacks); ``market``
puts them together as the single-pair market dict the app saves and loads,
and ``market_bundle`` does it for every pair in the snapshot.
"""
import pandas as pd
import QuantLib as ql

from helper_functions.fx_conventions import joint_calendar, option_dates, spot_date
from market_data.vol_tickers import TENORS


#: The pillars the app quotes; ON is stored but the app derives its short end from 1W.
APP_TENORS = tuple(tenor for tenor in TENORS if tenor != "ON")
COLUMNS = ("tenor", "expiry", "delivery_date", "atm", "rr_25", "bf_25", "rr_10", "bf_10")


def _ql_date(value):
    stamp = pd.Timestamp(value)
    return ql.Date(stamp.day, stamp.month, stamp.year)


def _timestamp(date):
    return pd.Timestamp(date.year(), date.month(), date.dayOfMonth())


def tenor_dates(pair, valuation_date, tenor):
    """``(expiry, delivery)`` of ``tenor`` traded on ``valuation_date``."""
    trade = _ql_date(valuation_date)
    if tenor.upper() == "ON":
        expiry = joint_calendar(pair).advance(trade, ql.Period(1, ql.Days), ql.Following)
        return _timestamp(expiry), _timestamp(spot_date(pair, expiry))
    dates = option_dates(pair, trade, tenor.upper())
    return _timestamp(dates.expiry), _timestamp(dates.delivery)


def vol_table(surface, pair, valuation_date, tenors=APP_TENORS):
    """The app's vol table for one pair from ``{tenor: {field: vol}}``.

    Tenors the surface does not have are left out; rows are in tenor order.
    """
    rows = []
    for tenor in tenors:
        if tenor not in surface:
            continue
        expiry, delivery = tenor_dates(pair, valuation_date, tenor)
        rows.append({"tenor": tenor, "expiry": expiry, "delivery_date": delivery,
                     **{field: surface[tenor][field] for field in COLUMNS[3:]}})
    return pd.DataFrame(rows, columns=list(COLUMNS))


def vol_tables(snapshot, pairs=None, tenors=APP_TENORS):
    """``{pair: vol table}`` for a stored snapshot (all its pairs by default)."""
    pairs = sorted(snapshot.vols) if pairs is None else pairs
    return {pair: vol_table(snapshot.vols[pair], pair, snapshot.date, tenors)
            for pair in pairs if pair in snapshot.vols}


FORWARD_COLUMNS = ("tenor", "delivery_date", "fwd_pts", "term_discount")


def forward_table(snapshot, pair, tenors=APP_TENORS, curves=None):
    """``(table, notes, source)``: the pair's forward points with each tenor's
    delivery date and the term currency's discount rate as ``term_discount``.

    ``source`` is the discount curve's (``sofr``, ``implied``, ``quoted``,
    ``stale`` or ``unavailable``); when unavailable the table is empty.
    Tenors without forward points are left out.
    """
    from market_data.discount_curves import UNAVAILABLE, DiscountCurves
    curve = (curves or DiscountCurves(snapshot)).curve(pair[3:])
    if curve.source == UNAVAILABLE:
        return pd.DataFrame(columns=list(FORWARD_COLUMNS)), list(curve.notes), curve.source
    points = snapshot.forward_points.get(pair, {})
    rows = []
    for tenor in tenors:
        if tenor not in points:
            continue
        _, delivery = tenor_dates(pair, snapshot.date, tenor)
        rows.append({"tenor": tenor, "delivery_date": delivery, "fwd_pts": points[tenor],
                     "term_discount": curve.rates[tenor]})
    return pd.DataFrame(rows, columns=list(FORWARD_COLUMNS)), list(curve.notes), curve.source


def market(snapshot, pair, broker_strangle=None, event_rows=None, tenors=APP_TENORS,
           curves=None):
    """``(market dict, notes)`` for ``pair`` in the app's saved-market format,
    or ``(None, notes)`` when the snapshot cannot build it.

    Only tenors with both vols and forward points are pillars. Butterflies
    follow the pair's convention (``universe.broker_strangle``: broker
    strangles for G10, smile strangles for EM) unless ``broker_strangle`` is
    given. Events are empty unless ``event_rows`` (``event_date``,
    ``variance_days``) is given. ``curves`` (a ``DiscountCurves``) lets
    several markets share one snapshot's discount curves and fallbacks.
    """
    from app.market_context import EVENT_COLUMNS
    from app.market_service import pair_divisor
    from app.market_store import market_payload
    from app.usd_spots import pair_spot
    from market_data.fwd_tickers import SHORT_TENORS
    from market_data.universe import broker_strangle as quoted_broker_strangle

    if broker_strangle is None:
        broker_strangle = quoted_broker_strangle(pair)
    spot = pair_spot(pair, snapshot.spots)
    if not spot == spot:            # nan: no spot for either leg
        return None, [f"{pair}: no spot in market {snapshot.market_id}"]
    vols = vol_table(snapshot.vols.get(pair, {}), pair, snapshot.date, tenors)
    forwards, notes, source = forward_table(snapshot, pair, tenors, curves)
    if source == "unavailable":
        return None, notes + [f"{pair}: unavailable, no {pair[3:]} discount rate"]
    both = [tenor for tenor in vols["tenor"] if tenor in set(forwards["tenor"])]
    if not both:
        return None, notes + [f"{pair}: no tenor with both vols and forward points"]
    dropped = [tenor for tenor in tenors
               if tenor not in both and (tenor in set(vols["tenor"]) or tenor in set(forwards["tenor"]))]
    if dropped:
        notes.append(f"{pair}: {', '.join(dropped)} left out (vols or forward points missing)")
    vols = vols[vols["tenor"].isin(both)].drop(columns=["delivery_date"])
    forwards = forwards[forwards["tenor"].isin(both)]
    events = pd.DataFrame(columns=list(EVENT_COLUMNS)) if event_rows is None else event_rows
    # The short swaps date value-dated deals around spot (``short_points``).
    quoted = snapshot.forward_points.get(pair, {})
    short_points = {tenor: quoted[tenor] for tenor in SHORT_TENORS if tenor in quoted}
    payload = market_payload(pair, snapshot.date, spot, pair_divisor(pair),
                             broker_strangle, vols, forwards, events, short_points)
    return payload, notes


def market_bundle(snapshot, pairs=None, broker_strangle=None, store=None):
    """``(bundle, notes)``: every buildable pair as one market bundle, the file
    the Markets tab loads (``app.market_store`` bundle format).

    With ``store``, a currency without a live rate can use an earlier
    snapshot's (``discount_curves`` step 3).
    """
    from app.market_store import bundle_payload
    from market_data.discount_curves import DiscountCurves
    pairs = sorted(set(snapshot.vols) | set(snapshot.forward_points)) if pairs is None else pairs
    curves = DiscountCurves(snapshot, store)
    markets, notes = {}, []
    for pair in pairs:
        payload, pair_notes = market(snapshot, pair, broker_strangle, curves=curves)
        notes += pair_notes
        if payload is not None:
            markets[pair] = payload
    return bundle_payload(markets, snapshot.date), list(dict.fromkeys(notes))

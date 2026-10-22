"""The app's markets, built from the market data store.

``load_store_markets`` turns one stored snapshot into the app's market
library (``{pair: saved-market dict}``, the shape ``MarketSet.from_payloads``
and the Markets tab use), with a reason for every pair it could not build.

A market's data may be from the valuation date or earlier, never later:
its ``date_time``'s date must be on or before the valuation date.

Without a market data store (no ``$VV_MARKET_STORE`` and no file at the
default path: a fresh checkout, the Streamlit Cloud deploy) the app uses an
illustrative sample market instead (``market_data.sample_market``), kept in
a temporary folder and dated on the valuation date.

It never raises. Every failure is reported instead:

* the store cannot be opened or read -> ``problem``, no markets;
* no market with data on or before the valuation date -> ``problem``;
* the latest such market is more than ``MAX_SNAPSHOT_AGE_BDAYS`` business
  days older than the valuation date -> ``problem`` (an explicitly chosen
  market is allowed, with a note);
* a chosen market does not exist, or its data is from after the valuation
  date -> ``problem``;
* the market has no vols or forward points -> ``problem``;
* a pair's events: its own calendar version saved on or before the
  valuation date (``store.load_pair_events``); else the market's pinned
  event set; else none, noted. A pair's calendar that cannot be read makes
  that pair unavailable, and a pinned set that cannot be read makes every
  pair relying on it unavailable (pricing without the events it should
  have would be wrong). The calendar used is named in the market's
  ``event_source``;
* a pair cannot be built (no spot, no discount rate ...), its build raises,
  or the result fails the app's market validation -> that pair is in
  ``unavailable`` with the reason; the other pairs still load.
"""
from collections import namedtuple

import pandas as pd
from pandas.tseries.offsets import BDay

MAX_SNAPSHOT_AGE_BDAYS = 5

#: ``snapshot`` is the stored snapshot used (or ``None``), ``payloads`` the
#: buildable pairs' market dicts, ``unavailable`` ``{pair: reason}``,
#: ``notes`` informational messages, ``problem`` why nothing loaded (or ``None``).
StoreMarkets = namedtuple("StoreMarkets", "snapshot payloads unavailable notes problem")


def _failed(problem, snapshot=None, notes=()):
    return StoreMarkets(snapshot, {}, {}, list(notes), problem)


def app_market_store(valuation_date):
    """The app's market data store, or the sample store when there is none.

    Opening a missing store would create an empty one, so the default path
    is checked first; the sample store gets a sample market for the date.
    """
    import os
    from market_data.store import DEFAULT_PATH, MarketDataStore, open_market_data_store
    if os.environ.get("VV_MARKET_STORE") or DEFAULT_PATH.exists():
        return open_market_data_store()
    from market_data.sample_market import ensure_sample_market, sample_store_path
    store = MarketDataStore(sample_store_path())
    ensure_sample_market(store, valuation_date)
    return store


def with_fx_date(snapshot):
    """``snapshot`` with its FX date attached (17:00 New York rollover); ``None`` stays."""
    if snapshot is None:
        return None
    from apeiron.gaia.market_dates.rollover import fx_date
    return snapshot._replace(fx_date=str(fx_date(snapshot.date_time, snapshot.timezone)))


def fx_cutoff(valuation_date):
    """The last stored date-time whose FX date is on or before ``valuation_date``."""
    from apeiron.gaia.market_dates.rollover import cutoff
    from market_data.store import STORE_TIMEZONE
    return cutoff(valuation_date, STORE_TIMEZONE)


def load_store_markets(valuation_date, market_id=None, store=None, pairs=None):
    """The markets of one stored market snapshot for ``valuation_date``.

    By default the latest market whose FX date is on or before
    ``valuation_date``: taken before 17:00 New York that day (the rollover,
    ``apeiron.gaia.market_dates.rollover``); ``market_id`` picks one explicitly.
    The snapshot carries its FX date (``Snapshot.fx_date``), which dates its
    markets. ``store`` defaults to the app's
    market data store (``app_market_store``); ``pairs`` limits which pairs are built.
    """
    valuation = pd.Timestamp(valuation_date).normalize()
    try:
        if store is None:
            store = app_market_store(valuation)
        if market_id is not None:
            snapshot = store.load(market_id=market_id)
        else:
            snapshot = store.load(at_or_before=fx_cutoff(valuation))
    except Exception as exc:                 # unreadable, locked or corrupt store
        return _failed(f"market data store unavailable: {exc}")
    snapshot = with_fx_date(snapshot)

    if snapshot is None:
        if market_id is not None:
            return _failed(f"market {market_id} does not exist")
        return _failed(f"no market data on or before {valuation:%Y-%m-%d}; "
                       "run python -m market_data.bloomberg_fetch")
    label = f"market {snapshot.market_id} ({snapshot.market_source}, {snapshot.date_time})"
    snapshot_date = pd.Timestamp(snapshot.date)
    if snapshot_date > valuation:
        return _failed(f"{label} is from after the valuation date {valuation:%Y-%m-%d}; "
                       "future market data cannot be used", snapshot)
    notes = []
    if snapshot_date < valuation:
        too_old = snapshot_date < valuation - BDay(MAX_SNAPSHOT_AGE_BDAYS)
        if too_old and market_id is None:
            return _failed(f"the latest market data, {label}, is more than "
                           f"{MAX_SNAPSHOT_AGE_BDAYS} business days before "
                           f"{valuation:%Y-%m-%d}; run the fetch or choose a market", snapshot)
        notes.append(f"{label} valued as of {valuation:%Y-%m-%d}")
    if not snapshot.vols or not snapshot.forward_points:
        return _failed(f"{label} has no vol surfaces or forward points; "
                       "fetch without --no-vols", snapshot, notes)

    try:
        from app.market_store import read_market_file
        from market_data.discount_curves import DiscountCurves
        from market_data.events import event_rows_table, event_table
        from market_data.market_tables import market
        curves = DiscountCurves(snapshot, store)
    except Exception as exc:
        return _failed(f"cannot build markets: {exc}", snapshot, notes)

    wanted = sorted(set(snapshot.vols) | set(snapshot.forward_points)) if pairs is None else pairs
    event_set, events_problem = None, None
    if snapshot.event_set_id is None:
        notes.append(f"{label} has no event set pinned; markets built without events")
    else:
        try:
            event_set = store.load_event_set(snapshot.event_set_id)
            if event_set is None:
                events_problem = f"its event set {snapshot.event_set_id} does not exist"
        except Exception as exc:
            events_problem = f"its event set {snapshot.event_set_id} cannot be read: {exc}"
    # Each pair's own calendar first (as of the valuation date), then the
    # pinned set, then none.
    own, unavailable = {}, {}
    for pair in wanted:
        try:
            found = store.load_pair_events(pair, as_of=valuation)
        except Exception as exc:
            unavailable[pair] = f"{pair}: its event calendar cannot be read: {exc}"
            continue
        if found is not None:
            own[pair] = found
    relying = [pair for pair in wanted if pair not in own and pair not in unavailable]
    if events_problem:
        for pair in relying:
            unavailable[pair] = f"{pair}: {label} {events_problem}"
        if not own:
            return StoreMarkets(snapshot, {}, unavailable, notes, f"{label} {events_problem}")
    elif event_set is not None:
        without = [pair for pair in relying if pair not in event_set.events]
        if without:
            notes.append(f"event set {event_set.event_set_id} has no events for "
                         f"{', '.join(without)}; built without events")
    if own:
        notes.append("own event calendars: " + ", ".join(
            f"{pair} v{found.version}" for pair, found in sorted(own.items())))
    payloads = {}
    for pair in wanted:
        if pair in unavailable:
            continue
        if pair in own:
            found = own[pair]
            rows = event_rows_table(found.rows)
            source = (f"{pair} calendar v{found.version}, saved {found.saved_at[:16]}"
                      + (f" by {found.user}" if found.user else ""))
        elif event_set is not None:
            rows = event_table(event_set, pair)
            source = (f"event set {event_set.event_set_id} (pinned to market "
                      f"{snapshot.market_id})" if not rows.empty else "no events")
        else:
            rows, source = event_table(None, pair), "no events"
        try:
            payload, pair_notes = market(snapshot, pair, curves=curves, event_rows=rows)
            if payload is not None:
                payload["event_source"] = source
        except Exception as exc:             # one pair's failure must not stop the others
            unavailable[pair] = f"{pair} could not be built from {label}: {exc}"
            continue
        if payload is None:
            unavailable[pair] = pair_notes[-1] if pair_notes else f"{pair}: not in snapshot"
            continue
        try:
            markets, errors = read_market_file({"format": "market_bundle", "version": 1,
                                                "markets": {pair: payload}})
        except Exception as exc:             # raises when its only market is invalid
            markets, errors = {}, {pair: exc}
        if errors or pair not in markets:
            unavailable[pair] = f"{pair} failed market validation: " + "; ".join(
                str(error) for error in errors.values())
            continue
        payloads[pair] = payload
        notes.extend(pair_notes)
    if not payloads:
        return StoreMarkets(snapshot, {}, unavailable, list(dict.fromkeys(notes)),
                            f"no pair could be built from {label}")
    return StoreMarkets(snapshot, payloads, unavailable, list(dict.fromkeys(notes)), None)

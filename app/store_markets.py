"""The app's markets, built from the market data store.

``load_store_markets`` turns one stored snapshot into the app's market
library (``{pair: saved-market dict}``, the shape ``MarketSet.from_payloads``
and the Markets tab use), with a reason for every pair it could not build.

A market's data may be from the valuation date or earlier, never later:
its ``date_time``'s date must be on or before the valuation date.

It never raises. Every failure is reported instead:

* the store cannot be opened or read -> ``problem``, no markets;
* no market with data on or before the valuation date -> ``problem``;
* the latest such market is more than ``MAX_SNAPSHOT_AGE_BDAYS`` business
  days older than the valuation date -> ``problem`` (an explicitly chosen
  market is allowed, with a note);
* a chosen market does not exist, or its data is from after the valuation
  date -> ``problem``;
* the market has no vols or forward points -> ``problem``;
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


def load_store_markets(valuation_date, market_id=None, store=None, pairs=None):
    """The markets of one stored market snapshot for ``valuation_date``.

    By default the latest market with data on or before ``valuation_date``;
    ``market_id`` picks one explicitly. ``store`` defaults to the app's
    market data store; ``pairs`` limits which pairs are built.
    """
    valuation = pd.Timestamp(valuation_date).normalize()
    try:
        if store is None:
            from market_data.store import open_market_data_store
            store = open_market_data_store()
        if market_id is not None:
            snapshot = store.load(market_id=market_id)
        else:
            snapshot = store.load(on_or_before=valuation)
    except Exception as exc:                 # unreadable, locked or corrupt store
        return _failed(f"market data store unavailable: {exc}")

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
        from market_data.market_tables import market
        curves = DiscountCurves(snapshot, store)
    except Exception as exc:
        return _failed(f"cannot build markets: {exc}", snapshot, notes)

    wanted = sorted(set(snapshot.vols) | set(snapshot.forward_points)) if pairs is None else pairs
    payloads, unavailable = {}, {}
    for pair in wanted:
        try:
            payload, pair_notes = market(snapshot, pair, curves=curves)
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

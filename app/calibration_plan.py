"""Which pairs to calibrate up front, and calibrating only what changed.

The app calibrates surfaces before pricing so a progress bar can show it.
Only the pairs in use are calibrated: the Markets tab's active pair, the
Pricer lines' pairs and the selected Portfolio trades' pairs. A pair is
calibrated again only when its market changed (its fingerprint), so adding
a pair calibrates just that pair. Anything not calibrated here still
calibrates on demand when it is priced, so a missed pair costs time, never
a wrong or missing price.

A pair whose market came unchanged from the store (``origins``: pair ->
market_id) goes through the disk cache (``app.calibration_cache``):
restored when calibrated before, saved when not. Any other pair (uploaded,
edited) is calibrated for the session only.

Fail-safes: a pair that fails to calibrate is reported in the result and
skipped (pricing then reports its error per line); it never stops the
other pairs. A failing disk cache falls back to calibrating in the session.
"""
from collections import namedtuple

from app.market_store import normalize_pair

#: ``source``: ``disk``, ``calibrated``, ``session`` or ``failed``.
PairCalibration = namedtuple("PairCalibration", "pair source notes")


def pairs_in_use(available, active_pair=None, pricer_rows=None, trades=()):
    """Sorted pairs to calibrate: in use and loaded (``available``)."""
    active = normalize_pair(active_pair) if active_pair else ""
    wanted = {active} if active else set()
    if pricer_rows is not None and "currency_pair" in getattr(pricer_rows, "columns", ()):
        for value in pricer_rows["currency_pair"]:
            pair = normalize_pair(value) if isinstance(value, str) else ""
            wanted.add(pair or active)          # a blank pair prices on the active one
    wanted.update(normalize_pair(trade.currency_pair) for trade in trades)
    return sorted(pair for pair in wanted if pair and pair in set(available))


def calibrate_pairs(market_set, pairs, fingerprints, done, settings, valuation_date,
                    surface_cache, market_service, origins=None, payloads=None, cache=None,
                    progress=None):
    """Calibrate ``pairs`` whose fingerprint is not in ``done`` (updated in place).

    ``origins``/``payloads`` (pair -> market_id / saved market) route store
    markets through the disk cache. ``progress(position, count, pair)`` is
    called before each pair. Returns ``[PairCalibration]`` for the pairs done.
    """
    from app.calibration_cache import ensure_calibrated
    origins, payloads = origins or {}, payloads or {}
    todo = [pair for pair in pairs if done.get(pair) != fingerprints.get(pair)]
    results = []
    for position, pair in enumerate(todo):
        if progress is not None:
            progress(position, len(todo), pair)
        try:
            # A market still to roll calibrates rolled, in the session: the
            # disk cache holds the quotes' own dates.
            rolls = getattr(market_set, "rolls_when_priced", lambda name: False)(pair)
            if pair in origins and pair in payloads and not rolls:
                try:
                    outcome = ensure_calibrated(payloads[pair], origins[pair], valuation_date,
                                                settings, surface_cache, market_service, cache)
                    results.append(PairCalibration(pair, outcome.source, list(outcome.notes)))
                except Exception as exc:
                    # The disk-cache path failed: calibrate in the session instead.
                    _calibrate_in_session(market_set, pair)
                    results.append(PairCalibration(pair, "session", [
                        f"{pair}: disk cache bypassed ({type(exc).__name__}: {exc})"]))
            else:
                _calibrate_in_session(market_set, pair)
                results.append(PairCalibration(pair, "session", []))
        except Exception as exc:
            results.append(PairCalibration(pair, "failed", [f"{pair}: {exc}"]))
        # Marked done either way: a failing market is not retried every rerun;
        # pricing reports its error per line until the market changes.
        done[pair] = fingerprints.get(pair)
    return results


def _calibrate_in_session(market_set, pair):
    context = market_set.market(pair)
    for expiry in context.pillar_expiries():
        context.calibrate_pillar(expiry)

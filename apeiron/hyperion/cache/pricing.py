"""Cached book pricing: each trade priced once per state, only changes repriced.

``price_trades`` has the signature and result of ``app.trades.service.
price_trades`` plus a ``cache``: every trade's outcome (an option result, a
forward row, an error, or nothing for a cancelled trade) is stored under its
key (``keys.trade_key``); a call prices only the trades it has no outcome for,
in one batch, and returns all of them in the original order. ``verify``
reprices a sample fresh and lists any difference from the cache.
"""
import random

from apeiron.hyperion.cache.keys import market_fingerprint, trade_key
from apeiron.hyperion.cache.store import default_cache

OPTION, FORWARD, ERROR, NONE = "option", "forward", "error", "none"


def _keys(markets, trades, mods, extras, spot_path):
    hashes, keys = {}, {}
    for trade in trades:
        pair = trade.currency_pair
        if pair not in hashes:
            try:
                hashes[pair] = market_fingerprint(markets.market(pair))
            except Exception:
                hashes[pair] = None
        keys[trade.trade_id] = trade_key(trade, hashes[pair], mods, extras, spot_path,
                                         markets.valuation_date)
    return keys


def _outcomes(trades, results, forwards, errors):
    """``({trade_id: (kind, value)}, unmatched errors)`` from a ``price_trades`` result."""
    found = {trade.trade_id: (NONE, None) for trade in trades}
    unmatched = []
    for row in results:
        found[row["trade_id"]] = (OPTION, row)
    for row in forwards:
        found[row["trade_id"]] = (FORWARD, row)
    ids = sorted((t.trade_id for t in trades), key=len, reverse=True)
    for message in errors:
        owner = next((i for i in ids if message.startswith(f"{i}:")), None)
        if owner is not None:
            found[owner] = (ERROR, message)
        else:
            unmatched.append(message)
    return found, unmatched


def price_trades(markets, trades, mods, extras=(), spot_path=None, progress=None, cache=None):
    """``(option results, forward rows, errors)`` as ``service.price_trades``,
    each trade's outcome read from ``cache`` when known, priced otherwise."""
    from app import pricer
    from app.progress import safe
    from app.trades import service
    spot_path = pricer.SMILE_PATH if spot_path is None else spot_path
    cache = default_cache() if cache is None else cache
    report = safe(progress)
    trades = list(trades)
    try:
        keys = _keys(markets, trades, mods, extras, spot_path)
    except Exception:                       # keys unavailable: price everything fresh
        keys = {trade.trade_id: None for trade in trades}
    known, missing = {}, []
    for trade in trades:
        outcome = cache.get(keys[trade.trade_id])
        if outcome is None:
            missing.append(trade)
        else:
            known[trade.trade_id] = outcome
    report(len(known), len(trades), "cached")
    if missing:
        done = len(known)

        def priced(count, _total, label):
            report(done + count, len(trades), label)
        results, forwards, errors = service.price_trades(
            markets, missing, mods, extras=extras, spot_path=spot_path, progress=priced)
        fresh, unmatched = _outcomes(missing, results, forwards, errors)
        for trade in missing:
            known[trade.trade_id] = fresh[trade.trade_id]
            # An error naming no trade could be any "nothing" outcome's: those
            # are not cached, so the next call prices them again.
            if not (unmatched and fresh[trade.trade_id][0] == NONE):
                cache.put(keys[trade.trade_id], fresh[trade.trade_id])
    else:
        unmatched = []
    results, forwards, errors = [], [], []
    rows = _line_numbers(trades, markets.valuation_date)
    for trade in trades:
        kind, value = known[trade.trade_id]
        if kind == OPTION:
            # ``row`` is the line's place in the batch: numbered as a fresh
            # call over these trades would, not as in the batch it was cached from.
            if "row" in value:
                value = dict(value, row=rows[trade.trade_id])
            results.append(value)
        elif kind == FORWARD:
            forwards.append(value)
        elif kind == ERROR:
            errors.append(value)
    return results, forwards, errors + unmatched


def _line_numbers(trades, valuation_date):
    """``{trade_id: n}``: the line number ``service.price_trades`` gives each
    option it prices (live, booked by the valuation date, not expired)."""
    from app.trades.model import LIVE, as_date
    valuation = as_date(valuation_date)
    numbers, count = {}, 0
    for trade in trades:
        if (trade.status == LIVE and trade.is_option and trade.trade_date <= valuation
                and trade.expiry > valuation):
            count += 1
            numbers[trade.trade_id] = count
    return numbers


#: Fields describing where a result sat in its batch, not its value.
_POSITION_FIELDS = {"row"}


def _same(a, b):
    """Equal, counting NaN as equal to NaN (a blank value is the same value)."""
    try:
        if a != a and b != b:
            return True
    except Exception:
        pass
    return a == b


def verify(markets, trades, mods, extras=(), spot_path=None, cache=None, sample=10, seed=0):
    """Reprice up to ``sample`` cached trades fresh; ``[(trade_id, field, cached, fresh)]``
    for every difference (empty when the cache agrees)."""
    from app import pricer
    from app.trades import service
    spot_path = pricer.SMILE_PATH if spot_path is None else spot_path
    cache = default_cache() if cache is None else cache
    trades = list(trades)
    keys = _keys(markets, trades, mods, extras, spot_path)
    cached = [(t, cache.get(keys[t.trade_id])) for t in trades]
    cached = [(t, outcome) for t, outcome in cached if outcome is not None]
    picked = random.Random(seed).sample(cached, min(sample, len(cached)))
    differences = []
    for trade, (kind, value) in picked:
        with pricer.bypass_greek_memo():        # fresh, not from the Greek memo
            results, forwards, errors = service.price_trades(
                markets, [trade], mods, extras=extras, spot_path=spot_path)
        fresh_kind, fresh = _outcomes([trade], results, forwards, errors)[0][trade.trade_id]
        if fresh_kind != kind:
            differences.append((trade.trade_id, "kind", kind, fresh_kind))
        elif isinstance(value, dict):
            for field in sorted((set(value) | set(fresh)) - _POSITION_FIELDS):
                if not _same(value.get(field), fresh.get(field)):
                    differences.append((trade.trade_id, field, value.get(field), fresh.get(field)))
        elif value != fresh:
            differences.append((trade.trade_id, "value", value, fresh))
    return differences

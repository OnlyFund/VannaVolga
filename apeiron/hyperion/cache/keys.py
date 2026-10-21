"""Cache keys: everything a trade's valuation depends on, hashed.

A trade's result is determined by its own terms (id and version included),
its pair's market (every quote, the spot and any spot edit, the settings,
the valuation date), the smile dynamics, the requested Greeks, how a moved
spot reprices, and the code. ``trade_key`` hashes all of them, so a change in
any one gives a different key: a stale result can never be served.
``CACHE_VERSION`` (with the smile cache's version) stands for the code: bump
it when pricing changes.
"""
import hashlib
import json

#: Bump when anything that changes a priced result changes in the code.
CACHE_VERSION = 1


def _text(value):
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))


def _table(frame):
    if frame is None:
        return ""
    try:
        return frame.to_json(orient="split", date_format="iso", double_precision=15)
    except Exception:
        return repr(frame)


def market_fingerprint(context):
    """A hash of everything in a pair's market that pricing reads, or ``None``
    for a market that cannot price (nothing is cached for it)."""
    if getattr(context, "reason", None) is not None or not hasattr(context, "vol_rows"):
        return None
    parts = {
        "pair": context.currency_pair,
        "valuation": str(context.valuation_date),
        "spot": repr(float(context.spot)),
        # The smile dynamics anchor on the market's own spot (a spot edit moves
        # the spot, not the anchor).
        "anchor": repr(float(context.__dict__.get("spot_anchor", context.spot))),
        "divisor": repr(float(context.forward_divisor)),
        "broker_strangle": bool(context.broker_strangle),
        "short_points": context.__dict__.get("short_points") or {},
        "settings": repr(context.settings),
        "vols": _table(context.vol_rows),
        "forwards": _table(context.forward_rows),
        "events": _table(context.event_rows),
    }
    return hashlib.sha256(_text(parts).encode("utf-8")).hexdigest()


def trade_key(trade, market_hash, mods, extras, spot_path, valuation_date):
    """The cache key of one trade's valuation; ``None`` when its market can't price."""
    if market_hash is None:
        return None
    from app.smile_service import SURFACE_CACHE_VERSION
    parts = {
        "v": [CACHE_VERSION, SURFACE_CACHE_VERSION],
        "trade": {name: getattr(trade, name) for name in trade.__dataclass_fields__},
        "market": market_hash,
        "mods": dict(mods) if mods is not None else None,
        "extras": sorted(extras or ()),
        "spot_path": str(spot_path),
        "valuation": str(valuation_date),
    }
    return hashlib.sha256(_text(parts).encode("utf-8")).hexdigest()

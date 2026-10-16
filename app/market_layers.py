"""Where each pair's market in the session came from, and which layer wins.

A pair's market comes from one **source**:

* ``store``: a stored market (``market_id``), loaded for the valuation date;
* ``file``: an uploaded market file (``file``);
* ``manual``: typed into the Markets tab with no store or file behind it;

and is **edited** when its Markets tab tables differ from that source.
Edits win over files, files over the store. A pair that is a file, or
edited, is kept when the store reloads (another valuation date or market);
an unedited store pair is replaced by the new market's copy, or dropped
(and reported unavailable) when the new market cannot build it.

Edits are found by comparing the tables with the source through the app's
own normalisation (``validate_market`` and ``market_payload``) ignoring the
valuation date, so a mere rerun, or a valuation date other than the
market's, never counts as an edit; undoing the edits makes the pair clean
again. Layers are plain dicts so they live in the session state.
"""
import json

STORE, FILE, MANUAL = "store", "file", "manual"


def comparable(payload):
    """A market's content as text, normalised the way the app reads it,
    without its valuation date; ``None`` when it is not a valid market."""
    from app.market_store import market_payload, validate_market
    try:
        loaded = validate_market(payload)
        rebuilt = market_payload(loaded["currency_pair"], "2000-01-03", loaded["spot"],
                                 loaded["forward_divisor"], loaded["broker_strangle"],
                                 loaded["vol_rows"], loaded["forward_rows"], loaded["event_rows"])
    except Exception:
        return None
    rebuilt.pop("valuation_date", None)
    return json.dumps(rebuilt, sort_keys=True, default=str)


def _layer(source, payload, market_id=None, file=None):
    return {"source": source, "market_id": market_id, "file": file,
            "base": comparable(payload) if payload is not None else None,
            "edited": source == MANUAL, "unavailable": None}


def keeps_own_copy(layer):
    """A file or an edited pair: not replaced when the store reloads."""
    return layer is not None and (layer["source"] in (FILE, MANUAL) or layer["edited"])


def merge_store(library, layers, payloads, unavailable, market_id):
    """Bring a store load into the session; returns notes.

    Unedited store pairs take the new copies; store pairs the new market
    cannot build leave the library and are marked unavailable with the
    reason; files and edited pairs are kept.
    """
    notes = []
    for pair, layer in list(layers.items()):
        if layer["source"] == STORE and not layer["edited"] and pair not in payloads:
            library.pop(pair, None)
            reason = unavailable.get(pair, f"{pair} is not in market {market_id}")
            layers[pair] = dict(layer, unavailable=reason)
    for pair, payload in payloads.items():
        layer = layers.get(pair)
        if keeps_own_copy(layer):
            kind = "edited" if layer["edited"] else f"from {layer['file'] or 'a file'}"
            notes.append(f"{pair} kept as {kind}; not reloaded from market {market_id}")
            continue
        library[pair] = payload
        layers[pair] = _layer(STORE, payload, market_id=market_id)
    for pair, reason in unavailable.items():
        if pair not in layers:
            layers[pair] = dict(_layer(STORE, None, market_id=market_id), unavailable=reason)
    return notes


def drop_store(library, layers, reason):
    """No usable stored market (e.g. no data on or before an earlier
    valuation date): clean store pairs leave the library, unavailable with
    ``reason``, so nothing prices on data from another date. Edited pairs
    are the user's own and are kept; returns notes naming them."""
    notes = []
    for pair, layer in list(layers.items()):
        if layer["source"] != STORE:
            continue
        if layer["edited"]:
            notes.append(f"{pair} kept as edited (on market {layer['market_id']}), "
                         f"although no stored market applies now: {reason}")
            continue
        library.pop(pair, None)
        layers[pair] = dict(layer, unavailable=reason)
    return notes


def mark_files(layers, markets, file_name):
    """Uploaded markets: their source becomes the file, unedited."""
    for pair, payload in markets.items():
        layers[pair] = _layer(FILE, payload, file=file_name)


def record_tables(layers, pair, payload):
    """The active pair's tables at the end of a run: returns whether they
    belong in the library (``False`` keeps an unavailable, unedited store
    pair out of pricing)."""
    layer = layers.get(pair)
    if layer is None:
        layers[pair] = _layer(MANUAL, None)
        return True
    edited = layer["source"] == MANUAL or comparable(payload) != layer["base"]
    layers[pair] = dict(layer, edited=edited)
    return edited or not layer["unavailable"]


def origins(layers):
    """``{pair: market_id}`` for unedited, available store pairs: the pairs
    whose smiles may be shared through the disk calibration cache."""
    return {pair: layer["market_id"] for pair, layer in layers.items()
            if layer["source"] == STORE and not layer["edited"] and not layer["unavailable"]}


def unavailable(layers):
    """``{pair: reason}`` for store pairs the current market cannot build
    and that have no copy of their own."""
    return {pair: layer["unavailable"] for pair, layer in layers.items()
            if layer["unavailable"] and not keeps_own_copy(layer)}


def describe(layer):
    """Where a pair's market came from, for the user."""
    if layer is None:
        return "no market"
    if layer["source"] == STORE:
        text = f"market {layer['market_id']}"
    elif layer["source"] == FILE:
        text = f"file {layer['file']}"
    else:
        return "typed in"
    return f"edited, on {text}" if layer["edited"] else text


def group_by_message(messages):
    """``[(pairs, text)]`` from ``{pair: message}``: pairs whose message is the
    same once the leading pair name is removed share one line."""
    groups = {}
    for pair, message in messages.items():
        text = str(message)
        if text.startswith(pair):
            text = text[len(pair):].lstrip(": ").strip()
        groups.setdefault(text, []).append(pair)
    return [(sorted(pairs), text) for text, pairs in groups.items()]


def pair_list(pairs, limit=6):
    """``EURUSD, USDJPY`` or ``EURUSD, USDJPY ... and 58 more``."""
    pairs = sorted(pairs)
    shown = ", ".join(pairs[:limit])
    return shown if len(pairs) <= limit else f"{shown} and {len(pairs) - limit} more"

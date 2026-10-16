"""Bring stored market data into the app session (Streamlit glue).

The session's market library is filled from the market data store (the
latest market on or before the valuation date) whenever the valuation date
or the chosen market changes, through ``app.market_layers``: files and
edited pairs are kept, clean store pairs take the new market's copy. When
the Markets tab's active pair is a clean store pair, its tables are
reloaded too.

Fail-safes: a store problem (no data for the date, unreadable store ...) is
shown in the sidebar; clean store pairs then become unavailable with that
reason, so nothing prices on another date's (possibly future) data, while
files and edited pairs keep working.
"""
import pandas as pd
import streamlit as st

from app.market_layers import STORE, drop_store, merge_store
from app.market_store import normalize_pair

LAYERS, STATUS, KEY = "_market_layers", "_store_status", "_store_markets_key"


def sync_store_markets(valuation_date, apply_loaded_market, market_id=None):
    """Load the store's markets for ``valuation_date`` once per (date, market)."""
    key = (pd.Timestamp(valuation_date).strftime("%Y-%m-%d"), market_id)
    if st.session_state.get(KEY) == key:
        return st.session_state.get(STATUS, {})
    st.session_state[KEY] = key
    try:
        from app.store_markets import load_store_markets
        result = load_store_markets(valuation_date, market_id=market_id)
    except Exception as exc:                     # the loader never raises; belt and braces
        st.session_state[STATUS] = {"problem": f"market data unavailable: {exc}", "notes": []}
        return st.session_state[STATUS]
    snapshot = result.snapshot
    status = {"problem": result.problem, "notes": list(result.notes),
              "market": None if snapshot is None else
              f"market {snapshot.market_id} ({snapshot.market_source}, "
              f"{snapshot.date_time.replace('T', ' ')})",
              "unavailable": dict(result.unavailable)}
    library = st.session_state.setdefault("market_library", {})
    layers = st.session_state.setdefault(LAYERS, {})
    if snapshot is not None and (result.payloads or result.unavailable):
        status["notes"] += merge_store(library, layers, result.payloads, result.unavailable,
                                       snapshot.market_id)
    elif result.problem:
        # Nothing usable for this date: clean store pairs from another date
        # must not keep pricing (they could be future data).
        status["notes"] += drop_store(library, layers, result.problem)
    st.session_state[STATUS] = status
    active = normalize_pair(st.session_state.get("currency_pair", ""))
    layer = layers.get(active)
    if (active in result.payloads and layer is not None and layer["source"] == STORE
            and not layer["edited"]):
        payload = dict(result.payloads[active],
                       valuation_date=pd.Timestamp(valuation_date).strftime("%Y-%m-%d"))
        try:
            apply_loaded_market(payload)
        except Exception as exc:
            status["notes"].append(f"{active} tables not reloaded: {exc}")
            return status
        st.experimental_rerun()
    return status


def render_store_status(status):
    """The market data in use, or why there is none, under the valuation date."""
    if status.get("problem"):
        st.warning("Market data: " + status["problem"])
    elif status.get("market"):
        st.caption("Market data: " + status["market"])
    notes = status.get("notes") or []
    if notes:
        with st.expander(f"Notes ({len(notes)})", expanded=False):
            for note in notes:
                st.caption(note)

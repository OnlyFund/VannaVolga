"""Sidebar market picker: choose which stored snapshot the app prices on.

Under the valuation date: a quick filter (``market_data.snapshot_filter``)
and a picker listing "Latest" and the usable snapshots on or before the
valuation date, newest first, each day's close marked. Nothing loads until
**Load** is clicked, so browsing never recalibrates a market.

"Latest" is resolved once per valuation date (``store_sync``); a snapshot
fetched while the app is open shows as new and loads only when chosen.

Fail-safes, each with a warning: a picked snapshot that is after a new
valuation date, has left the store, or does not load falls back to Latest;
an unreadable store hides the picker and the app runs on Latest as before.
Falling back happens within the run, never by a rerun from the sidebar: a
rerun before the Markets tab draws its widgets would drop their values (the
spot) and leave default tables that look edited.
"""
import pandas as pd
import streamlit as st

from market_data.snapshot_filter import apply_filter, close_of_business, usable_snapshots

#: The loaded pick (``None`` = Latest); the sync key in ``store_sync`` is cleared to reload.
PICKED = "_picked_market_id"
FILTER, CHOICE_VERSION, NOTICE = "market_filter", "_market_choice_version", "_market_picker_notice"
LATEST = None


def _snapshots(valuation_date):
    from app.store_markets import app_market_store
    return app_market_store(pd.Timestamp(valuation_date).normalize()).snapshots()


#: Three-letter source codes; any other source shows its first three letters.
SOURCE_CODES = {"bloomberg": "BBG", "import": "IMP", "sample": "SMP"}


def source_code(source):
    text = str(source)
    return SOURCE_CODES.get(text.lower(), text[:3].upper())


def label(row, closes=frozenset()):
    """``[23] BBG 16-Oct 15:28``, ``· close`` for a day's close."""
    stamp = pd.Timestamp(row["date_time"])
    text = f"[{row['market_id']}] {source_code(row['market_source'])} {stamp:%d-%b %H:%M}"
    return text + (" · close" if int(row["market_id"]) in closes else "")


def _closes(usable):
    """Market ids that are their day's close for their source."""
    ids = set()
    for (day, source), _ in usable.groupby([usable["date_time"].str[:10], "market_source"]):
        best = close_of_business(usable, day, source)
        if best is not None:
            ids.add(int(best["market_id"]))
    return frozenset(ids)


def fall_back(reason):
    """Return to Latest; the warning shows under the picker. The caller loads
    Latest in the same run."""
    from app.ui.store_sync import KEY
    st.session_state[PICKED] = LATEST
    st.session_state.pop(KEY, None)
    st.session_state[NOTICE] = reason


def _load(choice):
    from app.ui.store_sync import KEY
    st.session_state[PICKED] = choice
    st.session_state.pop(KEY, None)
    st.session_state[CHOICE_VERSION] = st.session_state.get(CHOICE_VERSION, 0) + 1


def checked_pick(valuation_date):
    """The picked market id to load, or ``None`` (Latest) after falling back
    when it is after the valuation date or no longer stored."""
    picked = st.session_state.get(PICKED, LATEST)
    if picked is None:
        return LATEST
    try:
        snapshots = _snapshots(valuation_date)
    except Exception:             # the load itself reports a store problem
        return picked
    if int(picked) in set(int(i) for i in usable_snapshots(snapshots, valuation_date)["market_id"]):
        return picked
    why = ("is after the valuation date" if int(picked) in set(snapshots["market_id"])
           else "is no longer in the market data store")
    fall_back(f"Market {picked} {why}; using Latest.")
    return LATEST


def render_market_picker(valuation_date, loaded_latest_id=None):
    """Draw the filter, picker and Load button; return the market id to load
    (``None`` for Latest). ``loaded_latest_id`` is the market Latest resolved to."""
    notice = st.session_state.pop(NOTICE, None)
    if notice:
        st.warning(notice)
    picked = st.session_state.get(PICKED, LATEST)
    try:
        snapshots = _snapshots(valuation_date)
    except Exception as exc:             # the store problem shows in the status below
        st.caption(f"Market picker unavailable: {exc}")
        return picked
    usable = usable_snapshots(snapshots, valuation_date)
    usable_ids = [int(i) for i in usable["market_id"]]

    query = st.text_input("Find market", key=FILTER, placeholder="e.g. bloomberg 16oct",
                          help="A source (bloomberg), a date (16oct, 2026-10-16) or both. "
                               "A date proposes that day's close: the snapshot nearest "
                               "15:00 New York. Nothing loads until you click Load.")
    result = apply_filter(snapshots, query, valuation_date)
    if result.message:
        st.caption(result.message)
    closes = _closes(usable)
    rows = {int(r["market_id"]): r for _, r in usable.iterrows()}
    newest = usable_ids[0] if usable_ids else None
    latest_text = "Latest"
    if loaded_latest_id is not None and int(loaded_latest_id) in rows:
        latest_text += f" ({label(rows[int(loaded_latest_id)], closes)})"
    if newest is not None and loaded_latest_id is not None and newest != int(loaded_latest_id):
        latest_text += f" · new: [{newest}]"
    options = [LATEST] + [int(i) for i in result.snapshots["market_id"]]
    if picked is not None and picked not in options:
        options.append(picked)               # the loaded pick stays visible
    proposed = result.selected if result.selected is not None else picked
    index = options.index(proposed) if proposed in options else 0
    choice = st.selectbox(
        "Market data", options, index=index,
        format_func=lambda i: latest_text if i is None else label(rows[i], closes),
        key=f"market_choice_{st.session_state.get(CHOICE_VERSION, 0)}_{query}",
        help="Latest: the newest snapshot on or before the valuation date. Spot/depo-only "
             "snapshots cannot price and are not listed.")
    hidden = len(snapshots[pd.to_datetime(snapshots["date_time"].str[:10]) <=
                           pd.Timestamp(valuation_date).normalize()]) - len(usable)
    if hidden:
        st.caption(f"{hidden} spot/depo-only snapshots not listed.")
    stale_latest = choice is LATEST and newest is not None and loaded_latest_id is not None \
        and newest != int(loaded_latest_id)
    pending = choice != picked or stale_latest
    # A callback, not a rerun from here: it runs before the next run, which
    # then loads the choice at the top (see the module docstring).
    st.button("Load", key="picker_load_market", disabled=not pending, on_click=_load,
              args=(choice,), help="Load the chosen snapshot. Loaded files and edited "
                                   "pairs stay.")
    if pending:
        st.caption("Chosen, not loaded: click Load.")
    return picked

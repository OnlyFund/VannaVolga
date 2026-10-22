"""Sidebar market picker: choose which stored snapshot the app prices on.

Under the valuation date: a quick filter (``market_data.snapshot_filter``)
and a picker listing "Latest" and the usable snapshots on or before the
valuation date, newest first, each day's close marked. Nothing loads until
**Load** is clicked, so browsing never recalibrates a market.

"Latest" is resolved once per valuation date (``store_sync``); a snapshot
fetched while the app is open shows as new and loads only when chosen.

**Link to market data date** (on by default): the valuation date is the
loaded snapshot's date. Loading a snapshot (or Latest) moves the date to it;
changing the date loads that day's close, or, when the day has none, the
last good snapshot before it and its date, with a note. The list then runs
to today, since choosing a snapshot moves the date. Unlinked, the date is
free and a snapshot on or before it is valued forward (the old behaviour);
re-linking moves the date back to the loaded snapshot's date.

Fail-safes, each with a warning: a picked snapshot that is after a new
valuation date, has left the store, or does not load falls back to Latest;
an unreadable store hides the picker and the app runs on Latest as before.
Falling back happens within the run, never by a rerun from the sidebar: a
rerun before the Markets tab draws its widgets would drop their values (the
spot) and leave default tables that look edited.
"""

import pandas as pd
import streamlit as st

from market_data.snapshot_filter import (  # noqa: F401  (label, source_code re-exported)
    apply_filter, close_of_business, data_day, data_days, label, linked_snapshot, snapshot_day,
    source_code, usable_snapshots,
)

#: The loaded pick (``None`` = Latest); the sync key in ``store_sync`` is cleared to reload.
PICKED = "_picked_market_id"
FILTER, CHOICE_VERSION, NOTICE = "market_filter", "_market_choice_version", "_market_picker_notice"
LATEST = None
#: The link checkbox; the valuation date widget's key.
LINK, DATE = "link_market_date", "valuation_date"


def _snapshots(valuation_date):
    """The store's snapshots, each with its FX date (``fx_date``, 17:00 New York rollover)."""
    from app.store_markets import app_market_store
    return with_fx_dates(app_market_store(pd.Timestamp(valuation_date).normalize()).snapshots())


def with_fx_dates(frame):
    """``frame`` (``MarketDataStore.snapshots`` rows) with an ``fx_date`` column."""
    from apeiron.gaia.market_dates.rollover import fx_date
    frame = frame.copy()
    zones = frame["timezone"] if "timezone" in frame.columns else [None] * len(frame)
    frame["fx_date"] = [str(fx_date(stamp, zone)) for stamp, zone in zip(frame["date_time"], zones)]
    return frame




def _closes(usable):
    """Market ids that are their day's close for their source."""
    ids = set()
    for (day, source), _ in usable.groupby([data_days(usable), "market_source"]):
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


def linked():
    return bool(st.session_state.get(LINK, True))


def _today():
    """Today's FX date (17:00 New York rollover; weekends are Monday's)."""
    from apeiron.gaia.market_dates.rollover import fx_today
    return pd.Timestamp(fx_today())


def horizon(valuation_date):
    """The last date a snapshot may be from: today when linked (choosing one
    moves the valuation date), else the valuation date."""
    return _today() if linked() else pd.Timestamp(valuation_date)


def _set_date(day):
    st.session_state[DATE] = pd.Timestamp(day).date()


def _pick(choice):
    from app.ui.store_sync import KEY
    st.session_state[PICKED] = choice
    st.session_state.pop(KEY, None)
    st.session_state[CHOICE_VERSION] = st.session_state.get(CHOICE_VERSION, 0) + 1


def init_dates():
    """Before the date widget: link on by default and, on the first run when
    linked, the valuation date set to Latest's date."""
    st.session_state.setdefault(LINK, True)
    if DATE in st.session_state or not linked():
        return
    try:
        usable = usable_snapshots(_snapshots(_today()), _today())
    except Exception:                    # no store: today, as before
        return
    if not usable.empty:
        _set_date(data_day(usable.iloc[0]))


def on_date_change():
    """Linked: the new date loads its close, or the last good snapshot and its date."""
    if not linked():
        return
    day = pd.Timestamp(st.session_state.get(DATE))
    try:
        snapshots = _snapshots(day)
    except Exception:                    # the load reports the store problem
        return
    market_id, linked_day, note = linked_snapshot(snapshots, day)
    _pick(market_id)
    _set_date(linked_day)
    if note:
        st.session_state[NOTICE] = note


def on_link_change():
    """Re-linking moves the valuation date to the loaded snapshot's date."""
    loaded = st.session_state.get("_loaded_market_id")
    if not linked() or loaded is None:
        return
    try:
        day = snapshot_day(_snapshots(_today()), loaded)
    except Exception:
        return
    if day is not None:
        _set_date(day)


def _load(choice):
    _pick(choice)
    if not linked():
        return
    try:
        snapshots = _snapshots(_today())
        if choice is LATEST:
            usable = usable_snapshots(snapshots, _today())
            day = None if usable.empty else data_day(usable.iloc[0])
        else:
            day = snapshot_day(snapshots, choice)
    except Exception:
        return
    if day is not None:
        _set_date(day)


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


def render_market_picker(valuation_date, loaded_latest_id=None, last_date=None):
    """Draw the filter, picker and Load button; return the market id to load
    (``None`` for Latest). ``loaded_latest_id`` is the market Latest resolved
    to; ``last_date`` the last date a listed snapshot may be from (``horizon``)."""
    last_date = pd.Timestamp(valuation_date if last_date is None else last_date)
    notice = st.session_state.pop(NOTICE, None)
    if notice:
        st.warning(notice)
    picked = st.session_state.get(PICKED, LATEST)
    try:
        snapshots = _snapshots(valuation_date)
    except Exception as exc:             # the store problem shows in the status below
        st.caption(f"Market picker unavailable: {exc}")
        return picked
    usable = usable_snapshots(snapshots, last_date)
    usable_ids = [int(i) for i in usable["market_id"]]

    query = st.text_input("Find market", key=FILTER, placeholder="e.g. bloomberg 16oct",
                          help="A source (bloomberg), a date (16oct, 2026-10-16) or both. "
                               "A date proposes that day's close: the snapshot nearest "
                               "15:00 New York. Nothing loads until you click Load.")
    result = apply_filter(snapshots, query, last_date)
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
        help="Latest: the newest snapshot (on or before the valuation date when not "
             "linked). Spot/depo-only snapshots cannot price and are not listed.")
    hidden = len(snapshots[pd.to_datetime(data_days(snapshots)) <=
                           last_date.normalize()]) - len(usable)
    if hidden:
        st.caption(f"{hidden} spot/depo-only snapshots not listed.")
    stale_latest = choice is LATEST and newest is not None and loaded_latest_id is not None \
        and newest != int(loaded_latest_id)
    pending = choice != picked or stale_latest
    # A callback, not a rerun from here: it runs before the next run, which
    # then loads the choice at the top (see the module docstring).
    load_col, refresh_col = st.columns(2)
    with load_col:
        st.button("Load", key="picker_load_market", disabled=not pending, on_click=_load,
                  args=(choice,), help="Load the chosen snapshot. Loaded files and edited "
                                       "pairs stay.")
    with refresh_col:
        # The list is read from the store on every run, so the click lists a
        # fetch made while the app is open. On Latest it also loads it (as
        # restarting the app would); a picked market stays loaded.
        st.button("Refresh", key="picker_refresh_snapshots",
                  on_click=_load if picked is LATEST else None,
                  args=(LATEST,) if picked is LATEST else None,
                  help="Re-read the market data snapshots, e.g. after a fetch. On Latest, "
                       "the newest snapshot loads; a picked market stays loaded (pick the "
                       "new one and click Load).")
    if pending:
        st.caption("Chosen, not loaded: click Load.")
    return picked

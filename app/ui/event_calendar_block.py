"""The Markets tab's event calendar tools: source, save, versions, CSV.

Logic is ``app.event_calendar``; this block draws it under the event table.
Every store call is guarded: a failure is shown and changes nothing.
"""
import getpass
import hashlib

import pandas as pd
import streamlit as st

from app import event_calendar

PENDING_KEY = "_pending_event_rows"
MESSAGES_KEY = "_event_calendar_messages"


def adopt_pending_event_rows():
    """Make a table queued by the last run (a CSV or a version) the event table.

    Called before the event editor is drawn, so the editor starts from it.
    """
    pending = st.session_state.pop(PENDING_KEY, None)
    if pending is not None:
        st.session_state.event_rows = pending
        st.session_state.pop("event_editor", None)


def _queue(table, message):
    st.session_state[PENDING_KEY] = table
    st.session_state[MESSAGES_KEY] = [("success", message)]
    st.experimental_rerun()


def _store(valuation_date):
    from app.store_markets import app_market_store
    return app_market_store(pd.Timestamp(valuation_date).normalize())


def render_event_tools(pair, valuation_date, event_rows):
    """Source caption, Save, version history and CSV for ``pair``'s calendar."""
    for level, message in st.session_state.pop(MESSAGES_KEY, []):
        getattr(st, level)(message)
    source = st.session_state.get("event_source")
    if source and st.session_state.get("_active_market_pair") == pair:
        st.caption(f"Loaded from: {source}. Edits apply now; save them to keep them.")
    if len(pair) != 6 or not pair.isalpha():
        return
    note_col, save_col = st.columns([3, 1])
    with note_col:
        note = st.text_input("Note for this version", key="event_calendar_note",
                             placeholder="e.g. NFP weight raised to 1.5")
    with save_col:
        st.write("")
        clicked = st.button(f"Save {pair} calendar", key="event_calendar_save",
                            help="Saves the table as a new version of this pair's calendar. "
                                 "Earlier versions are kept. A version reaches valuation "
                                 "dates from its save date on.")
    if clicked:
        try:
            user = getpass.getuser()
        except Exception:
            user = ""
        try:
            version, count = event_calendar.save(_store(valuation_date), pair, event_rows,
                                                 user=user, note=note)
        except Exception as exc:                 # invalid rows or store trouble: nothing saved
            st.error(f"Calendar not saved: {exc}")
        else:
            saved = pd.Timestamp.now()
            st.session_state["event_source"] = (f"{pair} calendar v{version}, saved "
                                                f"{saved:%Y-%m-%dT%H:%M}"
                                                + (f" by {user}" if user else ""))
            st.session_state["_active_market_pair"] = pair
            later = pd.Timestamp(valuation_date).normalize() < event_calendar.applies_from(saved)
            st.success(f"Saved {pair} calendar version {version} ({count} events). It applies "
                       f"to valuation dates from {saved:%d-%b-%Y}"
                       + ("; this earlier valuation date keeps the calendar saved by then "
                          "when the market is reloaded." if later else "."))

    with st.expander("Calendar versions and CSV", expanded=False):
        try:
            versions = _store(valuation_date).pair_event_versions(pair)
        except Exception as exc:
            versions = None
            st.warning(f"Versions unavailable: {exc}")
        if versions is not None and versions.empty:
            st.caption(f"No saved {pair} calendar yet: markets use their pinned event set.")
        elif versions is not None:
            st.dataframe(versions, hide_index=True, use_container_width=True)
            chosen = st.selectbox("Version", list(versions["version"]),
                                  key=f"event_version_{pair}")
            try:
                found = _store(valuation_date).load_pair_events(pair, version=chosen)
            except Exception as exc:
                found = None
                st.warning(f"Version {chosen} unavailable: {exc}")
            if found is not None:
                table = event_calendar.event_rows_table(found.rows)
                st.dataframe(table, hide_index=True, use_container_width=True)
                if st.button(f"Use version {chosen} in the table", key="event_version_use"):
                    _queue(table, f"Version {chosen} is in the table; save to make it the "
                                  "latest.")
        st.download_button("Download the table as CSV",
                           event_calendar.csv_text(pair, event_rows),
                           file_name=f"{pair}_events.csv", mime="text/csv",
                           key="event_calendar_download")
        upload = st.file_uploader("Load a CSV into the table (pair,date,event,variance_days)",
                                  type=["csv"], key="event_calendar_upload")
        if upload is not None:
            data = upload.getvalue()
            digest = hashlib.sha1(data).hexdigest()
            if st.session_state.get("_event_csv_digest") != digest:
                st.session_state["_event_csv_digest"] = digest     # load each file once
                try:
                    table, notes = event_calendar.rows_from_csv(data, pair)
                except Exception as exc:
                    st.error(f"CSV not loaded: {exc}")
                else:
                    _queue(table, f"Loaded {len(table)} {pair} events from the CSV"
                                  + (f" ({'; '.join(notes)})" if notes else "")
                                  + "; save to keep them.")

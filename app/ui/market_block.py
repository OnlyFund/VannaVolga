"""Market input block for the Streamlit application."""

import copy
import json
import numpy as np
import pandas as pd
import streamlit as st
from pandas.tseries.offsets import BDay
from app.market_service import pair_delivery, pair_divisor
from app.market_store import (
    bundle_payload, choose_active_pair, is_pair, market_payload as build_market_payload,
    pair_broker_strangle, read_market_file, validate_market,
)
from app.smile_service import DiagnosticsCollector


def _store_active_market(library, currency_pair, payload):
    """Keep the market set's copy of the active pair in step with its edits.

    Only a valid market replaces the stored one, so a half-edited table never
    overwrites the last good copy that switching back or saving all would use.
    The pair's layer records whether the tables differ from their source
    (store market or file); an unedited store pair the current market cannot
    build stays out of the market set.
    """
    from app.market_layers import describe, record_tables
    if not is_pair(currency_pair):
        return
    try:
        validate_market(payload)
    except Exception as exc:
        if currency_pair in library:
            st.caption(f"Market set keeps the last valid {currency_pair} copy: {exc}")
        else:
            st.caption(f"{currency_pair} is not in the market set until it is valid: {exc}")
        return
    layers = st.session_state.setdefault("_market_layers", {})
    if record_tables(layers, currency_pair, payload):
        library[currency_pair] = payload
    else:
        library.pop(currency_pair, None)
        st.warning(f"{currency_pair}: {layers[currency_pair]['unavailable']}. These tables are "
                   "from an earlier market; edit them to price on them.")
    st.caption(f"{currency_pair} source: {describe(layers.get(currency_pair))}")
    st.session_state._active_market_pair = currency_pair


def _activate_market(payload, apply_loaded_market, valuation_date=None):
    """Queue ``payload`` as the active market; report failure without raising.

    With ``valuation_date`` the market keeps that date instead of the one it
    was saved for: switching pairs must not move the date every pair is
    valued at.
    """
    if valuation_date is not None:
        payload = dict(payload, valuation_date=pd.Timestamp(valuation_date).date().isoformat())
    try:
        apply_loaded_market(payload)
    except Exception as exc:
        st.error(f"Could not switch to {payload.get('currency_pair', 'market')}: {exc}")
        return False
    return True


def _load_market_file(raw, library, current_pair, apply_loaded_market, file_name=""):
    """Load a single-pair or multi-pair file into the market set.

    Valid pairs are added (replacing stored copies) and one becomes active:
    the current pair when the file has it, else the file's first. Invalid
    pairs are skipped and reported. Nothing changes when the file is
    unreadable or has no valid pair. Returns whether a rerun is needed.
    """
    try:
        markets, errors = read_market_file(raw)
        active = choose_active_pair(markets, current_pair)
        apply_loaded_market(markets[active])
    except Exception as exc:
        st.error(f"Could not load market: {exc}")
        return False
    library.update(markets)
    from app.market_layers import mark_files
    mark_files(st.session_state.setdefault("_market_layers", {}), markets, file_name or "file")
    messages = [("success", "Loaded {} market{}: {}; active {}.".format(
        len(markets), "" if len(markets) == 1 else "s", ", ".join(markets), active))]
    messages += [("warning", f"Skipped {label}: {error}") for label, error in errors.items()]
    st.session_state._market_load_messages = messages
    return True


def _persist_editor_changes(data_key, editor_key):
    """Apply new data-editor changes while preserving the active editor cell."""
    editor_state = st.session_state.get(editor_key) or {}
    previous_key = f"_last_delta_{editor_key}"
    previous_state = st.session_state.get(previous_key, {})
    frame = st.session_state[data_key].copy(deep=True).reset_index(drop=True)

    previous_edits = previous_state.get("edited_rows", {})
    for row_index, changes in editor_state.get("edited_rows", {}).items():
        row_index = int(row_index)
        for column, value in changes.items():
            old_value = previous_edits.get(row_index, previous_edits.get(str(row_index), {}))
            if (
                column in frame.columns
                and 0 <= row_index < len(frame)
                and (column not in old_value or repr(old_value[column]) != repr(value))
            ):
                frame.at[row_index, column] = value

    old_deleted = previous_state.get("deleted_rows", [])
    new_deleted = editor_state.get("deleted_rows", [])
    deleted = sorted(
        (int(row_index) for row_index in new_deleted[len(old_deleted):]),
        reverse=True,
    )
    for row_index in deleted:
        if 0 <= row_index < len(frame):
            frame = frame.drop(index=row_index)

    old_added_count = len(previous_state.get("added_rows", []))
    for added_row in editor_state.get("added_rows", [])[old_added_count:]:
        frame = pd.concat(
            [frame, pd.DataFrame([added_row], columns=frame.columns)],
            ignore_index=True,
        )

    st.session_state[data_key] = frame.reset_index(drop=True)
    # Streamlit accumulates editor deltas. Keep a snapshot so older changes are
    # not replayed while the widget key and its cursor remain stable.
    st.session_state[previous_key] = copy.deepcopy(editor_state)


def _render_bf_convention(currency_pair):
    """The active pair's BF quote convention, editable for already converted
    two-vol butterflies.

    A loaded market brings its own flag; a pair typed without one starts at
    its convention. Anything unreadable falls back to the pair convention.
    """
    convention = pair_broker_strangle(currency_pair)
    if st.session_state.get("_broker_strangle_pair") != currency_pair:
        st.session_state["broker_strangle"] = convention
        st.session_state["_broker_strangle_pair"] = currency_pair
    if not isinstance(st.session_state.get("broker_strangle"), bool):
        st.session_state["broker_strangle"] = convention
    st.checkbox(
        f"{currency_pair} BF quotes are broker strangles", key="broker_strangle",
        help="Follows the pair convention (G10 broker strangles, EM smile strangles). "
             "Untick only for already converted two-vol butterflies.",
    )
    if st.session_state.get("broker_strangle", True) != convention:
        st.caption("Differs from the {} convention ({}).".format(
            currency_pair, "broker strangles" if convention else "smile strangles"))


def render_market_block(markets_tab, valuation_date, apply_loaded_market):
    """Render market inputs and return the current market tables and spot."""
    with markets_tab:
        st.subheader("Market inputs")
        market_col1, market_col2, market_col3 = st.columns(3)
        with market_col1:
            spot = st.number_input("Spot", format="%.6f", key="spot")
        with market_col2:
            divisor_placeholder = st.empty()
        with market_col3:
            currency_pair = st.text_input("Currency pair", key="currency_pair")
        currency_pair = currency_pair.replace("/", "").replace(" ", "").upper()
        library = st.session_state.setdefault("market_library", {})
        active_pair = st.session_state.get("_active_market_pair")
        if active_pair and currency_pair != active_pair and currency_pair in library:
            # Typing a stored pair switches to its market; the previous pair's
            # edits were stored at the end of the last run.
            if _activate_market(library[currency_pair], apply_loaded_market, valuation_date):
                st.experimental_rerun()
        for level, message in st.session_state.pop("_market_load_messages", []):
            getattr(st, level)(message)
        forward_divisor = pair_divisor(currency_pair, st.session_state.get("forward_divisor", 10000.0))
        st.session_state.forward_divisor = forward_divisor
        divisor_placeholder.caption("Forward divisor (automatic): {:g}".format(forward_divisor))
        delivery_diagnostics = DiagnosticsCollector()
        try:
            from helper_functions.fx_conventions import get_pair
            convention = get_pair(currency_pair)
            st.caption("Automatic {}: ATM {}; premium adjusted {}; delta switches beyond {}; settlement T+{}.".format(
                currency_pair, convention.atm, convention.premium_adjusted,
                convention.fwd_delta_beyond, convention.spot_lag))
        except (ImportError, KeyError, ValueError, RuntimeError) as exc:
            st.warning("Unsupported pair {}: using default conventions.".format(currency_pair))
        _render_bf_convention(currency_pair)
    
        vol_tab, forwards_tab, events_tab = st.tabs(
            ["Vol data", "Forward points", "Event calendar"]
        )
    
    
    def default_market_tables(as_of):
        tenors = ["1W", "2W", "1M", "2M", "3M", "6M", "9M", "1Y"]
        business_days = [5, 10, 21, 42, 63, 126, 189, 252]
        expiries = [pd.Timestamp(as_of) + BDay(days) for days in business_days]
        vol_rows = pd.DataFrame({
            "tenor": tenors, "expiry": expiries,
            "atm": [7.0] * 8, "rr_25": [0.5] * 8, "bf_25": [0.2] * 8,
            "rr_10": [0.925] * 8, "bf_10": [0.76] * 8,
        })
        forward_rows = pd.DataFrame({
            "tenor": tenors,
            "delivery_date": [pair_delivery(currency_pair, expiry, delivery_diagnostics) for expiry in expiries],
            "fwd_pts": [10.0] * 8,
            "term_discount": [0.05] * 8,
        })
        event_rows = pd.DataFrame({
            "event_date": pd.Series(dtype="datetime64[ns]"),
            "variance_days": pd.Series(dtype=float),
        })
        return vol_rows, forward_rows, event_rows
    
    
    if "vol_rows" not in st.session_state:
        st.session_state.vol_rows, st.session_state.forward_rows, st.session_state.event_rows = default_market_tables(valuation_date)

    # Streamlit's editor requires its configured column type to match the
    # dataframe dtype. Added rows arrive through the editor delta as strings,
    # so restore the market table's date and numeric dtypes before rendering.
    vol_input = st.session_state.vol_rows.copy(deep=True)
    vol_input["expiry"] = pd.to_datetime(
        vol_input["expiry"], errors="coerce", utc=True
    ).dt.tz_localize(None)
    for column in ("atm", "rr_25", "bf_25", "rr_10", "bf_10"):
        vol_input[column] = pd.to_numeric(vol_input[column], errors="coerce")
    st.session_state.vol_rows = vol_input

    forward_input = st.session_state.forward_rows.copy(deep=True)
    if "term_discount" not in forward_input:
        forward_input["term_discount"] = 0.05
    forward_input["delivery_date"] = pd.to_datetime(
        forward_input["delivery_date"], errors="coerce", utc=True
    ).dt.tz_localize(None)
    forward_input["fwd_pts"] = pd.to_numeric(forward_input["fwd_pts"], errors="coerce")
    forward_input["term_discount"] = pd.to_numeric(
        forward_input["term_discount"], errors="coerce"
    )
    st.session_state.forward_rows = forward_input

    event_input = st.session_state.event_rows.copy(deep=True)
    event_input["event_date"] = pd.to_datetime(
        event_input["event_date"], errors="coerce", utc=True
    ).dt.tz_localize(None)
    event_input["variance_days"] = pd.to_numeric(
        event_input["variance_days"], errors="coerce"
    )
    st.session_state.event_rows = event_input

    with vol_tab:
        st.caption("Expiry dates use MM/DD/YYYY. Volatility quotes are percentage points.")
        st.caption("BF input: " + ("broker common-vol strangles (converted per quoted expiry)"
                                  if st.session_state.get("broker_strangle", True)
                                  else "already converted two-vol butterflies"))
        vol_rows = st.data_editor(
            vol_input.copy(deep=True),
            num_rows="dynamic", use_container_width=True, hide_index=True,
            key="vol_editor",
            on_change=_persist_editor_changes,
            args=("vol_rows", "vol_editor"),
            column_config={
                "tenor": st.column_config.TextColumn("Tenor"),
                "expiry": st.column_config.DateColumn("Expiry date", format="MM/DD/YYYY"),
                "atm": st.column_config.NumberColumn("ATM vol", min_value=0.0001, format="%.4f"),
                "rr_25": st.column_config.NumberColumn("25D RR", format="%.4f"),
                "bf_25": st.column_config.NumberColumn("25D BF", format="%.4f"),
                "rr_10": st.column_config.NumberColumn("10D RR", format="%.4f"),
                "bf_10": st.column_config.NumberColumn("10D BF", format="%.4f"),
            },
        )
        st.session_state.vol_rows = vol_rows.copy(deep=True)
    
    with forwards_tab:
        forward_rows = forward_input.copy(deep=True)
        expiry_by_tenor = {
            str(tenor).strip().upper(): expiry
            for tenor, expiry in zip(vol_rows["tenor"], vol_rows["expiry"])
        }
        derived_delivery = forward_rows["tenor"].map(
            lambda tenor: pair_delivery(currency_pair, expiry_by_tenor[str(tenor).strip().upper()], delivery_diagnostics)
            if str(tenor).strip().upper() in expiry_by_tenor else pd.NaT
        )
        current_delivery = pd.to_datetime(forward_rows["delivery_date"], errors="coerce").dt.normalize()
        desired_delivery = pd.to_datetime(derived_delivery, errors="coerce").dt.normalize()
        if not current_delivery.reset_index(drop=True).equals(
            desired_delivery.reset_index(drop=True)
        ):
            forward_rows["delivery_date"] = derived_delivery
            st.session_state.forward_rows = forward_rows
        # Expiry edits change the disabled delivery-date column. Give the
        # editor a new key only when its tenor-to-expiry mapping changes;
        # clearing a live widget's state during an unrelated rerun can restore
        # its previous edit for one run.
        forward_editor_key = "forward_editor_" + currency_pair + "_" + "_".join(
            pd.to_datetime(vol_rows["expiry"], errors="coerce")
            .dt.strftime("%Y%m%d")
            .fillna("")
            .tolist()
        )
        forward_rows = st.data_editor(
            st.session_state.forward_rows.copy(deep=True),
            num_rows="dynamic", use_container_width=True, hide_index=True,
            key=forward_editor_key,
            on_change=_persist_editor_changes,
            args=("forward_rows", forward_editor_key),
            column_config={
                "tenor": st.column_config.TextColumn("Tenor"),
                "delivery_date": st.column_config.DateColumn(
                    "Delivery date (pair calendar/lag)", format="MM/DD/YYYY", disabled=True
                ),
                "fwd_pts": st.column_config.NumberColumn("Forward points", format="%.4f"),
                "term_discount": st.column_config.NumberColumn(
                    "Simple term deposit rate (0.05 = 5%)", format="%.4f"
                ),
            },
        )
        st.session_state.forward_rows = forward_rows.copy(deep=True)
    
    with events_tab:
        event_rows = st.data_editor(
            event_input.copy(deep=True),
            num_rows="dynamic", use_container_width=True, hide_index=True,
            key="event_editor",
            on_change=_persist_editor_changes,
            args=("event_rows", "event_editor"),
            column_config={
                "event_date": st.column_config.DateColumn("Event date", format="MM/DD/YYYY"),
                "variance_days": st.column_config.NumberColumn("Variance days", format="%.4f"),
            },
        )
        st.session_state.event_rows = event_rows.copy(deep=True)
    
    with markets_tab:
        pair_token = "".join(ch for ch in currency_pair.upper() if ch.isalnum()) or "MARKET"
        date_token = f"{pd.Timestamp(valuation_date):%Y%m%d}"
        market_filename = f"{pair_token}_{date_token}.json"
        st.caption(f"Market name: {pair_token}_{date_token}")
        market_payload = build_market_payload(
            currency_pair, valuation_date, spot, forward_divisor,
            st.session_state.get("broker_strangle", True),
            vol_rows, forward_rows, event_rows,
        )
        _store_active_market(library, currency_pair, market_payload)

        st.markdown("**Market set**")
        if library:
            pairs = list(library)
            current = pairs.index(currency_pair) if currency_pair in pairs else None
            set_col, remove_col = st.columns([2, 2])
            with set_col:
                chosen = st.selectbox(
                    "Active market", pairs, index=0 if current is None else current,
                    help="Markets held in this session. Typing a stored pair "
                         "in Currency pair also switches to it; typing a new "
                         "pair copies the current tables to it.",
                ) if current is not None else None
                if chosen is not None and chosen != currency_pair:
                    if _activate_market(library[chosen], apply_loaded_market, valuation_date):
                        st.experimental_rerun()
            with remove_col:
                removable = [pair for pair in pairs if pair != currency_pair]
                # A fresh key after each removal clears the selection.
                version = st.session_state.get("_market_set_version", 0)
                to_remove = st.multiselect(
                    "Remove from set", removable, key=f"remove_market_pairs_{version}"
                )
                if st.button("Remove", key="remove_markets", disabled=not to_remove):
                    for pair in to_remove:
                        library.pop(pair, None)
                        st.session_state.get("_market_layers", {}).pop(pair, None)
                    st.session_state._market_set_version = version + 1
                    st.experimental_rerun()

        save_col, save_all_col, upload_col, load_col = st.columns([1, 1, 2, 1])
        with save_col:
            st.download_button(
                "Save market", data=json.dumps(market_payload, indent=2),
                file_name=market_filename, mime="application/json", key="save_market",
                help=f"Only {currency_pair}, as currently edited.",
            )
        with save_all_col:
            st.download_button(
                f"Save all markets ({len(library)})",
                data=json.dumps(bundle_payload(library, valuation_date), indent=2),
                file_name=f"MARKETS_{date_token}.json", mime="application/json",
                key="save_market_bundle", disabled=not library,
                help="Every pair in the market set in one file: " + ", ".join(library),
            )
        with upload_col:
            uploaded_market = st.file_uploader(
                "Choose saved market or market-set JSON", type=["json"], key="load_market_file"
            )
        with load_col:
            if st.button("Load market", key="load_market"):
                if uploaded_market is None:
                    st.error("Choose a saved market JSON file first.")
                elif _load_market_file(uploaded_market.getvalue(), library,
                                       currency_pair, apply_loaded_market,
                                       file_name=uploaded_market.name):
                    st.experimental_rerun()

    return spot, forward_divisor, currency_pair, vol_rows, forward_rows, event_rows

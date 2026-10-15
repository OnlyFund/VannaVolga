"""Blotter and Portfolio tabs: the booked trades, outside the Pricer scratchpad."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app import pricer
from app.trades import service
from app.trades.model import AMENDABLE, CANCELLED, FX_FORWARD, FX_OPTION, LIVE
from app.trades.store import TradeStoreError, open_trade_store

#: Where the app keeps its trade store unless ``VV_TRADE_STORE`` says otherwise.
DEFAULT_STORE = Path(__file__).resolve().parents[2] / "data" / "local" / "trades.sqlite"

ALL = "All"
TRADE_COLUMNS = [
    "trade_id", "version", "status", "product_type", "currency_pair", "buy_sell", "notional",
    "strike", "call_put", "contract_rate", "expiry", "premium", "premium_ccy", "book",
    "counterparty", "trade_date", "package_id",
]


def trade_store():
    """The session's trade store and any problem opening it.

    Opened once per session. A store that cannot be written opens read-only;
    one that cannot be read at all is ``None`` with the reason.
    """
    import os
    if "_trade_store" not in st.session_state:
        path = Path(os.environ.get("VV_TRADE_STORE", str(DEFAULT_STORE)))
        st.session_state["_trade_store"] = open_trade_store(path)
    return st.session_state["_trade_store"]


def _store_banner(store, problem):
    if store is None:
        st.error(f"Trades cannot be shown: {problem}")
        return False
    if problem:
        (st.warning if store.read_only else st.info)(problem)
    return True


def _amount(value, decimals=0):
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    number = float(value)
    if abs(number) < 0.5 * 10 ** -decimals:
        number = 0.0          # no "-0"
    return f"{number:,.{decimals}f}"


def _rate(pair, value):
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    return f"{float(value):.{3 if 'JPY' in str(pair) else 5}f}"


def trades_frame(trades):
    """Trades for display: amounts with separators, rates at pair precision."""
    rows = []
    for trade in trades:
        row = {column: getattr(trade, column) for column in TRADE_COLUMNS}
        row["notional"] = _amount(trade.notional)
        row["premium"] = _amount(trade.premium, 2)
        row["strike"] = "" if trade.strike is None else pricer.format_strike_for_pair(
            trade.currency_pair, trade.strike)
        row["contract_rate"] = _rate(trade.currency_pair, trade.contract_rate)
        rows.append(row)
    return pd.DataFrame(rows, columns=TRADE_COLUMNS)


def _display(rows, first=("trade_id",), amounts=(), rates=()):
    """Result rows for display, ``first`` columns leading."""
    frame = pd.DataFrame(rows)
    for column in amounts:
        if column in frame:
            frame[column] = frame[column].map(_amount)
    for column in rates:
        if column in frame:
            frame[column] = [_rate(pair, value) for pair, value in
                             zip(frame.get("currency_pair", [""] * len(frame)), frame[column])]
    leading = [c for c in first if c in frame]
    return frame[leading + [c for c in frame.columns if c not in leading]]


def _choice(label, values, key):
    options = [ALL, *sorted({value for value in values if value})]
    choice = st.selectbox(label, options, key=key)
    return None if choice == ALL else choice


def _filters(trades, prefix):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        pair = _choice("Currency pair", [t.currency_pair for t in trades], f"{prefix}_pair")
    with col2:
        book = _choice("Book", [t.book for t in trades], f"{prefix}_book")
    with col3:
        counterparty = _choice("Counterparty", [t.counterparty for t in trades],
                               f"{prefix}_counterparty")
    with col4:
        product = _choice("Product", [FX_OPTION, FX_FORWARD], f"{prefix}_product")
    return [t for t in trades
            if (pair is None or t.currency_pair == pair) and (book is None or t.book == book)
            and (counterparty is None or t.counterparty == counterparty)
            and (product is None or t.product_type == product)]


def render_blotter(tab):
    """Booked trades: filter, inspect, amend, cancel, export and import."""
    store, problem = trade_store()
    with tab:
        st.subheader("Blotter")
        if not _store_banner(store, problem):
            return
        status = st.radio("Status", ["live", "cancelled", "all"], horizontal=True,
                          key="blotter_status")
        everything = store.trades(status=None)
        chosen = [t for t in everything if status == "all" or t.status == status]
        shown = _filters(chosen, "blotter")
        st.dataframe(trades_frame(shown), use_container_width=True, hide_index=True)
        st.caption(f"{len(shown)} of {len(everything)} trades.")
        for hedge, option_id in store.orphaned_hedges():
            st.warning(f"{hedge.trade_id} hedges {option_id}, which is no longer live: "
                       "amend or cancel the hedge.")
        if shown:
            _trade_detail(store, shown)
        _export_import(store)


def _trade_detail(store, shown):
    trade_id = st.selectbox("Trade", [t.trade_id for t in shown], key="blotter_trade")
    trade = store.get(trade_id)
    st.markdown(f"**{trade.trade_id}** — version {trade.version}, {trade.status}")
    if trade.is_option:
        hedges = store.hedges_for(trade.trade_id)
        if hedges:
            st.caption("Hedged by: " + ", ".join(
                f"{hedge.trade_id} ({amount:,.0f} {trade.base})" for hedge, amount in hedges))
    else:
        links = store.allocations(trade.trade_id)
        if links:
            st.caption("Hedges: " + ", ".join(
                f"{link.option} ({link.base_amount:,.0f} {trade.base})" for link in links))
    history = store.history(trade.trade_id)
    st.dataframe(pd.DataFrame([{
        "version": v.trade.version, "action": v.action,
        "recorded (UTC)": v.recorded_at.strftime("%Y-%m-%d %H:%M:%S"), "user": v.user,
        "reason": v.reason, "notional": _amount(v.trade.notional),
        "strike": "" if v.trade.strike is None else pricer.format_strike_for_pair(
            v.trade.currency_pair, v.trade.strike),
        "contract rate": _rate(v.trade.currency_pair, v.trade.contract_rate),
        "expiry": v.trade.expiry, "status": v.trade.status,
    } for v in history]), use_container_width=True, hide_index=True)
    if trade.status == CANCELLED or store.read_only:
        return
    amend_col, cancel_col = st.columns(2)
    with amend_col, st.form(f"amend_{trade.trade_id}_{trade.version}"):
        st.markdown("Amend")
        editable = [name for name in AMENDABLE
                    if (trade.is_option and name != "contract_rate")
                    or (trade.is_forward and name not in ("strike", "call_put", "premium",
                                                          "premium_ccy"))]
        values = {name: st.text_input(name.replace("_", " ").title(),
                                      "" if getattr(trade, name) is None else str(getattr(trade, name)))
                  for name in editable}
        reason = st.text_input("Reason", key=f"amend_reason_{trade.trade_id}")
        if st.form_submit_button("Save amendment"):
            changes = {name: value for name, value in values.items()
                       if value != ("" if getattr(trade, name) is None else str(getattr(trade, name)))}
            if not changes:
                st.info("Nothing changed.")
            else:
                _write(lambda: store.amend(trade.trade_id, trade.version, changes,
                                           user=_user(), reason=reason),
                       f"{trade.trade_id} amended")
    with cancel_col, st.form(f"cancel_{trade.trade_id}_{trade.version}"):
        st.markdown("Cancel")
        reason = st.text_input("Reason", key=f"cancel_reason_{trade.trade_id}")
        confirm = st.checkbox(f"Cancel {trade.trade_id}", key=f"cancel_confirm_{trade.trade_id}")
        if st.form_submit_button("Cancel trade", disabled=False):
            if not confirm:
                st.warning("Tick the box to confirm the cancellation.")
            else:
                _write(lambda: store.cancel(trade.trade_id, trade.version, user=_user(),
                                            reason=reason), f"{trade.trade_id} cancelled")


def _user():
    import getpass
    try:
        return getpass.getuser()
    except Exception:
        return ""


def _write(action, success):
    try:
        action()
    except (TradeStoreError, ValueError) as exc:
        st.error(str(exc))
        return
    st.session_state["_blotter_message"] = success
    st.experimental_rerun()


def _export_import(store):
    message = st.session_state.pop("_blotter_message", None)
    if message:
        st.success(message)
    st.markdown("**Export / import**")
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        st.download_button("Export book (JSON)", json.dumps(service.export_book(store), indent=1),
                           file_name="trade_book.json", mime="application/json",
                           key="export_book_json")
    with col2:
        st.download_button("Export book (CSV)", service.export_csv(store),
                           file_name="trade_book.csv", mime="text/csv", key="export_book_csv")
    with col3:
        upload = st.file_uploader("Import a trade book export (books new trade ids)",
                                  type=["json"], key="import_book_file")
        if upload is not None and st.button("Import", key="import_book",
                                            disabled=store.read_only):
            try:
                mapping, notes = service.import_book(store, upload.getvalue(), user=_user())
            except (TradeStoreError, ValueError, KeyError, TypeError) as exc:
                st.error(f"Import failed, nothing booked: {exc}")
            else:
                st.session_state["_blotter_message"] = (
                    f"Imported {len(mapping)} trades." + "".join(f" {note}." for note in notes))
                st.experimental_rerun()


BY_PAIR, LINE_ITEMS = "By pair", "Line items"
#: Portfolio sub-tabs; the first is shown by default.
PORTFOLIO_VIEWS = (BY_PAIR, LINE_ITEMS)


def pair_totals(results, forwards):
    """Totals per currency pair with the number of options and forwards in each."""
    totals = pricer.aggregate_by_pair(results, forwards)
    count = lambda rows, pair: sum(row["currency_pair"] == pair for row in rows)  # noqa: E731
    totals.insert(1, "options", [count(results, pair) for pair in totals["currency_pair"]])
    totals.insert(2, "forwards", [count(forwards, pair) for pair in totals["currency_pair"]])
    return totals.sort_values("currency_pair").reset_index(drop=True)


def results_strikes(results):
    return [result["strike"] for result in results]


def render_portfolio(tab, market_set, mods, premium_quote):
    """Price a selection of the book through the market set."""
    store, problem = trade_store()
    with tab:
        st.subheader("Portfolio")
        if not _store_banner(store, problem):
            return
        live = store.trades(status=LIVE)
        selected = _filters(live, "portfolio")
        # The Risk tab reports on this same selection.
        st.session_state["_portfolio_selection"] = [t.trade_id for t in selected]
        st.caption(f"{len(selected)} live trades selected "
                   f"({sum(t.is_option for t in selected)} options, "
                   f"{sum(t.is_forward for t in selected)} forwards).")
        if st.button("Price selection", type="primary", key="price_book",
                     disabled=not selected):
            results, forwards, errors = service.price_trades(market_set, selected, mods,
                                                             premium_quote)
            st.session_state["_portfolio_book_result"] = (
                [t.trade_id for t in selected], results, forwards, errors)
        cached = st.session_state.get("_portfolio_book_result")
        if not cached:
            return
        ids, results, forwards, errors = cached
        if ids != [t.trade_id for t in selected]:
            st.info("The selection changed since it was priced; price it again.")
            return
        for error in errors:
            st.error("Not priced — " + error)
        if not (results or forwards):
            return
        pair_tab, items_tab = st.tabs(list(PORTFOLIO_VIEWS))
        with pair_tab:
            totals = pair_totals(results, forwards)
            st.dataframe(_display(totals.to_dict("records"), first=("currency_pair",),
                                  amounts=[c for c in totals.columns
                                           if c not in ("currency_pair", "options", "forwards")]),
                         use_container_width=True, hide_index=True)
            st.caption("One row per currency pair, hedge forwards included. MTM, P&L and "
                       "Greeks are in the pair's base currency (theta in its premium "
                       "currency); pairs are not converted to one currency.")
        with items_tab:
            _line_items(results, forwards)


def _line_items(results, forwards):
    """One pair's options and forwards, by trade id."""
    pairs = sorted({row["currency_pair"] for row in (*results, *forwards)})
    pair = st.selectbox("Currency pair", pairs, key="portfolio_items_pair")
    options = [row for row in results if row["currency_pair"] == pair]
    hedges = [row for row in forwards if row["currency_pair"] == pair]
    if options:
        st.markdown("**Options**")
        columns = ["trade_id", "buy_sell", "call_put", "strike", "expiry", "notional",
                   "vol_pct", "mtm_base", "pnl_base", *pricer.GREEK_NAMES, "smile_status"]
        frame = _display(options, amounts=("mtm_base", "pnl_base", *pricer.GREEK_NAMES))
        frame["vol_pct"] = frame["vol_pct"].map(lambda v: f"{float(v):.4f}")
        frame["strike"] = [pricer.format_strike_for_pair(pair, k) for k in results_strikes(options)]
        st.dataframe(frame[[c for c in columns if c in frame]], use_container_width=True,
                     hide_index=True)
    if hedges:
        st.markdown("**Forwards**")
        st.dataframe(_display(hedges, first=("trade_id", "expiry", "direction"),
                              amounts=("base_amount", "delta_offset", "mtm_term", "mtm_base",
                                       "pnl_term", "pnl_base", "theta"),
                              rates=("forward_rate",)).drop(columns=["currency_pair"]),
                     use_container_width=True, hide_index=True)

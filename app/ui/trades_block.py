"""Blotter and Portfolio tabs: the booked trades, outside the Pricer scratchpad."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app import pricer, usd_spots
from app.ui.negatives import red_negatives
from pricing import greek_catalog
from app.premium_side import SIDES, premium_currency, premium_side
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
    opened = st.session_state.get("_trade_store")
    if opened is None:
        path = Path(os.environ.get("VV_TRADE_STORE", str(DEFAULT_STORE)))
        opened = open_trade_store(path)
        # Returned from the local name: a session cleared mid-run (browser
        # closed) must not turn this into a KeyError.
        st.session_state["_trade_store"] = opened
    return opened


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


def _filters(trades, prefix, by_counterparty=True):
    """Pair, book, counterparty (optional) and product filters; the trades matching."""
    columns = iter(st.columns(4 if by_counterparty else 3))
    with next(columns):
        pair = _choice("Currency pair", [t.currency_pair for t in trades], f"{prefix}_pair")
    with next(columns):
        book = _choice("Book", [t.book for t in trades], f"{prefix}_book")
    counterparty = None             # without the filter, a choice made earlier is ignored
    if by_counterparty:
        with next(columns):
            counterparty = _choice("Counterparty", [t.counterparty for t in trades],
                                   f"{prefix}_counterparty")
    with next(columns):
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
        values = {}
        for name in editable:
            current = "" if getattr(trade, name) is None else str(getattr(trade, name))
            if name == "premium_ccy":
                # Base or term of the pair only; a wrong currency cannot be chosen.
                side, _ = premium_side(trade.currency_pair, current)
                values[name] = st.selectbox(
                    "Premium in", SIDES, index=SIDES.index(side),
                    format_func=lambda s: f"{s.title()} ({premium_currency(trade.currency_pair, s)})",
                    help="Base or term currency of the trade's pair (of the new pair "
                         "when the amendment changes it).")
            else:
                values[name] = st.text_input(name.replace("_", " ").title(), current)
        reason = st.text_input("Reason", key=f"amend_reason_{trade.trade_id}")
        if st.form_submit_button("Save amendment"):
            if "premium_ccy" in values:
                pair = values.get("currency_pair") or trade.currency_pair
                values["premium_ccy"] = premium_currency(pair, values["premium_ccy"])
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
    """Totals per currency pair with the number of options and forwards in each,
    and every Greek the results carry (extras only when priced)."""
    totals = pricer.aggregate_by_pair(results, forwards, greeks=list(greek_catalog.CATALOG))
    count = lambda rows, pair: sum(row["currency_pair"] == pair for row in rows)  # noqa: E731
    totals.insert(1, "options", [count(results, pair) for pair in totals["currency_pair"]])
    totals.insert(2, "forwards", [count(forwards, pair) for pair in totals["currency_pair"]])
    return totals.sort_values("currency_pair").reset_index(drop=True)


def results_strikes(results):
    return [result["strike"] for result in results]


#: ``{pair: spot}`` the Portfolio prices at instead of the market data spot.
SPOT_OVERRIDES = "_portfolio_spot_overrides"
_SPOT_EDITOR = "_portfolio_spot_editor_version"
TOTAL_ROW = "Total (USD)"


def spot_overrides():
    return dict(st.session_state.get(SPOT_OVERRIDES, {}))


def _reset_spot_editor():
    # A new key drops the browser's own edit state with the old editor.
    st.session_state[_SPOT_EDITOR] = st.session_state.get(_SPOT_EDITOR, 0) + 1


def refresh_spots():
    """Back to the market data spots."""
    st.session_state[SPOT_OVERRIDES] = {}
    _reset_spot_editor()


def apply_spot_edits(edited_rows, pairs, market_spots, overrides):
    """Overrides after editor edits ``{row: {"spot": value}}`` on rows ``pairs``.

    A spot set back to the market spot, cleared, or not a positive number
    drops that pair's override; edits to the total row are ignored.
    """
    overrides = dict(overrides)
    for row, change in edited_rows.items():
        if "spot" not in change or int(row) >= len(pairs) or pairs[int(row)] not in market_spots:
            continue
        pair = pairs[int(row)]
        try:
            value = float(change["spot"])
        except (TypeError, ValueError):
            value = float("nan")
        if not (0 < value < float("inf")) or value == market_spots[pair]:
            overrides.pop(pair, None)
        else:
            overrides[pair] = value
    return overrides


def _on_spot_edit(key, pairs, market):
    state = st.session_state.get(key) or {}
    st.session_state[SPOT_OVERRIDES] = apply_spot_edits(
        state.get("edited_rows", {}), pairs, market, spot_overrides())
    _reset_spot_editor()


def market_spots(market_set):
    """``{pair: spot}`` from the market data, for pairs that can price."""
    return {pair: float(market_set.market(pair).spot) for pair in market_set.available()}


def render_portfolio(tab, market_set, mods, premium_quote):
    """Price a selection of the book through the market set, at the Portfolio spots."""
    store, problem = trade_store()
    with tab:
        st.subheader("Portfolio")
        if not _store_banner(store, problem):
            return
        live = store.trades(status=LIVE)
        selected = _filters(live, "portfolio", by_counterparty=False)
        # The Risk tab reports on this same selection.
        st.session_state["_portfolio_selection"] = [t.trade_id for t in selected]
        st.caption(f"{len(selected)} live trades selected "
                   f"({sum(t.is_option for t in selected)} options, "
                   f"{sum(t.is_forward for t in selected)} forwards).")
        ids = [t.trade_id for t in selected]
        market = market_spots(market_set)
        overrides = {pair: spot for pair, spot in spot_overrides().items() if pair in market}
        cached = st.session_state.get("_portfolio_book_result")
        if cached and len(cached) != 5:
            cached = None          # priced before Portfolio spots existed
        clicked = st.button("Price selection", type="primary", key="price_book",
                            disabled=not selected)
        # A moved or refreshed spot reprices the selection already priced.
        respot = bool(cached) and cached[0] == ids and cached[1] != overrides
        if clicked or respot:
            extras = chosen_extras(st.session_state.get(_PAIR_COLUMNS, DEFAULT_COLUMNS))
            results, forwards, errors = service.price_trades(
                market_set.with_spots(overrides), selected, mods, premium_quote,
                extras=extras)
            cached = (ids, overrides, results, forwards, errors)
            st.session_state["_portfolio_book_result"] = cached
        if not cached:
            return
        priced_ids, _, results, forwards, errors = cached
        if priced_ids != ids:
            st.info("The selection changed since it was priced; price it again.")
            return
        for error in errors:
            st.error("Not priced — " + error)
        problems = {name: reason for result in results
                    for name, reason in result.get("greek_problems", {}).items()}
        for name, reason in problems.items():
            st.warning(f"{greek_catalog.CATALOG[name].label} not available: {reason}")
        if not (results or forwards):
            return
        pair_tab, items_tab = st.tabs(list(PORTFOLIO_VIEWS))
        with pair_tab:
            _pair_view(pair_totals(results, forwards), market, overrides,
                       market_set.valuation_date)
        with items_tab:
            _line_items(results, forwards)


def usd_totals(totals, spots):
    """Pair totals with spot and USD columns, plus a USD total row across pairs."""
    frame = usd_spots.add_usd_columns(totals, spots)
    usd = [target for _, target in usd_spots.USD_COLUMNS if target in frame]
    counts = [c for c in ("options", "forwards") if c in frame]
    total = {"currency_pair": TOTAL_ROW, **frame[counts + usd].sum(min_count=1).to_dict()}
    return pd.concat([frame, pd.DataFrame([total])], ignore_index=True), usd


def portfolio_spots(market, overrides, snapshot_spots=None):
    """Spots the Portfolio converts at: snapshot, then market data, then overrides."""
    return {**(snapshot_spots or {}), **market, **overrides}


#: By-pair columns always shown, and the optional ones shown until the user changes them.
FIXED_COLUMNS = ("currency_pair", "spot")
RATE_COLUMNS = ("currency_pair", "spot", "market_spot")
DEFAULT_COLUMNS = ("pnl_usd", "delta", "fixed_delta", "gamma", "vega_usd", "vanna_usd",
                   "volga_usd", "rega_usd", "sega_usd")
_PAIR_COLUMNS = "_portfolio_pair_columns_choice"


def _greek_columns(names):
    usd = {source: target for source, target in usd_spots.USD_COLUMNS}
    return [column for name in names
            for column in (f"fixed_{name}", name, *([usd[name]] if name in usd else []))]


def extra_columns():
    """By-pair columns of the extra Greeks: fixed, smile and USD where converted."""
    return _greek_columns(greek_catalog.EXTRA_GREEKS)


def column_options(present=()):
    """Every by-pair column the picker offers, in one fixed order.

    The list must not depend on what has been priced: Streamlit resets a
    multiselect whose options change. Columns outside it that a pricing
    still produced (``present``) are appended, so none is ever hidden.
    """
    canonical = ["options", "forwards", "market_spot", "mtm_base", "pnl_base", "mtm_usd",
                 "pnl_usd", *_greek_columns(greek_catalog.CATALOG)]
    ordered = [*DEFAULT_COLUMNS, *(c for c in canonical if c not in DEFAULT_COLUMNS)]
    return ordered + [c for c in present if c not in ordered and c not in FIXED_COLUMNS]


def chosen_extras(columns):
    """The extra Greeks the chosen by-pair ``columns`` need priced."""
    return greek_catalog.extras(greek_catalog.greek_of(column) for column in columns
                                if greek_catalog.greek_of(column))


def _table_height(rows):
    """Pixel height that shows every row and the header without scrolling."""
    return 35 * (rows + 1) + 3


def _pair_view(totals, market, overrides, valuation_date=None):
    try:
        snapshot = usd_spots.latest_snapshot(valuation_date)
    except Exception as exc:          # an unreadable store must not stop the page
        st.warning(f"Market data store unavailable: {exc}")
        snapshot = None
    spots = portfolio_spots(market, overrides, snapshot.spots if snapshot else None)
    totals, usd = usd_totals(totals, spots)
    totals.insert(totals.columns.get_loc("spot") + 1, "market_spot",
                  [market.get(pair, float("nan")) for pair in totals["currency_pair"]])
    is_pair = totals["currency_pair"] != TOTAL_ROW
    pairs = list(totals["currency_pair"][is_pair])
    # A fixed list: extra Greeks can be chosen before they are priced (computed
    # on the next pricing), and the choice survives the pricing.
    optional = column_options(totals.columns)
    # Kept outside the widget too: Streamlit drops a widget's state on runs
    # that do not draw it (an unpriced selection).
    if "portfolio_pair_columns" not in st.session_state:
        saved = st.session_state.get(_PAIR_COLUMNS, DEFAULT_COLUMNS)
        st.session_state["portfolio_pair_columns"] = [c for c in saved if c in optional]
    picker_col, _ = st.columns([1, 2])
    with picker_col, st.expander("Columns", expanded=False):
        chosen = st.multiselect(
            "Output columns", optional, key="portfolio_pair_columns",
            label_visibility="collapsed",
            help="Add or remove output columns; pair and spot are always shown.")
    st.session_state[_PAIR_COLUMNS] = list(chosen)
    unpriced = [c for c in chosen if c not in totals]
    if unpriced:
        st.caption("Price selection to compute: " + ", ".join(unpriced))
    unconverted = list(totals["currency_pair"][is_pair & totals[usd[0]].isna()])
    if unconverted:
        st.warning("No USD spot for " + ", ".join(unconverted) + ": USD columns left blank"
                   + ("." if snapshot else "; run python -m market_data.bloomberg_fetch for a USD spot snapshot."))
    moved = [f"{pair} {pricer.format_strike_for_pair(pair, spot)} "
             f"(market {pricer.format_strike_for_pair(pair, market[pair])})"
             for pair, spot in sorted(overrides.items())]
    button_col, note_col = st.columns([1, 5])
    with button_col:
        st.button("Refresh spot", key="refresh_portfolio_spot", on_click=refresh_spots,
                  help="Reprice at the market data spots.")
    with note_col:
        if moved:
            st.info("Priced at Portfolio spots: " + ", ".join(moved) + ".")
    shown = [*FIXED_COLUMNS, *(c for c in optional if c in chosen and c in totals)]
    # Spot is a text cell so the total row stays blank; edits are parsed back.
    frame = _display(totals[shown].to_dict("records"), first=shown,
                     rates=("spot", "market_spot"),
                     amounts=[c for c in shown if c not in RATE_COLUMNS])
    key = f"portfolio_spots_{st.session_state.get(_SPOT_EDITOR, 0)}"
    st.data_editor(
        red_negatives(frame, [c for c in frame.columns if c not in RATE_COLUMNS]), key=key, use_container_width=True, hide_index=True,
        height=_table_height(len(frame)),
        disabled=[c for c in frame.columns if c != "spot"],
        column_config={"spot": st.column_config.TextColumn(
            "spot ✎",
            help="Edit to reprice the portfolio at this spot; Refresh spot reverts.")},
        on_change=_on_spot_edit, args=(key, pairs, market))
    st.caption("One row per currency pair, hedge forwards included. Edit spot to reprice "
               "the portfolio at it; market_spot and the Markets tab keep the market data "
               "spot. Columns ending _usd are converted from the pair's base currency at "
               "the Portfolio spots (other currencies at the USD spot snapshot) and summed "
               "in the Total row; the other MTM, P&L and Greek columns, added from "
               "Columns, stay in the pair's base currency (theta in its premium currency).")
    if snapshot:
        st.caption(f"USD spots from market {snapshot.market_id} ({snapshot.market_source}, "
                   f"{snapshot.date_time.replace('T', ' ')}).")


def _line_items(results, forwards):
    """One pair's options and forwards, by trade id."""
    pairs = sorted({row["currency_pair"] for row in (*results, *forwards)})
    pair = st.selectbox("Currency pair", pairs, key="portfolio_items_pair")
    options = [row for row in results if row["currency_pair"] == pair]
    hedges = [row for row in forwards if row["currency_pair"] == pair]
    if options:
        st.markdown("**Options**")
        # Smile Greeks: the standard ones and any extras the pricing carried.
        greeks = [name for name in greek_catalog.CATALOG if name in options[0]]
        columns = ["trade_id", "buy_sell", "call_put", "strike", "expiry", "notional",
                   "vol_pct", "mtm_base", "pnl_base", *greeks, "smile_status"]
        frame = _display(options, amounts=("mtm_base", "pnl_base", *greeks))
        frame["vol_pct"] = frame["vol_pct"].map(lambda v: f"{float(v):.4f}")
        frame["strike"] = [pricer.format_strike_for_pair(pair, k) for k in results_strikes(options)]
        st.dataframe(red_negatives(frame[[c for c in columns if c in frame]],
                                   ["mtm_base", "pnl_base", *greeks]),
                     use_container_width=True, hide_index=True)
    if hedges:
        st.markdown("**Forwards**")
        amounts = ("base_amount", "delta_offset", "mtm_term", "mtm_base", "pnl_term",
                   "pnl_base", "theta")
        st.dataframe(red_negatives(_display(hedges, first=("trade_id", "expiry", "direction"),
                                            amounts=amounts, rates=("forward_rate",))
                                   .drop(columns=["currency_pair"]), amounts),
                     use_container_width=True, hide_index=True)

"""The Pricer's Spot / forward grid: enter, price and book spot and forward deals.

Lines and booking are ``app.spot_forwards``; this block only draws them.
Results are kept while the lines they were priced from are unchanged.
"""
import getpass

import pandas as pd
import streamlit as st

from app import spot_forwards
from app.premium_side import BASE, TERM
from app.ui.negatives import red_negatives

ROWS_KEY, EDITOR_KEY = "spot_forward_rows", "spot_forward_editor"
PENDING_KEY, RESULTS_KEY = "_pending_spot_forward_rows", "_spot_forward_results"
MESSAGES_KEY = "_spot_forward_messages"
TICK = "book_line"
INPUTS = [c for c in spot_forwards.COLUMNS if c != "trade_id"]


def _default_rows(pair):
    return pd.DataFrame([{TICK: False, "currency_pair": pair, "buy_sell": "BUY",
                          "amount": "1m", "amount_ccy": BASE, "value_date": "SPOT",
                          "rate": "", "book": "", "counterparty": "", "trade_id": ""}])


def _inputs(rows):
    """The priced inputs of ``rows`` as comparable text (ticks and ids aside)."""
    frame = rows.reindex(columns=INPUTS).fillna("").astype(str)
    return frame.apply(lambda column: column.str.strip()).to_json()


RESULT_COLUMNS = {
    "row": st.column_config.NumberColumn("Line", width="small"),
    "currency_pair": st.column_config.TextColumn("Pair", width="small"),
    "value_date": st.column_config.DateColumn("Value date", format="DD-MMM-YYYY"),
    "base_amount": st.column_config.NumberColumn("Base amount", format="%.2f"),
    "term_amount": st.column_config.NumberColumn("Term amount", format="%.2f"),
    "contract_rate": st.column_config.NumberColumn("Rate", format="%.6f"),
    "outright": st.column_config.NumberColumn("Market outright", format="%.6f"),
    "mtm_term": st.column_config.NumberColumn("MTM (term)", format="%.2f"),
    "mtm_base": st.column_config.NumberColumn("MTM (base)", format="%.2f"),
    "mtm_usd": st.column_config.NumberColumn("MTM (USD)", format="%.2f"),
    "settlement_term": st.column_config.NumberColumn(
        "Settlement P&L (term)", format="%.2f",
        help="Amount × (outright − rate), paid on the value date (undiscounted)."),
    "spot_delta": st.column_config.NumberColumn("Spot delta", format="%.0f"),
    "fwd_delta": st.column_config.NumberColumn("Fwd delta", format="%.0f"),
    "theta": st.column_config.NumberColumn("Theta (prem ccy)", format="%.2f"),
}


def render_spot_forward_block(tab, market_set, store, can_book, default_pair,
                              default_book="", default_counterparty=""):
    """Draw the grid, its Price / Book buttons and the priced results."""
    pending = st.session_state.pop(PENDING_KEY, None)
    if pending is not None:
        st.session_state[ROWS_KEY] = pending
        st.session_state.pop(EDITOR_KEY, None)
    if ROWS_KEY not in st.session_state:
        st.session_state[ROWS_KEY] = _default_rows(default_pair)
    with tab:
        st.subheader("Spot / forward")
        rows = st.data_editor(
            st.session_state[ROWS_KEY],
            num_rows="dynamic", use_container_width=True, hide_index=True,
            key=EDITOR_KEY, disabled=["trade_id"],
            column_order=[TICK, *spot_forwards.COLUMNS],
            column_config={
                TICK: st.column_config.CheckboxColumn("Book?", width="small", default=False),
                "currency_pair": st.column_config.TextColumn("Pair", width="small"),
                "buy_sell": st.column_config.SelectboxColumn(
                    "Buy/Sell", options=["BUY", "SELL"], width="small", required=True,
                    help="Of the amount's currency: buying term is selling base."),
                "amount": st.column_config.TextColumn("Amount", width="small",
                                                      help="e.g. 10m, 250k"),
                "amount_ccy": st.column_config.SelectboxColumn(
                    "Amount ccy", options=[BASE, TERM], width="small"),
                "value_date": st.column_config.TextColumn(
                    "Value date", width="small",
                    help="SPOT, TOM, SN, a tenor (1W, 3M, 18M, 1Y: spot plus the tenor) "
                         "or a business-day date. Before spot needs the market's TN points."),
                "rate": st.column_config.TextColumn(
                    "Rate", width="small", help="Blank: the market outright at the value date."),
                "book": st.column_config.TextColumn("Book", width="small"),
                "counterparty": st.column_config.TextColumn("Counterparty", width="small"),
                "trade_id": st.column_config.TextColumn("Trade id", width="small"),
            },
        )
        ticked = [p for p, v in enumerate(rows.get(TICK, [])) if v is True or v == 1]
        price_col, book_col = st.columns([1, 3])
        with price_col:
            price_clicked = st.button("Price spot / forwards", key="price_spot_forwards")
        with book_col:
            book_clicked = st.button(f"Book ticked deals ({len(ticked)})",
                                     key="book_spot_forwards",
                                     disabled=not ticked or not can_book)
        for level, message in st.session_state.pop(MESSAGES_KEY, []):
            getattr(st, level)(message)

        if price_clicked:
            results, errors = spot_forwards.price_lines(market_set, rows,
                                                        default_pair=default_pair)
            st.session_state[RESULTS_KEY] = {"inputs": _inputs(rows), "results": results,
                                             "errors": errors}
        if book_clicked:
            _book(store, market_set, rows, ticked, default_pair, default_book,
                  default_counterparty)
            st.experimental_rerun()

        held = st.session_state.get(RESULTS_KEY)
        if held is not None and held["inputs"] != _inputs(rows):
            held = None                    # the lines changed since they were priced
        if held is not None:
            for error in held["errors"]:
                st.error("Not priced — " + error)
            if held["results"]:
                frame = pd.DataFrame(held["results"])
                notes = frame.get("forward_note")
                frame = frame.reindex(columns=list(RESULT_COLUMNS))
                st.dataframe(red_negatives(_formatted(frame)), hide_index=True,
                             use_container_width=True, column_config=RESULT_COLUMNS)
                if notes is not None:
                    for note in notes.dropna().unique():
                        st.caption(note)
                st.caption("MTM is the outright less the rate on the amount, discounted on "
                           "the term curve to the value date; settlement P&L is the same "
                           "undiscounted. Theta is the next business day's value on the "
                           "rolled curve, or the whole MTM for a deal settling before then.")


#: Display formats: the styled table ignores the column config's formats.
_FORMATS = {
    **{c: "{:,.2f}" for c in ("base_amount", "term_amount", "mtm_term", "mtm_base", "mtm_usd",
                              "settlement_term", "theta")},
    **{c: "{:,.0f}" for c in ("spot_delta", "fwd_delta")},
    **{c: "{:.6f}" for c in ("contract_rate", "outright")},
    "value_date": lambda d: pd.Timestamp(d).strftime("%d-%b-%Y"),
}


def _formatted(frame):
    """``frame`` as a Styler with the display formats and a stable id."""
    from app.ui.negatives import stable_uuid
    try:
        styler = frame.style.format(
            {c: f for c, f in _FORMATS.items() if c in frame}, na_rep="")
        styler.set_uuid(stable_uuid(frame.columns))
        return styler
    except Exception:              # cosmetic: never break the table over formats
        return frame


def _book(store, market_set, rows, ticked, default_pair, book, counterparty):
    try:
        user = getpass.getuser()
    except Exception:
        user = ""
    result = spot_forwards.book_lines(store, market_set, rows.iloc[ticked],
                                      book=book, counterparty=counterparty, user=user,
                                      default_pair=default_pair)
    messages = []
    if not result.ok:
        for error in result.errors:
            head, _, rest = error.partition(" ")
            number = rest.split(":")[0].split(" ")[0]
            if head == "Line" and number.isdigit() and int(number) <= len(ticked):
                error = error.replace(f"Line {number}", f"Line {ticked[int(number) - 1] + 1}", 1)
            messages.append(("error", "Nothing booked — " + error))
    else:
        updated = rows.copy()
        trades = {t.trade_id: t for t in result.booked}
        for number, trade_id in result.lines.items():
            index = updated.index[ticked[number - 1]]
            trade = trades[trade_id]
            updated.at[index, "trade_id"] = trade_id
            updated.at[index, "value_date"] = trade.value_date.isoformat()
            decimals = spot_forwards.rate_decimals(trade.currency_pair)
            updated.at[index, "rate"] = f"{trade.contract_rate:.{decimals}f}"
            updated.at[index, TICK] = False
        st.session_state[PENDING_KEY] = updated
        messages.append(("success", "Booked " + ", ".join(trades) + "."))
    st.session_state[MESSAGES_KEY] = messages

"""Portfolio input block for the Streamlit application."""

import pandas as pd
import streamlit as st

from app.premium_side import SIDES, TERM, premium_currency
from app.pricer import GREEK_NAMES
from app.ui.greek_picker import render_greek_picker
from app.ui.negatives import red_negatives
from pricing import greek_catalog as catalog


REMOVE_COLUMN = "_remove"
#: Tick box choosing the lines "Book ticked lines" books.
BOOK_COLUMN = "_book"
TRADE_ID_COLUMN = "trade_id"
#: Read-only display of the premium currency a line's BASE/TERM stands for.
PREMIUM_CODE_COLUMN = "_premium_code"
BOOKING_COLUMNS = ("book", "counterparty")
#: Always shown after pricing; the Greek columns follow the Greek picker.
SUMMARY_RESULT_COLUMNS = ("vol_pct", "mtm_base", "pnl_base", "mtm_quote", "pnl_quote")
#: Every read-only result column a line can carry (all catalogue Greeks).
LINE_RESULT_COLUMNS = (*SUMMARY_RESULT_COLUMNS, *catalog.columns(catalog.CATALOG))
PRICER_GREEKS_KEY = "pricer_greeks"

#: Input cells whose edits can change a resolved pair, expiry or strike.
RESOLVED_INPUT_COLUMNS = ("currency_pair", "expiry", "call_put", "strike")

#: Columns the pricer supplies already formatted for display.
PREFORMATTED_RESULT_COLUMNS = ("mtm_quote", "pnl_quote")


def _with_line_results(rows, results, greek_columns=None):
    """Add read-only pricing outputs to their matching visible input rows:
    the summary columns and ``greek_columns`` (the base Greeks by default)."""
    if greek_columns is None:
        greek_columns = catalog.columns(catalog.BASE_GREEKS)
    shown = (*SUMMARY_RESULT_COLUMNS, *greek_columns)
    display = rows.drop(columns=LINE_RESULT_COLUMNS, errors="ignore").copy()
    for column in shown:
        display[column] = pd.NA
    for result in results:
        row_number = result.get("row")
        if not isinstance(row_number, int) or not 1 <= row_number <= len(display):
            continue
        row_index = row_number - 1
        for column in shown:
            if column in result:
                value = result[column]
                if column in PREFORMATTED_RESULT_COLUMNS:
                    # Already rendered in the active premium quote upstream.
                    display.at[row_index, column] = (
                        pd.NA if value is None else str(value)
                    )
                elif value is None or pd.isna(value):
                    display.at[row_index, column] = pd.NA
                elif column == "vol_pct":
                    display.at[row_index, column] = f"{float(value):.4f}"
                else:
                    display.at[row_index, column] = f"{float(value):,.0f}"
    return display


#: Cells a user types; an added line without any of them is a placeholder.
LINE_INPUT_COLUMNS = (
    "currency_pair", "strike", "expiry", "call_put", "buy_sell", "notional", "premium",
)


#: Cells pricing reads: a line's results stay valid while these are unchanged
#: (ticks, book, counterparty and trade id do not change a price).
PRICED_INPUT_COLUMNS = (*LINE_INPUT_COLUMNS, "premium_ccy")


def _comparable(value):
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    if isinstance(value, str):
        return value.strip()
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def same_priced_inputs(rows, other):
    """True when ``rows`` and ``other`` have the same lines with the same
    priced inputs, whatever their dtypes or other columns."""
    if rows is None or other is None or len(rows) != len(other):
        return False
    columns = [c for c in PRICED_INPUT_COLUMNS if c in rows.columns or c in other.columns]

    def cells(frame):
        frame = frame.reindex(columns=columns)
        return [[_comparable(value) for value in row] for row in frame.itertuples(index=False)]
    return cells(rows) == cells(other)


def _has_value(value):
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    try:
        return not pd.isna(value)
    except (TypeError, ValueError):
        return True


def _only_placeholder_rows_added(editor_state):
    """True when the only change is new lines carrying no typed values."""
    added = editor_state.get("added_rows", [])
    return (
        bool(added)
        and not editor_state.get("edited_rows")
        and not editor_state.get("deleted_rows")
        and not any(_has_value(row.get(column)) for row in added
                    for column in LINE_INPUT_COLUMNS)
    )


def _marked_row_indices(editor_state):
    """Return rows selected in the explicit transient removal column."""
    return [
        int(raw_index)
        for raw_index, changes in editor_state.get("edited_rows", {}).items()
        if changes.get(REMOVE_COLUMN) is True
    ]


def _apply_editor_state(rows, editor_state):
    """Merge Streamlit data-editor changes into a persistent dataframe."""
    updated = rows.copy()
    added_rows = [
        {
            column: value
            for column, value in row.items()
            if column not in (REMOVE_COLUMN, PREMIUM_CODE_COLUMN)
        }
        for row in editor_state.get("added_rows", [])
    ]
    if added_rows:
        new_columns = {
            column
            for row in added_rows
            for column in row
            if column not in updated.columns
        }
        if new_columns:
            updated = updated.reindex(
                columns=[*updated.columns, *sorted(new_columns)]
            )
        additions = pd.DataFrame(added_rows).reindex(columns=updated.columns)
        updated = pd.concat([updated, additions], ignore_index=True)

    # Streamlit indexes edits and deletions against the full displayed table,
    # including dynamically added rows. Apply them after appending additions.
    # Rows saved after pricing carry no tick column: a tick made afterwards
    # must not be dropped as an edit to an unknown column.
    if (BOOK_COLUMN not in updated.columns
            and any(BOOK_COLUMN in changes
                    for changes in editor_state.get("edited_rows", {}).values())):
        updated[BOOK_COLUMN] = False
    for raw_index, changes in editor_state.get("edited_rows", {}).items():
        index = int(raw_index)
        if 0 <= index < len(updated):
            for column, value in changes.items():
                if column in updated.columns:
                    updated.iat[index, updated.columns.get_loc(column)] = value
            # An edited booked line no longer matches its trade: it becomes a
            # new, unbooked line (amend booked trades in the Blotter).
            if (TRADE_ID_COLUMN in updated.columns
                    and any(column in (*LINE_INPUT_COLUMNS, *BOOKING_COLUMNS) for column in changes)):
                updated.iat[index, updated.columns.get_loc(TRADE_ID_COLUMN)] = ""

    deleted_indices = sorted(
        (int(index) for index in editor_state.get("deleted_rows", [])),
        reverse=True,
    )
    for index in deleted_indices:
        if 0 <= index < len(updated):
            updated = updated.drop(updated.index[index])

    updated = updated.reset_index(drop=True)
    if "currency_pair" in updated.columns:
        updated["currency_pair"] = updated["currency_pair"].map(
            lambda value: (
                value if pd.isna(value) else str(value).strip().upper()
            )
        )
    return updated


def with_premium_codes(rows, market_pair):
    """``rows`` with ``PREMIUM_CODE_COLUMN`` (the currency each line's
    BASE/TERM stands for) right after ``premium_ccy``; a blank pair uses
    ``market_pair``, as pricing does."""
    shown = rows.drop(columns=[PREMIUM_CODE_COLUMN], errors="ignore").copy()
    pairs = shown["currency_pair"] if "currency_pair" in shown else [market_pair] * len(shown)
    sides = shown["premium_ccy"] if "premium_ccy" in shown else [TERM] * len(shown)
    codes = []
    for pair, side in zip(pairs, sides):
        text = "" if pair is None or (not isinstance(pair, str) and pd.isna(pair)) else str(pair)
        codes.append(premium_currency(text.strip() or market_pair, side))
    columns = list(shown.columns)
    at = columns.index("premium_ccy") + 1 if "premium_ccy" in columns else len(columns)
    shown.insert(at, PREMIUM_CODE_COLUMN, codes)
    return shown


def render_portfolio_block(portfolio_tab, market_pair, quote_label="quote",
                           pairs=()):
    """Render the Pricer lines and the hedge choice.

    The premium currency is chosen as BASE or TERM of the line's pair
    (``app.premium_side``), shown with the currency it stands for.
    """
    market_pair = str(market_pair).upper()

    def resolve_strikes_after_edit():
        """Request input resolution after an edit to a resolvable cell.

        Pair, expiry, call/put and strike all feed the resolved strike, so a
        delta strike typed before its expiry still resolves once the expiry
        arrives.
        """
        editor_state = st.session_state.get("portfolio_editor_v3", {})
        if _only_placeholder_rows_added(editor_state):
            # Clicking into the blank row adds a line holding only column
            # defaults before the typed value arrives. Committing it would
            # reset the editor and drop the cell being typed; the editor keeps
            # the line until it carries a value.
            return
        editor_state = dict(editor_state)
        editor_state["deleted_rows"] = [
            *editor_state.get("deleted_rows", []),
            *_marked_row_indices(editor_state),
        ]
        st.session_state._pending_editor_portfolio_rows = _apply_editor_state(
            st.session_state.option_rows, editor_state
        )
        # Results are dropped at render time only if a priced input changed.
        edited_rows = editor_state.get("edited_rows", {})
        input_was_edited = any(
            column in RESOLVED_INPUT_COLUMNS
            for changes in edited_rows.values() for column in changes
        )
        added_rows = editor_state.get("added_rows", [])
        input_was_added = any(
            str(row.get(column) or "").strip()
            for row in added_rows for column in RESOLVED_INPUT_COLUMNS
        )
        if input_was_edited or input_was_added:
            st.session_state._portfolio_strike_resolution_requested = True

    with portfolio_tab:
        st.subheader("Pricer")
        st.caption(
            "Add a line with the + control or the blank row at the bottom. "
            "Tick Remove to delete a line."
        )
        greeks = render_greek_picker(PRICER_GREEKS_KEY)
        pending_resolved_rows = st.session_state.pop(
            "_pending_resolved_portfolio_rows", None
        )
        pending_priced_rows = st.session_state.pop(
            "_pending_priced_portfolio_rows", None
        )
        pending_editor_rows = st.session_state.pop(
            "_pending_editor_portfolio_rows", None
        )
        pending_line_results = st.session_state.pop(
            "_pending_portfolio_line_results", None
        )
        pending_rows = (
            pending_resolved_rows
            if pending_resolved_rows is not None
            else pending_priced_rows
            if pending_priced_rows is not None
            else pending_editor_rows
        )
        if pending_rows is not None:
            st.session_state.option_rows = pending_rows
            st.session_state.pop("portfolio_editor_v3", None)
        if pending_line_results is not None:
            st.session_state._portfolio_line_results = pending_line_results
        line_results = st.session_state.get("_portfolio_line_results")
        # Ticking, booking or naming a book does not change a price, so
        # results stay shown until a priced input changes.
        if (
            line_results is not None
            and same_priced_inputs(st.session_state.option_rows, line_results["rows"])
        ):
            editor_input = _with_line_results(
                st.session_state.option_rows, line_results["results"],
                catalog.columns(greeks),
            )
        else:
            st.session_state.pop("_portfolio_line_results", None)
            editor_input = st.session_state.option_rows.copy()
        editor_input = editor_input.drop(
            columns=[REMOVE_COLUMN], errors="ignore"
        )
        editor_input[REMOVE_COLUMN] = False
        editor_input = with_premium_codes(editor_input, market_pair)
        for note in st.session_state.pop("_premium_notes", []):
            st.warning(note)
        ticks = editor_input[BOOK_COLUMN] if BOOK_COLUMN in editor_input else False
        editor_input[BOOK_COLUMN] = pd.Series(ticks, index=editor_input.index).map(
            lambda value: value is True or value == 1)
        editor_column_config = {
                "strike": st.column_config.TextColumn(
                    "Strike",
                    help="Strike, delta, or ATM alias (for example 1.18, 25d, atm, atmf, atms).",
                    width="small",
                ),
                "currency_pair": st.column_config.TextColumn(
                    "Currency pair",
                    help="Six-letter FX currency pair, such as EURUSD.",
                    width="small",
                ),
                "expiry": st.column_config.TextColumn(
                    "Expiry",
                    help="Expiry, such as 1w, 2w, 1m, 1y, 09/18/2025, or 18sep25.",
                    width="small",
                ),
                "call_put": st.column_config.SelectboxColumn(
                    "Call/Put", help="Option type: call or put.", options=["CALL", "PUT"], width="small",
                ),
                "buy_sell": st.column_config.SelectboxColumn(
                    "Buy/Sell", help="Trade direction: buy or sell.", options=["BUY", "SELL"], width="small",
                ),
                "notional": st.column_config.TextColumn(
                    "Notional", help="Base-currency notional, for example 100k or 1m.", width="small",
                ),
                "premium": st.column_config.NumberColumn(
                    "Premium", help="Absolute premium amount. Its cashflow sign follows Side.",
                    min_value=0.0, format="%.2f", width="small",
                ),
                "premium_ccy": st.column_config.SelectboxColumn(
                    "Premium in",
                    help="Premium paid or received in the pair's base or term currency.",
                    options=list(SIDES),
                    default=TERM,
                    width="small",
                ),
                PREMIUM_CODE_COLUMN: st.column_config.TextColumn(
                    "Premium ccy", help="The currency BASE or TERM stands for on this line.",
                    width="small",
                ),
                "book": st.column_config.TextColumn(
                    "Book", help="Book to book the line into; empty uses the default below.",
                    width="small",
                ),
                "counterparty": st.column_config.TextColumn(
                    "Counterparty", help="Counterparty; empty uses the default below.",
                    width="small",
                ),
                TRADE_ID_COLUMN: st.column_config.TextColumn(
                    "Trade ID", help="Set when the line is booked. Editing the line "
                                     "clears it: amend booked trades in the Blotter.",
                    width="small",
                ),
                BOOK_COLUMN: st.column_config.CheckboxColumn(
                    "Book?", help="Tick lines, then use Book ticked lines below.",
                    default=False, width="small",
                ),
                REMOVE_COLUMN: st.column_config.CheckboxColumn(
                    "Remove",
                    help="Tick to remove this line immediately.",
                    default=False,
                    width="small",
                ),
                "vol_pct": st.column_config.TextColumn(
                    "Implied volatility (%)", help="Solved implied volatility in percent.",
                    width="small",
                ),
                "mtm_base": st.column_config.TextColumn(
                    "MTM (base)", help="Mark-to-market in base currency.",
                    width="small",
                ),
                "pnl_base": st.column_config.TextColumn(
                    "P&L (base)", help="P&L in base currency, including premium.",
                    width="small",
                ),
                "mtm_quote": st.column_config.TextColumn(
                    "MTM ({})".format(quote_label),
                    help="Mark-to-market in the selected premium quote.",
                    width="small",
                ),
                "pnl_quote": st.column_config.TextColumn(
                    "P&L ({})".format(quote_label),
                    help="P&L in the selected premium quote, including premium.",
                    width="small",
                ),
        }
        for greek_name in greeks:
            # Theta is a P&L amount in the pair's premium currency; the other
            # Greeks are in the base currency.
            unit = (
                " (prem ccy)" if greek_name == "theta" else ""
            )
            unit_help = (
                " In the premium currency: USD when USD is the term currency, "
                "otherwise the base currency." if greek_name == "theta" else ""
            )
            greek = catalog.CATALOG[greek_name]
            editor_column_config[f"fixed_{greek_name}"] = (
                st.column_config.TextColumn(
                    catalog.label(f"fixed_{greek_name}") + unit,
                    help=f"Fixed-volatility {greek.label.lower()} ({greek.unit}).{unit_help}",
                    width="small",
                )
            )
            editor_column_config[greek_name] = st.column_config.TextColumn(
                catalog.label(greek_name) + unit,
                help=f"Smile-adjusted {greek.label.lower()} ({greek.unit}).{unit_help}",
                width="small",
            )
        edited_rows = st.data_editor(
            red_negatives(editor_input, LINE_RESULT_COLUMNS),
            num_rows="dynamic",
            use_container_width=True,
            column_config=editor_column_config,
            hide_index=True,
            disabled=[*LINE_RESULT_COLUMNS, TRADE_ID_COLUMN, PREMIUM_CODE_COLUMN],
            key="portfolio_editor_v3",
            on_change=resolve_strikes_after_edit,
        )
        ticked = [position for position, value in enumerate(edited_rows[BOOK_COLUMN])
                  if value is True or value == 1] if BOOK_COLUMN in edited_rows else []
        edited_rows = edited_rows.drop(
            columns=[REMOVE_COLUMN, BOOK_COLUMN, PREMIUM_CODE_COLUMN, *LINE_RESULT_COLUMNS],
            errors="ignore"
        )
        delta_hedge = st.checkbox(
            "Add forward hedge",
            key="delta_hedge",
            help="Adds offsetting base-currency forward trades grouped by pair and expiry.",
        )
    return edited_rows, delta_hedge, ticked

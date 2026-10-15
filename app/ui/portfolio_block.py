"""Portfolio input block for the Streamlit application."""

import pandas as pd
import streamlit as st

from app.pricer import GREEK_NAMES


REMOVE_COLUMN = "_remove"
#: Tick box choosing the lines "Book ticked lines" books.
BOOK_COLUMN = "_book"
TRADE_ID_COLUMN = "trade_id"
BOOKING_COLUMNS = ("book", "counterparty")
LINE_RESULT_COLUMNS = (
    "vol_pct",
    "mtm_base",
    "pnl_base",
    "mtm_quote",
    "pnl_quote",
    *(column for greek in GREEK_NAMES for column in (f"fixed_{greek}", greek)),
)

#: Input cells whose edits can change a resolved pair, expiry or strike.
RESOLVED_INPUT_COLUMNS = ("currency_pair", "expiry", "call_put", "strike")

#: Columns the pricer supplies already formatted for display.
PREFORMATTED_RESULT_COLUMNS = ("mtm_quote", "pnl_quote")


def _with_line_results(rows, results):
    """Add read-only pricing outputs to their matching visible input rows."""
    display = rows.drop(columns=LINE_RESULT_COLUMNS, errors="ignore").copy()
    for column in LINE_RESULT_COLUMNS:
        display[column] = pd.NA
    for result in results:
        row_number = result.get("row")
        if not isinstance(row_number, int) or not 1 <= row_number <= len(display):
            continue
        row_index = row_number - 1
        for column in LINE_RESULT_COLUMNS:
            if column in result:
                value = result[column]
                if column in PREFORMATTED_RESULT_COLUMNS:
                    # Already rendered in the active premium quote upstream.
                    display.at[row_index, column] = (
                        pd.NA if value is None else str(value)
                    )
                elif pd.isna(value):
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


def _without_ticks(rows):
    return rows.drop(columns=[BOOK_COLUMN], errors="ignore")


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
            if column != REMOVE_COLUMN
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


def _is_pair_code(pair):
    return len(pair) == 6 and pair.isascii() and pair.isalpha()


def premium_currency_options(market_pair, pairs=()):
    """Premium currencies offered in the grid: every loaded pair's two.

    The active pair's currencies come first and its term currency is the
    default for new lines.
    """
    codes = [str(pair).upper() for pair in (market_pair, *pairs)]
    options = list(dict.fromkeys(
        currency for pair in codes if _is_pair_code(pair) for currency in (pair[:3], pair[3:])
    ))
    active = codes[0]
    default = active[3:] if _is_pair_code(active) else None
    return (options, default) if options else (["BASE", "TERM"], "TERM")


def render_portfolio_block(portfolio_tab, ladder_tab, market_pair, quote_label="quote",
                           pairs=()):
    """Render pricer lines, hedge choice, and spot ladder inputs.

    ``pairs`` are the loaded markets: the grid offers their currencies and
    the spot ladder chooses among them.
    """
    market_pair = str(market_pair).upper()
    premium_currencies, premium_default = premium_currency_options(market_pair, pairs)

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
        st.session_state.pop("_portfolio_line_results", None)
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
        # Ticking lines to book does not change them, so results stay shown.
        if (
            line_results is not None
            and _without_ticks(st.session_state.option_rows).equals(
                _without_ticks(line_results["rows"]))
        ):
            editor_input = _with_line_results(
                st.session_state.option_rows, line_results["results"]
            )
        else:
            st.session_state.pop("_portfolio_line_results", None)
            editor_input = st.session_state.option_rows.copy()
        editor_input = editor_input.drop(
            columns=[REMOVE_COLUMN], errors="ignore"
        )
        editor_input[REMOVE_COLUMN] = False
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
                    "Premium currency",
                    help="Currency in which the premium is paid or received.",
                    options=premium_currencies,
                    default=premium_default,
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
        for greek_name in GREEK_NAMES:
            title = greek_name.title()
            # Theta is a P&L amount in the pair's premium currency; the other
            # Greeks are in the base currency.
            unit = (
                " (prem ccy)" if greek_name == "theta" else ""
            )
            unit_help = (
                " In the premium currency: USD when USD is the term currency, "
                "otherwise the base currency." if greek_name == "theta" else ""
            )
            editor_column_config[f"fixed_{greek_name}"] = (
                st.column_config.TextColumn(
                    f"Fixed {title}{unit}",
                    help=f"Fixed-volatility {greek_name}.{unit_help}",
                    width="small",
                )
            )
            editor_column_config[greek_name] = st.column_config.TextColumn(
                f"Smile {title}{unit}",
                help=f"Smile-adjusted {greek_name}.{unit_help}",
                width="small",
            )
        edited_rows = st.data_editor(
            editor_input,
            num_rows="dynamic",
            use_container_width=True,
            column_config=editor_column_config,
            hide_index=True,
            disabled=[*LINE_RESULT_COLUMNS, TRADE_ID_COLUMN],
            key="portfolio_editor_v3",
            on_change=resolve_strikes_after_edit,
        )
        ticked = [position for position, value in enumerate(edited_rows[BOOK_COLUMN])
                  if value is True or value == 1] if BOOK_COLUMN in edited_rows else []
        edited_rows = edited_rows.drop(
            columns=[REMOVE_COLUMN, BOOK_COLUMN, *LINE_RESULT_COLUMNS], errors="ignore"
        )
        delta_hedge = st.checkbox(
            "Add forward hedge",
            key="delta_hedge",
            help="Adds offsetting base-currency forward trades grouped by pair and expiry.",
        )
    with ladder_tab:
        st.subheader("Spot ladder")
        ladder_pairs = list(dict.fromkeys(
            [pair for pair in (market_pair, *pairs) if _is_pair_code(str(pair))]
        )) or [market_pair]
        if st.session_state.get("ladder_pair") not in ladder_pairs:
            st.session_state["ladder_pair"] = ladder_pairs[0]
        ladder_col0, ladder_col1, ladder_col2 = st.columns(3)
        with ladder_col0:
            ladder_pair = st.selectbox(
                "Currency pair", ladder_pairs, key="ladder_pair",
                help="Shocks this pair's spot and revalues its lines (and its "
                     "forward hedges). Lines on other pairs are not included.",
            )
        with ladder_col1:
            ladder_steps = st.number_input(
                "Steps on each side of spot",
                min_value=0,
                max_value=20,
                value=5,
                step=1,
                key="ladder_steps",
            )
        with ladder_col2:
            ladder_increment = st.number_input(
                "Spot increment (%)",
                min_value=0.0001,
                value=0.50,
                step=0.05,
                format="%.2f",
                key="ladder_increment",
            )
        ladder_calculate = st.button(
            "Calculate spot ladder", type="primary", key="calculate_spot_ladder"
        )
    
    return (edited_rows, ladder_steps, ladder_increment, ladder_calculate, delta_hedge,
            ladder_pair, ticked)

"""Portfolio input block for the Streamlit application."""

import numpy as np
import pandas as pd
import streamlit as st

from app.option_types import LABELS as OPTION_TYPE_LABELS, VANILLA
from app.premium_side import BASE, SIDES, TERM, premium_currency
from app.pricer import AUTO, GREEK_NAMES, PREMIUM_MODE, TYPED
from app.hedge_delta import HEDGE_DELTA, HEDGE_MODE, shown_pct
from app.vol_override import MARKET, MAX_VOL, VOL_INPUT, VOL_MODE
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
#: The vol a line priced at is its Vol % cell (typed, or the market's).
SUMMARY_RESULT_COLUMNS = ("mkt_vol_pct", "vol_spread", "prem_at_vol", "mtm_base", "pnl_base",
                          "delta_convention")
#: Every read-only result column a line can carry (all catalogue Greeks).
LINE_RESULT_COLUMNS = (*SUMMARY_RESULT_COLUMNS, "vol_pct", *catalog.columns(catalog.CATALOG))
#: A line at market has no override fields: its market vol is its vol.
RESULT_FALLBACKS = {"mkt_vol_pct": "vol_pct"}
PRICER_GREEKS_KEY = "pricer_greeks"
#: A line shows its delta in its pair's convention, when the pricing has it.
PAIR_DELTA_COLUMNS = {"delta": "pair_delta", "fixed_delta": "fixed_pair_delta"}

#: Input cells whose edits can change a resolved pair, expiry or strike.
RESOLVED_INPUT_COLUMNS = ("currency_pair", "expiry", "call_put", "strike")


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
            source = PAIR_DELTA_COLUMNS.get(column, column)
            source = source if source in result else RESULT_FALLBACKS.get(column, column)
            if source in result:
                value = result[source]
                if value is None or (not isinstance(value, str) and pd.isna(value)):
                    display.at[row_index, column] = pd.NA
                elif column == "delta_convention":
                    display.at[row_index, column] = str(value)
                elif column in ("vol_pct", "mkt_vol_pct"):
                    display.at[row_index, column] = f"{float(value):.4f}"
                elif column == "vol_spread":
                    display.at[row_index, column] = f"{float(value):+.4f}"
                elif column == "prem_at_vol":
                    display.at[row_index, column] = f"{float(value):,.2f}"
                else:
                    # A rounding residue below half a unit shows as 0, not "-0".
                    number = float(value)
                    display.at[row_index, column] = f"{0.0 if abs(number) < 0.5 else number:,.0f}"
    return display


#: Cells a user types; an added line without any of them is a placeholder.
LINE_INPUT_COLUMNS = (
    "currency_pair", "strike", "expiry", "call_put", "buy_sell", "notional", "premium",
)


#: Cells pricing reads: a line's results stay valid while these are unchanged
#: (ticks, book, counterparty and trade id do not change a price).
PRICED_INPUT_COLUMNS = (*LINE_INPUT_COLUMNS, "premium_ccy", PREMIUM_MODE, VOL_INPUT, VOL_MODE,
                        HEDGE_DELTA, HEDGE_MODE, "option_type", "payout_ccy")
#: A line's terms: changing any of them resets its premium to auto.
TERM_COLUMNS = ("currency_pair", "expiry", "call_put", "strike", "buy_sell", "notional",
                "premium_ccy", "option_type", "payout_ccy")


def _premium_entered(value):
    """True for a premium cell holding a positive amount."""
    try:
        return _has_value(value) and float(value) > 0
    except (TypeError, ValueError):
        return False


def premium_mode_of(mode, premium):
    """A line's premium mode: its own when valid, else auto for an empty or
    zero premium and typed for an entered one."""
    if isinstance(mode, str) and mode.strip().lower() in (AUTO, TYPED):
        return mode.strip().lower()
    return TYPED if _premium_entered(premium) else AUTO


def with_premium_modes(rows):
    """``rows`` with a valid premium mode on every line (see ``premium_mode_of``)."""
    rows = rows.copy()
    modes = rows[PREMIUM_MODE] if PREMIUM_MODE in rows else pd.Series([None] * len(rows),
                                                                     index=rows.index)
    premiums = rows["premium"] if "premium" in rows else pd.Series([None] * len(rows),
                                                                  index=rows.index)
    rows[PREMIUM_MODE] = [premium_mode_of(mode, premium)
                          for mode, premium in zip(modes, premiums)]
    return rows


def _vol_entered(value):
    """True for a vol cell holding a usable vol (percent)."""
    try:
        return _has_value(value) and 0.0 < float(value) <= MAX_VOL
    except (TypeError, ValueError):
        return False


def vol_mode_of(mode, vol):
    """A line's vol mode: market without a usable vol; else its own when
    valid, and typed for a line that brings a vol but no mode."""
    if not _vol_entered(vol):
        return MARKET
    if isinstance(mode, str) and mode.strip().lower() in (MARKET, TYPED):
        return mode.strip().lower()
    return TYPED


def with_vol_modes(rows):
    """``rows`` with a valid vol mode on every line (see ``vol_mode_of``)."""
    rows = rows.copy()
    blank = pd.Series([None] * len(rows), index=rows.index)
    modes = rows[VOL_MODE] if VOL_MODE in rows else blank
    vols = rows[VOL_INPUT] if VOL_INPUT in rows else blank
    rows[VOL_MODE] = [vol_mode_of(mode, vol) for mode, vol in zip(modes, vols)]
    return rows


def apply_market_vols(rows, results):
    """``rows`` with each market line's Vol % set to the vol it priced at;
    typed lines keep their vol."""
    if VOL_INPUT not in rows:
        return rows
    rows = rows.copy()
    modes = rows[VOL_MODE] if VOL_MODE in rows else pd.Series([MARKET] * len(rows),
                                                              index=rows.index)
    for result in results:
        number = result.get("row")
        if (isinstance(number, int) and 1 <= number <= len(rows)
                and not result.get("vol_override") and modes.iloc[number - 1] != TYPED
                and result.get("vol_pct") is not None):
            rows.iat[number - 1, rows.columns.get_loc(VOL_INPUT)] = float(result["vol_pct"])
    return rows


def _hedge_entered(value):
    """True for a hedge delta cell holding a usable percent (sign ignored;
    pricing checks a vanilla's against ``MAX_PCT``)."""
    try:
        return _has_value(value) and np.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def hedge_mode_of(mode, pct):
    """A line's hedge delta mode: auto without a usable delta; else its own
    when valid, and typed for a line that brings a delta but no mode."""
    if not _hedge_entered(pct):
        return AUTO
    if isinstance(mode, str) and mode.strip().lower() in (AUTO, TYPED):
        return mode.strip().lower()
    return TYPED


def with_hedge_modes(rows):
    """``rows`` with a valid hedge delta mode on every line."""
    rows = rows.copy()
    blank = pd.Series([None] * len(rows), index=rows.index)
    modes = rows[HEDGE_MODE] if HEDGE_MODE in rows else blank
    pcts = rows[HEDGE_DELTA] if HEDGE_DELTA in rows else blank
    rows[HEDGE_MODE] = [hedge_mode_of(mode, pct) for mode, pct in zip(modes, pcts)]
    return rows


def apply_auto_hedge_deltas(rows, results):
    """``rows`` with each auto line's hedge delta set to its shown delta (a
    percent of notional in its convention); typed lines keep theirs."""
    if HEDGE_DELTA not in rows:
        return rows
    from app.pricer import parse_notional
    rows = rows.copy()
    modes = rows[HEDGE_MODE] if HEDGE_MODE in rows else pd.Series([AUTO] * len(rows),
                                                                  index=rows.index)
    for result in results:
        number = result.get("row")
        if (not isinstance(number, int) or not 1 <= number <= len(rows)
                or result.get("hedge_override") or modes.iloc[number - 1] == TYPED):
            continue
        try:
            notional = parse_notional(rows.iloc[number - 1]["notional"])
        except Exception:
            continue
        pct = shown_pct(result, notional)
        if pct is not None:
            rows.iat[number - 1, rows.columns.get_loc(HEDGE_DELTA)] = pct
    return rows


def apply_auto_premiums(rows, results):
    """``rows`` with each auto line's premium set to the one its pricing
    found (``auto_premium``); other lines unchanged."""
    rows = rows.copy()
    for result in results:
        number = result.get("row")
        if (result.get("premium_auto") and isinstance(number, int)
                and 1 <= number <= len(rows) and "premium" in rows):
            rows.iat[number - 1, rows.columns.get_loc("premium")] = float(result["auto_premium"])
    return rows


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
                    and any(column in (*LINE_INPUT_COLUMNS, *BOOKING_COLUMNS, "option_type",
                                       "payout_ccy") for column in changes)):
                updated.iat[index, updated.columns.get_loc(TRADE_ID_COLUMN)] = ""
            # Typing a premium makes it the line's (clearing it returns to
            # auto); any other change to the terms resets it to auto.
            if PREMIUM_MODE in updated.columns:
                mode = updated.columns.get_loc(PREMIUM_MODE)
                if "premium" in changes:
                    updated.iat[index, mode] = (TYPED if _premium_entered(changes["premium"])
                                                else AUTO)
                elif any(column in TERM_COLUMNS for column in changes):
                    updated.iat[index, mode] = AUTO
                    if "premium" in updated.columns:
                        updated.iat[index, updated.columns.get_loc("premium")] = float("nan")
            # Typing a vol makes it the line's (clearing it returns to the
            # market); a change to the terms returns it to the market too.
            if VOL_MODE in updated.columns:
                mode = updated.columns.get_loc(VOL_MODE)
                if VOL_INPUT in changes:
                    updated.iat[index, mode] = TYPED if _vol_entered(changes[VOL_INPUT]) else MARKET
                elif any(column in TERM_COLUMNS for column in changes):
                    updated.iat[index, mode] = MARKET
                    if VOL_INPUT in updated.columns:
                        updated.iat[index, updated.columns.get_loc(VOL_INPUT)] = float("nan")
            # Typing a hedge delta makes it the line's (clearing it returns to
            # auto); a change to the terms or the vol returns it to auto, as
            # the agreed delta belonged to the old ones.
            if HEDGE_MODE in updated.columns:
                mode = updated.columns.get_loc(HEDGE_MODE)
                if HEDGE_DELTA in changes:
                    updated.iat[index, mode] = (TYPED if _hedge_entered(changes[HEDGE_DELTA])
                                                else AUTO)
                elif any(column in (*TERM_COLUMNS, VOL_INPUT) for column in changes):
                    updated.iat[index, mode] = AUTO
                    if HEDGE_DELTA in updated.columns:
                        updated.iat[index, updated.columns.get_loc(HEDGE_DELTA)] = float("nan")

    deleted_indices = sorted(
        (int(index) for index in editor_state.get("deleted_rows", [])),
        reverse=True,
    )
    for index in deleted_indices:
        if 0 <= index < len(updated):
            updated = updated.drop(updated.index[index])

    updated = updated.reset_index(drop=True)
    if PREMIUM_MODE in updated.columns or added_rows:
        # Added lines: typed when they carry a premium, auto otherwise.
        updated = with_premium_modes(updated)
    if VOL_MODE in updated.columns or added_rows:
        # Added lines: typed when they carry a vol, market otherwise.
        updated = with_vol_modes(updated)
    if HEDGE_MODE in updated.columns or added_rows:
        # Added lines: typed when they carry a hedge delta, auto otherwise.
        updated = with_hedge_modes(updated)
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


def adopt_pending_rows():
    """Make lines left by the last run (resolved, priced, or committed edits)
    the grid's lines, and its pricing results the shown ones.

    The app calls this before it sets up the grid's columns: lines adopted
    after that would show unprepared for one run, and the editor, whose
    identity is its contents, would be rebuilt on the next run, dropping a
    line being added. Calling it again in the same run does nothing.
    """
    pending_resolved_rows = st.session_state.pop("_pending_resolved_portfolio_rows", None)
    pending_priced_rows = st.session_state.pop("_pending_priced_portfolio_rows", None)
    pending_editor_rows = st.session_state.pop("_pending_editor_portfolio_rows", None)
    pending_line_results = st.session_state.pop("_pending_portfolio_line_results", None)
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


def render_portfolio_block(portfolio_tab, market_pair, pairs=()):
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
        # Normally done already, before the app sets up the grid's columns.
        adopt_pending_rows()
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
                "option_type": st.column_config.SelectboxColumn(
                    "Type", options=list(OPTION_TYPE_LABELS.values()),
                    default=OPTION_TYPE_LABELS[VANILLA], width="small",
                    help="Option type. Vanilla and Digital (European, replicated with "
                         "vanilla spreads on the smile) are priced; the others are listed "
                         "for what comes next and report that they are not priced yet."),
                "payout_ccy": st.column_config.SelectboxColumn(
                    "Payout in", options=list(SIDES), default=BASE, width="small",
                    help="Binary-payout types (digital, touches): the notional is the "
                         "payout, paid in the pair's base (default) or term currency. "
                         "Vanillas ignore it."),
                "call_put": st.column_config.SelectboxColumn(
                    "Call/Put", help="Option type: call or put.", options=["CALL", "PUT"], width="small",
                ),
                "buy_sell": st.column_config.SelectboxColumn(
                    "Buy/Sell", help="Trade direction: buy or sell.", options=["BUY", "SELL"], width="small",
                ),
                "notional": st.column_config.TextColumn(
                    "Notional", help="Base-currency notional, for example 100k or 1m.", width="small",
                ),
                VOL_INPUT: st.column_config.NumberColumn(
                    "Vol %", min_value=0.0, max_value=MAX_VOL, format="%.4f", width="small",
                    help="The vol the line prices at: the market's at the strike, or "
                         "yours. A typed vol shifts this line's smile so its strike is at "
                         "your vol; Greeks follow it, an auto premium stays at the market "
                         "price, so P&L is your vol against market. Clear it, or change "
                         "a term, to return to the market."),
                VOL_MODE: st.column_config.TextColumn(
                    "Vol src", width="small",
                    help="market: the smile's vol at the strike. typed: the vol you entered."),
                HEDGE_DELTA: st.column_config.NumberColumn(
                    "Hedge Δ %", min_value=0.0, format="%.2f",
                    width="small",
                    help="The delta the line's hedge flattens, in percent of notional, in "
                         "the line's convention (Delta conv.), without sign: a bought "
                         "call or sold put is long delta, a bought put or sold call "
                         "short. auto: the line's delta. Type the agreed delta to hedge "
                         "and book that; clearing it, or changing a term or the vol, "
                         "returns to auto."),
                HEDGE_MODE: st.column_config.TextColumn(
                    "Delta src", width="small",
                    help="auto: the line's delta. typed: the delta you entered."),
                "mkt_vol_pct": st.column_config.TextColumn(
                    "Mkt vol %", width="small", help="The market smile's vol at the strike."),
                "vol_spread": st.column_config.TextColumn(
                    "Vol spread", width="small",
                    help="Your vol minus the market's, in vol points (typed lines)."),
                "prem_at_vol": st.column_config.TextColumn(
                    "Prem @ vol", width="small",
                    help="The premium at your vol, in the premium currency (typed lines). "
                         "Type it into Premium to book that price."),
                "premium": st.column_config.NumberColumn(
                    "Premium", help="Absolute premium amount. Its cashflow sign follows Side.",
                    min_value=0.0, format="%.2f", width="small",
                ),
                PREMIUM_MODE: st.column_config.TextColumn(
                    "Premium src", width="small",
                    help="auto: pricing fills the premium that makes P&L zero (the "
                         "option's value), and books it. typed: the premium you entered. "
                         "Typing a premium makes it typed; clearing it or changing any "
                         "other term returns the line to auto."),
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
                "delta_convention": st.column_config.TextColumn(
                    "Delta conv.", width="small",
                    help="The line's delta convention, the pair's for this expiry: "
                         "S spot delta, F forward delta (G10 spot up to 1Y, EM forward).",
                ),
        }
        for greek_name in greeks:
            # Theta is a P&L amount in the pair's premium currency; the other
            # Greeks are in the base currency.
            unit = "prem ccy" if greek_name == "theta" else None
            unit_help = (
                " In the premium currency: USD when USD is the term currency, "
                "otherwise the base currency." if greek_name == "theta" else ""
            )
            greek = catalog.CATALOG[greek_name]
            editor_column_config[f"fixed_{greek_name}"] = (
                st.column_config.TextColumn(
                    catalog.label(f"fixed_{greek_name}", unit),
                    help=f"Sticky-delta {greek.label.lower()} ({greek.unit}): the smile "
                         f"quotes stay unchanged as spot moves.{unit_help}",
                    width="small",
                )
            )
            convention_help = (
                " In the line's convention (Delta conv.); totals add spot-equivalent delta."
                if greek_name == "delta" else "")
            fixed_config = editor_column_config[f"fixed_{greek_name}"]
            if convention_help and isinstance(fixed_config, dict):
                fixed_config["help"] = (fixed_config.get("help") or "") + convention_help
            editor_column_config[greek_name] = st.column_config.TextColumn(
                catalog.label(greek_name, unit),
                help=f"Smile {greek.label.lower()} ({greek.unit}): the smile moved by "
                     f"the betas.{unit_help}"
                     f"{convention_help}",
                width="small",
            )
        edited_rows = st.data_editor(
            red_negatives(editor_input, LINE_RESULT_COLUMNS),
            num_rows="dynamic",
            use_container_width=True,
            column_config=editor_column_config,
            hide_index=True,
            disabled=[*LINE_RESULT_COLUMNS, TRADE_ID_COLUMN, PREMIUM_CODE_COLUMN,
                      PREMIUM_MODE, VOL_MODE, HEDGE_MODE],
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

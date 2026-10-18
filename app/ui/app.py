"""Streamlit user interface for the VannaVolga portfolio pricer."""

import json
from datetime import date, timedelta
import pandas as pd
import streamlit as st

from market_data import Modifications
from app import pricer
from app.market_context import (
    PricingSettings, build_context, interpolate_delivery_curve, naive_date,
)
from app.market_set import MarketSet
from app.market_service import MarketService
from app.market_store import validate_market
from app.curve_roll import next_business_day
from app.smile_service import (
    SURFACE_CACHE_VERSION, DiagnosticsCollector, SurfaceCache, bind_cache,
)
from app.ui.market_block import render_market_block
from app.ui.model_block import render_model_tab
from app.premium_side import TERM, premium_side
from app.ui.greek_picker import chosen_greeks, missing_extras
from app.ui.negatives import red_negatives
from pricing import greek_catalog
from app.ui.portfolio_block import (
    BOOK_COLUMN, PRICER_GREEKS_KEY, TRADE_ID_COLUMN, render_portfolio_block,
    same_priced_inputs,
)
from app.ui.trades_block import render_blotter, render_portfolio, trade_store
from app.ui.risk_block import render_risk
from app.ui.store_sync import LAYERS, render_store_status, sync_store_markets
from app.market_layers import origins as store_origins, unavailable as store_unavailable
from app.market_context import UnavailableMarket
from app.trades import service as trade_service

# Pricing helpers now live in the backend; kept here under their old names.
format_strike_for_pair = pricer.format_strike_for_pair
_is_complete_strike_alias = pricer.is_complete_strike_alias


aggregate_pricer_results_by_pair = pricer.aggregate_by_pair


def book_ticked_lines(store, markets, rows, ticked, mods, book, counterparty, with_hedges,
                      default_pair):
    """Book the ticked grid lines; returns ``[(level, message)]`` to show.

    Lines already carrying a trade id are not booked again (add a new line
    to book the same terms twice). On success the booked values and trade
    ids are written back to the grid; on failure nothing is booked.
    """
    import getpass
    messages = []
    ids = rows[TRADE_ID_COLUMN] if TRADE_ID_COLUMN in rows else pd.Series([None] * len(rows))
    booked_before = {p: ids.iloc[p] for p in ticked
                     if isinstance(ids.iloc[p], str) and ids.iloc[p].strip()}
    for position, trade_id in booked_before.items():
        messages.append(("warning", f"Line {position + 1} is already booked as {trade_id}; "
                                    "add a new line to book the same terms again."))
    to_book = [p for p in ticked if p not in booked_before]
    if not to_book:
        return messages
    try:
        user = getpass.getuser()
    except Exception:
        user = ""
    result = trade_service.book_lines(
        store, markets, rows.iloc[to_book], mods, book=book, counterparty=counterparty,
        with_hedges=with_hedges, user=user, default_pair=default_pair)
    if not result.ok:
        for error in result.errors:
            head, _, rest = error.partition(": ")
            if head.startswith("Line ") and head[5:].isdigit() and int(head[5:]) <= len(to_book):
                error = f"Line {to_book[int(head[5:]) - 1] + 1}: {rest}"
            messages.append(("error", "Nothing booked — " + error))
        return messages
    trades = {trade.trade_id: trade for trade in result.booked}
    updated = rows.copy()
    for number, trade_id in result.lines.items():
        index = updated.index[to_book[number - 1]]
        trade = trades[trade_id]
        for column, value in (
                ("currency_pair", trade.currency_pair),
                ("strike", pricer.format_strike_for_pair(trade.currency_pair, trade.strike)),
                ("expiry", trade.expiry.isoformat()), ("book", trade.book),
                ("counterparty", trade.counterparty), (TRADE_ID_COLUMN, trade_id)):
            if column in updated:
                updated.at[index, column] = value
    st.session_state._pending_resolved_portfolio_rows = updated
    # Booking writes back the booked values (strike 1.18 -> 1.1800, the trade
    # id) without changing a price: the results follow the rewritten lines.
    for key in ("_portfolio_line_results", "_portfolio_result_cache"):
        held = getattr(st.session_state, key, None)
        if held is not None and same_priced_inputs(held["rows"], rows):
            setattr(st.session_state, key, dict(held, rows=updated.copy()))
    options = [t.trade_id for t in result.booked if t.is_option]
    hedges = [t.trade_id for t in result.booked if t.is_forward]
    messages.append(("success", "Booked {}{}.".format(
        ", ".join(options), f"; hedges {', '.join(hedges)}" if hedges else "")))
    return messages


def format_pricer_aggregation(aggregation):
    """Format aggregate read-only monetary and risk values for display."""
    display = aggregation.copy()
    for column in display.columns:
        if column != "currency_pair":
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):,.0f}"
            )
    return display


def render_app():
    """Render the complete Streamlit application."""
    st.title("FXO Risk Analyzer")
    st.caption("Prices a portfolio of vanilla FX options and aggregates MTM/P&L and Greeks.")

    for key, default in (("spot", 1.1700), ("forward_divisor", 10000.0),
                         ("currency_pair", "EURUSD")):
        if key not in st.session_state:
            st.session_state[key] = default


    def apply_loaded_market(payload):
        """Validate one market dict and queue it as the active market."""
        loaded = validate_market(payload)
        loaded["_active_market_pair"] = loaded["currency_pair"]
        # The market's own BF flag, not the pair default, for this pair.
        loaded["_broker_strangle_pair"] = loaded["currency_pair"]
        st.session_state._pending_market_load = loaded

    pending_market = st.session_state.pop("_pending_market_load", None)
    if pending_market is not None:
        editor_keys = [
            key for key in st.session_state
            if (
                key in {"vol_editor", "event_editor"}
                or key.startswith("forward_editor_")
                or key in {"_last_delta_vol_editor", "_last_delta_event_editor"}
                or key.startswith("_last_delta_forward_editor_")
            )
        ]
        for key in editor_keys:
            st.session_state.pop(key, None)
        for key, value in pending_market.items():
            st.session_state[key] = value


    def default_rows():
        expiry = (date.today() + timedelta(days=180)).strftime("%m/%d/%Y")
        return pd.DataFrame([{
            "currency_pair": str(st.session_state.currency_pair).upper(),
            "strike": "1.18",
            "expiry": expiry,
            "call_put": "CALL",
            "buy_sell": "BUY",
            "notional": "1m",
            "premium": 0.0,
            "premium_ccy": TERM,
        }])

    if "option_rows" not in st.session_state:
        st.session_state.option_rows = default_rows()

    # Migrate existing in-session rows when new pricing inputs are introduced.
    if "currency_pair" not in st.session_state.option_rows:
        st.session_state.option_rows["currency_pair"] = str(
            st.session_state.currency_pair
        ).upper()
    if "premium" not in st.session_state.option_rows:
        st.session_state.option_rows["premium"] = 0.0
    # The grid holds the premium currency as BASE/TERM; lines from before
    # (currency codes) convert, and a code outside the line's pair falls
    # back to TERM with a note.
    if "premium_ccy" not in st.session_state.option_rows:
        st.session_state.option_rows["premium_ccy"] = TERM
    sides = [premium_side(pair, value) for pair, value in zip(
        st.session_state.option_rows["currency_pair"],
        st.session_state.option_rows["premium_ccy"])]
    st.session_state.option_rows["premium_ccy"] = [side for side, _ in sides]
    premium_notes = [f"Line {number}: {note}"
                     for number, (_, note) in enumerate(sides, start=1) if note]
    if premium_notes:
        st.session_state["_premium_notes"] = premium_notes
    if st.session_state.get("_portfolio_editor_schema") != 6:
        st.session_state.pop("portfolio_editor_v2", None)
        st.session_state.pop("portfolio_editor_v3", None)
        st.session_state._portfolio_editor_schema = 6

    portfolio_columns = [
        "currency_pair", "expiry", "call_put", "strike", "buy_sell",
        "notional", "premium", "premium_ccy", "book", "counterparty", TRADE_ID_COLUMN,
    ]
    portfolio_columns += [
        column for column in st.session_state.option_rows.columns
        if column not in portfolio_columns
    ]
    portfolio_columns = [c for c in portfolio_columns if c != BOOK_COLUMN] + [BOOK_COLUMN]
    st.session_state.option_rows = st.session_state.option_rows.reindex(
        columns=portfolio_columns
    )
    # Text columns must hold text for the editor: empty cells are "".
    for column in ("book", "counterparty", TRADE_ID_COLUMN):
        st.session_state.option_rows[column] = st.session_state.option_rows[column].map(
            lambda value: "" if value is None or (not isinstance(value, str) and pd.isna(value))
            else str(value))

    # The strike editor accepts both numeric values and delta strings such as
    # ``25d``, so keep this column as text even when the initial value is numeric.
    if "strike" in st.session_state.option_rows:
        # Empty cells stay empty: str() would turn them into the text "nan".
        st.session_state.option_rows["strike"] = (
            st.session_state.option_rows["strike"].map(
                lambda value: "" if not isinstance(value, str) and pd.isna(value) else str(value)
            )
        )

    with st.sidebar:
        st.header("Valuation")
        valuation_date = st.date_input("Valuation date", key="valuation_date")
        # Markets come from the market data store (latest on or before the
        # date); files and edits sit on top. A store problem only shows here.
        render_store_status(sync_store_markets(valuation_date, apply_loaded_market))

        st.header("Reports")
        spot_ladder_mode = st.selectbox(
            "Spot ladder calculation mode",
            ["shadow", "optimized", "reference"],
            index=1,
            help="Shadow compares optimized and reference results before returning.",
            key="spot_ladder_mode",
        )
        st.caption("Pricing model and smile dynamics: Model tab.")

    surface_progress = st.empty()
    book_tab, risk_tab, portfolio_tab, blotter_tab, markets_tab, model_tab = st.tabs(
        ["Portfolio", "Risk", "Pricer", "Blotter", "Markets", "Model"]
    )
    # Global model settings (Model tab); invalid or missing values fall back
    # to the defaults. The app fixes the rest.
    model = render_model_tab(model_tab)
    pricing_model = model["pricing_model"]
    beta_atm, beta_rr, beta_bf = model["beta_atm"], model["beta_rr"], model["beta_bf"]
    atm_type, delta_base = "dn", True
    smooth_pillar_interpolation = True
    # Deltas are computed spot-equivalent, so totals add up; each line
    # shows its pair's convention. Hedge amounts are forward deltas.
    delta_reporting = "spot"

    spot, forward_divisor, currency_pair, vol_rows, forward_rows, event_rows = render_market_block(
        markets_tab, valuation_date, apply_loaded_market
    )
    # The active pair's BF convention (Markets tab); other pairs keep their own.
    broker_strangle = bool(st.session_state.get("broker_strangle", True))
    diagnostics = DiagnosticsCollector()
    # Calibrated surfaces live for the whole session, so reruns and pricing
    # reuse them instead of recalibrating per option or per strike.
    # The key carries the cache version: after a code change, surfaces built
    # by the old code are dropped instead of reused.
    cache_key = f"_surface_cache_v{SURFACE_CACHE_VERSION}"
    service_key = f"_market_service_v{SURFACE_CACHE_VERSION}"
    if cache_key not in st.session_state:
        for stale in [key for key in st.session_state
                      if key.startswith(("_surface_cache_v", "_market_service_v"))]:
            st.session_state.pop(stale, None)
        st.session_state[cache_key] = SurfaceCache()
        st.session_state[service_key] = MarketService(diagnostics)
    bind_cache(diagnostics.token, st.session_state[cache_key])
    market_service = st.session_state[service_key]
    market_service.collector = diagnostics
    market_service.progress = None
    settings = PricingSettings(
        pricing_model=pricing_model, atm_type=atm_type, delta_base=delta_base,
        smooth_pillar_interpolation=smooth_pillar_interpolation,
        delta_reporting=delta_reporting,
    )
    # The active pair's market as one backend object. Inputs that cannot form
    # a market at all (a half-typed pair, a zero spot) give a stand-in that
    # reports the reason on every line instead of stopping the page.
    market_context = build_context(
        currency_pair, valuation_date, spot, forward_divisor, broker_strangle,
        vol_rows, forward_rows, event_rows, settings=settings,
        market_service=market_service, diagnostics=diagnostics,
    )
    if getattr(market_context, "reason", None):
        with markets_tab:
            st.warning("Market unavailable for pricing: " + market_context.reason)
    # Every loaded pair, valued as of the sidebar date. The pair being edited
    # comes from the live tables; the others from the session's market set.
    market_library = st.session_state.get("market_library", {})
    layers = st.session_state.get(LAYERS, {})
    missing = store_unavailable(layers)
    # A store pair the current market cannot build prices nowhere, not even
    # from the Markets tab's leftover tables, unless the user edited them.
    live = None if market_context.currency_pair in missing else market_context
    market_set = MarketSet.from_payloads(
        market_library, valuation_date, settings, market_service=market_service,
        diagnostics=diagnostics, live=live,
    )
    for pair, reason in missing.items():
        if pair not in market_set:
            market_set.add(UnavailableMarket(pair, reason, valuation_date))
    st.session_state["_market_origins"] = store_origins(layers)


    def prepare_market_surfaces():
        """Calibrate the pairs in use, each once per market state, with progress.

        Only the active pair, the Pricer lines' pairs and the selected
        Portfolio trades' pairs are calibrated here, and a pair only when its
        market changed (fingerprint), so adding a pair calibrates just that
        pair. Store markets restore from the disk cache. Other pairs
        calibrate on demand if something prices them. A pair that fails is
        skipped here; pricing reports its errors per line.
        """
        from app.calibration_plan import calibrate_pairs, pairs_in_use
        from app.ui.trades_block import trade_store
        rows = st.session_state.get("option_rows")
        trades = []
        trades_store, _ = trade_store()
        if trades_store is not None:
            try:
                live = trades_store.trades(status="live")
                selected = st.session_state.get("_portfolio_selection")
                trades = live if selected is None else [
                    trade for trade in live if trade.trade_id in set(selected)]
            except Exception:
                trades = []           # an unreadable book only means less pre-calibration
        pairs = pairs_in_use(market_set.available(), market_context.currency_pair, rows, trades)
        common = (repr(settings), str(valuation_date))
        fingerprints = {}
        for pair in pairs:
            if pair == market_context.currency_pair:
                state = (spot, forward_divisor, broker_strangle,
                         vol_rows.to_json(date_format="iso"),
                         forward_rows.to_json(date_format="iso"),
                         event_rows.to_json(date_format="iso"))
            else:
                state = json.dumps(market_library.get(pair), sort_keys=True, default=str)
            fingerprints[pair] = repr((common, state))
        done = st.session_state.setdefault("_surface_fingerprints", {})
        if all(done.get(pair) == fingerprints[pair] for pair in pairs):
            return
        bar = surface_progress.progress(0.0, text="Preparing volatility surfaces...")

        def report(position, count, pair):
            bar.progress(min(position / max(count, 1), 1.0),
                         text=f"{pair}: preparing {pricing_model} smiles ({position + 1}/{count})")
        try:
            outcomes = calibrate_pairs(
                market_set, pairs, fingerprints, done, settings, valuation_date,
                st.session_state[cache_key], market_service,
                origins=st.session_state.get("_market_origins", {}),
                payloads=market_library, progress=report)
        finally:
            market_service.progress = None
            surface_progress.empty()
        st.session_state["_last_calibration"] = outcomes

    prepare_market_surfaces()


    mods = Modifications(beta_atm=beta_atm, beta_rr=beta_rr, beta_bf=beta_bf)
    # Pricer results are kept across reruns only while the markets, settings
    # and smile dynamics they were priced with are unchanged.
    pricing_state = repr((sorted(st.session_state.get("_surface_fingerprints", {}).items()),
                          repr(mods)))
    for key in ("_portfolio_line_results", "_portfolio_result_cache"):
        held = st.session_state.get(key)
        if held is not None and held.get("state") != pricing_state:
            st.session_state.pop(key, None)
    edited_rows, delta_hedge, ticked_lines = render_portfolio_block(
        portfolio_tab,
        market_pair=str(st.session_state.currency_pair).strip().upper(),
        pairs=sorted(market_set.pairs),
    )
    if st.session_state.pop("_portfolio_strike_resolution_requested", False):
        resolved_rows, strikes_changed, strike_failures = (
            pricer.resolve_line_inputs(market_set, edited_rows, market_context.currency_pair)
        )
        if strike_failures:
            st.session_state._portfolio_strike_failures = strike_failures
        else:
            st.session_state.pop("_portfolio_strike_failures", None)
        if strikes_changed:
            # Carry the complete edited frame forward before resetting the keyed
            # widget, preserving edits to every other column and dynamic row.
            st.session_state._pending_resolved_portfolio_rows = resolved_rows
            st.experimental_rerun()


    with portfolio_tab:
        from app.market_layers import group_by_message, pair_list
        if market_set.available():
            st.caption("Markets: {} (valued as of {:%d-%b-%Y})".format(
                pair_list(market_set.available()), naive_date(valuation_date)))
        # Pairs sharing a reason share one line (a date with no market data
        # would otherwise list every pair).
        for pairs, reason in group_by_message(market_set.problems()):
            st.warning(f"{pair_list(pairs)} cannot price: {reason}")
        for pairs, note in group_by_message(market_set.notes()):
            st.info(f"{pair_list(pairs)}: {note}")
        for message in st.session_state.get("_portfolio_strike_failures", []):
            st.warning(message)


    with portfolio_tab:
        calculate = st.button(
            "Price portfolio", type="primary", key="calculate_portfolio"
        )

    store, store_problem = trade_store()
    can_book = store is not None and not store.read_only
    with portfolio_tab:
        st.markdown("**Book**")
        book_col, counterparty_col, hedge_col, button_col = st.columns(4)
        with book_col:
            default_book = st.text_input("Default book", key="booking_book")
        with counterparty_col:
            default_counterparty = st.text_input("Default counterparty",
                                                 key="booking_counterparty")
        with hedge_col:
            book_hedges = st.checkbox(
                "Book delta hedges", key="booking_hedges",
                help="Also books one forward per pair and expiry that flattens the "
                     "booked lines' delta, linked to each option it hedges.")
        with button_col:
            book_clicked = st.button(
                f"Book ticked lines ({len(ticked_lines)})", key="book_lines",
                disabled=not ticked_lines or not can_book)
        if not can_book:
            st.caption("Booking is unavailable: " + (store_problem or "trade store is read-only"))
        for level, message in st.session_state.pop("_booking_messages", []):
            getattr(st, level)(message)

    if book_clicked:
        st.session_state._booking_messages = book_ticked_lines(
            store, market_set, edited_rows, ticked_lines, mods, default_book,
            default_counterparty, book_hedges, market_context.currency_pair)
        st.experimental_rerun()


    if calculate:
        # Book what is priced: tenors, delta strikes and blank pairs are
        # resolved first and written back, so the grid shows the expiry and
        # strike each line was priced at. A line that cannot resolve keeps
        # its input and pricing reports why.
        edited_rows, _, _ = pricer.resolve_line_inputs(
            market_set, edited_rows, market_context.currency_pair
        )
        # Pricing reads no session state, but stays on the script thread so
        # progress and diagnostics belong to this run.
        results, errors = pricer.price_lines(
            market_set, edited_rows, mods,
            extras=greek_catalog.extras(chosen_greeks(PRICER_GREEKS_KEY)))
        hedge_trades = []
        if delta_hedge:
            hedge_trades, hedge_errors = pricer.hedge_trades_for(market_set, results)
            errors = errors + hedge_errors

        st.session_state._portfolio_result_cache = {
            "rows": edited_rows.copy(), "results": results, "errors": errors,
            "hedge_trades": hedge_trades,
            "diagnostics": diagnostics.records(), "state": pricing_state,
        }
        # Dynamic data-editor rows live in widget state until explicitly
        # committed. Preserve the complete frame before this rerun so pricing
        # a portfolio cannot drop rows added after the initial dataframe.
        st.session_state._pending_priced_portfolio_rows = edited_rows.copy()
        st.session_state._pending_portfolio_line_results = {
            "rows": edited_rows.copy(),
            "results": results, "state": pricing_state,
        }
        st.experimental_rerun()

    # Kept while the priced inputs are unchanged, so ticking or booking a
    # line does not lose the results; dropped once a priced input changes.
    cached_calculation = st.session_state.get("_portfolio_result_cache")
    if cached_calculation is not None and not same_priced_inputs(
            edited_rows, cached_calculation["rows"]):
        st.session_state.pop("_portfolio_result_cache", None)
        cached_calculation = None
    if cached_calculation is not None:
        for diagnostic in cached_calculation.get("diagnostics", ()):
            diagnostics.append(diagnostic)
        calculation_matches_inputs = True
        results = cached_calculation["results"]
        errors = cached_calculation["errors"]
        hedge_trades = cached_calculation.get("hedge_trades", [])
    else:
        calculation_matches_inputs = False
        results = []
        errors = []
        hedge_trades = []

    if calculation_matches_inputs:
        st.info(
            "Priced {} of {} input lines.".format(
                len(results), len(cached_calculation["rows"])
            )
        )
    for error in errors:
        st.error("Not priced — " + error)

    pricer_greeks = chosen_greeks(PRICER_GREEKS_KEY)
    if results:
        to_compute = missing_extras(pricer_greeks, results)
        if to_compute:
            st.caption("Price portfolio to compute: " + ", ".join(
                greek_catalog.CATALOG[name].label for name in to_compute))
        problems = {name: reason for result in results
                    for name, reason in result.get("greek_problems", {}).items()}
        for name, reason in problems.items():
            st.warning(f"{greek_catalog.CATALOG[name].label} not available: {reason}")
        aggregation = aggregate_pricer_results_by_pair(results, hedge_trades, pricer_greeks)
        aggregation_display = format_pricer_aggregation(aggregation)
        st.subheader("By pair")
        aggregation_column_config = {
            "currency_pair": st.column_config.TextColumn(
                "Currency pair", help="Currency pair aggregation.", width="small"
            ),
            "mtm_base": st.column_config.TextColumn(
                "MTM (base)", help="Aggregate mark-to-market in base currency.",
                width="small",
            ),
            "pnl_base": st.column_config.TextColumn(
                "P&L (base)", help="Aggregate P&L in base currency, including premium.",
                width="small",
            ),
        }
        for greek_name in pricer_greeks:
            for column in (f"fixed_{greek_name}", greek_name):
                aggregation_column_config[column] = st.column_config.TextColumn(
                    greek_catalog.label(column),
                    help=f"Aggregate {greek_catalog.label(column).lower()}.",
                    width="small",
                )
        st.dataframe(
            red_negatives(aggregation_display),
            use_container_width=True,
            hide_index=True,
            column_config=aggregation_column_config,
        )
        if hedge_trades:
            st.subheader("Delta hedge forward trades")
            st.dataframe(
                red_negatives(pd.DataFrame(hedge_trades)),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "base_amount": st.column_config.NumberColumn(
                        "Base amount", format="%.2f"
                    ),
                    "forward_rate": st.column_config.NumberColumn(
                        "Forward rate", format="%.6f"
                    ),
                    "delta_offset": st.column_config.NumberColumn(
                        "Delta offset", format="%.2f"
                    ),
                    "mtm_term": st.column_config.NumberColumn(
                        "MTM (term)", format="%.2f"
                    ),
                    "mtm_base": st.column_config.NumberColumn(
                        "MTM (base)", format="%.2f"
                    ),
                    "pnl_term": st.column_config.NumberColumn(
                        "P&L (term)", format="%.2f"
                    ),
                    "pnl_base": st.column_config.NumberColumn(
                        "P&L (base)", format="%.2f"
                    ),
                    "theta": st.column_config.NumberColumn(
                        "Roll theta (prem ccy)", format="%.2f"
                    ),
                },
            )
        theta_date = next_business_day(currency_pair, valuation_date)
        theta_days = (pd.Timestamp(theta_date) - pd.Timestamp(valuation_date)).days
        st.caption(
            "Line results are shown in the pricer grid. Sticky-delta Greeks keep the "
            "ATM/RR/BF quotes unchanged as spot moves; smile Greeks move the smile by "
            "the betas (equal when every beta is 0). "
            "Premium is negative for buys and positive for sells. MTM and P&L are "
            "shown in base currency. "
            "Line deltas are in the pair's convention for the expiry (Delta conv.: "
            "S spot, F forward); the By pair totals add spot-equivalent delta and "
            "gamma. Hedge forwards are sized by forward delta. "
            "Gamma is spot-equivalent delta sensitivity per 1% spot move. "
            f"Smile theta is a {theta_days}-calendar-day roll to {theta_date}, "
            "re-interpolating volatility and forward curves and re-solving the "
            "fixed strike on the rolled skew. Pair totals include forward-hedge "
            "roll theta when a hedge is enabled."
        )

    render_blotter(blotter_tab)
    render_portfolio(book_tab, market_set, mods)
    render_risk(risk_tab, market_set, mods, spot_ladder_mode)
    if diagnostics.records():
        st.warning("Some calculations used legacy fallback. The selected pricing model was retained.")

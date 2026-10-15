"""Streamlit user interface for the VannaVolga portfolio pricer."""

from datetime import date, timedelta
import pandas as pd
import streamlit as st

from market_data import Modifications
from app import pricer
from app.market_context import (
    PricingSettings, build_context, interpolate_delivery_curve, naive_date,
)
from app.market_service import MarketService
from app.market_store import validate_market
from app.curve_roll import next_business_day
from app.smile_service import DiagnosticsCollector, SurfaceCache, bind_cache
from helper_functions.vanna_volga_helpers import get_model_map
from helper_functions.premium_quotes import PREMIUM_QUOTES, premium_quote_label
from app.reports.spot_ladder import calculate_spot_ladder, render_spot_ladder
from app.ui.market_block import render_market_block
from app.ui.portfolio_block import (
    GREEK_NAMES, render_portfolio_block,
)

# Pricing helpers now live in the backend; kept here under their old names.
format_strike_for_pair = pricer.format_strike_for_pair
_is_complete_strike_alias = pricer.is_complete_strike_alias


def aggregate_pricer_results_by_pair(results, hedge_trades=()):
    """Aggregate base-currency line results and hedge risks by currency pair."""
    columns = [
        "currency_pair",
        "mtm_base",
        "pnl_base",
        *(column for greek in GREEK_NAMES for column in (f"fixed_{greek}", greek)),
    ]
    if not results:
        return pd.DataFrame(columns=columns)
    result_df = pd.DataFrame(results)
    aggregation = (
        result_df[columns]
        .groupby("currency_pair", as_index=False, sort=True)
        .sum(min_count=1)
    )
    for trade in hedge_trades:
        pair = str(trade["currency_pair"]).upper()
        matches = aggregation["currency_pair"] == pair
        if not matches.any():
            aggregation = pd.concat(
                [
                    aggregation,
                    pd.DataFrame(
                        [{column: pair if column == "currency_pair" else 0.0
                          for column in columns}]
                    ),
                ],
                ignore_index=True,
            )
            matches = aggregation["currency_pair"] == pair
        aggregation.loc[matches, "mtm_base"] += float(trade["mtm_base"])
        aggregation.loc[matches, "pnl_base"] += float(trade["pnl_base"])
        aggregation.loc[matches, "fixed_delta"] += float(trade["delta_offset"])
        aggregation.loc[matches, "delta"] += float(trade["delta_offset"])
        aggregation.loc[matches, "theta"] += float(trade.get("theta", 0.0))
    return aggregation


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
            "premium_ccy": str(st.session_state.currency_pair).upper()[3:6],
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
    if "premium_ccy" not in st.session_state.option_rows:
        pair_values = st.session_state.option_rows["currency_pair"].astype(str).str.upper()
        st.session_state.option_rows["premium_ccy"] = pair_values.str[3:6]
    else:
        pair_values = st.session_state.option_rows["currency_pair"].astype(str).str.upper()
        old_premium_ccy = st.session_state.option_rows["premium_ccy"].astype(str).str.upper()
        st.session_state.option_rows["premium_ccy"] = [
            pair[:3] if currency == "BASE" else pair[3:6] if currency == "TERM" else currency
            for pair, currency in zip(pair_values, old_premium_ccy)
        ]
    if st.session_state.get("_portfolio_editor_schema") != 4:
        st.session_state.pop("portfolio_editor_v2", None)
        st.session_state.pop("portfolio_editor_v3", None)
        st.session_state._portfolio_editor_schema = 4

    portfolio_columns = [
        "currency_pair", "strike", "expiry", "call_put", "buy_sell",
        "notional", "premium", "premium_ccy",
    ]
    portfolio_columns += [
        column for column in st.session_state.option_rows.columns
        if column not in portfolio_columns
    ]
    st.session_state.option_rows = st.session_state.option_rows.reindex(
        columns=portfolio_columns
    )

    # The strike editor accepts both numeric values and delta strings such as
    # ``25d``, so keep this column as text even when the initial value is numeric.
    if "strike" in st.session_state.option_rows:
        st.session_state.option_rows["strike"] = (
            st.session_state.option_rows["strike"].map(str)
        )

    with st.sidebar:
        st.header("Valuation")
        valuation_date = st.date_input("Valuation date", key="valuation_date")

        st.header("Conventions")
        pricing_model = st.selectbox("Pricing model", ["vv", "vv-sabr"], key="pricing_model")
        atm_type, delta_base = "dn", True
        delta_reporting = st.selectbox(
            "Delta convention", ["forward", "spot"], index=0, key="delta_reporting",
            format_func=lambda value: "Forward delta" if value == "forward" else "Spot delta",
            help="Applies to reported deltas and hedges only. Premium adjustment "
                 "uses the FX pair convention; smile pillars remain unchanged.",
        )
        broker_strangle = st.checkbox(
            "BF quotes are broker strangles", value=True, key="broker_strangle",
            help="Disable only for already converted two-vol butterflies.",
        )
        premium_quote_options = list(PREMIUM_QUOTES)
        _pair = str(st.session_state.currency_pair).replace("/", "").replace(" ", "").upper()
        _base_ccy, _term_ccy = (_pair[:3], _pair[3:]) if len(_pair) == 6 else ("BASE", "TERM")
        # A value persisted by an older build would make the widget raise, so
        # reset anything unrecognised before the selectbox is constructed.
        if st.session_state.get("premium_quote") not in premium_quote_options:
            st.session_state["premium_quote"] = premium_quote_options[0]
        premium_quote = st.selectbox(
            "Premium quote",
            premium_quote_options,
            key="premium_quote",
            format_func=lambda q: premium_quote_label(q, _base_ccy, _term_ccy),
            help=(
                "Units used to report premium, MTM and P&L. Presentation only: "
                "it does not change the trade value, the delta convention, or "
                "the currency the premium settles in."
            ),
        )
        if premium_quote not in PREMIUM_QUOTES:
            premium_quote = "term_cash"
        st.header("Smile dynamics")
        beta_atm = st.slider("beta_atm", 0.0, 5.0, 0.7, 0.05, key="beta_atm")
        beta_rr = st.slider("beta_rr", 0.0, 2.0, 1.2, 0.05, key="beta_rr")
        beta_bf = st.slider("beta_bf", 0.0, 2.0, 0.0, 0.05, key="beta_bf")
        smooth_pillar_interpolation = True
        spot_ladder_mode = st.selectbox(
            "Spot ladder calculation mode",
            ["shadow", "optimized", "reference"],
            index=1,
            help="Shadow compares optimized and reference results before returning.",
            key="spot_ladder_mode",
        )

    surface_progress = st.empty()
    markets_tab, portfolio_tab, ladder_tab = st.tabs(
        ["Markets", "Pricer", "Spot ladder"]
    )

    spot, forward_divisor, currency_pair, vol_rows, forward_rows, event_rows = render_market_block(
        markets_tab, valuation_date, apply_loaded_market
    )
    diagnostics = DiagnosticsCollector()
    # Calibrated surfaces live for the whole session, so reruns and pricing
    # reuse them instead of recalibrating per option or per strike.
    if "_surface_cache_v1" not in st.session_state:
        st.session_state["_surface_cache_v1"] = SurfaceCache()
        st.session_state["_market_service_v1"] = MarketService(diagnostics)
    bind_cache(diagnostics.token, st.session_state["_surface_cache_v1"])
    market_service = st.session_state["_market_service_v1"]
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
    market_for_expiry = market_context.market_for_expiry
    resolve_market_expiry = market_context.resolve_market_expiry


    def prepare_market_surface():
        """Calibrate every quoted pillar once per market state, with progress."""
        rows = st.session_state.vol_rows
        fingerprint = repr((
            pricing_model, broker_strangle, str(valuation_date), spot, currency_pair,
            forward_divisor, rows.to_json(date_format="iso"),
            st.session_state.forward_rows.to_json(date_format="iso"),
            st.session_state.event_rows.to_json(date_format="iso"),
        ))
        if st.session_state.get("_surface_fingerprint") == fingerprint:
            return
        expiries = pd.to_datetime(rows["expiry"], errors="coerce", utc=True).dt.tz_localize(None)
        expiries = sorted({naive_date(e) for e in expiries.dropna()
                           if naive_date(e) > naive_date(valuation_date)})
        if not expiries:
            return
        total = 2 * len(expiries)
        bar = surface_progress.progress(0.0, text="Preparing volatility surface...")

        def report(index, count, label):
            bar.progress(index / total,
                         text=f"Converting quotes and calibrating {label} ({index + 1}/{count})")

        market_service.progress = report
        try:
            for index, expiry in enumerate(expiries):
                market = market_for_expiry(expiry)
                market_service.progress = None
                bar.progress((len(expiries) + index) / total,
                             text=f"Building {pricing_model} smile for {expiry:%d-%b-%Y} "
                                  f"({index + 1}/{len(expiries)})")
                forward = market.s + market.fwd_pts / market.forward_divisor
                get_model_map(market)[pricing_model](
                    market=market, K=forward, expiry_date=expiry.date().isoformat(),
                    valuation_date=naive_date(valuation_date).date().isoformat(),
                    CallPut="CALL", atm_conv=market.atm_conv, deltaBase=market.deltaBase,
                ).get_vol_for_strike()
        except Exception:
            # Incomplete or invalid market inputs: pricing reports the error.
            surface_progress.empty()
            return
        finally:
            market_service.progress = None
        st.session_state["_surface_fingerprint"] = fingerprint
        surface_progress.empty()

    prepare_market_surface()


    edited_rows, ladder_steps, ladder_increment, ladder_calculate, delta_hedge = render_portfolio_block(
        portfolio_tab, ladder_tab,
        market_pair=str(st.session_state.currency_pair).strip().upper(),
        quote_label=premium_quote_label(premium_quote, _base_ccy, _term_ccy),
    )
    if st.session_state.pop("_portfolio_strike_resolution_requested", False):
        resolved_rows, strikes_changed, strike_failures = (
            pricer.resolve_strike_aliases(market_context, edited_rows)
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
        for message in st.session_state.get("_portfolio_strike_failures", []):
            st.warning(message)


    with portfolio_tab:
        calculate = st.button(
            "Price portfolio", type="primary", key="calculate_portfolio"
        )


    if calculate:
        mods = Modifications(beta_atm=beta_atm, beta_rr=beta_rr, beta_bf=beta_bf)
        # Pricing reads no session state, but stays on the script thread so
        # progress and diagnostics belong to this run.
        results, errors = pricer.price_lines(market_context, edited_rows, mods, premium_quote)
        hedge_trades = []
        if delta_hedge:
            hedge_trades, hedge_errors = pricer.hedge_trades_for(market_context, results)
            errors = errors + hedge_errors

        st.session_state._portfolio_result_cache = {
            "rows": edited_rows.copy(), "results": results, "errors": errors,
            "hedge_trades": hedge_trades,
            "diagnostics": diagnostics.records(),
        }
        # Dynamic data-editor rows live in widget state until explicitly
        # committed. Preserve the complete frame before this rerun so pricing
        # a portfolio cannot drop rows added after the initial dataframe.
        st.session_state._pending_priced_portfolio_rows = edited_rows.copy()
        st.session_state._pending_portfolio_line_results = {
            "rows": edited_rows.copy(),
            "results": results,
        }
        st.experimental_rerun()

    cached_calculation = st.session_state.pop("_portfolio_result_cache", None)
    if cached_calculation is not None and edited_rows.equals(cached_calculation["rows"]):
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

    if results:
        aggregation = aggregate_pricer_results_by_pair(results, hedge_trades)
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
        for greek_name in GREEK_NAMES:
            title = greek_name.title()
            aggregation_column_config[f"fixed_{greek_name}"] = (
                st.column_config.TextColumn(
                    f"Fixed {title}", help=f"Aggregate fixed-volatility {greek_name}.",
                    width="small",
                )
            )
            aggregation_column_config[greek_name] = st.column_config.TextColumn(
                f"Smile {title}", help=f"Aggregate smile-adjusted {greek_name}.",
                width="small",
            )
        st.dataframe(
            aggregation_display,
            use_container_width=True,
            hide_index=True,
            column_config=aggregation_column_config,
        )
        if hedge_trades:
            st.subheader("Delta hedge forward trades")
            st.dataframe(
                pd.DataFrame(hedge_trades),
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
            "Line results are shown in the pricer grid. f-prefix Greek columns are "
            "fixed-volatility risks; unprefixed columns are smile-adjusted risks. "
            "Premium is negative for buys and positive for sells. MTM and P&L are "
            "shown in base currency. "
            "Gamma is quote-delta sensitivity per 1% spot move. "
            f"Smile theta is a {theta_days}-calendar-day roll to {theta_date}, "
            "re-interpolating volatility and forward curves and re-solving the "
            "fixed strike on the rolled skew. Pair totals include forward-hedge "
            "roll theta when a hedge is enabled."
        )

    if ladder_calculate:
        with st.spinner("Calculating spot ladder..."):
            mods = Modifications(beta_atm=beta_atm, beta_rr=beta_rr, beta_bf=beta_bf)
            ladder_spots = [
                spot * (1.0 + step * float(ladder_increment) / 100.0)
                for step in range(-int(ladder_steps), int(ladder_steps) + 1)
            ]
            forward_positions = []
            hedge_errors = []
            if delta_hedge:
                forward_positions, hedge_errors = pricer.build_delta_hedges(
                    market_context, edited_rows, mods
                )
            if hedge_errors:
                ladder_results, ladder_errors = [], hedge_errors
            else:
                ladder_results, ladder_errors = calculate_spot_ladder(
                    list(edited_rows.iterrows()),
                    ladder_spots,
                    valuation_date,
                    market_for_expiry,
                    resolve_market_expiry,
                    pricing_model,
                    atm_type,
                    delta_base,
                    mods,
                    mode=spot_ladder_mode,
                    forward_positions=forward_positions,
                )
        render_spot_ladder(ladder_tab, ladder_results, ladder_errors, spot)
    if diagnostics.records():
        st.warning("Some calculations used legacy fallback. The selected pricing model was retained.")

"""Streamlit user interface for the VannaVolga portfolio pricer."""

import json
import re
from datetime import date, timedelta
import numpy as np
import pandas as pd
import streamlit as st

from market_data import MarketData, Modifications
from app.market_service import MarketService, pair_delivery
from app.market_store import validate_market
from app.curve_roll import (
    anchor_forward_rows, next_business_day, roll_forward_rows, roll_vol_rows,
)
from app.smile_service import (
    DiagnosticsCollector, SurfaceCache, bind_cache, reuse_prepared, market_diagnostics,
)
from helper_functions.vanna_volga_helpers import get_model_map
from pricing import Pricable
from pricing.fx_forward import FxForwardPosition
from pricing.greeks import calculate_greeks, calculate_forward_theta, forward_delta_factor
from helper_functions.input_resolvers import parse_notional, resolve_expiry, resolve_strike
from helper_functions.premium_quotes import (
    PREMIUM_QUOTES, format_premium_quote, premium_quote_label,
)
from app.reports.spot_ladder import calculate_spot_ladder, render_spot_ladder
from app.ui.market_block import render_market_block
from app.ui.portfolio_block import (
    GREEK_NAMES, render_portfolio_block,
)


def _is_complete_strike_alias(row):
    """Report whether a row carries a fully specified strike alias.

    Half-typed cells must stay silent while editing, but a complete alias that
    fails to resolve is a genuine error worth showing.
    """
    value = str(row.get("strike", "")).strip().lower().replace(" ", "")
    if value not in {"atm", "atmf", "atms"} and not re.fullmatch(
        r"(\d+(?:\.\d+)?)d", value
    ):
        return False
    expiry = row.get("expiry")
    call_put = str(row.get("call_put", "")).strip().upper()
    return (
        not pd.isna(expiry)
        and bool(str(expiry).strip())
        and call_put in {"CALL", "PUT"}
    )


def format_strike_for_pair(currency_pair, strike):
    """Format FX strikes using JPY and non-JPY market conventions."""
    pair = str(currency_pair).strip().upper()
    decimals = 2 if "JPY" in pair else 4
    return f"{float(strike):.{decimals}f}"


def interpolate_delivery_curve(rows, date_column, value_column, target_date, name):
    """Linearly interpolate a curve only within its supplied delivery range."""
    dates = pd.to_datetime(rows[date_column], errors="coerce")
    values = pd.to_numeric(rows[value_column], errors="coerce")
    valid = dates.notna() & values.notna() & np.isfinite(values)
    curve = rows.loc[valid].copy()
    if curve.empty:
        raise ValueError(f"{name} rows must have valid delivery dates and finite values")
    curve["_date"] = dates.loc[valid].map(pd.Timestamp.toordinal)
    curve["_value"] = values.loc[valid]
    curve.sort_values("_date", inplace=True)
    if curve["_date"].duplicated().any():
        raise ValueError("Forward delivery dates must be unique")
    target = pd.Timestamp(target_date).toordinal()
    first, last = curve["_date"].iloc[0], curve["_date"].iloc[-1]
    if target < first or target > last:
        raise ValueError(f"{name} delivery date is outside the quoted forward range")
    return float(np.interp(target, curve["_date"], curve["_value"]))


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


    def format_notional(value):
        """Format a numeric notional with comma separators for display."""
        amount = parse_notional(value)
        if not np.isfinite(amount):
            raise ValueError("Notional must be finite")
        return f"{amount:,.0f}"


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
    def tenor_days(tenor):
        units = {"W": 7, "M": 30, "Y": 365}
        text = str(tenor).strip().upper()
        if len(text) < 2 or text[-1] not in units:
            raise ValueError(f"Invalid tenor: {tenor}")
        return float(text[:-1]) * units[text[-1]]


    def naive_date(value):
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_localize(None)
        return timestamp.normalize()


    def variance_days_between(start, end):
        start = naive_date(start)
        end = naive_date(end)
        if end < start:
            raise ValueError("End date must not precede start date")
        total = float((end - start).days)
        events = st.session_state.event_rows
        if not events.empty:
            event_dates = pd.to_datetime(
                events["event_date"], errors="coerce", utc=True
            ).dt.tz_localize(None)
            weights = pd.to_numeric(events["variance_days"], errors="coerce").fillna(0.0)
            if not np.isfinite(weights.to_numpy(dtype=float)).all():
                raise ValueError("Event variance days must be finite")
            total += float(weights.loc[(event_dates > start) & (event_dates <= end)].sum())
        return float(total)


    def forward_curve(curve_date=None, rows=None):
        """Forward rows for ``curve_date``, anchored at its spot date."""
        rows = st.session_state.forward_rows if rows is None else rows
        return anchor_forward_rows(
            rows, currency_pair, valuation_date if curve_date is None else curve_date
        )


    def forward_points_for_expiry(expiry, curve_rows=None):
        return interpolate_delivery_curve(
            forward_curve() if curve_rows is None else curve_rows,
            "delivery_date",
            "fwd_pts",
            pair_delivery(currency_pair, expiry, diagnostics),
            "Forward points",
        )


    def term_discount_for_expiry(expiry, curve_rows=None):
        """Interpolate decimal SIMPLE term deposit rates by delivery date."""
        return interpolate_delivery_curve(
            forward_curve() if curve_rows is None else curve_rows,
            "delivery_date",
            "term_discount",
            pair_delivery(currency_pair, expiry, diagnostics),
            "Term discount rate",
        )


    def market_for_expiry(
        expiry, as_of=None, allow_short_end=False, spot_value=None
    ):
        """Build a MarketData object from the editable vol/forward/event tables.

        With ``as_of`` after the valuation date the curves are rolled to it
        (sticky tenor): pillar expiries and forward deliveries move with their
        tenors while every tenor keeps its quotes, so a fixed expiry picks up
        the vol and forward-point roll-down. Pillar total variance is fixed by
        the quotes; events only distribute it between pillar dates, so a
        passed event's share moves to the remaining days. Expiries before the
        first pillar take the first pillar's total variance, distributed by
        the variance-day weights.
        """
        curve_date = naive_date(valuation_date if as_of is None else as_of)
        requested_date = naive_date(expiry)
        if requested_date == curve_date and allow_short_end and as_of is not None:
            return market_for_expiry(expiry, spot_value=spot_value)
        rows = st.session_state.vol_rows.copy()
        if rows.empty:
            raise ValueError("Enter at least one volatility expiry")
        rows["expiry"] = pd.to_datetime(
            rows["expiry"], errors="coerce", utc=True
        ).dt.tz_localize(None)
        rows = rows.dropna(subset=["expiry"]).sort_values("expiry").reset_index(drop=True)
        if rows.empty:
            raise ValueError("Volatility expiries must be valid dates")
        forward_rows = st.session_state.forward_rows
        roll_days = (curve_date - naive_date(valuation_date)).days
        if curve_date != naive_date(valuation_date):
            forward_rows = forward_curve(valuation_date, forward_rows)
            rows = roll_vol_rows(rows, currency_pair, valuation_date, curve_date)
            forward_rows = roll_forward_rows(
                forward_rows, rows, currency_pair, valuation_date, curve_date,
                lambda pillar_expiry: pair_delivery(currency_pair, pillar_expiry, diagnostics),
            )
        curve_rows = forward_rows if roll_days else forward_curve(curve_date, forward_rows)
        curve_forward_points = lambda value: forward_points_for_expiry(value, curve_rows)
        curve_term_discount = lambda value: term_discount_for_expiry(value, curve_rows)
        def pillar_delivery(value):
            # Synthetic risk nodes retain their original settlement offsets.
            original_expiry = naive_date(value) - pd.Timedelta(days=roll_days)
            return pair_delivery(currency_pair, original_expiry, diagnostics) + pd.Timedelta(days=roll_days)

        def pillar_forward_points(value):
            return interpolate_delivery_curve(
                curve_rows, "delivery_date", "fwd_pts", pillar_delivery(value),
                "Forward points",
            )

        def pillar_term_discount(value):
            return interpolate_delivery_curve(
                curve_rows, "delivery_date", "term_discount", pillar_delivery(value),
                "Term discount rate",
            )
        if requested_date < curve_date or (requested_date == curve_date and not allow_short_end):
            raise ValueError("Expiry must be after the valuation date")
        target_date = requested_date
        # Before the first pillar the variance interpolation below runs from
        # zero at the curve date, so the short end carries the first pillar's
        # variance scaled by its own variance-day weights.
        if target_date < rows["expiry"].iloc[0] and not allow_short_end:
            raise ValueError("Expiry is before the first volatility pillar")
        if target_date > rows["expiry"].iloc[-1]:
            raise ValueError("Expiry is outside the quoted volatility range")
        for column in ("atm", "rr_25", "bf_25", "rr_10", "bf_10"):
            rows[column] = pd.to_numeric(rows[column], errors="coerce")
        rows, curve_reasons, curve_prepared = market_service.convert_curve(
            rows, curve_date.date(), spot,
            currency_pair, forward_divisor, pricing_model, broker_strangle,
            pillar_forward_points, pillar_term_discount,
        )

        quotes = pd.DataFrame({
            "PUT_10": rows["atm"] + rows["bf_10"] - rows["rr_10"] / 2,
            "PUT_25": rows["atm"] + rows["bf_25"] - rows["rr_25"] / 2,
            "ATM": rows["atm"],
            "CALL_25": rows["atm"] + rows["bf_25"] + rows["rr_25"] / 2,
            "CALL_10": rows["atm"] + rows["bf_10"] + rows["rr_10"] / 2,
        })
        values = quotes.to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("Vol quotes must reconstruct finite positive wing vols")
        pillar_vdays = np.array([
            variance_days_between(curve_date, pillar_date)
            for pillar_date in rows["expiry"]
        ], dtype=float)
        if (pillar_vdays <= 0).any() or np.any(np.diff(pillar_vdays) <= 0):
            raise ValueError("Volatility pillars must have increasing positive variance days")
        pillar_calendar_days = np.array([
            (pillar_date - curve_date).days for pillar_date in rows["expiry"]
        ], dtype=float)
        if (pillar_calendar_days <= 0).any():
            raise ValueError("Volatility pillars must have positive calendar days")
        target_vdays = variance_days_between(curve_date, target_date)
        calendar_days = (target_date - curve_date).days
        if target_vdays <= 0 or calendar_days <= 0:
            raise ValueError("Expiry must have positive calendar and variance days")
        x = np.concatenate(([0.0], pillar_vdays))
        interpolated = {}
        for name in quotes:
            # Quotes are annualized over calendar time, so total variance must be
            # accumulated in calendar-annualized units. Only the interpolation
            # clock is the event-weighted variance-day measure; mixing the two
            # measures here would rescale every quote by sqrt(vdays/caldays).
            total_variance = quotes[name].to_numpy(dtype=float) ** 2 * pillar_calendar_days
            y = np.concatenate(([0.0], total_variance))
            interpolated[name] = float(np.sqrt(np.interp(target_vdays, x, y) / calendar_days))
        atm = interpolated["ATM"]
        rr25 = interpolated["CALL_25"] - interpolated["PUT_25"]
        bf25 = (interpolated["CALL_25"] + interpolated["PUT_25"]) / 2.0 - atm
        rr10 = interpolated["CALL_10"] - interpolated["PUT_10"]
        bf10 = (interpolated["CALL_10"] + interpolated["PUT_10"]) / 2.0 - atm
        market_args = dict(
            s=spot if spot_value is None else spot_value,
            fwd_pts=curve_forward_points(requested_date), v_atm=atm,
            rr_25=rr25, rr_10=rr10, bf_25=bf25, bf_10=bf10,
            forward_divisor=forward_divisor, atm_conv=atm_type, deltaBase=delta_base,
        )
        term_discount = curve_term_discount(requested_date)
        try:
            market = MarketData(**market_args, term_discount=term_discount)
        except TypeError as exc:
            # Streamlit can retain this imported class until its process
            # restarts, even after the app module has reloaded.
            if "term_discount" not in str(exc) or "unexpected keyword" not in str(exc):
                raise
            market = MarketData(**market_args)
            market.term_discount = term_discount
        market.smooth_pillar_interpolation = smooth_pillar_interpolation
        market.delta_reporting = getattr(st.session_state, "delta_reporting", None)
        # Lets the pricers discount to the option's delivery date on the pair's
        # own day-count basis. Assigned after construction so a stale imported
        # MarketData class retained across a Streamlit reload cannot raise; an
        # unrecognised pair simply falls back to continuous discounting.
        try:
            _pair_code = currency_pair
            market.currency_pair = _pair_code if re.fullmatch(r"[A-Z]{6}", _pair_code) else None
        except Exception:
            market.currency_pair = None
        market.legacy_target_blend = not smooth_pillar_interpolation
        market.compatibility_mode = "auto" if smooth_pillar_interpolation else "legacy"
        market.wrapped_smile = True
        market.broker_strangle = False
        market.force_legacy_smile = bool(curve_reasons)
        market.smile_reasons = curve_reasons
        market.smile_model = pricing_model
        market.diagnostics_token = diagnostics.token
        if not curve_reasons and requested_date > curve_date:
            from resolved_smile_inputs import resolve_conventions, MarketSnapshot
            context = resolve_conventions(MarketSnapshot.capture(market),
                                          requested_date.date(), curve_date.date())
            market.deltaBase, market.atm_conv = context.premium_adjusted, context.atm_convention
            for pillar_date, prepared in zip(rows.expiry, curve_prepared):
                if (naive_date(pillar_date) == requested_date and market.s == spot and
                        curve_date == naive_date(valuation_date)):
                    reuse_prepared(market, prepared)
                    break
        return market


    def resolve_market_expiry(value):
        """Resolve pillar tenor strings to their configured expiry dates."""
        token = str(value).strip().upper()
        rows = st.session_state.vol_rows
        matches = rows[rows["tenor"].astype(str).str.strip().str.upper() == token]
        if len(matches) > 1:
            raise ValueError(f"Tenor {token} appears more than once in Vol data")
        if len(matches) == 1:
            return pd.Timestamp(matches.iloc[0]["expiry"]).date()
        return resolve_expiry(value, valuation_date, rows["tenor"])


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


    def auto_resolve_portfolio_strikes(rows):
        """Convert resolvable delta and ATM aliases to numeric input strikes."""
        resolved_rows = rows.copy()
        changed = False
        failures = []
        for index, source_row in rows.iterrows():
            value = source_row.get("strike")
            if pd.isna(value) or not str(value).strip():
                continue
            try:
                numeric_strike = float(str(value).strip().replace(",", ""))
                if np.isfinite(numeric_strike) and numeric_strike > 0:
                    continue
            except (TypeError, ValueError):
                pass
            try:
                row = source_row.copy()
                row["valuation"] = valuation_date
                row["expiry"] = resolve_market_expiry(row["expiry"])
                market = market_for_expiry(row["expiry"])
                strike = resolve_strike(market, row, atm_type, delta_base)
            except Exception as exc:
                # A well-formed alias that still fails is a real problem, not a
                # half-typed cell. Staying silent makes it indistinguishable
                # from "nothing happened" and hides the cause.
                if _is_complete_strike_alias(source_row):
                    failures.append("Line {}: {}".format(index + 1, exc))
                continue
            resolved_rows.at[index, "strike"] = format_strike_for_pair(
                source_row.get("currency_pair", ""), strike
            )
            changed = True
        return resolved_rows, changed, failures


    def build_delta_hedges(rows, mods):
        """Create fixed forward hedge positions from current option deltas."""
        grouped = {}
        errors = []
        for line_number, (_, source_row) in enumerate(rows.iterrows(), start=1):
            try:
                row = source_row.copy()
                row["valuation"] = valuation_date
                pair = str(row.get("currency_pair", "")).strip().upper()
                if not re.fullmatch(r"[A-Z]{6}", pair):
                    raise ValueError("Currency pair must contain two 3-letter ISO codes")
                if pair != str(st.session_state.currency_pair).strip().upper():
                    raise ValueError("Market data does not match this currency pair")
                row["currency_pair"] = pair
                expiry = resolve_market_expiry(row["expiry"])
                row["expiry"] = expiry
                row["notional"] = parse_notional(row["notional"])
                market = market_for_expiry(expiry)
                row["strike"] = resolve_strike(market, row, atm_type, delta_base)
                greeks = calculate_greeks(
                    market, row, None, pricing_model, atm_type, delta_base, mods
                )
                smile_delta = float(greeks["delta"])
                # Convert selected delta units back to forward hedge notional.
                forward_rate = float(market.s) + float(market.fwd_pts) / float(market.forward_divisor)
                if not np.isfinite(smile_delta):
                    raise ValueError("Smile-adjusted delta is invalid")
                key = (pair, expiry, forward_rate)
                grouped[key] = grouped.get(key, 0.0) - smile_delta / forward_delta_factor(
                    market, expiry, valuation_date
                )
            except Exception as exc:
                errors.append("Line {} hedge: {}".format(line_number, exc))

        positions = []
        for (pair, expiry, rate), base_amount in grouped.items():
            if not np.isfinite(base_amount):
                errors.append("{} {} hedge amount is not finite".format(pair, expiry))
                continue
            if abs(base_amount) < 1e-12:
                continue
            positions.append({
                "currency_pair": pair,
                "expiry": expiry,
                "base_amount": float(base_amount),
                "contract_rate": float(rate),
            })
        return positions, errors


    edited_rows, ladder_steps, ladder_increment, ladder_calculate, delta_hedge = render_portfolio_block(
        portfolio_tab, ladder_tab,
        market_pair=str(st.session_state.currency_pair).strip().upper(),
        quote_label=premium_quote_label(premium_quote, _base_ccy, _term_ccy),
    )
    if st.session_state.pop("_portfolio_strike_resolution_requested", False):
        resolved_rows, strikes_changed, strike_failures = (
            auto_resolve_portfolio_strikes(edited_rows)
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
        def calculate_row(item):
            line_number, source_row = item
            resolved_strike = None
            try:
                row = source_row.copy()
                row["valuation"] = valuation_date
                currency_pair = str(row.get("currency_pair", "")).strip().upper()
                if not re.fullmatch(r"[A-Z]{6}", currency_pair):
                    raise ValueError("Currency pair must contain two 3-letter ISO codes")
                if currency_pair != str(st.session_state.currency_pair).strip().upper():
                    raise ValueError(
                        f"Market data is configured for {st.session_state.currency_pair}, "
                        f"not {currency_pair}"
                    )
                row["currency_pair"] = currency_pair
                row["expiry"] = resolve_market_expiry(row["expiry"])
                row["notional"] = parse_notional(row["notional"])
                premium = pd.to_numeric(row.get("premium", 0.0), errors="coerce")
                if pd.isna(premium) or not np.isfinite(float(premium)) or premium < 0:
                    raise ValueError("Premium must be a finite non-negative amount")
                premium_ccy = str(row.get("premium_ccy", currency_pair[3:])).strip().upper()
                if premium_ccy not in {currency_pair[:3], currency_pair[3:]}:
                    raise ValueError(
                        f"Premium currency must be {currency_pair[:3]} or {currency_pair[3:]}"
                    )
                buy_sell = str(row["buy_sell"]).strip().upper()
                if buy_sell not in {"BUY", "SELL"}:
                    raise ValueError("Buy/Sell must be BUY or SELL")
                market = market_for_expiry(row["expiry"])
                resolved_strike = resolve_strike(market, row, atm_type, delta_base)
                # Pass a numeric strike into every downstream calculator.
                row["strike"] = resolved_strike
                next_valuation = next_business_day(currency_pair, valuation_date)
                next_market = market_for_expiry(
                    row["expiry"], as_of=next_valuation, allow_short_end=True
                )
                pricing_result = Pricable(
                    row=row,
                    premium=float(premium),
                    premium_ccy=premium_ccy,
                ).price(
                    market, pricing_model, atm_type, delta_base, mods,
                    next_market=next_market,
                    next_valuation=next_valuation,
                )
                forward_rate = market.s + market.fwd_pts / market.forward_divisor
                # Map the chosen quote onto stable column names so the display
                # layer never has to know which convention is active.  Missing
                # conversions degrade to the term-cash figures.
                quoted_mtm = pricing_result.get(
                    "mtm_" + premium_quote, pricing_result.get("mtm_term")
                )
                quoted_pnl = pricing_result.get(
                    "pnl_" + premium_quote, pricing_result.get("pnl_term")
                )
                return line_number, {
                    "row": line_number,
                    **pricing_result,
                    "strike": resolved_strike,
                    "call_put": row["call_put"],
                    "buy_sell": row["buy_sell"],
                    "notional": format_notional(row["notional"]),
                    "mtm_quote": format_premium_quote(quoted_mtm, premium_quote),
                    "pnl_quote": format_premium_quote(quoted_pnl, premium_quote),
                    "_forward_rate": forward_rate,
                    "smile_status": market_diagnostics(market, row["expiry"])[0],
                    "smile_reason": market_diagnostics(market, row["expiry"])[1],
                }, None, resolved_strike
            except Exception as exc:
                return line_number, None, "Line {}: {}".format(line_number, exc), resolved_strike

        # Use visible row positions rather than the DataFrame index. Dynamic
        # editor rows can have non-consecutive indices after lines are removed.
        rows = [
            (line_number, row)
            for line_number, (_, row) in enumerate(edited_rows.iterrows(), start=1)
        ]
        # Market and pricing helpers read Streamlit session state. Keep them on
        # the script thread; worker threads do not have the active session context.
        calculated = [calculate_row(item) for item in rows]
        calculated.sort(key=lambda item: item[0])
        results = [item[1] for item in calculated if item[1] is not None]
        errors = [item[2] for item in calculated if item[2] is not None]

        hedge_trades = []
        if delta_hedge:
            grouped_deltas = {}
            for result in results:
                key = (
                    result["currency_pair"],
                    result["expiry"],
                    float(result["_forward_rate"]),
                )
                grouped_deltas[key] = grouped_deltas.get(key, 0.0) + float(result["delta"])
            for (pair, expiry, forward_rate), net_smile_delta in grouped_deltas.items():
                if not np.isfinite(net_smile_delta):
                    raise ValueError("Smile-adjusted delta is invalid")
                market = market_for_expiry(expiry)
                hedge_delta_factor = forward_delta_factor(market, expiry, valuation_date)
                base_amount = -net_smile_delta / hedge_delta_factor
                if abs(base_amount) < 1e-12:
                    continue
                position = FxForwardPosition(
                    currency_pair=pair,
                    expiry=expiry,
                    base_amount=base_amount,
                    contract_rate=forward_rate,
                )
                value = position.value(market, valuation_date)
                next_valuation = next_business_day(pair, valuation_date)
                next_market = market_for_expiry(
                    expiry, as_of=next_valuation, allow_short_end=True
                )
                hedge_theta = calculate_forward_theta(
                    position, market, next_market, valuation_date, next_valuation
                )
                hedge_trades.append({
                    "currency_pair": pair,
                    "expiry": expiry,
                    "direction": "BUY BASE" if base_amount > 0 else "SELL BASE",
                    "base_amount": base_amount,
                    "forward_rate": forward_rate,
                    "delta_offset": base_amount * hedge_delta_factor,
                    "mtm_term": value["mtm_term"],
                    "mtm_base": value["mtm_base"],
                    "pnl_term": value["mtm_term"],
                    "pnl_base": value["mtm_base"],
                    "theta": hedge_theta,
                })

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
                forward_positions, hedge_errors = build_delta_hedges(edited_rows, mods)
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

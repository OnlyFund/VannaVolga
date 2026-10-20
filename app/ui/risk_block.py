"""Risk tab: spot ladder, strike grid and Greek buckets for one currency pair.

Reports cover the trades selected in the Portfolio tab, one pair at a time.
Hedge forwards are included only when "Include hedges" is ticked; in the
strike grid they sit in the last call bucket at their own expiry.
"""
import pandas as pd
import streamlit as st

from app.option_types import DIGITAL, PRICED, VANILLA
from app.reports import risk_grid
from app.reports.spot_ladder import calculate_spot_ladder, render_spot_ladder
from app.trades import service
from app.ui.progress_bar import ProgressBar
from app.ui.greek_picker import render_greek_picker
from app.ui.negatives import red_negatives
from app.ui.trades_block import _store_banner, trade_store
from pricing import greek_catalog
from pricing.greek_catalog import CATALOG
from pricing.digital import leg_rows

STEPS = (5, 10, 25)


def _formatted(frame, decimals=0):
    def cell(value):
        if pd.isna(value):
            return ""
        number = float(value)
        if abs(number) < 0.5 * 10 ** -decimals:
            number = 0.0
        return f"{number:,.{decimals}f}"
    return frame.applymap(cell)


def _selection(store, valuation_date):
    """The Portfolio selection in the book on ``valuation_date``, and the
    note naming trades booked after it (``service.book_on``)."""
    ids = st.session_state.get("_portfolio_selection")
    trades, later = service.book_on(store.trades(), valuation_date)
    if ids is not None:
        wanted = set(ids)
        trades = [trade for trade in trades if trade.trade_id in wanted]
        later = [trade for trade in later if trade.trade_id in wanted]
    return trades, service.not_booked_note(later, valuation_date)


def risk_options(trades, pair, include_exotics=False):
    """``(options, left_out)``: the pair's options in the risk reports, and
    its exotic options left out of them.

    Vanillas are always in. Exotics only when ``include_exotics``, and then
    only the types the app prices (``app.option_types.PRICED``): a type not
    priced yet is left out, never reported as a vanilla.
    """
    pair_options = [t for t in trades if t.currency_pair == pair and t.is_option]
    included = [t for t in pair_options
                if t.is_vanilla or (include_exotics and t.option_type in PRICED)]
    ids = {id(t) for t in included}
    return included, [t for t in pair_options if id(t) not in ids]


def ladder_rows(options):
    """Spot ladder rows for ``options``: a vanilla is its own row, a digital
    its replicating vanilla legs (``pricing.digital.leg_rows``), which the
    ladder sums like any other rows."""
    rows = []
    for trade in options:
        row = trade.to_pricer_row()
        if (trade.option_type or VANILLA) == DIGITAL:
            rows.extend(leg_rows(row, trade.payout_ccy == trade.currency_pair[:3]))
        else:
            rows.append(row)
    return rows


def render_risk(tab, market_set, mods, ladder_mode):
    store, problem = trade_store()
    with tab:
        st.subheader("Risk")
        if not _store_banner(store, problem):
            return
        trades, note = _selection(store, market_set.valuation_date)
        if note:
            st.info(note)
        if not trades:
            st.info("No live trades selected: choose trades in the Portfolio tab.")
            return
        pairs = sorted({trade.currency_pair for trade in trades})
        pair_col, hedge_col, exotic_col, fixed_col = st.columns(4)
        with pair_col:
            if st.session_state.get("risk_pair") not in pairs:
                st.session_state["risk_pair"] = pairs[0]
            pair = st.selectbox("Currency pair", pairs, key="risk_pair")
        with hedge_col:
            include_hedges = st.checkbox(
                "Include hedges", key="risk_hedges",
                help="Adds the selected hedge forwards: to the ladder and the Greek "
                     "buckets, and to the last call bucket of the strike grid.")
        with exotic_col:
            include_exotics = st.checkbox(
                "Include exotics", key="risk_exotics",
                help="Adds the selected exotic options the app prices (digitals) to the "
                     "spot ladder, strike grid and Greek buckets, through their "
                     "replicating vanilla legs. Types not priced yet stay out.")
        with fixed_col:
            fixed = st.checkbox("Sticky-delta Greeks", key="risk_fixed",
                                help="Use sticky-delta Greeks (smile quotes unchanged as "
                                     "spot moves) instead of smile Greeks (smile moved by "
                                     "the betas).")
        options, left_out = risk_options(trades, pair, include_exotics)
        exotics_in = sum(not t.is_vanilla for t in options)
        hedges = [t for t in trades if t.currency_pair == pair and t.is_forward]
        not_priced = [t for t in left_out if t.option_type not in PRICED]
        st.caption(f"{pair}: {len(options)} options"
                   + (f" ({exotics_in} exotic)" if exotics_in else "")
                   + (f" and {len(hedges)} hedge forwards" if include_hedges else
                      f" ({len(hedges)} hedge forwards not included)")
                   + " from the Portfolio selection."
                   + (f" {len(left_out) - len(not_priced)} exotic options left out: tick "
                      "Include exotics." if len(left_out) > len(not_priced) else "")
                   + (f" {len(not_priced)} options of types not priced yet are left out."
                      if not_priced else "")
                   + (" Exotic payouts are not notionals: the strike grid's Notional "
                      "leaves them out." if exotics_in else ""))
        if pair not in market_set.available():
            st.error(f"No usable market for {pair}: load it in the Markets tab.")
            return
        context = market_set.market(pair)
        ladder_tab, grid_tab, bucket_tab = st.tabs(["Spot ladder", "Strike grid", "Greek buckets"])
        _spot_ladder(ladder_tab, market_set, context, options, hedges if include_hedges else [],
                     mods, ladder_mode)
        # Drawn before pricing so their choices decide which extras are priced.
        with bucket_tab:
            bucket_greeks = render_greek_picker("risk_bucket_greeks", risk_grid.BUCKET_GREEKS)
        with grid_tab:
            measure_col, step_col = st.columns(2)
            with measure_col:
                measure = st.selectbox("Show", list(risk_grid.MEASURES), key="risk_measure",
                                       format_func=risk_grid.MEASURES.get, index=1)
            with step_col:
                step = st.selectbox("Delta step", STEPS, key="risk_step")
        greeks = greek_catalog.known([*bucket_greeks, measure])
        lines, line_errors = _risk_lines(market_set, pair, options, hedges, include_hedges,
                                         fixed, mods, greeks)
        pillars = risk_grid.market_tenors(context)
        with grid_tab:
            tenors_text = st.text_input(
                "Tenor buckets", "", key="risk_tenors", placeholder=", ".join(pillars),
                help="Comma-separated tenors or dates, e.g. 1W, 1M, 3M. Empty uses the "
                     "market's pillars.")
        typed = [t.strip() for t in tenors_text.split(",") if t.strip()]
        labels, dates, tenor_errors = risk_grid.tenor_axis(context, typed or pillars)
        for tab_ in (grid_tab, bucket_tab):
            with tab_:
                for message in (*line_errors, *tenor_errors):
                    st.warning(message)
        if not labels:
            with grid_tab:
                st.error("No tenor bucket could be resolved on this market.")
            return
        with grid_tab:
            grid = risk_grid.strike_grid(lines, labels, dates, measure, step)
            st.dataframe(red_negatives(_formatted(grid)), use_container_width=True)
            st.caption("Rows: expiry buckets; columns: strike buckets by call delta, low "
                       "strike to high. A position between two buckets is split between "
                       "them in proportion to its distance on each axis."
                       + (" Hedge forwards are in the last call bucket." if include_hedges else ""))
        with bucket_tab:
            buckets = risk_grid.greek_buckets(lines, labels, dates, bucket_greeks)
            buckets.index = [CATALOG[name].label for name in buckets.index]
            st.dataframe(red_negatives(_formatted(buckets)), use_container_width=True)
            st.caption("Greeks per expiry bucket (" + ("sticky-delta" if fixed else "smile")
                       + "), split between neighbouring buckets by expiry. Theta is in "
                         "the premium currency; the rest in the base currency.")


def _risk_lines(market_set, pair, options, hedges, include_hedges, fixed, mods,
                greeks=risk_grid.BUCKET_GREEKS):
    """Price the pair's selection once per inputs (and extra Greeks); cached
    across reruns."""
    extras = tuple(greek_catalog.extras(greeks))
    key = (tuple(t.trade_id + f"v{t.version}" for t in (*options, *hedges)), pair,
           str(market_set.valuation_date), repr(mods),
           repr(st.session_state.get("_surface_fingerprints", {}).get(pair)), extras)
    cached = st.session_state.get("_risk_pricing")
    if not cached or cached[0] != key:
        with ProgressBar(f"Pricing {pair} risk") as bar:
            results, forwards, errors = service.price_trades(
                market_set, [*options, *hedges], mods, extras=extras, progress=bar)
        cached = (key, results, forwards, errors)
        st.session_state["_risk_pricing"] = cached
    _, results, forwards, errors = cached
    lines, line_errors = risk_grid.risk_lines(market_set, pair, [*options, *hedges], results,
                                              forwards, include_hedges, fixed, greeks)
    problems = {name: reason for result in results
                for name, reason in result.get("greek_problems", {}).items()}
    notes = [f"{CATALOG[name].label} not available: {reason}" for name, reason in problems.items()]
    return lines, [*errors, *line_errors, *notes]


def _spot_ladder(tab, market_set, context, options, hedges, mods, ladder_mode):
    with tab:
        steps_col, increment_col = st.columns(2)
        with steps_col:
            steps = st.number_input("Steps on each side of spot", 0, 20, 5, 1,
                                    key="risk_ladder_steps")
        with increment_col:
            increment = st.number_input("Spot increment (%)", min_value=0.0001, value=0.50,
                                        step=0.05, format="%.2f", key="risk_ladder_increment")
        greeks = render_greek_picker("risk_ladder_greeks")
        run = st.button("Calculate spot ladder", type="primary", key="risk_ladder_run",
                        disabled=not options)
    if not run:
        return
    spot = float(context.spot)
    spots = [spot * (1.0 + step * float(increment) / 100.0)
             for step in range(-int(steps), int(steps) + 1)]
    rows = pd.DataFrame(ladder_rows(options))
    positions = [{"currency_pair": h.currency_pair, "expiry": h.expiry,
                  "base_amount": h.signed_base_amount(), "contract_rate": h.contract_rate,
                  "value_date": h.value_date}
                 for h in hedges]
    settings = context.settings
    with ProgressBar("Calculating spot ladder") as bar:
        results, errors = calculate_spot_ladder(
            list(rows.iterrows()), spots, market_set.valuation_date, context.market_for_expiry,
            context.resolve_market_expiry, settings.pricing_model, settings.atm_type,
            settings.delta_base, mods, mode=ladder_mode, forward_positions=positions,
            extras=greek_catalog.extras(greeks),
            market_for_delivery=getattr(context, "market_for_delivery", None), progress=bar)
    render_spot_ladder(tab, results, errors, spot, greeks)

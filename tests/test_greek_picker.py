"""Greek picker helpers and the Pricer's Greek columns and By pair totals."""
from types import SimpleNamespace

import pandas as pd
import pytest

import app.ui.greek_picker as picker
from app import pricer
from app.ui.portfolio_block import SUMMARY_RESULT_COLUMNS, _with_line_results
from pricing import greek_catalog as catalog


@pytest.fixture
def session(monkeypatch):
    state = {}
    monkeypatch.setattr(picker, "st", SimpleNamespace(session_state=state))
    return state


def test_chosen_greeks_default_and_stale_names(session):
    assert picker.chosen_greeks("k") == list(catalog.BASE_GREEKS)
    session["k"] = ["vega_decay", "renamed_greek", "delta"]
    assert picker.chosen_greeks("k") == ["delta", "vega_decay"]     # catalogue order, stale dropped
    session["k"] = []                                   # cleared (Escape): defaults, not none
    assert picker.chosen_greeks("k") == list(catalog.BASE_GREEKS)


def test_missing_extras_names_what_pricing_has_not_computed():
    results = [{"row": 1, "delta": 1.0, "vega_decay": 2.0, "fixed_vega_decay": 2.0}]
    chosen = ["delta", "vega_decay", "gamma_vol"]
    assert picker.missing_extras(chosen, results) == ["gamma_vol"]
    assert picker.missing_extras(chosen, []) == []


def result(pair, row, amount, greeks=catalog.BASE_GREEKS):
    return {"row": row, "currency_pair": pair, "mtm_base": amount, "pnl_base": amount,
            **{column: amount for column in catalog.columns(greeks)}}


def test_by_pair_shows_chosen_greeks_that_were_priced():
    results = [result("EURUSD", 1, 10.0, (*catalog.BASE_GREEKS, "vega_decay")),
               result("EURUSD", 2, 5.0, (*catalog.BASE_GREEKS, "vega_decay"))]
    table = pricer.aggregate_by_pair(results, greeks=["delta", "vega_decay", "gamma_vol"])
    assert list(table.columns) == ["currency_pair", "mtm_base", "pnl_base", "fixed_delta",
                                   "delta", "fixed_vega_decay", "vega_decay"]   # gamma_vol not priced
    assert table.loc[0, "vega_decay"] == 15.0


def test_hedges_add_only_to_columns_shown():
    hedge = {"currency_pair": "EURUSD", "mtm_base": 1.0, "pnl_base": 1.0,
             "delta_offset": -3.0, "theta": 0.5}
    table = pricer.aggregate_by_pair([result("EURUSD", 1, 10.0)], [hedge], greeks=["vega"])
    assert list(table.columns) == ["currency_pair", "mtm_base", "pnl_base", "fixed_vega", "vega"]
    assert table.loc[0, "mtm_base"] == 11.0
    full = pricer.aggregate_by_pair([result("EURUSD", 1, 10.0)], [hedge])
    assert full.loc[0, "delta"] == 7.0 and full.loc[0, "theta"] == 10.5


def test_default_aggregation_is_unchanged():
    table = pricer.aggregate_by_pair([result("EURUSD", 1, 10.0)])
    assert list(table.columns) == ["currency_pair", "mtm_base", "pnl_base",
                                   *catalog.columns(catalog.BASE_GREEKS)]


def test_line_results_show_the_chosen_greek_columns_only():
    rows = pd.DataFrame([{"currency_pair": "EURUSD", "strike": "1.10"}])
    priced = result("EURUSD", 1, 1234.0, (*catalog.BASE_GREEKS, "gamma_vol"))
    priced["gamma_vol"] = None                                 # failed extra -> blank
    display = _with_line_results(rows, [priced], catalog.columns(["delta", "gamma_vol"]))
    added = [c for c in display.columns if c not in rows.columns]
    assert added == [*SUMMARY_RESULT_COLUMNS, "fixed_delta", "delta",
                     "fixed_gamma_vol", "gamma_vol"]
    assert display.loc[0, "delta"] == "1,234" and pd.isna(display.loc[0, "gamma_vol"])


# -- Portfolio ---------------------------------------------------------------------

def test_columns_map_to_their_greek():
    assert catalog.greek_of("fixed_vega_decay") == "vega_decay"
    assert catalog.greek_of("vega_decay_usd") == "vega_decay"
    assert catalog.greek_of("weighted_vega_usd") == "weighted_vega"
    assert catalog.greek_of("vega") == "vega"
    assert catalog.greek_of("pnl_usd") is None and catalog.greek_of("spot") is None


def test_portfolio_prices_only_the_extras_its_columns_need():
    from app.ui.trades_block import DEFAULT_COLUMNS, chosen_extras, extra_columns
    # The defaults show weighted vega, rho and the delta and gamma decays: those are priced.
    assert chosen_extras(DEFAULT_COLUMNS) == ["delta_decay", "gamma_decay", "weighted_vega", "rho"]
    assert chosen_extras(["pnl_usd", "vega_decay_usd", "fixed_gamma_vol"]) == [
        "vega_decay", "gamma_vol"]
    columns = extra_columns()
    assert "vega_decay_usd" in columns and "weighted_vega_usd" in columns
    assert "delta_decay_usd" not in columns            # delta is not converted to USD
    assert {"fixed_gamma_vol", "gamma_vol"} <= set(columns)


def test_portfolio_totals_carry_priced_extras_and_convert_them():
    from app.ui.trades_block import pair_totals, usd_totals
    priced = [result("EURUSD", 1, 100.0, (*catalog.BASE_GREEKS, "vega_decay")),
              result("USDJPY", 2, 50.0)]
    totals = pair_totals(priced, [])
    assert "vega_decay" in totals and "gamma_vol" not in totals
    frame, usd = usd_totals(totals, {"EURUSD": 1.10, "USDJPY": 150.0})
    eur = frame[frame["currency_pair"] == "EURUSD"].iloc[0]
    assert eur["vega_decay_usd"] == pytest.approx(110.0)
    assert "vega_decay_usd" in usd


def test_portfolio_column_options_do_not_depend_on_what_was_priced():
    """Streamlit resets a multiselect whose options change."""
    from app.ui.trades_block import DEFAULT_COLUMNS, column_options, pair_totals, usd_totals
    plain, _ = usd_totals(pair_totals([result("EURUSD", 1, 1.0)], []), {"EURUSD": 1.1})
    rich, _ = usd_totals(pair_totals(
        [result("EURUSD", 1, 1.0, (*catalog.BASE_GREEKS, *catalog.EXTRA_GREEKS))], []),
        {"EURUSD": 1.1})
    assert column_options(plain.columns) == column_options(rich.columns)
    options = column_options(rich.columns)
    assert options[:len(DEFAULT_COLUMNS)] == list(DEFAULT_COLUMNS)
    assert set(rich.columns) - {"currency_pair", "spot"} <= set(options)
    assert column_options(["surprise"])[-1] == "surprise"     # never hidden


@pytest.mark.parametrize("column, directional", [
    ("vanna", True), ("fixed_vanna_usd", True), ("rega_usd", True), ("vanna_decay_usd", True),
    ("rega_decay", True), ("delta", True), ("fixed_delta", True), ("delta_decay", True),
    ("spot_delta", True), ("rho", True), ("phi", True),
    ("vega_usd", False), ("volga_usd", False), ("sega_usd", False), ("gamma", False),
    ("theta_usd", False), ("vega_decay_usd", False), ("weighted_vega_usd", False),
    ("pnl_usd", False), ("mtm_usd", False),
])
def test_directional_greeks(column, directional):
    from pricing.greek_catalog import is_directional
    assert is_directional(column) is directional


def test_the_total_row_leaves_directional_exposures_blank():
    from app.ui.trades_block import TOTAL_ROW, pair_totals, usd_totals
    totals = pair_totals([result("EURUSD", 1, 10.0), result("USDJPY", 2, 20.0)], [])
    frame, usd = usd_totals(totals, {"EURUSD": 1.10, "USDJPY": 150.0})
    total = frame[frame["currency_pair"] == TOTAL_ROW].iloc[0]
    pairs = frame[frame["currency_pair"] != TOTAL_ROW]
    for column in usd:
        if column in ("vanna_usd", "rega_usd", "vanna_decay_usd", "rega_decay_usd"):
            assert pd.isna(total[column]), column
            assert pairs[column].notna().all(), column        # per pair still shown
        else:
            assert total[column] == pytest.approx(pairs[column].sum()), column

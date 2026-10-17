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

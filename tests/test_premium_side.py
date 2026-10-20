"""Premium currency as BASE/TERM: conversion, fallbacks, pricing and booking parity."""
from datetime import date

import pandas as pd
import pytest

from app import pricer
from app.market_set import MarketSet
from app.premium_side import BASE, TERM, premium_currency, premium_side
from app.trades.service import booking_from_lines
from app.ui.portfolio_block import PREMIUM_CODE_COLUMN, with_premium_codes
from golden.app_harness import load_markets
from market_data import Modifications

pytestmark = pytest.mark.filterwarnings("ignore")
MODS = Modifications(beta_atm=0.7, beta_rr=1.2, beta_bf=0.0)
VALUATION = date(2026, 10, 15)


@pytest.fixture(scope="module")
def markets():
    return MarketSet.from_payloads(load_markets("MARKETS_20261015.json"), VALUATION)


def line(**changes):
    base = dict(currency_pair="EURUSD", strike="1.18", expiry="2026-12-17", call_put="CALL",
                buy_sell="BUY", notional="1m", premium=12000.0, premium_ccy=TERM)
    return dict(base, **changes)


# -- conversion ----------------------------------------------------------------

@pytest.mark.parametrize("value, side", [
    ("BASE", BASE), ("term", TERM), ("EUR", BASE), ("usd", TERM), (" USD ", TERM),
    ("", TERM), (None, TERM), (float("nan"), TERM)])
def test_side_from_a_side_a_code_or_blank(value, side):
    assert premium_side("EURUSD", value) == (side, None)


def test_a_code_outside_the_pair_falls_back_to_term_with_a_note():
    side, note = premium_side("EURUSD", "JPY")
    assert side == TERM and "JPY is not in EURUSD" in note
    side, note = premium_side("", "JPY")              # no pair to place it in
    assert side == TERM and "JPY" in note


@pytest.mark.parametrize("pair, value, code", [
    ("EURUSD", BASE, "EUR"), ("EURUSD", TERM, "USD"), ("USDJPY", TERM, "JPY"),
    ("usdjpy", "base", "USD"), ("EURUSD", "", "USD"), ("EURUSD", "USD", "USD"),
    ("EURUSD", "JPY", "JPY"),                         # passed on, rejected downstream
    ("", BASE, ""), ("EUR", TERM, "")])
def test_currency_from_a_side_or_a_code(pair, value, code):
    assert premium_currency(pair, value) == code


def test_the_currency_follows_a_pair_change():
    assert premium_currency("EURUSD", TERM) == "USD"
    assert premium_currency("USDJPY", TERM) == "JPY"


# -- the grid --------------------------------------------------------------------

def test_grid_shows_the_currency_next_to_the_side():
    rows = pd.DataFrame([line(), line(currency_pair="USDJPY", premium_ccy=BASE),
                         line(currency_pair="", premium_ccy=BASE)])
    shown = with_premium_codes(rows, "GBPUSD")
    assert list(shown[PREMIUM_CODE_COLUMN]) == ["USD", "USD", "GBP"]
    columns = list(shown.columns)
    assert columns[columns.index("premium_ccy") + 1] == PREMIUM_CODE_COLUMN
    again = with_premium_codes(shown.assign(currency_pair=["USDJPY", "EURUSD", ""]), "GBPUSD")
    assert list(again[PREMIUM_CODE_COLUMN]) == ["JPY", "EUR", "GBP"]   # recomputed, not kept


# -- pricing and booking ---------------------------------------------------------

def test_pricing_a_side_equals_pricing_its_code(markets):
    rows = pd.DataFrame([line(), line(premium_ccy="USD"), line(premium_ccy=BASE, premium=10000.0),
                         line(premium_ccy="EUR", premium=10000.0)])
    results, errors = pricer.price_lines(markets, rows, MODS)
    assert errors == []
    for side, code in ((0, 1), (2, 3)):
        assert results[side]["premium_ccy"] == results[code]["premium_ccy"]
        assert results[side]["pnl_base"] == results[code]["pnl_base"]
        assert results[side]["mtm_base"] == results[code]["mtm_base"]
    assert results[0]["premium_ccy"] == "USD" and results[2]["premium_ccy"] == "EUR"


def test_a_blank_pair_takes_the_default_pairs_currency(markets):
    rows = pd.DataFrame([line(currency_pair="", premium_ccy=BASE)])
    resolved, _, _ = pricer.resolve_line_inputs(markets, rows, "EURUSD")
    results, errors = pricer.price_lines(markets, resolved, MODS)
    assert errors == [] and results[0]["premium_ccy"] == "EUR"


def test_a_code_outside_the_pair_is_still_rejected(markets):
    _, errors = pricer.price_lines(markets, pd.DataFrame([line(premium_ccy="JPY")]), MODS)
    assert errors and "Premium currency must be EUR or USD" in errors[0]


def test_booking_stores_the_currency_the_side_stands_for(markets):
    rows = pd.DataFrame([line(premium_ccy=BASE), line(currency_pair="USDJPY", strike="150",
                                                      premium_ccy=TERM)])
    booking, numbers, errors = booking_from_lines(markets, rows, MODS)
    assert errors == []
    assert [trade.premium_ccy for trade in booking.trades] == ["EUR", "JPY"]


# -- the pair's convention (Pricer step 4) ------------------------------------

@pytest.mark.parametrize("pair, side, code", [
    ("EURUSD", TERM, "USD"), ("AUDUSD", TERM, "USD"), ("USDJPY", BASE, "USD"),
    ("EURGBP", BASE, "EUR"), ("AUDJPY", BASE, "AUD"), ("", TERM, "")])
def test_a_blank_premium_currency_is_the_pairs_convention(pair, side, code):
    from app.premium_side import conventional_side
    assert conventional_side(pair) == side
    assert premium_side(pair, "") == (side, None) and premium_side(pair, None) == (side, None)
    assert premium_currency(pair, "") == code


def test_a_premium_on_convention_follows_a_pair_change_a_chosen_one_stays():
    import pandas as pd
    from app.ui.portfolio_block import _apply_editor_state
    rows = pd.DataFrame([line(currency_pair="EURUSD", premium_ccy=TERM),     # convention
                         line(currency_pair="EURUSD", premium_ccy=BASE),     # chosen
                         line(currency_pair="EURUSD", premium_ccy=TERM)])
    changed = _apply_editor_state(rows, {"edited_rows": {
        0: {"currency_pair": "USDJPY"}, 1: {"currency_pair": "USDJPY"},
        2: {"currency_pair": "USDJPY", "premium_ccy": TERM}}})          # chosen with it
    sides = [premium_side(p, s)[0] for p, s in zip(changed["currency_pair"],
                                                    changed["premium_ccy"])]
    assert sides == [BASE, BASE, TERM]                  # USD (USDJPY's), EUR->USD base, JPY
    assert changed.at[0, "premium_ccy"] == ""            # blank until the app fills the side


def test_book_and_remove_sit_after_the_trade_details():
    from app.ui.portfolio_block import TRADE_DETAIL_COLUMNS
    assert TRADE_DETAIL_COLUMNS == ("currency_pair", "expiry", "option_type", "call_put",
                                    "strike", "buy_sell", "notional", "payout_ccy")

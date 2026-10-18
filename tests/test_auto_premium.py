"""Premium auto-fill: a Pricer line marked auto takes the premium that makes
its P&L zero, a typed premium is kept, and any change to the terms resets the
line to auto. Lines without a mode (stored trades) price as before."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from app import pricer
from app.market_set import MarketSet
from app.trades import service
from app.ui.portfolio_block import (
    PREMIUM_MODE, _apply_editor_state, apply_auto_premiums, premium_mode_of,
    same_priced_inputs, with_premium_modes,
)
from golden.app_harness import load_markets
from market_data import Modifications
from pricing import Pricable

VALUATION = date(2026, 10, 15)
MODS = Modifications(beta_atm=0.7, beta_rr=1.2, beta_bf=0.0)


@pytest.fixture(scope="module")
def markets():
    return MarketSet.from_payloads(load_markets("MARKETS_20261015.json"), VALUATION)


def line(pair="EURUSD", buy_sell="BUY", premium=0.0, mode=None, premium_ccy="TERM", **terms):
    row = dict(currency_pair=pair, expiry="3M", call_put="PUT", strike="25d", buy_sell=buy_sell,
               notional="10m", premium=premium, premium_ccy=premium_ccy, **terms)
    if mode is not None:
        row[PREMIUM_MODE] = mode
    return row


def priced(markets, lines):
    rows, _, _ = pricer.resolve_line_inputs(markets, pd.DataFrame(lines))
    results, errors = pricer.price_lines(markets, rows, MODS)
    assert errors == []
    return rows, results


@pytest.mark.parametrize("pair, premium_ccy", [("EURUSD", "TERM"), ("EURUSD", "BASE"),
                                               ("USDJPY", "BASE"), ("USDMXN", "TERM")])
@pytest.mark.parametrize("buy_sell", ["BUY", "SELL"])
def test_auto_premium_is_the_option_value_and_pnl_is_zero(markets, pair, premium_ccy, buy_sell):
    _, (plain, auto) = priced(markets, [line(pair, buy_sell, premium_ccy=premium_ccy),
                                        line(pair, buy_sell, premium_ccy=premium_ccy,
                                             mode="auto")])
    assert auto["premium_auto"] and "premium_auto" not in plain
    assert auto["auto_premium"] == abs(plain["mtm"]) > 0
    assert auto["pnl"] == auto["pnl_base"] == auto["pnl_term"] == 0.0
    # The premium is paid on a buy and received on a sale.
    assert auto["premium"] == (-1 if buy_sell == "BUY" else 1) * auto["auto_premium"]
    # Nothing but the premium and P&L moves.
    for key, value in plain.items():
        if not key.startswith(("premium", "pnl")) and key != "row":
            assert auto[key] == value, key


def test_typed_and_unmarked_lines_price_as_before(markets):
    _, (plain, typed, unmarked_auto_mode) = priced(markets, [
        line(premium=50_000.0), line(premium=50_000.0, mode="typed"),
        line(premium=50_000.0, mode="nonsense")])
    strip = lambda result: {k: v for k, v in result.items() if k != "row"}  # noqa: E731
    assert strip(typed) == strip(plain) == strip(unmarked_auto_mode)
    assert plain["premium"] == -50_000.0


def test_pricable_without_a_premium_is_auto(markets):
    context = pricer.market_for_pair(markets, "EURUSD")
    expiry = context.resolve_market_expiry("3M")
    row = pd.Series(dict(currency_pair="EURUSD", expiry=expiry, call_put="CALL", strike=1.20,
                         buy_sell="SELL", notional=1e6, valuation=VALUATION))
    market = context.market_for_expiry(expiry)
    result = Pricable(row=row, premium=None, premium_ccy="EUR").price(market, "vv", "dn", True, MODS)
    assert result["pnl"] == 0.0 and result["auto_premium"] == -result["mtm"]
    with pytest.raises(ValueError, match="Buy/Sell"):
        Pricable(row=row.copy().replace("SELL", "HOLD"), premium=None,
                 premium_ccy="EUR").price(market, "vv", "dn", True, MODS)


def test_booking_takes_the_auto_premium(markets):
    lines = [line(mode="auto"), line(buy_sell="SELL", premium=12_345.0, mode="typed"), line()]
    rows, results = priced(markets, lines)
    booking, numbers, errors = service.booking_from_lines(markets, rows, MODS)
    assert errors == []
    premiums = [trade.premium for trade in booking.trades]
    assert premiums == [pytest.approx(results[0]["auto_premium"]), 12_345.0, 0.0]


def test_pricing_writes_the_auto_premium_back(markets):
    rows, results = priced(markets, [line(mode="auto"), line(premium=10.0, mode="typed")])
    filled = apply_auto_premiums(rows, results)
    assert filled.at[0, "premium"] == results[0]["auto_premium"]
    assert filled.at[1, "premium"] == 10.0
    assert apply_auto_premiums(rows, [{"row": 9, "premium_auto": True, "auto_premium": 1.0}]
                               ).equals(rows)


@pytest.mark.parametrize("mode, premium, expected", [
    (None, 0.0, "auto"), (None, None, "auto"), (None, np.nan, "auto"), (None, "", "auto"),
    (None, 25.0, "typed"), ("typed", 0.0, "typed"), ("AUTO", 25.0, "auto"),
    ("bad", 25.0, "typed"), ("bad", 0.0, "auto"), (None, "x", "auto"),
])
def test_premium_mode_defaults(mode, premium, expected):
    assert premium_mode_of(mode, premium) == expected


def grid(*lines):
    return with_premium_modes(pd.DataFrame(list(lines)))


def test_typing_a_premium_makes_it_typed_and_clearing_returns_to_auto():
    rows = grid(line(premium=100.0, mode="auto"), line(premium=5.0, mode="typed"))
    typed = _apply_editor_state(rows, {"edited_rows": {0: {"premium": 250.0}}})
    assert typed.at[0, PREMIUM_MODE] == "typed" and typed.at[0, "premium"] == 250.0
    cleared = _apply_editor_state(rows, {"edited_rows": {1: {"premium": None}}})
    assert cleared.at[1, PREMIUM_MODE] == "auto"


@pytest.mark.parametrize("column, value", [
    ("currency_pair", "USDJPY"), ("expiry", "6M"), ("call_put", "CALL"), ("strike", "1.10"),
    ("buy_sell", "SELL"), ("notional", "5m"), ("premium_ccy", "BASE"),
])
def test_any_change_to_the_terms_resets_the_premium_to_auto(column, value):
    rows = grid(line(premium=5.0, mode="typed"), line(premium=7.0, mode="typed"))
    changed = _apply_editor_state(rows, {"edited_rows": {0: {column: value}}})
    assert changed.at[0, PREMIUM_MODE] == "auto" and pd.isna(changed.at[0, "premium"])
    # Other lines, and edits that are not terms, keep their premium.
    assert changed.at[1, PREMIUM_MODE] == "typed" and changed.at[1, "premium"] == 7.0
    booked = _apply_editor_state(rows, {"edited_rows": {1: {"book": "FX1"}}})
    assert booked.at[1, PREMIUM_MODE] == "typed" and booked.at[1, "premium"] == 7.0


def test_added_lines_are_auto_unless_they_carry_a_premium():
    rows = grid(line(premium=5.0, mode="typed"))
    added = _apply_editor_state(rows, {"added_rows": [
        {"currency_pair": "EURUSD", "strike": "25d"},
        {"currency_pair": "EURUSD", "strike": "25d", "premium": 40.0}]})
    assert list(added[PREMIUM_MODE]) == ["typed", "auto", "typed"]


def test_a_mode_change_is_a_priced_input():
    rows = grid(line(premium=5.0, mode="typed"))
    auto = rows.copy()
    auto.at[0, PREMIUM_MODE] = "auto"
    assert same_priced_inputs(rows, rows.copy()) and not same_priced_inputs(rows, auto)

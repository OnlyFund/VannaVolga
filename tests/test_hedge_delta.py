"""Hedge delta override: a Pricer line's hedge flattens its typed (agreed)
delta, a percent of notional in the line's convention, instead of its priced
delta. Results, premiums and lines without a typed delta are unchanged."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from app import hedge_delta, pricer
from app.hedge_delta import HEDGE_DELTA, HEDGE_MODE
from app.market_set import MarketSet
from app.trades import service
from app.ui.portfolio_block import (
    _apply_editor_state, apply_auto_hedge_deltas, hedge_mode_of, same_priced_inputs,
    with_hedge_modes, with_premium_modes, with_vol_modes,
)
from app.vol_override import VOL_INPUT, VOL_MODE
from golden.app_harness import load_markets
from market_data import Modifications

VALUATION = date(2026, 10, 15)
MODS = Modifications(beta_atm=0.7, beta_rr=1.2, beta_bf=0.0)
NOTIONAL = 10e6


@pytest.fixture(scope="module")
def markets():
    return MarketSet.from_payloads(load_markets("MARKETS_20261015.json"), VALUATION)


def line(pair="EURUSD", call_put="PUT", buy_sell="BUY", expiry="3M", pct=None, mode=None,
         **terms):
    row = dict(currency_pair=pair, expiry=expiry, call_put=call_put, strike="25d",
               buy_sell=buy_sell, notional="10m", premium=np.nan, premium_mode="auto",
               premium_ccy="TERM", **terms)
    if mode is not None:
        row.update({HEDGE_DELTA: pct, HEDGE_MODE: mode})
    return row


def priced(markets, lines):
    rows, _, _ = pricer.resolve_line_inputs(markets, pd.DataFrame(lines))
    results, errors = pricer.price_lines(markets, rows, MODS)
    return rows, results, errors


OWN = {"row", "hedge_override", "hedge_delta_pct", "hedge_delta"}


def same_results(a, b):
    return ({k: v for k, v in a.items() if k not in OWN}
            == {k: v for k, v in b.items() if k not in OWN})


@pytest.mark.parametrize("pair, expiry", [("EURUSD", "3M"), ("USDJPY", "1Y"),
                                          ("USDMXN", "6M")])
@pytest.mark.parametrize("call_put, buy_sell, sign", [("CALL", "BUY", 1), ("PUT", "BUY", -1),
                                                      ("CALL", "SELL", -1), ("PUT", "SELL", 1)])
def test_typed_delta_sizes_the_hedge(markets, pair, expiry, call_put, buy_sell, sign):
    _, (plain, typed, negative), errors = priced(markets, [
        line(pair, call_put, buy_sell, expiry),
        line(pair, call_put, buy_sell, expiry, pct=30.0, mode="typed"),
        line(pair, call_put, buy_sell, expiry, pct=-30.0, mode="typed")])
    assert errors == []
    assert same_results(plain, typed) and "hedge_override" not in plain
    # 30% of notional in the line's convention, converted like its own delta.
    expected = sign * 0.30 * NOTIONAL * plain["delta"] / plain["pair_delta"]
    assert typed["hedge_delta"] == pytest.approx(expected, rel=1e-12)
    assert hedge_delta.hedge_delta(typed) == typed["hedge_delta"]
    assert hedge_delta.hedge_delta(plain) == plain["delta"]
    # The sign comes from call/put and buy/sell, not from what was typed.
    assert negative["hedge_delta"] == typed["hedge_delta"]
    assert np.sign(typed["hedge_delta"]) == np.sign(plain["delta"]) == sign


def test_typed_at_the_shown_delta_hedges_as_before(markets):
    rows, results, _ = priced(markets, [line(), line("USDJPY", "CALL", "SELL", "1Y")])
    filled = apply_auto_hedge_deltas(with_hedge_modes(rows.assign(**{HEDGE_DELTA: np.nan})),
                                     results)
    typed = filled.assign(**{HEDGE_MODE: "typed"})
    _, again, _ = priced(markets, list(typed.to_dict("records")))
    before, _ = pricer.hedge_amounts(markets, results)
    after, _ = pricer.hedge_amounts(markets, again)
    for (key, members), (_, typed_members) in zip(before.items(), after.items()):
        for (_, amount), (_, typed_amount) in zip(members, typed_members):
            assert typed_amount == pytest.approx(amount, rel=1e-12)


def test_hedge_trades_and_booking_follow_the_typed_delta(markets):
    lines = [line(), line(pct=40.0, mode="typed", expiry="6M")]
    rows, results, _ = priced(markets, lines)
    trades, errors = pricer.hedge_trades_for(markets, results)
    assert errors == []
    by_expiry = {trade["expiry"]: trade["base_amount"] for trade in trades}
    factor_market = pricer.market_for_pair(markets, "EURUSD")
    expiry = results[1]["expiry"]
    factor = pricer.forward_delta_factor(factor_market.market_for_expiry(expiry), expiry,
                                         VALUATION)
    assert by_expiry[expiry] == pytest.approx(-results[1]["hedge_delta"] / factor)
    booking, _, errors = service.booking_from_lines(markets, rows, MODS, with_hedges=True)
    assert errors == []
    forwards = {t.expiry: t.signed_base_amount() for t in booking.trades if t.is_forward}
    assert forwards[expiry] == pytest.approx(by_expiry[expiry])


def test_zero_typed_delta_leaves_the_line_unhedged(markets):
    _, (zero,), _ = priced(markets, [line(pct=0.0, mode="typed")])
    assert zero["hedge_delta"] == 0.0
    trades, errors = pricer.hedge_trades_for(markets, [zero])
    assert trades == [] and errors == []


@pytest.mark.parametrize("cell", [None, np.nan, "x", 151.0])
def test_a_typed_line_without_a_valid_delta_fails_that_line(markets, cell):
    _, results, errors = priced(markets, [line(pct=cell, mode="typed"), line()])
    assert [r["row"] for r in results] == [2]
    assert errors[0].startswith("Line 1:") and "Hedge delta" in errors[0]


def test_only_an_explicit_typed_mode_overrides():
    for row in ({}, {HEDGE_DELTA: 25.0}, {HEDGE_DELTA: 25.0, HEDGE_MODE: "auto"},
                {HEDGE_DELTA: 25.0, HEDGE_MODE: "bad"}):
        assert hedge_delta.typed_hedge_pct(row) is None
    assert hedge_delta.typed_hedge_pct({HEDGE_DELTA: -25.0, HEDGE_MODE: "Typed"}) == 25.0


def test_a_line_without_a_convention_delta_reports_why():
    with pytest.raises(ValueError, match="cannot be converted"):
        hedge_delta.typed_hedge_fields(25.0, {"delta": 1.0, "pair_delta": 0.0}, 1e6,
                                       "CALL", "BUY")
    with pytest.raises(ValueError, match="cannot be converted"):
        hedge_delta.typed_hedge_fields(25.0, {"delta": 1.0}, 1e6, "CALL", "BUY")


def test_pricing_writes_auto_hedge_deltas_back(markets):
    rows, results, _ = priced(markets, [line(pct=None, mode="auto"),
                                        line(pct=40.0, mode="typed")])
    filled = apply_auto_hedge_deltas(rows, results)
    assert filled.at[0, HEDGE_DELTA] == pytest.approx(
        abs(results[0]["pair_delta"]) / NOTIONAL * 100)
    assert filled.at[1, HEDGE_DELTA] == 40.0
    plain = pd.DataFrame([line()])
    assert apply_auto_hedge_deltas(plain, results) is plain


@pytest.mark.parametrize("mode, pct, expected", [
    (None, None, "auto"), (None, np.nan, "auto"), (None, 25.0, "typed"), (None, 0.0, "typed"),
    ("auto", 25.0, "auto"), ("typed", 25.0, "typed"), ("typed", None, "auto"),
    ("bad", 25.0, "typed"), ("typed", 151.0, "typed"), ("typed", np.inf, "auto"),
    (None, -25.0, "typed"),
])
def test_hedge_mode_defaults(mode, pct, expected):
    assert hedge_mode_of(mode, pct) == expected


def grid(*lines):
    return with_hedge_modes(with_vol_modes(with_premium_modes(pd.DataFrame(list(lines)))))


def test_typing_a_delta_makes_it_typed_and_clearing_returns_to_auto():
    rows = grid(line(pct=25.0, mode="auto"), line(pct=30.0, mode="typed"))
    typed = _apply_editor_state(rows, {"edited_rows": {0: {HEDGE_DELTA: 27.5}}})
    assert typed.at[0, HEDGE_MODE] == "typed" and typed.at[0, HEDGE_DELTA] == 27.5
    cleared = _apply_editor_state(rows, {"edited_rows": {1: {HEDGE_DELTA: None}}})
    assert cleared.at[1, HEDGE_MODE] == "auto"


@pytest.mark.parametrize("column, value", [
    ("currency_pair", "USDJPY"), ("expiry", "6M"), ("call_put", "CALL"), ("strike", "1.10"),
    ("buy_sell", "SELL"), ("notional", "5m"), ("premium_ccy", "BASE"), (VOL_INPUT, 9.0),
])
def test_a_change_to_the_terms_or_the_vol_returns_the_delta_to_auto(column, value):
    rows = grid(line(pct=30.0, mode="typed"), line(pct=35.0, mode="typed"))
    changed = _apply_editor_state(rows, {"edited_rows": {0: {column: value}}})
    assert changed.at[0, HEDGE_MODE] == "auto" and pd.isna(changed.at[0, HEDGE_DELTA])
    assert changed.at[1, HEDGE_MODE] == "typed" and changed.at[1, HEDGE_DELTA] == 35.0


def test_premium_and_delta_edits_leave_each_other_alone():
    rows = grid(line(pct=30.0, mode="typed", **{VOL_INPUT: 8.0, VOL_MODE: "typed"}))
    premium = _apply_editor_state(rows, {"edited_rows": {0: {"premium": 7.0}}})
    assert premium.at[0, HEDGE_MODE] == "typed" and premium.at[0, HEDGE_DELTA] == 30.0
    delta = _apply_editor_state(rows, {"edited_rows": {0: {HEDGE_DELTA: 20.0}}})
    assert delta.at[0, VOL_MODE] == "typed" and delta.at[0, VOL_INPUT] == 8.0
    assert delta.at[0, "premium_mode"] == "auto"


def test_added_lines_are_auto_unless_they_carry_a_delta():
    rows = grid(line(pct=30.0, mode="typed"))
    added = _apply_editor_state(rows, {"added_rows": [
        {"currency_pair": "EURUSD", "strike": "25d"},
        {"currency_pair": "EURUSD", "strike": "25d", HEDGE_DELTA: 20.0}]})
    assert list(added[HEDGE_MODE]) == ["typed", "auto", "typed"]


def test_a_delta_change_is_a_priced_input():
    rows = grid(line(pct=30.0, mode="typed"))
    other = rows.copy()
    other.at[0, HEDGE_DELTA] = 31.0
    assert same_priced_inputs(rows, rows.copy()) and not same_priced_inputs(rows, other)

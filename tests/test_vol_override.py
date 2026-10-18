"""Vol override: a Pricer line with a typed vol prices on its own copy of the
smile, shifted so its strike is at that vol. Greeks and an auto premium (the
dealt price) follow it, MTM stays at market, so P&L is the market value
against the dealt price; lines without a typed vol (stored trades) price
exactly as before."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from app import pricer, vol_override
from app.market_context import PricingSettings
from app.market_set import MarketSet
from app.trades import service
from app.ui.portfolio_block import (
    _apply_editor_state, apply_market_vols, same_priced_inputs, vol_mode_of, with_premium_modes,
    with_vol_modes,
)
from app.vol_override import VOL_INPUT, VOL_MODE
from golden.app_harness import load_markets
from market_data import Modifications

VALUATION = date(2026, 10, 15)
MODS = Modifications(beta_atm=0.7, beta_rr=1.2, beta_bf=0.0)


@pytest.fixture(scope="module")
def payloads():
    return load_markets("MARKETS_20261015.json")


@pytest.fixture(scope="module")
def markets(payloads):
    return MarketSet.from_payloads(payloads, VALUATION)


def line(pair="EURUSD", buy_sell="BUY", vol=None, vol_mode=None, premium=np.nan,
         premium_mode="auto", premium_ccy="TERM", **terms):
    row = dict(currency_pair=pair, expiry="3M", call_put="PUT", strike="25d", buy_sell=buy_sell,
               notional="10m", premium=premium, premium_mode=premium_mode,
               premium_ccy=premium_ccy)
    row.update(terms)
    if vol_mode is not None:
        row.update({VOL_INPUT: vol, VOL_MODE: vol_mode})
    return row


def priced(markets, lines):
    rows, _, _ = pricer.resolve_line_inputs(markets, pd.DataFrame(lines))
    results, errors = pricer.price_lines(markets, rows, MODS)
    return rows, results, errors


def strip(result):
    return {k: v for k, v in result.items() if k != "row"}


def test_market_lines_price_as_before(markets):
    _, (plain, market, same), errors = priced(markets, [
        line(), line(vol=33.0, vol_mode="market"), line()])
    assert errors == []
    _, (typed_at_market,), _ = priced(markets, [line(vol=plain["vol_pct"], vol_mode="typed")])
    assert strip(market) == strip(plain) == strip(same)
    # A typed vol equal to the market's: no shift, the same numbers.
    own = {"vol_override", "mkt_vol_pct", "vol_shift", "vol_spread", "prem_at_vol",
           "value_at_vol"}
    assert {k: v for k, v in strip(typed_at_market).items() if k not in own} == strip(plain)
    assert typed_at_market["vol_shift"] == 0.0


@pytest.mark.parametrize("pair, premium_ccy", [("EURUSD", "TERM"), ("USDJPY", "BASE"),
                                               ("USDMXN", "TERM")])
@pytest.mark.parametrize("buy_sell, bump", [("BUY", 1.5), ("SELL", -1.0)])
def test_typed_vol_shifts_the_smile_to_the_strike(markets, pair, premium_ccy, buy_sell, bump):
    _, (plain,), _ = priced(markets, [line(pair, buy_sell, premium_ccy=premium_ccy)])
    target = plain["vol_pct"] + bump
    _, (typed,), errors = priced(markets, [line(pair, buy_sell, vol=target, vol_mode="typed",
                                                premium_ccy=premium_ccy)])
    assert errors == []
    assert typed["vol_pct"] == pytest.approx(target, abs=1e-6)
    assert typed["mkt_vol_pct"] == plain["vol_pct"]
    assert typed["vol_spread"] == pytest.approx(bump, abs=1e-6)
    # MTM stays at market; the auto premium is the price at the typed vol
    # (the dealt price), so P&L is the market value against it.
    for key in ("mtm", "mtm_base", "mtm_term"):
        assert typed[key] == plain[key]
    holder = 1 if buy_sell == "BUY" else -1
    assert typed["premium_auto"]
    assert typed["auto_premium"] == typed["prem_at_vol"] == abs(typed["value_at_vol"])
    assert typed["premium"] == -holder * typed["prem_at_vol"]
    assert typed["pnl"] == pytest.approx(typed["mtm"] + typed["premium"], abs=1e-9)
    assert (typed["prem_at_vol"] - plain["auto_premium"]) * bump > 0
    # Buying above mid, or selling below it, loses money on the day.
    assert typed["pnl"] * holder * bump < 0
    # Greeks follow the typed vol; the strike and terms do not move.
    assert typed["delta"] != plain["delta"] and typed["vega"] != plain["vega"]
    assert typed["theta"] != plain["theta"]
    for key in ("strike", "expiry", "notional", "premium_ccy"):
        assert typed[key] == plain[key]


def test_typed_premium_is_kept_with_a_typed_vol(markets):
    _, (plain,), _ = priced(markets, [line(premium=50_000.0, premium_mode="typed")])
    _, (typed,), _ = priced(markets, [line(premium=50_000.0, premium_mode="typed",
                                           vol=plain["vol_pct"] + 1, vol_mode="typed")])
    assert typed["premium"] == plain["premium"] == -50_000.0
    assert "premium_auto" not in typed


def test_booking_takes_the_dealt_premium_and_the_typed_vol_delta(markets):
    _, (plain,), _ = priced(markets, [line()])
    lines = [line(), line(vol=plain["vol_pct"] + 1.5, vol_mode="typed")]
    rows, results, _ = priced(markets, lines)
    booking, _, errors = service.booking_from_lines(markets, rows, MODS, with_hedges=True)
    assert errors == []
    options = [t for t in booking.trades if t.is_option]
    assert [t.premium for t in options] == [pytest.approx(plain["auto_premium"]),
                                            pytest.approx(results[1]["prem_at_vol"])]
    groups, _ = pricer.hedge_amounts(markets, results)
    amounts = [amount for members in groups.values() for _, amount in members]
    # Each line's hedge follows its own (typed-vol) delta.
    assert amounts[0] != pytest.approx(amounts[1])


def test_vv_sabr_matches_within_fit_noise(payloads):
    markets = MarketSet.from_payloads(payloads, VALUATION, PricingSettings(pricing_model="vv-sabr"))
    _, (plain,), _ = priced(markets, [line(strike="10d", call_put="CALL")])
    target = plain["vol_pct"] + 1.5
    _, (typed,), errors = priced(markets, [line(strike="10d", call_put="CALL", vol=target,
                                                vol_mode="typed")])
    assert errors == []
    assert typed["vol_pct"] == pytest.approx(target, abs=vol_override.ACCEPT)


def test_an_unreachable_vol_fails_that_line_only(markets):
    _, results, errors = priced(markets, [line(vol=0.01, vol_mode="typed"), line()])
    assert [r["row"] for r in results] == [2]
    assert len(errors) == 1 and errors[0].startswith("Line 1:")


@pytest.mark.parametrize("cell", [None, np.nan, "x", 0.0, -1.0, 250.0])
def test_a_typed_line_without_a_valid_vol_errors(cell):
    with pytest.raises(ValueError, match="Vol must be"):
        vol_override.typed_vol({VOL_MODE: "typed", VOL_INPUT: cell})


def test_only_an_explicit_typed_mode_overrides():
    for row in ({}, {VOL_INPUT: 12.0}, {VOL_INPUT: 12.0, VOL_MODE: "market"},
                {VOL_INPUT: 12.0, VOL_MODE: "bad"}):
        assert vol_override.typed_vol(row) is None
    assert vol_override.typed_vol({VOL_INPUT: 12.0, VOL_MODE: " Typed "}) == 12.0


class FakeMarket:
    def __init__(self, v_atm):
        self.v_atm = v_atm

    def copy_with(self, v_atm):
        return FakeMarket(v_atm)


@pytest.mark.parametrize("noise, matches", [(4e-4, True), (5e-3, False)])
def test_noisy_smiles_take_the_closest_shift_within_tolerance(monkeypatch, noise, matches):
    # A smile whose strike vol is ATM + 1 with fit noise that never settles.
    calls = []

    def strike_vol(market, *args):
        calls.append(market.v_atm)
        return market.v_atm + 1.0 + noise * (-1) ** len(calls), 0.0

    monkeypatch.setattr(vol_override, "strike_vol", strike_vol)
    args = (FakeMarket(10.0), {}, 13.0, "vv", "dn", True)
    if matches:
        market, shift = vol_override.shifted_market(*args)
        assert abs(market.v_atm + 1.0 - 13.0) <= noise + 1e-12
    else:
        with pytest.raises(ValueError, match="could not be matched"):
            vol_override.shifted_market(*args)
    assert len(calls) <= vol_override.MAX_ITERATIONS + 1


def test_next_day_market_moves_with_the_line():
    assert vol_override.shift_market(FakeMarket(10.0), 1.5).v_atm == 11.5
    with pytest.raises(ValueError, match="next day"):
        vol_override.shift_market(FakeMarket(1.0), -2.0)


@pytest.mark.parametrize("mode, vol, expected", [
    (None, None, "market"), (None, np.nan, "market"), (None, 12.0, "typed"),
    ("market", 12.0, "market"), ("typed", 12.0, "typed"), ("typed", None, "market"),
    ("bad", 12.0, "typed"), ("typed", 0.0, "market"), ("typed", 500.0, "market"),
])
def test_vol_mode_defaults(mode, vol, expected):
    assert vol_mode_of(mode, vol) == expected


def grid(*lines):
    return with_vol_modes(with_premium_modes(pd.DataFrame(list(lines))))


def test_typing_a_vol_makes_it_typed_and_clearing_returns_to_market():
    rows = grid(line(vol=7.0, vol_mode="market"), line(vol=8.0, vol_mode="typed"))
    typed = _apply_editor_state(rows, {"edited_rows": {0: {VOL_INPUT: 9.5}}})
    assert typed.at[0, VOL_MODE] == "typed" and typed.at[0, VOL_INPUT] == 9.5
    cleared = _apply_editor_state(rows, {"edited_rows": {1: {VOL_INPUT: None}}})
    assert cleared.at[1, VOL_MODE] == "market"


@pytest.mark.parametrize("column, value", [
    ("currency_pair", "USDJPY"), ("expiry", "6M"), ("call_put", "CALL"), ("strike", "1.10"),
    ("buy_sell", "SELL"), ("notional", "5m"), ("premium_ccy", "BASE"),
])
def test_any_change_to_the_terms_returns_the_vol_to_market(column, value):
    rows = grid(line(vol=8.0, vol_mode="typed"), line(vol=9.0, vol_mode="typed"))
    changed = _apply_editor_state(rows, {"edited_rows": {0: {column: value}}})
    assert changed.at[0, VOL_MODE] == "market" and pd.isna(changed.at[0, VOL_INPUT])
    assert changed.at[1, VOL_MODE] == "typed" and changed.at[1, VOL_INPUT] == 9.0


def test_vol_and_premium_edits_leave_each_other_alone():
    rows = grid(line(vol=8.0, vol_mode="typed", premium=5.0, premium_mode="typed"))
    premium = _apply_editor_state(rows, {"edited_rows": {0: {"premium": 7.0}}})
    assert premium.at[0, VOL_MODE] == "typed" and premium.at[0, VOL_INPUT] == 8.0
    vol = _apply_editor_state(rows, {"edited_rows": {0: {VOL_INPUT: 9.0}}})
    assert vol.at[0, "premium_mode"] == "typed" and vol.at[0, "premium"] == 5.0


def test_added_lines_are_market_unless_they_carry_a_vol():
    rows = grid(line(vol=8.0, vol_mode="typed"))
    added = _apply_editor_state(rows, {"added_rows": [
        {"currency_pair": "EURUSD", "strike": "25d"},
        {"currency_pair": "EURUSD", "strike": "25d", VOL_INPUT: 11.0}]})
    assert list(added[VOL_MODE]) == ["typed", "market", "typed"]


def test_pricing_writes_market_vols_back(markets):
    rows, results, _ = priced(markets, [line(vol=None, vol_mode="market"),
                                        line(vol=9.0, vol_mode="typed")])
    filled = apply_market_vols(rows, results)
    assert filled.at[0, VOL_INPUT] == results[0]["vol_pct"]
    assert filled.at[1, VOL_INPUT] == 9.0
    # Grids without the column (stored trades) are left as they are.
    plain = pd.DataFrame([line()])
    assert apply_market_vols(plain, results) is plain


def test_a_vol_change_is_a_priced_input():
    rows = grid(line(vol=8.0, vol_mode="typed"))
    other = rows.copy()
    other.at[0, VOL_INPUT] = 8.5
    assert same_priced_inputs(rows, rows.copy()) and not same_priced_inputs(rows, other)


def test_pricable_values_apart_only_when_asked(markets):
    from pricing import Pricable
    context = pricer.market_for_pair(markets, "EURUSD")
    expiry = context.resolve_market_expiry("3M")
    row = pd.Series(dict(currency_pair="EURUSD", expiry=expiry, call_put="CALL", strike=1.20,
                         buy_sell="BUY", notional=1e6, valuation=VALUATION))
    market = context.market_for_expiry(expiry)
    shifted = market.copy_with(v_atm=market.v_atm + 1.0)
    args = ("vv", "dn", True, MODS)
    plain = Pricable(row=row, premium=None, premium_ccy="USD").price(shifted, *args)
    apart = Pricable(row=row, premium=None, premium_ccy="USD").price(
        shifted, *args, value_market=market)
    at_market = Pricable(row=row, premium=None, premium_ccy="USD").price(market, *args)
    assert "value_at_vol" not in plain
    assert apart["value_at_vol"] == plain["mtm"] and apart["premium"] == plain["premium"]
    assert apart["mtm"] == at_market["mtm"] and apart["vol_pct"] == plain["vol_pct"]
    assert apart["delta"] == plain["delta"]
    assert apart["pnl"] == pytest.approx(at_market["mtm"] - plain["mtm"]) and apart["pnl"] < 0


class FakeState(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def test_pending_lines_are_adopted_once_before_the_grid_is_set_up(monkeypatch):
    from app.ui import portfolio_block
    state = FakeState(option_rows=pd.DataFrame([line()]), portfolio_editor_v3={"added_rows": []},
                      _pending_priced_portfolio_rows=pd.DataFrame([line(), line()]),
                      _pending_portfolio_line_results={"rows": None, "results": []})
    monkeypatch.setattr(portfolio_block.st, "session_state", state)
    portfolio_block.adopt_pending_rows()
    assert len(state["option_rows"]) == 2 and "portfolio_editor_v3" not in state
    assert state["_portfolio_line_results"] == {"rows": None, "results": []}
    # Again in the same run: nothing left to adopt, the editor state stays.
    state["portfolio_editor_v3"] = {"added_rows": [{"currency_pair": "USDJPY"}]}
    portfolio_block.adopt_pending_rows()
    assert len(state["option_rows"]) == 2 and "portfolio_editor_v3" in state

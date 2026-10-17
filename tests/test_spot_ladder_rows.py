"""Spot-ladder rows: Greeks on the smile the betas imply at each spot.

A row's P&L is priced along the smile dynamics from today's spot; its smile
delta and gamma are derivatives along that same path, so the deltas shown
add up to the P&L between rows. The other Greeks use the betas-moved smile.
"""
from datetime import date

import pandas as pd
import pytest

import app.reports.spot_ladder as spot_ladder
from app import pricer
from app.market_set import MarketSet
from app.reports.spot_ladder import calculate_spot_ladder
from golden.app_harness import load_markets
from market_data import Modifications
from pricing.fx_forward import term_discount_factor
from pricing.greeks import calculate_greeks

VALUATION = date(2026, 10, 15)
BETAS = Modifications(beta_atm=0.7, beta_rr=1.2, beta_bf=0.0)
FLAT = Modifications(beta_atm=0.0, beta_rr=0.0, beta_bf=0.0)


@pytest.fixture(scope="module")
def markets():
    return MarketSet.from_payloads(load_markets("MARKETS_20261015.json"), VALUATION)


def ladder(markets, mods, steps=(-2, -1, 0, 1, 2), step=0.005, strike=1.15, mode="reference"):
    context = pricer.market_for_pair(markets, "EURUSD")
    row = pd.Series(dict(currency_pair="EURUSD", strike=strike, expiry="2027-04-15",
                         call_put="PUT", buy_sell="BUY", notional=10_000_000.0, premium=0.0,
                         premium_ccy="USD"))
    spot = float(context.spot)
    spots = [spot * (1 + k * step) for k in steps]
    levels, errors = calculate_spot_ladder(
        [(0, row)], spots, VALUATION, context.market_for_expiry, context.resolve_market_expiry,
        "vv", "dn", True, mods, mode=mode)
    assert errors == []
    return context, row, sorted(levels, key=lambda level: level["spot"])


@pytest.mark.parametrize("mode", ["reference", "optimized"])
def test_row_deltas_add_up_to_the_ladder_pnl(markets, mode):
    context, row, levels = ladder(markets, BETAS, mode=mode)
    expiry = context.resolve_market_expiry(row.expiry)
    discount = term_discount_factor(context.market_for_expiry(expiry), expiry, VALUATION)
    for low, high in zip(levels, levels[1:]):
        pnl = high["pnl_base"] * high["spot"] - low["pnl_base"] * low["spot"]     # USD
        # Trapezoid rule with its end correction (exact for a cubic P&L):
        # integral of delta = h (d0 + d1) / 2 - h^2 / 12 (d1' - d0'),
        # with d' = gamma per unit spot = gamma / (S * 1%).
        width = high["spot"] - low["spot"]
        slope = {id(level): level["gamma"] / (level["spot"] * 0.01) for level in (low, high)}
        integral = (width * (low["delta"] + high["delta"]) / 2
                    - width ** 2 / 12 * (slope[id(high)] - slope[id(low)]))
        assert pnl == pytest.approx(integral * discount, rel=1e-3)


def test_base_spot_row_is_todays_greeks(markets):
    context, row, levels = ladder(markets, BETAS)
    middle = levels[2]
    market = context.market_for_expiry(context.resolve_market_expiry(row.expiry))
    prepared = row.copy()
    prepared["expiry"] = context.resolve_market_expiry(row.expiry)
    prepared["valuation"] = VALUATION
    prepared["notional"] = 10_000_000.0
    today = calculate_greeks(market, prepared, None, "vv", "dn", True, BETAS)
    for name in ("delta", "gamma", "vega", "fixed_delta", "fixed_gamma"):
        assert middle[name] == pytest.approx(today[name], rel=1e-9)


def test_without_dynamics_rows_use_the_quoted_smile(markets, monkeypatch):
    calls = []
    original = spot_ladder._smile_moved_markets

    def spy(*args, **kwargs):
        moved = original(*args, **kwargs)
        calls.append(moved[0] is args[1])          # unchanged scenario market
        return moved
    monkeypatch.setattr(spot_ladder, "_smile_moved_markets", spy)
    ladder(markets, FLAT, steps=(-1, 0, 1))
    assert calls and all(calls)


def test_rows_fall_back_to_the_quoted_smile_when_dynamics_fail(markets, monkeypatch):
    _, _, expected = ladder(markets, FLAT, steps=(-1, 0, 1))

    def broken(*args, **kwargs):
        raise RuntimeError("boom")
    import surface_models.smile_extension as smile_extension
    _, _, moved = ladder(markets, BETAS, steps=(-1, 0, 1))
    monkeypatch.setattr(smile_extension.SmileExtension, "get_local_skews", broken)
    monkeypatch.setattr(spot_ladder, "smile_path_delta_gamma", broken)
    context, row, levels = ladder(markets, BETAS, steps=(-1, 0, 1))
    # P&L is still priced; the rows' Greeks come back on the quoted smile.
    assert all(level["vega"] is not None for level in levels)
    assert levels[0]["vega"] != moved[0]["vega"]

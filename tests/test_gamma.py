"""Gamma is the change in the reported delta for a 1% spot move.

Fixed gamma follows the unchanged smile quotes, smile gamma the smile
dynamics: the paths the fixed and smile deltas, and the spot ladder's P&L,
are priced on. Both are in the reported delta's spot/forward and
premium-adjustment conventions.
"""
from datetime import date

import pandas as pd
import pytest

import pricing.greeks as greeks
from app import pricer
from app.market_set import MarketSet
from app.reports.spot_ladder import calculate_spot_ladder
from Greeks.fixed_greeks import FixedGamma
from golden.app_harness import load_markets
from market_data import MarketData, Modifications
from pricing.fx_forward import term_discount_factor
from pricing.greeks import calculate_greeks

VALUATION = date(2026, 10, 15)
BETAS = Modifications(beta_atm=0.7, beta_rr=1.2, beta_bf=0.0)
FLAT = Modifications(beta_atm=0.0, beta_rr=0.0, beta_bf=0.0)
BUMP = 1e-4


@pytest.fixture(scope="module")
def markets():
    return MarketSet.from_payloads(load_markets("MARKETS_20261015.json"), VALUATION)


def line_inputs(markets, pair, strike, call_put="PUT", buy_sell="BUY", expiry="2027-04-15",
          premium_ccy=None):
    context = pricer.market_for_pair(markets, pair)
    market = context.market_for_expiry(pd.Timestamp(expiry))
    row = pd.Series(dict(currency_pair=pair, strike=strike, expiry=expiry, call_put=call_put,
                         buy_sell=buy_sell, notional=10_000_000.0,
                         premium_ccy=premium_ccy or pair[3:], valuation=VALUATION))
    return context, market, row


def greeks_at(market, row, mods, bump=0.0):
    return calculate_greeks(market.copy_with(s=market.s * (1 + bump)), row, None, "vv", "dn",
                            True, mods)


def bumped_delta_gamma(market, row, mods, name):
    up, dn = greeks_at(market, row, mods, BUMP), greeks_at(market, row, mods, -BUMP)
    return (up[name] - dn[name]) / (2 * BUMP) * 0.01


@pytest.mark.parametrize("pair,strike,call_put", [
    ("EURUSD", 1.16, "PUT"),       # forward reporting, term premium
    ("EURUSD", 1.20, "CALL"),
    ("USDJPY", 150.0, "PUT"),      # premium-adjusted pair
])
def test_fixed_gamma_is_the_change_in_reported_fixed_delta(markets, pair, strike, call_put):
    _, market, row = line_inputs(markets, pair, strike, call_put)
    result = greeks_at(market, row, BETAS)
    # The reported fixed delta is itself a 0.1% central bump; its truncation
    # error moves with spot, about 0.3% of gamma on an OTM strike.
    assert result["fixed_gamma"] == pytest.approx(
        bumped_delta_gamma(market, row, BETAS, "fixed_delta"), rel=5e-3)


def test_gamma_follows_a_base_currency_premium_on_legacy_reporting():
    market = MarketData(s=16.6, fwd_pts=3150.0, v_atm=11.78, rr_25=2.03, rr_10=4.115,
                        bf_25=0.4, bf_10=1.3, currency_pair="USDZAR", wrapped_smile=True,
                        broker_strangle=False)
    row = pd.Series(dict(currency_pair="USDZAR", strike=16.95, expiry=date(2027, 7, 9),
                         valuation=date(2026, 10, 9), call_put="CALL", buy_sell="BUY",
                         notional=1_000_000.0, premium=0.0, premium_ccy="USD"))
    result = greeks_at(market, row, BETAS)
    assert result["fixed_gamma"] == pytest.approx(
        bumped_delta_gamma(market, row, BETAS, "fixed_delta"), rel=2e-4)


def test_without_smile_dynamics_smile_gamma_is_fixed_gamma(markets):
    _, market, row = line_inputs(markets, "EURUSD", 1.16)
    result = greeks_at(market, row, FLAT)
    assert result["gamma"] == pytest.approx(result["fixed_gamma"], rel=1e-6)


@pytest.mark.parametrize("strike", [1.12, 1.17, 1.22])
def test_smile_gamma_matches_the_ladder_pnl_curvature(markets, strike):
    """The ladder prices each spot on the smile dynamics path; its P&L
    curvature, in forward-delta units, is the smile gamma."""
    context, market, row = line_inputs(markets, "EURUSD", strike)
    spot, step = float(market.s), 1e-3
    levels, errors = calculate_spot_ladder(
        [(0, row.drop("valuation"))], [spot * (1 - step), spot, spot * (1 + step)], VALUATION,
        context.market_for_expiry, context.resolve_market_expiry, "vv", "dn", True, BETAS,
        mode="reference")
    assert errors == []
    low, mid, high = sorted(levels, key=lambda level: level["spot"])
    pnl = [level["pnl_base"] * level["spot"] for level in (low, mid, high)]   # term ccy
    discount = term_discount_factor(market, pd.Timestamp(row.expiry), VALUATION)
    curvature = (pnl[0] + pnl[2] - 2 * pnl[1]) / (spot * step) ** 2 * spot * 0.01 / discount
    assert mid["gamma"] == pytest.approx(curvature, rel=2e-3)


def test_gamma_falls_back_to_the_calculators_when_the_new_one_fails(markets, monkeypatch):
    _, market, row = line_inputs(markets, "EURUSD", 1.16)

    def broken(*args, **kwargs):
        raise RuntimeError("boom")
    monkeypatch.setattr(greeks, "_consistent_gammas", broken)
    result = greeks_at(market, row, BETAS)
    common = greeks._greek_inputs(market, row, "vv", "dn", True)
    assert result["fixed_gamma"] == pytest.approx(FixedGamma(**common).calculate())
    assert result["gamma"] is not None and result["delta"] is not None

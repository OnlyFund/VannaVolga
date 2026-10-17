"""Trade premium currency affects hedge delta, not smile construction."""

from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from app.reports.spot_ladder import calculate_spot_ladder
from market_data import MarketData
from pricing.greeks import calculate_greeks
from pricing.pricing import Pricable, price_option


VALUATION = date(2026, 10, 9)
EXPIRY = date(2027, 7, 9)
MODS = {"beta_atm": 1.0, "beta_rr": 0.5, "beta_bf": 0.0}


def market(spot=16.6):
    return MarketData(
        s=spot, fwd_pts=3150.0, v_atm=11.78, rr_25=2.03, rr_10=4.115,
        bf_25=0.4, bf_10=1.3, currency_pair="USDZAR",
        wrapped_smile=True, broker_strangle=False,
    )


def row(**changes):
    values = dict(
        currency_pair="USDZAR", strike=16.95, expiry=EXPIRY,
        valuation=VALUATION, call_put="CALL", buy_sell="BUY",
        notional=700_000_000.0, premium=0.0,
    )
    values.update(changes)
    return pd.Series(values)


@pytest.mark.parametrize("model", ["vv", "vv-sabr"])
@pytest.mark.parametrize("call_put", ["CALL", "PUT"])
@pytest.mark.parametrize("buy_sell", ["BUY", "SELL"])
def test_trade_premium_adjusts_both_deltas_only(model, call_put, buy_sell):
    m = market()
    trade = row(call_put=call_put, buy_sell=buy_sell)
    _, vol, value = price_option(m, trade, model, "dn", True)
    raw = calculate_greeks(m, trade, None, model, "dn", True, MODS)
    term = calculate_greeks(
        m, row(call_put=call_put, buy_sell=buy_sell, premium_ccy="ZAR"),
        None, model, "dn", True, MODS, option_value_term=value,
    )
    base_row = row(call_put=call_put, buy_sell=buy_sell, premium_ccy="USD")
    base = calculate_greeks(m, base_row, None, model, "dn", True, MODS)
    assert term == raw
    for name, result in raw.items():
        expected = result - value / m.s if name in ("delta", "fixed_delta") else result
        if name in ("gamma", "fixed_gamma"):
            # Gamma is the derivative of the reported delta, raw - V/S:
            # d(V/S)/dS per 1% spot is 1% x (raw delta - V/S).
            delta = name.replace("gamma", "delta")
            expected = result - 0.01 * (raw[delta] - value / m.s)
            assert base[name] == pytest.approx(expected, rel=1e-4)
        elif expected is None:
            assert base[name] is None
        else:
            assert base[name] == pytest.approx(expected)
    assert price_option(m, base_row, model, "dn", True) == pytest.approx(
        (trade["strike"], vol, value)
    )
    assert m.deltaBase is True


@pytest.mark.parametrize("premium_ccy", ["USD", "ZAR"])
def test_pricable_uses_its_premium_currency_and_unit_value(premium_ccy):
    m = market()
    trade = row(premium_ccy="ZAR" if premium_ccy == "USD" else "USD")
    result = Pricable(trade, premium=0.0, premium_ccy=premium_ccy).price(
        m, "vv", "dn", True, MODS
    )
    unit_row = row(notional=1.0, premium_ccy=premium_ccy)
    expected = calculate_greeks(
        m, unit_row, None, "vv", "dn", True, MODS,
        option_value_term=result["mtm_term"] / trade["notional"],
    )
    assert result["delta"] == pytest.approx(expected["delta"] * trade["notional"])
    assert result["fixed_delta"] == pytest.approx(
        expected["fixed_delta"] * trade["notional"]
    )
    nonzero_cashflow = Pricable(trade, premium=10_000_000.0, premium_ccy=premium_ccy).price(
        m, "vv", "dn", True, MODS
    )
    assert nonzero_cashflow["delta"] == result["delta"]
    assert nonzero_cashflow["fixed_delta"] == result["fixed_delta"]


@pytest.mark.parametrize("premium_ccy", ["EUR", "", float("nan")])
def test_invalid_trade_premium_currency_is_rejected(premium_ccy):
    with pytest.raises(ValueError, match="Premium currency"):
        calculate_greeks(
            market(), row(premium_ccy=premium_ccy), None, "vv", "dn", True, MODS
        )


@pytest.mark.parametrize("mode", ["reference", "optimized"])
def test_ladder_adjustment_uses_scenario_option_value(mode):
    def build(expiry, as_of=None, allow_short_end=False, spot_value=None):
        return market(16.6 if spot_value is None else spot_value)

    args = dict(
        ladder_spots=[16.5, 16.7], valuation_date=VALUATION,
        market_for_expiry=build, resolve_expiry=lambda expiry: expiry,
        pricing_model="vv", atm_type="dn", delta_base=True, mods=MODS,
        mode=mode, scenario_workers=1,
    )
    base, base_errors = calculate_spot_ladder(
        rows=[(0, row(premium_ccy="USD"))], **args
    )
    term, term_errors = calculate_spot_ladder(
        rows=[(0, row(premium_ccy="ZAR"))], **args
    )
    assert base_errors == term_errors == []
    assert len(base) == len(term) == 2
    for base_level, term_level in zip(base, term):
        # With zero entered premium, base P&L is the scenario's signed V/S.
        for name in ("delta", "fixed_delta"):
            assert term_level[name] - base_level[name] == pytest.approx(
                base_level["pnl_base"]
            )
        for name in ("gamma", "fixed_gamma"):
            # Gamma follows the reported (premium-adjusted) delta.
            assert base_level[name] == pytest.approx(
                term_level[name] - 0.01 * base_level[name.replace("gamma", "delta")],
                rel=1e-4)
        for name in base_level:
            if name not in ("delta", "fixed_delta", "gamma", "fixed_gamma"):
                assert base_level[name] == pytest.approx(term_level[name])


def test_expired_option_keeps_existing_zero_greeks():
    expired = row(expiry=VALUATION, premium_ccy="USD")
    assert set(calculate_greeks(
        market(), expired, None, "vv", "dn", True, MODS
    ).values()) == {0.0}


def test_zero_notional_does_not_create_premium_exposure():
    result = Pricable(row(notional=0.0), premium=0.0, premium_ccy="USD").price(
        market(), "vv", "dn", True, MODS
    )
    assert result["delta"] == result["fixed_delta"] == result["mtm_term"] == 0.0


def test_trade_adjustment_is_independent_of_unadjusted_pillar_convention():
    m = MarketData(
        s=1.1, fwd_pts=30.0, v_atm=10.0, rr_25=-0.5, rr_10=-1.0,
        bf_25=0.2, bf_10=0.6, currency_pair="EURUSD", deltaBase=False,
        wrapped_smile=True, broker_strangle=False,
    )
    trade = row(currency_pair="EURUSD", strike=1.1, premium_ccy="USD")
    term = Pricable(trade, premium=0.0, premium_ccy="USD").price(
        m, "vv", "dn", False, MODS
    )
    base = Pricable(trade, premium=0.0, premium_ccy="EUR").price(
        m, "vv", "dn", False, MODS
    )
    for name in ("delta", "fixed_delta"):
        assert term[name] - base[name] == pytest.approx(base["mtm_base"])
    assert m.deltaBase is False


@pytest.mark.parametrize("premium_ccy", ["USD", "ZAR"])
@pytest.mark.parametrize("reporting", [None, "forward", "spot"])
def test_ui_hedge_builder_uses_the_same_adjusted_delta_as_pricer(premium_ccy, reporting):
    from app.market_context import PricingSettings
    from app.pricer import build_delta_hedges
    from pricing.greeks import forward_delta_factor

    surface = market().copy_with(delta_reporting=reporting)
    context = SimpleNamespace(
        currency_pair="USDZAR", valuation_date=VALUATION, settings=PricingSettings(),
        resolve_market_expiry=lambda expiry: expiry,
        market_for_expiry=lambda expiry: surface,
    )
    trade = row(premium_ccy=premium_ccy)
    positions, errors = build_delta_hedges(context, pd.DataFrame([trade]), MODS)
    result = Pricable(trade, premium=0.0, premium_ccy=premium_ccy).price(
        surface, "vv", "dn", True, MODS
    )
    assert errors == []
    assert len(positions) == 1
    factor = forward_delta_factor(surface, EXPIRY, VALUATION)
    assert positions[0]["base_amount"] * factor == pytest.approx(-result["delta"])

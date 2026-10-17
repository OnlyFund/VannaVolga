"""Reporting conventions are separate from smile calibration and cashflows."""
import pandas as pd
import pytest

from market_data import MarketData
from pricing.greeks import calculate_greeks, forward_delta_factor
from pricing.pricing import price_option
from pricing_utils.BlackScholes import BlackScholes
from helper_functions.fx_discounting import discount_kwargs
from helper_functions.fx_conventions import get_pair
from app.reports.spot_ladder import _add_forward_pnl, _empty_level


@pytest.mark.parametrize("pair,spot", [("USDZAR", 16.6), ("EURUSD", 1.1)])
@pytest.mark.parametrize("call_put", ["CALL", "PUT"])
@pytest.mark.parametrize("buy_sell", ["BUY", "SELL"])
def test_flat_smile_matches_analytic_pair_delta(pair, spot, call_put, buy_sell):
    market = MarketData(
        s=spot, fwd_pts=100., v_atm=12., rr_25=0., rr_10=0.,
        bf_25=0., bf_10=0., currency_pair=pair, wrapped_smile=True,
        broker_strangle=False, deltaBase=get_pair(pair).premium_adjusted,
    )
    row = pd.Series(dict(
        currency_pair=pair, strike=spot, expiry="2027-09-28",
        valuation="2026-10-09", call_put=call_put, buy_sell=buy_sell,
        notional=700_000_000., premium_ccy=pair[3:],
    ))
    results = {}
    for convention in ("forward", "spot"):
        surface = market.copy_with(delta_reporting=convention)
        _, vol, value = price_option(surface, row, "vv", "dn", True)
        results[convention] = calculate_greeks(
            surface, row, None, "vv", "dn", True, {}, option_value_term=value
        )
        overrides = discount_kwargs(surface, row.expiry, row.valuation)
        overrides["delta_convention"] = convention
        bs = BlackScholes(
            S=spot, K=spot, fwd_pts=market.fwd_pts, v=vol,
            expiry_date=row.expiry, valuation_date=row.valuation,
            CallPut=call_put, BuySell=buy_sell, Notional=row.notional,
            deltaBase=get_pair(pair).premium_adjusted, **overrides
        )
        for name in ("fixed_delta", "delta"):
            assert results[convention][name] == pytest.approx(bs.delta(), rel=2e-5)
    factor = forward_delta_factor(
        market.copy_with(delta_reporting="spot"), row.expiry, row.valuation
    )
    for name in results["forward"]:
        if name in ("fixed_delta", "delta"):
            assert results["spot"][name] == pytest.approx(results["forward"][name] * factor)
        elif name in ("fixed_gamma", "gamma"):
            # Gamma is the derivative of the reported delta: spot delta =
            # forward delta x factor, and the factor (DF * F / S, forward
            # points fixed) itself moves with spot.
            def factor_at(bump):
                return forward_delta_factor(market.copy_with(
                    s=spot * (1 + bump), delta_reporting="spot"), row.expiry, row.valuation)
            factor_gamma = (factor_at(1e-4) - factor_at(-1e-4)) / 2e-4 * 0.01
            delta = name.replace("gamma", "delta")
            assert results["spot"][name] == pytest.approx(
                results["forward"][name] * factor + results["forward"][delta] * factor_gamma,
                rel=1e-5)
        else:
            assert results["spot"][name] == results["forward"][name]
    alternate_row = row.copy()
    alternate_row["premium_ccy"] = pair[:3]
    alternate = calculate_greeks(
        market.copy_with(delta_reporting="forward"),
        alternate_row, None, "vv", "dn", True, {},
    )
    assert alternate["delta"] == pytest.approx(results["forward"]["delta"])


@pytest.mark.parametrize("convention", ["spot", "forward"])
def test_forward_hedge_delta_uses_selected_units(convention):
    market = MarketData(
        s=16.6, fwd_pts=4215., v_atm=12., rr_25=0., rr_10=0.,
        bf_25=0., bf_10=0., currency_pair="USDZAR", delta_reporting=convention,
    )
    expiry, valuation = "2027-09-28", "2026-10-09"
    factor = forward_delta_factor(market, expiry, valuation)
    amount = -300_000_000. / factor
    position = dict(
        currency_pair="USDZAR", expiry=expiry,
        base_amount=amount, contract_rate=17.0215,
    )
    level = _empty_level(market.s)
    _add_forward_pnl(level, position, market, valuation)
    assert level["delta"] == pytest.approx(-300_000_000.)
    assert level["fixed_delta"] == pytest.approx(level["delta"])


def test_invalid_reporting_convention_is_rejected():
    with pytest.raises(ValueError, match="delta_reporting"):
        MarketData(
            s=1.1, fwd_pts=10., v_atm=10., rr_25=0., rr_10=0.,
            bf_25=0., bf_10=0., delta_reporting="invalid",
        )

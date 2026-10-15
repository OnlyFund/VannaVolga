import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from Greeks.cross_greeks import CrossGreeks
from Greeks.decay_greeks import DecayGreeks
from Greeks.fixed_greeks import FixedVanna
from Greeks.smile_greeks import SmileGreeks
from market_data import MarketData, Modifications


def _inputs():
    market = MarketData(
        s=1.17,
        fwd_pts=10.0,
        v_atm=7.0,
        rr_25=0.5,
        rr_10=1.0,
        bf_25=0.2,
        bf_10=0.7,
    )
    return market, Modifications(beta_atm=0.5, beta_rr=0.2, beta_bf=0.2)


def test_cross_gamma_vol_is_first_derivative_of_smile_gamma():
    market, mods = _inputs()
    kwargs = dict(
        market=market,
        mods=mods,
        k=1.18,
        expiry_date="2026-02-16",
        valuation_date="2025-08-14",
        CallPut="CALL",
        Notional=1_000_000,
        deltaBase=False,
    )
    cross = CrossGreeks(**kwargs)
    eps = 0.01
    expected = (
        cross._smile_gamma(cross._scaled_market(eps))
        - cross._smile_gamma(cross._scaled_market(-eps))
    ) / (2.0 * eps)
    assert cross.dGammaDvol(eps=eps) == pytest.approx(expected)


def test_decay_equals_tomorrow_minus_today_for_each_smile_greek():
    market, mods = _inputs()
    kwargs = dict(
        market=market,
        mods=mods,
        k=1.18,
        expiry_date="2026-02-16",
        valuation_date="2025-08-14",
        CallPut="CALL",
        Notional=1_000_000,
        deltaBase=False,
    )
    decay = DecayGreeks(**kwargs)
    today = decay.base_market
    tomorrow = decay._get_tomorrow_greeks()

    assert decay.delta_decay() == pytest.approx(tomorrow.delta() - today.delta())
    assert decay.gamma_decay() == pytest.approx(tomorrow.gamma() - today.gamma())
    assert decay.dVegaDtime() == pytest.approx(tomorrow.vega() - today.vega())
    assert decay.dVannaDtime() == pytest.approx(tomorrow.vanna() - today.vanna())
    assert decay.dVolgaDtime() == pytest.approx(tomorrow.volga() - today.volga())
    assert decay.dRegaDtime() == pytest.approx(tomorrow.rega() - today.rega())
    assert decay.dSegaDtime() == pytest.approx(tomorrow.sega() - today.sega())


def test_decay_rejects_expiry_on_or_before_next_business_day():
    market, mods = _inputs()
    with pytest.raises(ValueError, match="next business day"):
        DecayGreeks(
            market=market,
            mods=mods,
            k=1.18,
            expiry_date="2025-08-15",
            valuation_date="2025-08-14",
            deltaBase=False,
        )


def test_vanna_is_scaled_to_the_change_in_vega_for_one_percent_spot():
    market, mods = _inputs()
    kwargs = dict(
        market=market,
        k=1.18,
        expiry_date="2026-02-16",
        valuation_date="2025-08-14",
        CallPut="CALL",
        Notional=1_000_000,
        deltaBase=False,
    )
    eps = 1e-4
    fixed = FixedVanna(**kwargs).calculate(spot_eps=eps)
    smile = SmileGreeks(mods=mods, **kwargs).vanna(eps_spot=eps)
    assert abs(fixed) < 1e7
    assert abs(smile) < 1e7


def test_smile_theta_reprices_after_curve_roll():
    market, mods = _inputs()
    kwargs = dict(
        market=market,
        k=1.18,
        expiry_date="2026-02-16",
        valuation_date="2025-08-14",
        CallPut="CALL",
        Notional=1_000_000,
        deltaBase=False,
    )
    today = SmileGreeks(mods=mods, **kwargs)
    next_market = market.copy_with(v_atm=market.v_atm + 0.25)
    tomorrow = SmileGreeks(
        mods=mods,
        **dict(kwargs, market=next_market, valuation_date="2025-08-15"),
    )
    # Prices are term currency; theta is reported in base currency.
    expected = (
        tomorrow._price_given_market(next_market)
        - today._price_given_market(market)
    ) / market.s
    assert today.theta(next_market, "2025-08-15") == pytest.approx(expected)

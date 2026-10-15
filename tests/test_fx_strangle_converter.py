"""Broker quotes must be reproduced at their fixed, common-vol strikes."""

import os
import sys

import pytest
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from helper_functions.fx_strangle_converter import (
    ValidationError,
    _delta,
    _price,
    _vv_price_for_strike,
    _vv_vol_for_strike,
    broker_to_twovol_bf,
)


MARKET = dict(
    spot=1.1, fwd_pts=10, pip_factor=10000, v_atm=10,
    rr=2, bf_broker=0.5, valuation="10/02/2026",
    expiry="01/04/2027", delivery="01/06/2027",
    basis=360, depo_pct=4,
)


@pytest.mark.parametrize("delta", [10, 25])
@pytest.mark.parametrize("pa", [False, True])
@pytest.mark.parametrize("convention", ["spot", "forward"])
@pytest.mark.parametrize("rr", [-2, 0, 2])
def test_fixed_broker_premium_and_delta_consistent_smile(delta, pa, convention, rr):
    sm = broker_to_twovol_bf(
        **dict(MARKET, rr=rr), delta=delta, is_pa=pa,
        delta_convention=convention,
    )
    factor = sm.df_f if convention == "spot" else 1.0
    broker_vol = (MARKET["v_atm"] + MARKET["bf_broker"]) / 100.0
    for k, phi in [(sm.K_c_broker, 1), (sm.K_p_broker, -1)]:
        assert _delta(sm.F, k, broker_vol, sm.T, factor, phi, pa) == pytest.approx(
            phi * delta / 100.0, abs=1e-12,
        )
    expected = (
        _price(sm.F, sm.K_c_broker, broker_vol, sm.T, sm.df_d, 1)
        + _price(sm.F, sm.K_p_broker, broker_vol, sm.T, sm.df_d, -1)
    )
    assert sm.V_broker == pytest.approx(expected, abs=1e-12)
    assert sm.sigma_c - sm.sigma_p == pytest.approx(rr)
    assert sm.bf_2v == pytest.approx(sm.sigma_2v - MARKET["v_atm"])
    assert sm.K_p < sm.K_atm < sm.K_c
    for k, vol, phi in [(sm.K_c, sm.sigma_c, 1), (sm.K_p, sm.sigma_p, -1)]:
        assert _delta(sm.F, k, vol / 100.0, sm.T, factor, phi, pa) == pytest.approx(
            phi * delta / 100.0, abs=1e-12,
        )
    pivots = [(sm.K_p, sm.sigma_p), (sm.K_atm, MARKET["v_atm"]), (sm.K_c, sm.sigma_c)]
    # Independent price reconstruction using the smile's implied volatilities.
    premium = sum(
        _price(sm.F, k, _vv_vol_for_strike(k, sm.F, sm.T, sm.df_d, pivots) / 100.0,
               sm.T, sm.df_d, phi)
        for k, phi in [(sm.K_c_broker, 1), (sm.K_p_broker, -1)]
    )
    assert premium == pytest.approx(sm.V_broker, abs=1e-10)
    if rr == 0:
        assert sm.bf_2v == pytest.approx(MARKET["bf_broker"], abs=1e-10)
    else:
        assert sm.bf_2v != pytest.approx(MARKET["bf_broker"], abs=1e-5)


def test_unadjusted_forward_broker_strikes_have_common_vol_analytic_solution():
    import math

    sm = broker_to_twovol_bf(**MARKET, delta=25, delta_convention="forward")
    stddev = 0.105 * math.sqrt(sm.T)
    for strike, phi in [(sm.K_c_broker, 1), (sm.K_p_broker, -1)]:
        expected = sm.F * math.exp(0.5 * stddev**2 - phi * norm.ppf(0.25) * stddev)
        assert strike == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("atm_conv", ["dn", "atmf", "spot"])
def test_matching_price_reproduces_pivot_prices(atm_conv):
    sm = broker_to_twovol_bf(**MARKET, delta=25, atm_conv=atm_conv)
    pivots = [(sm.K_p, sm.sigma_p), (sm.K_atm, MARKET["v_atm"]), (sm.K_c, sm.sigma_c)]
    for strike, vol in pivots:
        for phi in [1, -1]:
            assert _vv_price_for_strike(strike, sm.F, sm.T, sm.df_d, pivots, phi) == pytest.approx(
                _price(sm.F, strike, vol / 100.0, sm.T, sm.df_d, phi), abs=1e-12,
            )


@pytest.mark.parametrize("changes", [
    {"bf_broker": -10},
    {"depo_pct": -400},
    {"refine_strikes": False},
    {"outer_tol": 0},
    {"outer_max_iter": 0},
    {"delta_convention": "invalid"},
])
def test_invalid_inputs_are_explicit(changes):
    with pytest.raises(ValidationError):
        broker_to_twovol_bf(**dict(MARKET, **changes), delta=25)

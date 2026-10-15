import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from market_data import MarketData
from VannaVolga_ImpliedVol import VannaVolga
from helper_functions.VannaVolga_function import (
    get_atm_strike, get_dn_strike, get_strike_from_delta, normalize_atm_type,
)
from helper_functions.helper_functions import year_fraction
from helper_functions.input_resolvers import resolve_strike
from helper_functions.vanna_volga_helpers import GetMainPillars
from pricing_utils.BlackScholes import BlackScholes
from pricing.pricing import price_option
from vanna_volga_sabr import VannaVolga_SABR


def test_atm_convention_normalization():
    assert normalize_atm_type('DN') == 'dn'
    assert normalize_atm_type('ATMF') == 'atmf'
    assert normalize_atm_type('forward') == 'atmf'
    assert normalize_atm_type('spot') == 'spot'


def test_delta_neutral_and_delta_strikes_support_long_high_volatility():
    kwargs = dict(
        S=1.0,
        fwd=0.0,
        v=100.0,
        expiry_date="2030-01-01",
        valuation_date="2025-01-01",
    )
    assert get_atm_strike("dn", **kwargs) == pytest.approx(
        np.exp(0.5 * year_fraction(kwargs["expiry_date"], kwargs["valuation_date"]))
    )
    strike = get_strike_from_delta(
        **kwargs,
        CallPut="CALL",
        delta=0.01,
        delta_base=True,
    )
    # The pillar solver targets a forward delta, so read the strike back in the
    # same convention; the default reporting delta is a spot delta.
    actual_delta = BlackScholes(
        S=kwargs["S"],
        K=strike,
        fwd_pts=kwargs["fwd"],
        v=kwargs["v"],
        expiry_date=kwargs["expiry_date"],
        valuation_date=kwargs["valuation_date"],
        CallPut="CALL",
        BuySell="BUY",
        Notional=1,
        deltaBase=True,
        delta_convention="forward",
    ).delta()
    assert actual_delta == pytest.approx(0.01)


def test_bs_delta_is_forward_delta_and_consistent():
    bs = BlackScholes(
        S=1.17,
        K=1.17,
        fwd_pts=10.0,
        v=7.0,
        expiry_date='2026-02-16',
        valuation_date='2025-08-14',
        CallPut='CALL',
        BuySell='BUY',
        Notional=1,
        deltaBase=False,
        fwd_divisor=10000,
    )
    bs_pa = BlackScholes(
        S=1.17,
        K=1.17,
        fwd_pts=10.0,
        v=7.0,
        expiry_date='2026-02-16',
        valuation_date='2025-08-14',
        CallPut='CALL',
        BuySell='BUY',
        Notional=1,
        deltaBase=True,
        fwd_divisor=10000,
    )
    assert abs(bs.delta() - 0.5) < 0.2
    assert abs(bs_pa.delta() - 0.5) < 0.2
    assert bs.delta() > 0
    assert bs_pa.delta() > 0


def test_expired_options_use_intrinsic_value_and_zero_residual_risks():
    params = dict(
        S=1.0,
        K=1.1,
        fwd_pts=0.0,
        v=10.0,
        expiry_date="2025-08-14",
        valuation_date="2025-08-14",
        CallPut="CALL",
        BuySell="BUY",
        Notional=1.0,
    )
    bs = BlackScholes(**params)
    assert bs.price() == pytest.approx(0.0)
    assert bs.vega() == pytest.approx(0.0)
    assert bs.gamma() == pytest.approx(0.0)

    market = MarketData(
        s=1.0, fwd_pts=0.0, v_atm=10.0, rr_25=0.0, rr_10=0.0,
        bf_25=0.0, bf_10=0.0,
    )
    row = {
        "strike": 0.9,
        "expiry": "2025-08-14",
        "valuation": "2025-08-14",
        "call_put": "CALL",
        "buy_sell": "BUY",
        "notional": 2.0,
    }
    _, vol, price = price_option(market, row, "vv", "dn", False)
    assert vol == pytest.approx(market.v_atm)
    assert price == pytest.approx(0.2)


def test_market_data_rejects_non_positive_derived_forward():
    with pytest.raises(ValueError, match="derived forward"):
        MarketData(
            s=1.0, fwd_pts=-10_000.0, v_atm=10.0, rr_25=0.0, rr_10=0.0,
            bf_25=0.0, bf_10=0.0,
        )


def test_vv_smile_computes_vols():
    market = MarketData(
        s=1.17,
        fwd_pts=10.0,
        v_atm=7.0,
        rr_25=0.5,
        rr_10=1.0,
        bf_25=0.2,
        bf_10=0.7,
    )
    vv = VannaVolga(
        market=market,
        K=1.15,
        expiry_date='2026-02-16',
        valuation_date='2025-08-14',
        CallPut='CALL',
        deltaBase=False,
        atm_conv='dn',
    )
    vol = vv.get_vol_for_strike()
    assert vol > 0
    assert vol < 100


def test_batch_vv_smile_returns_requested_strikes():
    market = MarketData(
        s=1.17,
        fwd_pts=10.0,
        v_atm=7.0,
        rr_25=0.5,
        rr_10=1.0,
        bf_25=0.2,
        bf_10=0.7,
    )
    vv = VannaVolga(
        market=market,
        K=1.15,
        expiry_date="2026-02-16",
        valuation_date="2025-08-14",
        CallPut="CALL",
        deltaBase=False,
        atm_conv="dn",
    )
    result = vv.get_vol_for_strikes(np.array([1.15, 1.18]))
    assert set(result) == {1.15, 1.18}
    assert all(np.isfinite(value) and value > 0 for value in result.values())


def test_smooth_pillars_use_premium_adjusted_delta_neutral_atm():
    market = MarketData(
        s=1.17,
        fwd_pts=10.0,
        v_atm=7.0,
        rr_25=0.5,
        rr_10=1.0,
        bf_25=0.2,
        bf_10=0.7,
    )
    pillars = GetMainPillars(
        k=1.18,
        expiry_date="2026-02-16",
        valuation_date="2025-08-14",
        atm_conv="dn",
        deltaBase=True,
        s=market.s,
        fwd_pts=market.fwd_pts,
        v_atm=market.v_atm,
        rr_25=market.rr_25,
        rr_10=market.rr_10,
        bf_25=market.bf_25,
        bf_10=market.bf_10,
        forward_divisor=market.forward_divisor,
    )
    expected = get_dn_strike(
        market.s,
        market.fwd_pts,
        market.v_atm,
        "2026-02-16",
        "2025-08-14",
        deltaBase=True,
        fwd_divisor=market.forward_divisor,
    )
    atm_strike = sorted(pillars.get_main_strikes_vols())[2]
    assert atm_strike == pytest.approx(expected)


def test_app_atm_resolver_matches_premium_adjusted_model_convention():
    market = MarketData(
        s=1.17,
        fwd_pts=10.0,
        v_atm=7.0,
        rr_25=0.5,
        rr_10=1.0,
        bf_25=0.2,
        bf_10=0.7,
    )
    row = {
        "strike": "atm",
        "expiry": "2026-02-16",
        "valuation": "2025-08-14",
    }
    assert resolve_strike(market, row, "dn", True) == pytest.approx(
        get_dn_strike(
            market.s,
            market.fwd_pts,
            market.v_atm,
            row["expiry"],
            row["valuation"],
            deltaBase=True,
            fwd_divisor=market.forward_divisor,
        )
    )


def test_greeks_match_quote_definitions_and_finite_differences():
    params = dict(
        S=1.17,
        K=1.18,
        fwd_pts=10.0,
        v=7.0,
        expiry_date='2026-02-16',
        valuation_date='2025-08-14',
        CallPut='CALL',
        BuySell='BUY',
        Notional=1_000_000,
        deltaBase=False,
    )
    bs = BlackScholes(**params)
    S = params['S']
    h_spot = 1e-6
    up = BlackScholes(**dict(params, S=S + h_spot))
    down = BlackScholes(**dict(params, S=S - h_spot))
    # Gamma and vanna are scaled to a 1% *relative* spot move, so the spot
    # derivative is multiplied by ``S * 0.01`` rather than by 0.01 alone.
    assert abs(
        bs.gamma() - (up.delta() - down.delta()) / (2 * h_spot) * S * 0.01
    ) < 1e-3
    # Vanna re-expresses each bumped vega at the valuation spot, because the
    # base-currency conversion inside vega() is a reporting unit, not a risk.
    assert abs(
        bs.vanna()
        - (up.vega() * (S + h_spot) / S - down.vega() * (S - h_spot) / S)
        / (2 * h_spot) * S * 0.01
    ) < 1e-4

    h_vol = 1e-3
    vol_up = BlackScholes(**dict(params, v=params['v'] + h_vol))
    vol_down = BlackScholes(**dict(params, v=params['v'] - h_vol))
    assert abs(bs.volga() - (vol_up.vega() - vol_down.vega()) / (2 * h_vol)) < 1e-4

    pa = BlackScholes(**dict(params, deltaBase=True))
    assert pa.delta() != bs.delta()


def test_frozen_sabr_params_reuse_across_spots():
    market = MarketData(
        s=1.17,
        fwd_pts=10.0,
        v_atm=7.0,
        rr_25=0.5,
        rr_10=1.0,
        bf_25=0.2,
        bf_10=0.7,
    )
    frozen = {"alpha": 0.12, "rho": -0.30, "nu": 0.85}
    vols = []
    for spot in np.linspace(0.95, 1.45, 9):
        model = VannaVolga_SABR(
            market=market.copy_with(s=spot),
            K=1.18,
            expiry_date="2026-02-16",
            valuation_date="2025-08-14",
            CallPut="CALL",
            deltaBase=False,
            atm_conv="dn",
            frozen_sabr_params=frozen,
        )
        vols.append(model.get_vol_for_strike())
    vols = np.asarray(vols, dtype=float)
    assert np.all(np.isfinite(vols))
    assert np.max(np.abs(np.diff(vols))) < 6.0
    assert model.calibrate_sabr()[0] == pytest.approx(frozen["alpha"], rel=1e-2, abs=1e-3)


def test_sabr_volga_uses_stable_central_bump():
    market = MarketData(
        s=1.17,
        fwd_pts=10.0,
        v_atm=7.0,
        rr_25=0.5,
        rr_10=1.0,
        bf_25=0.2,
        bf_10=0.7,
    )
    model = VannaVolga_SABR(
        market=market,
        K=1.18,
        expiry_date="2026-02-16",
        valuation_date="2025-08-14",
        CallPut="CALL",
        deltaBase=False,
        atm_conv="dn",
        frozen_sabr_params={"alpha": 0.12, "rho": -0.30, "nu": 0.85},
        central_bump_vol=0.10,
    )
    volga = model.volga()
    assert np.isfinite(volga)
    assert abs(volga) < 1e4
    assert model.central_bump_vol == pytest.approx(0.10)


def test_invalid_frozen_sabr_params_fall_back_to_legacy_surface():
    market = MarketData(
        s=1.17,
        fwd_pts=10.0,
        v_atm=7.0,
        rr_25=0.5,
        rr_10=1.0,
        bf_25=0.2,
        bf_10=0.7,
    )
    bad = {"alpha": float("nan"), "rho": 0.0, "nu": 0.85}
    model = VannaVolga_SABR(
        market=market,
        K=1.18,
        expiry_date="2026-02-16",
        valuation_date="2025-08-14",
        CallPut="CALL",
        deltaBase=False,
        atm_conv="dn",
        frozen_sabr_params=bad,
    )
    vol = model.get_vol_for_strike()
    assert np.isfinite(vol)
    assert vol > 0.0
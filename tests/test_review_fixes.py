"""Regression tests for the library-review fixes."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from market_data import MarketData
from VannaVolga_ImpliedVol import VannaVolga
from vanna_volga_sabr import VannaVolga_SABR
from perturbed_market import PerturbedMarket
from surface_models.smile_extension import SmileExtension
from Greeks.fixed_greeks import FixedTheta
from helper_functions.fx_discounting import discount_kwargs
from helper_functions.helper_functions import year_fraction
from pricing_utils.BlackScholes import BlackScholes
from pricing_utils.breeden_litzenberger import forward_call_price, smile_density

pytestmark = pytest.mark.filterwarnings("ignore")

EXPIRY, VALUATION = "2027-03-25", "2026-09-25"
QUOTES = dict(s=1.138, fwd_pts=89.32, v_atm=5.947, rr_25=-0.515,
              rr_10=-0.965, bf_25=0.233, bf_10=0.86)


def _market(**changes):
    return MarketData(**{**QUOTES, **changes})


def _vv(market, K=1.146, CallPut="CALL", BuySell="BUY"):
    return VannaVolga(market=market, K=K, expiry_date=EXPIRY,
                      valuation_date=VALUATION, CallPut=CallPut,
                      BuySell=BuySell, deltaBase=False)


# 1. Batch vols must not depend on the instance's option type or side.
@pytest.mark.parametrize("call_put,buy_sell", [("PUT", "BUY"), ("CALL", "SELL"), ("PUT", "SELL")])
def test_batch_vols_independent_of_option_type_and_side(call_put, buy_sell):
    strikes = [1.08, 1.11, 1.14, 1.17, 1.20]
    reference = _vv(_market()).get_vol_for_strikes(strikes)
    result = _vv(_market(), CallPut=call_put, BuySell=buy_sell).get_vol_for_strikes(strikes)
    for k in strikes:
        assert result[k] == pytest.approx(reference[k], abs=1e-6)
        assert 4.0 < result[k] < 9.0


# 2-3. Proportional smile scaling keeps RR signs and handles zero quotes.
def test_scale_all_vols_is_proportional_for_negative_rr():
    market = _market(v_atm=6.0, rr_25=-0.6, rr_10=-1.2, bf_25=0.2, bf_10=0.8)
    bumped = PerturbedMarket(market).scale_all_vols(0.6)
    assert bumped.v_atm == pytest.approx(6.6)
    assert bumped.rr_25 == pytest.approx(-0.66)
    assert bumped.rr_10 == pytest.approx(-1.32)
    assert bumped.bf_25 == pytest.approx(0.22)
    assert bumped.bf_10 == pytest.approx(0.88)


def test_scale_all_vols_handles_zero_quotes():
    market = _market(rr_25=0.0, rr_10=0.0, bf_25=0.0)
    bumped = PerturbedMarket(market).scale_all_vols(0.01)
    assert bumped.rr_25 == 0.0 and bumped.rr_10 == 0.0 and bumped.bf_25 == 0.0


# 4. Theta's next-day price must be discounted from the next day.
def test_fixed_theta_rolls_discount_factor():
    market = _market(currency_pair="EURUSD", term_discount=0.0429)
    expiry, valuation, next_day = "2027-03-25", "2026-10-07", "2026-10-08"
    assert (discount_kwargs(market, expiry, valuation)["term_discount_factor"]
            != discount_kwargs(market, expiry, next_day)["term_discount_factor"])

    def price_at(day):
        vol = VannaVolga(market=market, K=1.08, expiry_date=expiry, valuation_date=day,
                         CallPut="PUT", deltaBase=False, atm_conv="dn").get_vol_for_strike()
        return BlackScholes(S=market.s, K=1.08, fwd_pts=market.fwd_pts, v=vol,
                            expiry_date=expiry, valuation_date=day, CallPut="PUT",
                            BuySell="BUY", Notional=1e8, deltaBase=False,
                            term_discount=market.term_discount,
                            **discount_kwargs(market, expiry, day)).price()

    expected = (price_at(next_day) - price_at(valuation)) / market.s
    theta = FixedTheta(market=market, k=1.08, expiry_date=expiry, valuation_date=valuation,
                       CallPut="PUT", Notional=1e8, deltaBase=False).calculate()
    assert theta == pytest.approx(expected, rel=1e-9)


# 5-6. Smile-extension wing reconstruction.
def _new_pillars(keep_smile_ratio, mode):
    market = _market()
    extension = SmileExtension(
        market=market, K=1.146, expiry_date=EXPIRY, valuation_date=VALUATION,
        mods={"shifted_spot": market.s * 1.01, "beta_atm": 1.0},
        keep_smile_ratio=keep_smile_ratio, preserve_ratio_mode=mode, deltaBase=False,
    )
    skews = {"main_vols": (6.0, 5.0, 5.5), "local_skew": (-3.0, -2.0, -1.0),
             "rr_skew": 2.0, "bf_skew": 0.0, "rr_ratio": 1.9, "bf_ratio": 3.7}
    return extension.get_new_pillar_vols(skews)


def test_additive_mode_preserves_wing_spreads():
    new = _new_pillars(True, "additive")
    assert new["v_atm"] != pytest.approx(5.0)
    assert new["v_25p"] - new["v_atm"] == pytest.approx(1.0)
    assert new["v_25c"] - new["v_atm"] == pytest.approx(0.5)


def test_multiplicative_mode_preserves_wing_ratios():
    new = _new_pillars(True, "multiplicative")
    assert new["v_25p"] / new["v_atm"] == pytest.approx(6.0 / 5.0)
    assert new["v_25c"] / new["v_atm"] == pytest.approx(5.5 / 5.0)


def test_unratioed_mode_scales_rr_bf_by_relative_atm_move():
    new = _new_pillars(False, "additive")
    scale = new["v_atm"] / 5.0
    assert new["rr_25"] == pytest.approx(-0.5 * scale)
    assert new["bf_25"] == pytest.approx(0.75 * scale)


# 7. SABR calibrations with a poor fit are rejected.
def test_sabr_rejects_poor_fit_residual():
    model = VannaVolga_SABR(market=_market(), K=1.146, expiry_date=EXPIRY,
                            valuation_date=VALUATION, CallPut="CALL", deltaBase=False)
    assert model._validate_sabr_params(0.07, -0.2, 0.8, residual=(0.0002) ** 2)
    assert not model._validate_sabr_params(0.07, -0.2, 0.8, residual=(0.005) ** 2)


# 9. Breeden-Litzenberger densities.
def _flat_bs(K, vol=6.0):
    return BlackScholes(S=1.138, K=K, fwd_pts=89.32, v=vol, expiry_date=EXPIRY,
                        valuation_date=VALUATION, CallPut="CALL", BuySell="BUY",
                        Notional=1.0, deltaBase=False, term_discount=0.04)


def test_d2V_dK2_matches_finite_difference_of_price():
    K, h = 1.15, 1e-4
    fd = (_flat_bs(K + h).price() - 2 * _flat_bs(K).price() + _flat_bs(K - h).price()) / h ** 2
    assert float(_flat_bs(K).d2V_dK2()) == pytest.approx(float(fd), rel=1e-5)


def test_flat_density_integrates_to_one_with_forward_mean():
    strikes = np.linspace(0.6, 2.0, 4001)
    density = _flat_bs(strikes).risk_neutral_density()
    forward = 1.138 + 89.32 / 10000
    assert np.trapz(density, strikes) == pytest.approx(1.0, abs=1e-6)
    assert np.trapz(strikes * density, strikes) == pytest.approx(forward, abs=1e-6)


def test_smile_density_reduces_to_flat_density_for_flat_vol():
    forward, t = 1.138 + 89.32 / 10000, year_fraction(EXPIRY, VALUATION)
    strikes = np.array([1.05, 1.15, 1.25])
    numeric = smile_density(lambda k: 6.0, strikes, forward, t)
    # Central second differences carry an O(h**2) truncation error.
    assert numeric == pytest.approx(_flat_bs(strikes).risk_neutral_density(), rel=5e-4)


def test_vanna_volga_smile_density_is_positive_and_fat_tailed():
    market = _market()
    forward, t = 1.138 + 89.32 / 10000, year_fraction(EXPIRY, VALUATION)
    vol = lambda k: _vv(market, K=k).get_vol_for_strike()
    strikes = np.array([1.04, 1.10, 1.146, 1.20, 1.26])
    density = smile_density(vol, strikes, forward, t)
    assert np.all(density > 0.0)
    # A positive butterfly fattens the far downside wing relative to ATM vol.
    flat = BlackScholes(S=1.138, K=0.98, fwd_pts=89.32, v=market.v_atm, expiry_date=EXPIRY,
                        valuation_date=VALUATION, CallPut="CALL", BuySell="BUY",
                        Notional=1.0, deltaBase=False).risk_neutral_density()
    assert smile_density(vol, 0.98, forward, t) > float(flat)


def test_forward_call_price_rejects_bad_inputs():
    with pytest.raises(ValueError):
        forward_call_price(1.1, -1.0, 6.0, 0.5)

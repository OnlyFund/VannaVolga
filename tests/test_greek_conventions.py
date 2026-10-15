"""Currency and scaling conventions for the reported Greeks.

Prices are term currency, but market Greeks are quoted in base currency (the
same currency as the delta hedge). Gamma and vanna are scaled to a 1%
*relative* spot move. These tests pin both conventions, because mixing them
silently rescales gamma by ``1/S`` and vega/theta by ``S``.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from Greeks.fixed_greeks import FixedDelta, FixedGamma, FixedTheta, FixedVega
from market_data import MarketData
from pricing_utils.BlackScholes import BlackScholes
from VannaVolga_ImpliedVol import VannaVolga


SPOT = 1.1380
NOTIONAL = 1_000_000


def _market():
    return MarketData(
        s=SPOT, fwd_pts=88.75, v_atm=5.92, rr_25=-0.518,
        rr_10=-0.9650, bf_25=0.235, bf_10=0.8530,
    )


def _model_vol():
    """The smile vol at the benchmark strike, read back from the model.

    Hard-coding this number couples the Greek convention tests to whichever
    strike interpolation the smile happened to use when they were written,
    so a deliberate change to the smile shape shows up as a Greek regression.
    """
    return VannaVolga(
        market=_market(), K=1.148, expiry_date='2027-03-24',
        valuation_date='2026-09-25', CallPut='CALL',
        atm_conv='dn', deltaBase=True,
    ).get_vol_for_strike()


BASE_VOL = _model_vol()


def _bs(spot=SPOT, vol=BASE_VOL, deltaBase=True, delta_convention="spot"):
    return BlackScholes(
        S=spot, K=1.148, fwd_pts=88.75, v=vol,
        expiry_date='2027-03-24', valuation_date='2026-09-25',
        CallPut='CALL', BuySell='BUY', Notional=NOTIONAL,
        deltaBase=deltaBase, fwd_divisor=10000, term_discount=0.05,
        delta_convention=delta_convention,
    )


@pytest.mark.parametrize("delta_base", [True, False])
def test_analytic_gamma_is_per_one_percent_relative_spot_move(delta_base):
    h = 1e-6
    up = _bs(spot=SPOT + h, deltaBase=delta_base).delta()
    dn = _bs(spot=SPOT - h, deltaBase=delta_base).delta()
    expected = (up - dn) / (2 * h) * SPOT * 0.01
    assert _bs(deltaBase=delta_base).gamma() == pytest.approx(expected, rel=1e-6)


def test_analytic_vega_is_base_currency_price_sensitivity():
    h = 1e-4
    term_vega = (_bs(vol=BASE_VOL + h).price()
                 - _bs(vol=BASE_VOL - h).price()) / (2 * h)
    assert _bs().vega() == pytest.approx(term_vega / SPOT, rel=1e-6)


def test_analytic_vanna_is_base_vega_change_per_one_percent_spot_move():
    """The reporting-currency conversion is held at the valuation spot.

    ``vega()`` divides by its own spot, so a naive central difference of
    ``vega()`` also differentiates that conversion. That term carries no risk
    and near the money is large enough to flip vanna's sign, so each bumped
    vega is re-expressed at the valuation spot first.
    """
    h = 1e-6
    up = _bs(spot=SPOT + h).vega() * (SPOT + h) / SPOT
    dn = _bs(spot=SPOT - h).vega() * (SPOT - h) / SPOT
    expected = (up - dn) / (2 * h) * SPOT * 0.01
    assert _bs().vanna() == pytest.approx(expected, rel=1e-6)
    assert _bs().vanna() > 0.0


def test_analytic_vanna_matches_the_delta_per_vol_point_cross_derivative():
    """Vanna is a cross derivative, so both quoting routes must agree.

    Bloomberg quotes vanna as the delta change per volatility point; this
    repo quotes the vega change for a 1% relative spot move. They are the
    same mixed partial and differ only by the 0.01 scaling and by the term
    discount factor, because ``vega()`` is discounted.

    ``vanna()`` is deliberately left on the forward-delta cross partial even
    though ``delta()`` now reports a spot delta: vanna and vega feed the
    Vanna-Volga weights, which must not move with a delta *reporting*
    convention. The finite difference is therefore taken in the forward
    convention to compare like with like.
    """
    h = 1e-3
    base_vol = BASE_VOL
    bs = _bs(deltaBase=False)
    d_delta_d_vol = (
        _bs(vol=base_vol + h, deltaBase=False, delta_convention="forward").delta()
        - _bs(vol=base_vol - h, deltaBase=False, delta_convention="forward").delta()
    ) / (2 * h)
    assert bs.vanna() == pytest.approx(
        0.01 * d_delta_d_vol * bs.term_discount_factor, rel=1e-4
    )


def test_analytic_volga_is_the_vol_derivative_of_base_vega():
    h = 1e-4
    expected = (_bs(vol=BASE_VOL + h).vega()
                - _bs(vol=BASE_VOL - h).vega()) / (2 * h)
    assert _bs().volga() == pytest.approx(expected, rel=1e-6)


def _common(market):
    return dict(
        market=market, k=1.148, expiry_date='2027-03-24',
        valuation_date='2026-09-25', CallPut='CALL', BuySell='BUY',
        Notional=NOTIONAL, atm_type='dn', pricing_model='vv', deltaBase=True,
    )


def _reference_price(market, valuation_date):
    """Independent term-currency price via the public model + pricer."""
    vol = VannaVolga(
        market=market, K=1.148, expiry_date='2027-03-24',
        valuation_date=valuation_date, CallPut='CALL',
        atm_conv='dn', deltaBase=True,
    ).get_vol_for_strike()
    return BlackScholes(
        S=market.s, K=1.148, fwd_pts=market.fwd_pts, v=vol,
        expiry_date='2027-03-24', valuation_date=valuation_date,
        CallPut='CALL', BuySell='BUY', Notional=NOTIONAL, deltaBase=True,
        fwd_divisor=market.forward_divisor, term_discount=market.term_discount,
    ).price()


def test_fixed_vega_is_the_term_price_bump_divided_by_spot():
    market = _market()
    greek = FixedVega(**_common(market))
    bumps = greek._perturb_smile(method='atm')
    term_amount = (
        _reference_price(bumps["up"], '2026-09-25')
        - _reference_price(bumps["dn"], '2026-09-25')
    ) / (2 * bumps["eps"])
    assert greek.calculate() == pytest.approx(term_amount / market.s, rel=1e-12)
    assert greek.calculate() != pytest.approx(term_amount, rel=1e-3)


def test_fixed_theta_is_the_term_roll_divided_by_spot():
    market = _market()
    greek = FixedTheta(**_common(market))
    dates = greek._shift_date()
    term_roll = (
        _reference_price(market, dates["shift"])
        - _reference_price(market, dates["base"])
    )
    assert greek.calculate() == pytest.approx(term_roll / market.s, rel=1e-12)
    assert greek.calculate() < 0.0


def test_fixed_greeks_agree_with_the_analytic_reference_scale():
    """The finite-difference Greeks must share the analytic Greeks' units.

    This is a units check, not a precision check. The finite-difference
    Greeks re-strike the whole smile, so they legitimately carry a skew
    term that the analytic Greeks hold fixed, and that term grew once the
    strike interpolation stopped flattening the skew near the money. The
    tolerance only has to be tight enough to catch the regression it
    guards, which is a stray factor of spot, so it stays well inside the
    13.8% that such a rescale would produce.
    """
    market = _market()
    common = _common(market)
    analytic = _bs()
    assert FixedDelta(**common).calculate() == pytest.approx(
        analytic.delta(), rel=8e-2
    )
    assert FixedGamma(**common).calculate() == pytest.approx(
        analytic.gamma(), rel=8e-2
    )
    assert FixedVega(**common).calculate() == pytest.approx(
        analytic.vega(), rel=8e-2
    )


def test_fixed_gamma_is_not_rescaled_by_spot():
    """Guards the 1%-absolute vs 1%-relative spot-move regression.

    The two conventions differ by exactly a factor of spot, which is the
    discrepancy originally seen against market Greeks.
    """
    market = _market()
    greek = FixedGamma(**_common(market))
    bumps = greek._perturb_spot()
    d_delta_d_spot = (
        greek._delta_given_market(bumps["up"])
        - greek._delta_given_market(bumps["dn"])
    ) / (2 * market.s * bumps["eps"])
    assert greek.calculate() == pytest.approx(
        d_delta_d_spot * market.s * 0.01, rel=1e-12
    )
    assert greek.calculate() != pytest.approx(d_delta_d_spot * 0.01, rel=1e-3)

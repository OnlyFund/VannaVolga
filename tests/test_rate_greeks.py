"""FX rate risks: parity, discounting, smile dynamics and 10bp units."""
from math import exp, log, sqrt

import pytest
from scipy.special import ndtr

from apeiron.hyperion.resolve import attach
from Greeks.fixed_greeks import FixedRho, FixedPhi
from Greeks.smile_greeks import SmileGreeks
from helper_functions.fx_discounting import option_discount, CCY, _to_ql
from helper_functions.helper_functions import year_fraction
from market_data import MarketData
from perturbed_market import PerturbedMarket
from pricing.greeks import calculate_extra_greeks
from smile_perturbed_market import PerturbedMarket as SmilePerturbedMarket
from surface_models.smile_extension import SmileExtension


EXPIRY, VALUATION = "2027-09-28", "2026-10-19"
QUOTES = ("v_atm", "rr_25", "rr_10", "bf_25", "bf_10")


def market(**changes):
    # Resolved as the app's markets are (Hyperion), so the core discounts to
    # delivery on the pair's conventions; Atlas reads no calendar itself.
    return attach(MarketData(**dict(
        dict(s=16.6, fwd_pts=4215, forward_divisor=10000, v_atm=12.07,
             rr_25=0.0, rr_10=0.0, bf_25=0.0, bf_10=0.0,
             term_discount=0.05, currency_pair="USDZAR", wrapped_smile=False),
        **changes)), EXPIRY, VALUATION)


def common(mkt, **changes):
    return dict(dict(market=mkt, k=17.0, expiry_date=EXPIRY,
                     valuation_date=VALUATION, Notional=700e6,
                     CallPut="CALL", BuySell="BUY", deltaBase=True), **changes)


@pytest.mark.parametrize("settlement", [False, True])
@pytest.mark.parametrize("currency", ["domestic", "foreign"])
def test_bump_preserves_other_discount_factor_and_quotes(settlement, currency):
    mkt = market(settlement_discounting=settlement)
    bumped = PerturbedMarket(mkt).bump_rate(currency, 0.001, EXPIRY, VALUATION)
    f0 = mkt.s + mkt.fwd_pts / mkt.forward_divisor
    f1 = bumped.s + bumped.fwd_pts / bumped.forward_divisor
    assert bumped.s == mkt.s
    assert all(getattr(bumped, q) == getattr(mkt, q) for q in QUOTES)
    assert bumped.term_discount == pytest.approx(0.051 if currency == "domestic" else 0.05)
    if settlement:
        disc = option_discount("USDZAR", EXPIRY, VALUATION, 0.05)
        d0 = disc.discount_factor
        if currency == "domestic":
            d1 = option_discount("USDZAR", EXPIRY, VALUATION, 0.051).discount_factor
            assert f1 * d1 == pytest.approx(f0 * d0)
        else:
            tau = CCY["USD"].day_count.yearFraction(
                _to_ql(disc.valuation_date), _to_ql(disc.delivery_date))
            b0 = d0 * f0 / mkt.s
            b1 = d0 * f1 / mkt.s
            assert (1 / b1 - 1) / tau - (1 / b0 - 1) / tau == pytest.approx(0.001)
    else:
        t = year_fraction(EXPIRY, VALUATION)
        assert f1 / f0 == pytest.approx(exp((1 if currency == "domestic" else -1) * .001 * t))


@pytest.mark.parametrize("cp,sign", [("CALL", 1), ("PUT", -1)])
@pytest.mark.parametrize("side,position", [("BUY", 1), ("SELL", -1)])
@pytest.mark.parametrize("settlement", [False, True])
def test_flat_smile_rate_risks_match_analytic_black(cp, sign, side, position, settlement):
    mkt = market(settlement_discounting=settlement)
    f = mkt.s + mkt.fwd_pts / mkt.forward_divisor
    t = year_fraction(EXPIRY, VALUATION)
    v = mkt.v_atm / 100
    d1 = (log(f / 17) + v * v * t / 2) / (v * sqrt(t))
    d2 = d1 - v * sqrt(t)
    if settlement:
        disc = option_discount("USDZAR", EXPIRY, VALUATION, .05)
        dd = disc.discount_factor
        tau_d = disc.year_fraction
        tau_f = CCY["USD"].day_count.yearFraction(
            _to_ql(disc.valuation_date), _to_ql(disc.delivery_date))
        db = dd * f / mkt.s
        expected_rho = position * 700e6 * sign * 17 * ndtr(sign * d2) * tau_d * dd**2
        expected_phi = -position * 700e6 * sign * dd * f * ndtr(sign * d1) * tau_f * db
    else:
        dd = exp(-.05 * t)
        expected_rho = position * 700e6 * sign * 17 * t * dd * ndtr(sign * d2)
        expected_phi = -position * 700e6 * sign * f * t * dd * ndtr(sign * d1)
    args = common(mkt, CallPut=cp, BuySell=side)
    for fixed, method, expected in ((FixedRho, "rho", expected_rho),
                                     (FixedPhi, "phi", expected_phi)):
        # A small numerical step isolates the derivative; output stays per 10bp.
        value = fixed(**args).calculate(eps=1e-5)
        assert value == pytest.approx(expected / mkt.s * .001, rel=2e-6)
        smile = getattr(SmileGreeks(mods={"beta_atm": .7, "beta_rr": 1.2}, **args), method)
        assert smile(eps=1e-5) == pytest.approx(value, rel=1e-9)
        assert fixed(**args).calculate() == pytest.approx(value, rel=1e-4)


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("currency", ["domestic", "foreign"])
def test_zero_betas_equal_sticky_delta_on_skew(wrapped, currency):
    mkt = market(rr_25=2.01, rr_10=4.07, bf_25=.343, bf_10=1.148,
                 wrapped_smile=wrapped, broker_strangle=False)
    args = common(mkt)
    fixed = (FixedRho if currency == "domestic" else FixedPhi)(**args).calculate()
    smile = SmileGreeks(mods={}, **args)
    actual = smile.rho() if currency == "domestic" else smile.phi()
    assert actual == pytest.approx(fixed, rel=1e-12)


@pytest.mark.parametrize("currency", ["domestic", "foreign"])
def test_smile_quote_changes_use_original_skews_and_equivalent_forward(currency):
    mkt = market(rr_25=2.01, rr_10=4.07, bf_25=.343, bf_10=1.148)
    mods = {"beta_atm": .7, "beta_rr": 1.2, "beta_bf": .3}
    pm = SmilePerturbedMarket(mkt, mods=mods, K=17, expiry_date=EXPIRY,
                              valuation_date=VALUATION)
    bumped = pm.rate_market(currency, .001)
    sticky = PerturbedMarket(mkt).bump_rate(currency, .001, EXPIRY, VALUATION)
    f0 = mkt.s + mkt.fwd_pts / mkt.forward_divisor
    f1 = sticky.s + sticky.fwd_pts / sticky.forward_divisor
    extension = SmileExtension(mkt, 17, EXPIRY, VALUATION, mods=mods)
    skews = extension.get_local_skews()
    move = log(f1 / f0)
    assert bumped.v_atm - mkt.v_atm == pytest.approx(.7 * skews["local_skew"][1] * move)
    assert bumped.rr_25 - mkt.rr_25 == pytest.approx(1.2 * skews["rr_skew"] * move)
    assert bumped.bf_25 - mkt.bf_25 == pytest.approx(.3 * skews["bf_skew"] * move)
    assert bumped.fwd_pts == sticky.fwd_pts and bumped.s == mkt.s
    assert mods == {"beta_atm": .7, "beta_rr": 1.2, "beta_bf": .3}
    smile = SmileGreeks(mods=mods, **common(mkt))
    price_up = smile._price_given_market(pm.rate_market(currency, .001))
    price_dn = smile._price_given_market(pm.rate_market(currency, -.001))
    actual = smile.rho() if currency == "domestic" else smile.phi()
    assert actual == pytest.approx((price_up - price_dn) / (2 * mkt.s))
    sticky_risk = (FixedRho if currency == "domestic" else FixedPhi)(**common(mkt)).calculate()
    assert abs(actual - sticky_risk) > 1


def test_rate_extras_exposed_and_expiry_is_zero():
    mkt = market()
    row = dict(expiry=EXPIRY, valuation=VALUATION, strike=17, call_put="CALL",
               buy_sell="BUY", notional=700e6, currency_pair="USDZAR")
    values, problems = calculate_extra_greeks(
        mkt, row, ["rho", "phi"], {}, "vv", "dn", True, {})
    assert problems == {}
    # rho in its rate's (term) currency: the calculator's base value at spot.
    assert values["rho"] == pytest.approx(FixedRho(**common(mkt)).calculate() * mkt.s)
    assert values["phi"] == pytest.approx(FixedPhi(**common(mkt)).calculate())
    row["valuation"] = EXPIRY
    values, problems = calculate_extra_greeks(
        mkt, row, ["rho", "phi"], {}, "vv", "dn", True, {})
    assert problems == {} and all(v == 0 for v in values.values())


@pytest.mark.parametrize("step", [float("nan"), float("inf"), True])
def test_invalid_rate_bumps_raise(step):
    with pytest.raises(ValueError, match="finite"):
        PerturbedMarket(market()).bump_rate("domestic", step, EXPIRY, VALUATION)


def test_forward_rate_risks_match_analytic_linear_contract():
    from pricing.fx_forward import FxForwardPosition
    from pricing.greeks import calculate_forward_rate_risks

    mkt = market(settlement_discounting=False)
    position = FxForwardPosition("USDZAR", EXPIRY, -700e6, 17.2)
    result = calculate_forward_rate_risks(position, mkt, VALUATION)
    t = year_fraction(EXPIRY, VALUATION)
    f = mkt.s + mkt.fwd_pts / mkt.forward_divisor
    dd = exp(-.05 * t)
    assert result["rho"] == pytest.approx(-700e6 * 17.2 * t * dd * .001,       # term ccy
                                         rel=1e-6)
    assert result["phi"] == pytest.approx(700e6 * f * t * dd * .001 / mkt.s, rel=1e-6)
    assert result["fixed_rho"] == result["rho"]
    assert result["fixed_phi"] == result["phi"]


def test_rate_greeks_are_in_their_rates_currency_never_usd_never_summed():
    from app.usd_spots import ALL_USD_COLUMNS
    from pricing import greek_catalog as catalog
    assert catalog.label("rho") == "Rho (smile, term ccy)"
    assert catalog.label("fixed_rho") == "Rho (sticky, term ccy)"
    assert catalog.label("phi") == "Phi (smile, base ccy)"
    assert catalog.label("vega") == "Vega (smile)"                  # others unchanged
    converted = {column for pair in ALL_USD_COLUMNS for column in pair}
    for column in ("rho", "fixed_rho", "phi", "fixed_phi"):
        assert column not in converted and catalog.is_directional(column)

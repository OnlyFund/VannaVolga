"""Sticky-strike (Black-Scholes) Greeks, the Bloomberg screen's: delta at the
line's vol at its strike held fixed, with and without the premium, and its
gamma. Checked against finite differences of the Black-Scholes price at that
vol (rates fixed, so the forward moves with spot), the premium identity, and
forwards in the totals."""
from datetime import date

import pytest

from app import pricer
from app.market_set import MarketSet
from golden.app_harness import load_markets
from market_data import Modifications
from pricing import greek_catalog as catalog
from pricing_utils.BlackScholes import BlackScholes

VALUATION = date(2026, 10, 15)
MODS = Modifications(beta_atm=0.7, beta_rr=1.2, beta_bf=0.0)
EXTRAS = ("bs_delta", "bs_delta_prem", "bs_gamma")
LINES = [("EURUSD", "2027-01-15", "CALL", "1.15", "BUY"),
         ("EURUSD", "2027-01-15", "PUT", "1.10", "SELL"),
         ("USDJPY", "2026-12-15", "CALL", "160", "BUY"),        # premium-adjusted pair
         ("USDMXN", "2027-04-15", "PUT", "18.5", "BUY")]        # EM


@pytest.fixture(scope="module")
def markets():
    return MarketSet.from_payloads(load_markets("MARKETS_20261015.json"), VALUATION)


@pytest.fixture(scope="module")
def priced(markets):
    rows = [dict(currency_pair=p, expiry=e, call_put=cp, strike=k, buy_sell=bs,
                 notional="10m", premium=0.0, premium_ccy="TERM") for p, e, cp, k, bs in LINES]
    results, errors = pricer.price_lines(markets, rows, MODS, extras=EXTRAS)
    assert errors == [] and all(not r.get("greek_problems") for r in results)
    return results


def black_scholes(markets, result, spot, keep_forward_ratio):
    """The line at its own vol, at ``spot``: the forward moves with spot
    (rates fixed) or by the same points (points fixed)."""
    market = markets.market(result["currency_pair"]).market_for_expiry(result["expiry"])
    forward = market.s + market.fwd_pts / market.forward_divisor
    points = ((forward / market.s * spot - spot) * market.forward_divisor if keep_forward_ratio
              else market.fwd_pts)
    return BlackScholes(S=spot, K=float(result["strike"]), fwd_pts=points, v=result["vol_pct"],
                        expiry_date=str(result["expiry"]), valuation_date=str(VALUATION),
                        CallPut=result["call_put"], BuySell=result["buy_sell"], Notional=10e6,
                        fwd_divisor=market.forward_divisor, term_discount=market.term_discount,
                        currency_pair=result["currency_pair"], deltaBase=False), market


@pytest.mark.parametrize("line", range(len(LINES)))
def test_bs_delta_is_the_price_derivative_at_the_strike_vol(markets, priced, line):
    result = priced[line]
    _, market = black_scholes(markets, result, 1.0, True)      # only the market
    spot, h = market.s, market.s * 1e-5
    up, _ = black_scholes(markets, result, spot + h, True)
    dn, _ = black_scholes(markets, result, spot - h, True)
    assert result["bs_delta"] == pytest.approx((up.price() - dn.price()) / (2 * h), rel=1e-5)


@pytest.mark.parametrize("line", range(len(LINES)))
def test_bs_gamma_is_the_bs_deltas_change_per_1pct(markets, priced, line):
    result = priced[line]
    _, market = black_scholes(markets, result, 1.0, False)     # only the market
    spot, h = market.s, market.s * 1e-5
    up, _ = black_scholes(markets, result, spot + h, False)
    dn, _ = black_scholes(markets, result, spot - h, False)
    assert result["bs_gamma"] == pytest.approx(
        float(up.delta() - dn.delta()) / (2 * h) * spot * 0.01, rel=1e-4)


def test_delta_incl_premium_is_delta_less_value_over_spot(priced):
    for result in priced:
        assert result["bs_delta_prem"] == pytest.approx(result["bs_delta"] - result["mtm_base"],
                                                        rel=1e-9)
    sold_put = priced[1]
    assert sold_put["bs_delta"] > 0 and sold_put["bs_delta_prem"] > sold_put["bs_delta"]


def test_forwards_count_their_spot_delta_in_the_totals(markets, priced):
    hedge = pricer.value_forward(markets, "EURUSD", date(2027, 1, 15), -2e6, 1.13)
    assert hedge["bs_delta"] == hedge["bs_delta_prem"] == hedge["spot_delta"]
    assert hedge["bs_gamma"] == 0.0
    totals = pricer.aggregate_by_pair(priced, [hedge], greeks=list(EXTRAS)).set_index(
        "currency_pair")
    options = [r for r in priced if r["currency_pair"] == "EURUSD"]
    for name in ("bs_delta", "bs_delta_prem"):
        assert totals.at["EURUSD", name] == pytest.approx(
            sum(r[name] for r in options) + hedge["spot_delta"])
    assert totals.at["EURUSD", "bs_gamma"] == pytest.approx(sum(r["bs_gamma"] for r in options))
    assert "fixed_bs_delta" not in totals


def test_one_column_each_directional_and_zero_once_expired(markets):
    assert catalog.columns(["delta", "bs_delta"]) == ["fixed_delta", "delta", "bs_delta"]
    assert catalog.label("bs_delta") == "BS delta"
    assert catalog.label("bs_delta_prem") == "BS delta incl. premium"
    assert catalog.is_directional("bs_delta") and not catalog.is_directional("bs_gamma")
    rows = [dict(currency_pair="EURUSD", expiry="2026-10-15", call_put="CALL", strike="1.10",
                 buy_sell="BUY", notional="1m", premium=0.0, premium_ccy="TERM")]
    from pricing.greeks import calculate_extra_greeks
    market = markets.market("EURUSD").market_for_expiry(date(2027, 1, 15))
    row = dict(rows[0], expiry=VALUATION, valuation=VALUATION, notional=1e6)
    values, problems = calculate_extra_greeks(market, row, EXTRAS, {}, "vv", "dn", True, MODS)
    assert problems == {} and all(v == 0.0 for v in values.values())


def test_betas_one_one_one_are_close_to_sticky_strike_beta_atm_alone_is_not(markets):
    # The documented regimes: the smile delta at betas 1/1/1 is within a few
    # percent of the BS delta (premium matched to the pair's convention);
    # beta ATM 1 alone is further away on every line.
    from helper_functions.fx_conventions import get_pair
    rows = [dict(currency_pair=p, expiry=e, call_put=cp, strike=k, buy_sell="BUY",
                 notional="10m", premium=0.0, premium_ccy="TERM") for p, e, cp, k, _ in LINES]

    def priced_with(betas):
        mods = Modifications(beta_atm=betas[0], beta_rr=betas[1], beta_bf=betas[2])
        results, errors = pricer.price_lines(markets, rows, mods, extras=EXTRAS)
        assert errors == []
        return results
    for strike_like, atm_only in zip(priced_with((1, 1, 1)), priced_with((1, 0, 0))):
        adjusted = get_pair(strike_like["currency_pair"]).premium_adjusted
        bs = strike_like["bs_delta_prem" if adjusted else "bs_delta"]
        near = abs(strike_like["spot_delta"] - bs)
        assert near < 0.025 * abs(bs), strike_like["currency_pair"]
        assert abs(atm_only["spot_delta"] - bs) > near
    assert "1/1/1" in catalog.REGIMES["sticky strike"]

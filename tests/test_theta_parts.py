"""Theta split into static decay, vol roll and carry: each family's parts add
up to its theta, sticky theta has no carry, static is the plain one-day decay
at today's vol and forward, a hedge forward's roll is carry, and the parts
convert to USD as theta does."""
from datetime import date

import pytest

from app import pricer
from app.market_set import MarketSet
from app.trades import service
from app.trades.model import FX_FORWARD, FX_OPTION, Trade
from app.ui.trades_block import pair_totals
from golden.app_harness import load_markets
from market_data import Modifications
from pricing.greek_catalog import PREMIUM_CCY_GREEKS, THETA_PARTS

VALUATION = date(2026, 10, 15)
MODS = Modifications(beta_atm=0.7, beta_rr=1.2, beta_bf=0.0)


@pytest.fixture(scope="module")
def markets():
    return MarketSet.from_payloads(load_markets("MARKETS_20261015.json"), VALUATION)


def option(trade_id, pair, strike, call_put, buy_sell, expiry, premium_ccy, option_type=None):
    return Trade(product_type=FX_OPTION, currency_pair=pair, trade_date=VALUATION,
                 expiry=expiry, buy_sell=buy_sell, notional=10e6, strike=strike,
                 call_put=call_put, premium=0.0, premium_ccy=premium_ccy,
                 option_type=option_type, trade_id=trade_id).normalized()


BOOK = [option("E1", "EURUSD", 1.12, "CALL", "BUY", date(2027, 1, 15), "USD"),
        option("E2", "EURUSD", 1.08, "PUT", "SELL", date(2026, 12, 1), "USD"),
        option("E3", "EURUSD", 1.15, "CALL", "BUY", date(2027, 1, 15), "USD", "digital"),
        option("J1", "USDJPY", 150.0, "PUT", "BUY", date(2027, 4, 15), "JPY"),
        option("M1", "USDMXN", 19.0, "CALL", "SELL", date(2027, 1, 14), "USD"),
        Trade(product_type=FX_FORWARD, currency_pair="EURUSD", trade_date=VALUATION,
              expiry=date(2027, 1, 15), buy_sell="SELL", notional=5e6, contract_rate=1.13,
              trade_id="F1").normalized()]


@pytest.fixture(scope="module")
def priced(markets):
    results, forwards, errors = service.price_trades(markets, BOOK, MODS, extras=THETA_PARTS)
    assert errors == [] and not any(r.get("greek_problems") for r in results)
    return results, forwards


def test_each_familys_parts_add_up_to_its_theta(priced):
    results, _ = priced
    for result in results:
        for family in ("", "fixed_"):
            parts = sum(result[f"{family}{name}"] for name in THETA_PARTS)
            assert parts == pytest.approx(result[f"{family}theta"], rel=1e-9, abs=1e-6)
        assert result["fixed_theta_carry"] == 0.0
        assert result["fixed_theta_static"] == result["theta_static"]


def test_static_is_the_one_day_decay_at_todays_vol_and_forward(markets, priced):
    from pricing.pricing import discount_kwargs
    from pricing_utils.BlackScholes import BlackScholes
    from app.curve_roll import next_business_day
    results, _ = priced
    result = next(r for r in results if r["trade_id"] == "E1")
    context = markets.market("EURUSD")
    market = context.market_for_expiry(date(2027, 1, 15))

    def price(valuation):
        return BlackScholes(
            S=market.s, K=1.12, fwd_pts=market.fwd_pts, v=result["vol_pct"],
            expiry_date="2027-01-15", valuation_date=str(valuation), CallPut="CALL",
            BuySell="BUY", Notional=10e6, deltaBase=market.deltaBase,
            fwd_divisor=market.forward_divisor, term_discount=market.term_discount,
            **discount_kwargs(market, "2027-01-15", str(valuation))).price()
    decay = price(next_business_day("EURUSD", VALUATION)) - price(VALUATION)
    assert result["theta_static"] == pytest.approx(decay, rel=1e-9)     # USD premium: term


def test_pair_totals_count_the_hedge_forwards_roll_as_carry(priced):
    results, forwards = priced
    totals = pair_totals(results, forwards).set_index("currency_pair")
    for pair in ("EURUSD", "USDJPY", "USDMXN"):
        row = totals.loc[pair]
        assert sum(row[name] for name in THETA_PARTS) == pytest.approx(row["theta"], abs=1e-6)
    options_carry = sum(r["theta_carry"] for r in results if r["currency_pair"] == "EURUSD")
    forward_roll = sum(f["theta"] for f in forwards)
    assert totals.loc["EURUSD", "theta_carry"] == pytest.approx(options_carry + forward_roll)


def test_the_parts_are_premium_currency_greeks_with_usd_columns():
    from app.usd_spots import THETA_USD_COLUMNS
    assert set(THETA_PARTS) <= set(PREMIUM_CCY_GREEKS)
    converted = {source for source, _ in THETA_USD_COLUMNS}
    for name in THETA_PARTS:
        assert {name, f"fixed_{name}"} <= converted


def test_no_next_day_market_reports_the_split_unavailable(markets):
    from pricing.greeks import calculate_extra_greeks
    import pandas as pd
    row = pd.Series(dict(currency_pair="EURUSD", strike=1.12, expiry=date(2027, 1, 15),
                         valuation=VALUATION, call_put="CALL", buy_sell="BUY",
                         notional=1e6, premium_ccy="USD"))
    market = markets.market("EURUSD").market_for_expiry(date(2027, 1, 15))
    values, problems = calculate_extra_greeks(market, row, THETA_PARTS, {"theta": 1.0,
                                              "fixed_theta": 1.0}, "vv", "dn", True, MODS)
    assert all(values[name] is None for name in THETA_PARTS)
    assert set(problems) == set(THETA_PARTS)

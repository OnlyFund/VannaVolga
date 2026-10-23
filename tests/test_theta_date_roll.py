"""Theta is exactly the date roll (ROADMAP item 6, step 3).

A line's theta is its value on the market rolled to the next business day
(``app.rolled_market``, the market a later valuation date prices on) minus
its value today, with spot, strike, expiry and delivery fixed.
"""
from datetime import date

import pytest

from app import pricer
from app.curve_roll import next_business_day
from app.market_context import PricingSettings
from app.market_set import MarketSet
from app.rolled_market import roll_context
from golden.app_harness import load_markets

ROLLED = PricingSettings(theta_vol_roll="rolled")
from market_data import Modifications

pytestmark = pytest.mark.filterwarnings("ignore")
MODS = Modifications(beta_atm=0.7, beta_rr=1.2, beta_bf=0.0)
TODAY = date(2026, 10, 15)                      # Thursday


@pytest.fixture(scope="module")
def payloads():
    return load_markets("MARKETS_20261015.json")


def _value(row):
    """MTM in the currency theta is reported in."""
    return row["mtm_base"] if row["theta_ccy"] == row["currency_pair"][:3] else row["mtm_term"]


def _line(pair, tenor, strike, call_put):
    return dict(currency_pair=pair, strike=strike, expiry=tenor, call_put=call_put, buy_sell="BUY",
                notional="1m", premium=0.0, premium_ccy=pair[3:])


def _fixed(line, priced):
    expiry = priced["expiry"]
    return dict(line, strike=priced["strike"],
                expiry=expiry.strftime("%m/%d/%Y") if hasattr(expiry, "strftime") else str(expiry))


def _theta_and_roll(today_set, tomorrow_set, line):
    (today,), errors = pricer.price_lines(today_set, [dict(line)], MODS)
    assert errors == []
    (tomorrow,), errors = pricer.price_lines(tomorrow_set, [_fixed(line, today)], MODS)
    assert errors == []
    return today["theta"], _value(tomorrow) - _value(today)


@pytest.mark.parametrize("pair", ["EURUSD", "USDJPY", "USDMXN"])
@pytest.mark.parametrize("tenor,strike,call_put", [
    ("1W", "atm", "CALL"), ("1M", "25d", "PUT"), ("3M", "10d", "CALL"), ("1Y", "atm", "PUT")])
def test_theta_is_the_value_on_the_market_rolled_to_the_next_business_day(payloads, pair, tenor,
                                                                          strike, call_put):
    today_set = MarketSet.from_payloads(payloads, TODAY, settings=ROLLED)
    tomorrow_set = MarketSet.from_payloads(payloads, next_business_day(pair, TODAY), settings=ROLLED)
    theta, roll = _theta_and_roll(today_set, tomorrow_set, _line(pair, tenor, strike, call_put))
    assert theta == pytest.approx(roll, rel=1e-8, abs=1e-8)


@pytest.mark.parametrize("pair", ["EURUSD", "USDJPY"])
def test_over_a_weekend_too(payloads, pair):
    """Friday's market (itself rolled from Thursday) rolls to Monday."""
    friday = MarketSet.from_payloads(payloads, date(2026, 10, 16), settings=ROLLED)
    monday = date(2026, 10, 19)
    assert next_business_day(pair, friday.valuation_date) == monday
    rolled = MarketSet(monday, [roll_context(friday.market(pair), monday)[0]])
    for line in (_line(pair, "1M", "atm", "CALL"), _line(pair, "6M", "25d", "PUT")):
        theta, roll = _theta_and_roll(friday, rolled, line)
        assert theta == pytest.approx(roll, rel=1e-8, abs=1e-8)


def test_the_earlier_rolls_remain_for_comparison(payloads):
    line = _line("EURUSD", "1M", "atm", "CALL")
    thetas = {}
    for mode in ("rolled", "forward", "tenor"):
        markets = MarketSet.from_payloads(payloads, TODAY, settings=PricingSettings(theta_vol_roll=mode))
        (row,), errors = pricer.price_lines(markets, [dict(line)], MODS)
        assert errors == []
        thetas[mode] = row["theta"]
    assert len(set(round(value, 6) for value in thetas.values())) == 3


def test_a_market_that_cannot_roll_takes_the_forward_vol_theta_and_says_why(payloads, monkeypatch):
    import app.rolled_market as module
    from app.rolled_market import RollError

    def cannot(context, to_date):
        raise RollError("no conventions to roll with")
    monkeypatch.setattr(module, "roll_context", cannot)
    line = _line("EURUSD", "1M", "atm", "CALL")
    markets = MarketSet.from_payloads(payloads, TODAY, settings=ROLLED)
    (row,), errors = pricer.price_lines(markets, [dict(line)], MODS)
    forward = MarketSet.from_payloads(payloads, TODAY, settings=PricingSettings(theta_vol_roll="forward"))
    (expected,), _ = pricer.price_lines(forward, [dict(line)], MODS)
    assert errors == [] and row["theta"] == pytest.approx(expected["theta"], rel=1e-12)
    records = markets.market("EURUSD").diagnostics.records()
    assert any("does not roll to 2026-10-16" in str(record) for record in records)

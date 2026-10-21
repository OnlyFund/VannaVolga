"""Atlas untangling, sub-step 1: market objects carry their resolved dates and
conventions (``atlas.resolved``), filled by Hyperion. Nothing reads them yet,
so pricing is unchanged; here every carried value must equal exactly what the
core computes for itself today, for every pair, pillar, broken and short
expiry, the next-day (theta) market and value-dated deals."""
from datetime import date

import pandas as pd
import pytest

from apeiron.atlas.resolved import ResolvedInputs, simple_discount_factor
from apeiron.hyperion.resolve import attach, resolve_inputs
from app.market_set import MarketSet
from app.curve_roll import next_business_day
from golden.app_harness import load_markets
from helper_functions import fx_discounting
from resolved_smile_inputs import MarketSnapshot, resolve_conventions

VALUATION = date(2026, 10, 15)


@pytest.fixture(scope="module")
def markets():
    return MarketSet.from_payloads(load_markets("MARKETS_20261015.json"), VALUATION)


def expiries(context):
    pillars = [pd.Timestamp(e).date() for e in context.pillar_expiries()]
    broken = [date(2026, 10, 16), date(2026, 10, 19), date(2026, 11, 6), date(2027, 1, 20),
              date(2027, 3, 5), date(2027, 6, 30)]
    return sorted(set(pillars + [d for d in broken if d < pillars[-1]]))


def expected(market, expiry, valuation):
    discount = fx_discounting.option_discount(market.currency_pair, expiry, valuation,
                                              market.term_discount)
    conventions = resolve_conventions(MarketSnapshot.capture(market), expiry, valuation)
    return discount, conventions


@pytest.mark.parametrize("pair", ["EURUSD", "USDJPY", "USDMXN"])
def test_every_option_market_carries_what_the_core_resolves_itself(markets, pair):
    context = markets.market(pair)
    checked = 0
    for expiry in expiries(context):
        market = context.market_for_expiry(expiry)
        resolved = market.resolved
        assert isinstance(resolved, ResolvedInputs), (pair, expiry, market.resolution_note)
        discount, conventions = expected(market, expiry, VALUATION)
        assert resolved.delivery_date == discount.delivery_date
        assert resolved.year_fraction == discount.year_fraction            # exactly
        assert resolved.day_count == discount.day_count
        assert resolved.delta_convention == conventions.delta_convention
        assert resolved.premium_adjusted == conventions.premium_adjusted
        assert resolved.atm_convention == conventions.atm_convention
        assert resolved.matches(expiry, VALUATION)
        # The core's own discount factor follows from the rate and year fraction.
        assert simple_discount_factor(market.term_discount, resolved.year_fraction) == \
            discount.discount_factor
        checked += 1
    assert checked >= 8


@pytest.mark.parametrize("pair", ["EURUSD", "USDJPY"])
def test_the_next_day_market_is_resolved_for_the_next_day(markets, pair):
    context = markets.market(pair)
    tomorrow = next_business_day(pair, VALUATION)
    for expiry in expiries(context)[2:6]:
        market = context.market_for_expiry(expiry, as_of=tomorrow, allow_short_end=True)
        discount, conventions = expected(market, expiry, tomorrow)
        assert market.resolved.matches(expiry, tomorrow)
        assert market.resolved.year_fraction == discount.year_fraction
        assert market.resolved.delta_convention == conventions.delta_convention


def test_a_value_dated_deal_is_resolved_to_its_own_value_date(markets):
    context = markets.market("EURUSD")
    for value_date in (date(2026, 10, 19), date(2027, 3, 5)):
        market = context.market_for_delivery(value_date)
        resolved = market.resolved
        assert resolved.delivery_date == value_date and resolved.delta_convention is None
        assert resolved.year_fraction == fx_discounting.discount_year_fraction(
            "EURUSD", VALUATION, value_date)


def test_copies_and_bumps_keep_the_inputs(markets):
    market = markets.market("EURUSD").market_for_expiry(date(2027, 1, 15))
    bumped = market.copy_with(s=market.s * 1.01, term_discount=market.term_discount + 0.001)
    assert bumped.resolved == market.resolved


def test_the_calibration_cache_key_ignores_them(markets):
    from app.smile_service import cache_key
    market = markets.market("EURUSD").market_for_expiry(date(2027, 1, 15))
    without = market.copy_with(resolved=None)
    assert cache_key(market, "2027-01-15", VALUATION, "vv", False, .5) == \
        cache_key(without, "2027-01-15", VALUATION, "vv", False, .5)


def test_a_market_that_cannot_resolve_keeps_none_and_says_why(markets):
    market = markets.market("EURUSD").market_for_expiry(date(2027, 1, 15))
    broken = attach(market.copy_with(resolved=None, currency_pair="XXXYYY"),
                    date(2027, 1, 15), VALUATION)
    assert broken.resolved is None and "not resolved" in broken.resolution_note
    assert attach(market.copy_with(resolved=None, currency_pair=None),
                  date(2027, 1, 15), VALUATION).resolved is None


@pytest.mark.parametrize("bad, message", [
    (dict(delivery_date=date(2026, 10, 1)), "delivery must not precede"),
    (dict(expiry_date=date(2027, 2, 1)), "expiry must not follow"),
    (dict(year_fraction=float("nan")), "finite"),
    (dict(delta_convention="sideways"), "delta_convention"),
    (dict(premium_adjusted="yes"), "premium_adjusted"),
    (dict(atm_convention="xx"), "atm_convention"),
])
def test_resolved_inputs_validate_themselves(bad, message):
    good = dict(valuation_date=VALUATION, expiry_date=date(2027, 1, 15),
                delivery_date=date(2027, 1, 19), year_fraction=0.26, day_count="Actual/360",
                delta_convention="spot", premium_adjusted=False, atm_convention="dn")
    ResolvedInputs(**good)
    with pytest.raises(ValueError, match=message):
        ResolvedInputs(**dict(good, **bad))
    with pytest.raises(Exception):
        ResolvedInputs(**good).year_fraction = 1.0                  # immutable


@pytest.mark.parametrize("rate, fraction", [(0.05, 0.25), (-0.007, 1.02), (0.0, 0.0),
                                            (0.43, 0.004), (0.0429, 0.2611111)])
def test_atlas_discounting_is_the_same_arithmetic(rate, fraction):
    assert simple_discount_factor(rate, fraction) == \
        fx_discounting.simple_discount_factor(rate, fraction)
    with pytest.raises(ValueError):
        simple_discount_factor(0.05, -0.1)

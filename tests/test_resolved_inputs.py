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


# -- sub-step 2: the core discounts from the carried inputs ---------------------

def test_core_discounting_equals_the_calendar_path(markets):
    from apeiron.atlas import discounting as atlas
    context = markets.market("EURUSD")
    shift = (pd.Timestamp(VALUATION) + pd.offsets.BDay(1)).date()
    for expiry in expiries(context):
        market = context.market_for_expiry(expiry)
        for valuation in (VALUATION, shift):                 # today and sticky theta's date
            if expiry > valuation:
                assert atlas.discount_kwargs(market, expiry, valuation) == \
                    fx_discounting.discount_kwargs(market, expiry, valuation)
    deal = context.market_for_delivery(date(2027, 3, 5))
    assert atlas.discount_kwargs(deal, date(2027, 3, 5), VALUATION, delivery=date(2027, 3, 5)) == \
        fx_discounting.discount_kwargs(deal, date(2027, 3, 5), VALUATION, delivery=date(2027, 3, 5))


def test_without_resolved_inputs_the_core_uses_legacy_discounting_and_counts_it(markets):
    from apeiron.atlas import discounting as atlas
    market = markets.market("EURUSD").market_for_expiry(date(2027, 1, 15))
    before = sum(atlas.FALLBACKS.values())
    assert atlas.discount_kwargs(market.copy_with(resolved=None), "2027-01-15", VALUATION) == {}
    assert atlas.discount_kwargs(market, "2027-02-15", VALUATION) == {}         # another expiry
    assert sum(atlas.FALLBACKS.values()) == before + 2
    assert atlas.discount_kwargs(market.copy_with(settlement_discounting=False),
                                 "2027-01-15", VALUATION) == {}


def test_rate_bumps_from_carried_fractions_equal_the_calendar_ones(markets):
    from perturbed_market import PerturbedMarket
    market = markets.market("EURUSD").market_for_expiry(date(2027, 1, 15))
    resolved = market.resolved
    assert resolved.base_year_fraction == float(
        fx_discounting.CCY["EUR"].day_count.yearFraction(
            fx_discounting._to_ql(VALUATION), fx_discounting._to_ql(resolved.delivery_date)))
    for currency in ("domestic", "foreign"):
        bumped = PerturbedMarket(market).bump_rate(currency, 0.001, "2027-01-15", VALUATION)
        unresolved = PerturbedMarket(market.copy_with(resolved=None)).bump_rate(
            currency, 0.001, "2027-01-15", VALUATION)
        assert bumped.fwd_pts != unresolved.fwd_pts              # conventions vs legacy
        # Domestic: forward scaled by D_old / D_new on the term basis.
        if currency == "domestic":
            old = simple_discount_factor(market.term_discount, resolved.year_fraction)
            new = simple_discount_factor(market.term_discount + 0.001, resolved.year_fraction)
            forward = market.s + market.fwd_pts / market.forward_divisor
            assert bumped.s + bumped.fwd_pts / bumped.forward_divisor == pytest.approx(
                forward * old / new, rel=1e-14)


# -- sub-step 3: conventions from the carried inputs ---------------------------

def _conventions_by_calendar(market, expiry, valuation):
    """The smile's conventions as the core built them before sub-step 3."""
    import QuantLib as ql
    from helper_functions import fx_conventions as conventions
    from resolved_smile_inputs import ResolvedConventions
    pair = conventions.get_pair(market.currency_pair)
    discount = fx_discounting.option_discount(pair.name, expiry, valuation, market.term_discount)
    kind = conventions.delta_type(pair.name, fx_discounting._to_ql(expiry),
                                  fx_discounting._to_ql(valuation))
    forward_delta = (pair.fwd_delta_beyond == "0D"
                     or kind in (ql.DeltaVolQuote.Fwd, ql.DeltaVolQuote.PaFwd))
    forward = market.s + market.fwd_pts / pair.fwd_divisor
    return ResolvedConventions(
        pair.name, market.s, valuation, expiry, discount.delivery_date,
        (expiry - valuation).days / 365.0, forward, pair.fwd_divisor, discount.discount_factor,
        discount.discount_factor * forward / market.s, "forward" if forward_delta else "spot",
        pair.premium_adjusted, pair.atm.lower(),
        360 if conventions.CCY[pair.quote].day_count.name() == "Actual/360" else 365)


@pytest.mark.parametrize("pair", ["EURUSD", "USDJPY", "USDMXN"])
def test_smile_conventions_equal_the_calendar_ones_exactly(markets, pair):
    context = markets.market(pair)
    shift = (pd.Timestamp(VALUATION) + pd.offsets.BDay(1)).date()
    for expiry in expiries(context):
        market = context.market_for_expiry(expiry)
        for valuation in (VALUATION, shift):
            if expiry <= valuation:
                continue
            got = resolve_conventions(MarketSnapshot.capture(market), expiry, valuation)
            assert got == _conventions_by_calendar(market, expiry, valuation), (pair, expiry)


def test_without_resolved_inputs_the_smile_says_its_conventions_are_unavailable(markets):
    from resolved_smile_inputs import PreparationError
    market = markets.market("EURUSD").market_for_expiry(date(2027, 1, 15))
    with pytest.raises(PreparationError, match="not resolved"):
        resolve_conventions(MarketSnapshot.capture(market.copy_with(resolved=None)),
                            date(2027, 1, 15), VALUATION)


# -- sub-step 4: the core reads no calendar at all -----------------------------

@pytest.mark.parametrize("value", [
    "2026-10-15", " 2026-10-15 ", "15/10/2026", "10/25/2026", "15-Oct-2026", "15Oct2026",
    "15Oct26", date(2026, 10, 15), pd.Timestamp("2026-10-15 13:45"),
    __import__("datetime").datetime(2026, 10, 15, 9, 30), "2026-02-29", "", "not a date",
])
def test_atlas_parses_dates_exactly_as_the_calendar_layer(value):
    from apeiron.atlas.dates import to_date
    try:
        expected = fx_discounting._to_date(value)
    except ValueError as exc:
        with pytest.raises(ValueError, match=str(exc).split(" ")[0]):
            to_date(value)
        return
    assert to_date(value) == expected and type(to_date(value)) is date


def test_atlas_parses_quantlib_dates_without_importing_quantlib():
    import QuantLib as ql
    from apeiron.atlas.dates import to_date
    for day in (ql.Date(15, 10, 2026), ql.Date(29, 2, 2028), ql.Date(31, 12, 2030)):
        assert to_date(day) == fx_discounting._to_date(day)


@pytest.mark.parametrize("pair", ["EURUSD", "USDJPY", "USDMXN", "USDCAD", "EURGBP"])
def test_the_legacy_delta_convention_equals_the_resolved_one(markets, pair):
    for expiry in (date(2026, 11, 16), date(2027, 4, 15), date(2027, 10, 15), date(2028, 6, 1)):
        assert fx_discounting.option_delta_convention(pair, expiry, VALUATION) == \
            resolve_inputs(pair, expiry, VALUATION).delta_convention, (pair, expiry)


def test_the_core_modules_import_no_calendar_or_convention_table():
    from pathlib import Path
    from apeiron.scan import imports_of, library_modules
    modules = library_modules(Path(__file__).resolve().parents[1])
    calendars = {"helper_functions.fx_discounting", "helper_functions.fx_conventions"}
    for module in ("pricing_utils.BlackScholes", "pricing_utils.black_scholes_numba",
                   "wrapped_vol_resolver", "resolved_smile_inputs", "VannaVolga_ImpliedVol",
                   "pricing.greeks", "apeiron.atlas.dates", "apeiron.atlas.discounting"):
        assert not imports_of(modules[module], module, modules) & calendars, module

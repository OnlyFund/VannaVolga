"""Tests for delivery-date settlement discounting.

The library historically discounted with ``exp(-r * t)`` on an ACT/365 clock
running to the *expiry* date. Market convention discounts to the *delivery*
date (expiry plus the pair's spot lag on the joint calendar) using a *simple*
rate on the term currency's own day-count basis.

The feature is opt-in. Every test here also guards the far more important
property that with the feature off, nothing changes at all.
"""
import math

import pytest

from helper_functions import fx_discounting as fxd
from helper_functions.fx_discounting import (
    CONVENTIONS_AVAILABLE,
    delivery_date,
    discount_kwargs,
    discount_year_fraction,
    equivalent_continuous_rate,
    option_discount,
    safe_option_discount,
    simple_discount_factor,
)
from market_data import MarketData
from pricing_utils.BlackScholes import BlackScholes
from pricing_utils.black_scholes_numba import black_scholes_values

pytestmark = pytest.mark.skipif(
    not CONVENTIONS_AVAILABLE,
    reason="QuantLib / fx_conventions unavailable",
)

RATE = 0.0427
EXPIRY = "2027-03-24"
VALUATION = "2026-09-25"

BASE_MARKET = dict(
    s=1.1386, fwd_pts=88.75, v_atm=5.92, rr_25=-0.518, rr_10=-0.965,
    bf_25=0.235, bf_10=0.853, term_discount=RATE,
)

OPTION = dict(
    S=1.1386, K=1.1480, fwd_pts=88.75, v=5.92, expiry_date=EXPIRY,
    valuation_date=VALUATION, CallPut="CALL", BuySell="BUY",
    Notional=1_000_000, deltaBase=True,
)


@pytest.fixture
def legacy_market():
    return MarketData(**BASE_MARKET)


@pytest.fixture
def conv_market():
    return MarketData(currency_pair="EURUSD", settlement_discounting=True, **BASE_MARKET)


@pytest.fixture
def overrides(conv_market):
    return discount_kwargs(conv_market, EXPIRY, VALUATION)


# --------------------------------------------------------------------------
# Delivery dates
# --------------------------------------------------------------------------
class TestDeliveryDate:
    @pytest.mark.parametrize("pair,expected", [
        ("EURUSD", "2027-03-30"),
        ("USDJPY", "2027-03-26"),
        ("EURGBP", "2027-03-30"),
    ])
    def test_spot_lag_two_pairs(self, pair, expected):
        assert delivery_date(pair, EXPIRY).isoformat() == expected

    @pytest.mark.parametrize("pair", ["USDCAD", "USDTRY"])
    def test_t_plus_one_pairs(self, pair):
        """CAD and TRY settle T+1, not T+2."""
        assert delivery_date(pair, EXPIRY).isoformat() == "2027-03-25"

    def test_delivery_is_strictly_after_expiry(self):
        for pair in ("EURUSD", "USDJPY", "USDCAD", "EURGBP", "AUDUSD"):
            assert delivery_date(pair, EXPIRY).isoformat() > EXPIRY

    def test_lag_can_exceed_two_calendar_days(self):
        """A weekend between expiry and delivery stretches the calendar gap."""
        gap = (delivery_date("EURUSD", EXPIRY) - option_discount(
            "EURUSD", EXPIRY, VALUATION, RATE).expiry_date).days
        assert gap == 6

    def test_christmas_stretches_the_lag(self):
        """2026-12-24 expiry rolls past Christmas and the weekend."""
        assert delivery_date("EURUSD", "2026-12-24").isoformat() == "2026-12-29"

    def test_unknown_pair_raises(self):
        with pytest.raises(Exception):
            delivery_date("XAUUSD", EXPIRY)

    def test_inverted_pair_raises(self):
        with pytest.raises(Exception):
            delivery_date("USDEUR", EXPIRY)


# --------------------------------------------------------------------------
# Day-count basis
# --------------------------------------------------------------------------
class TestDayCount:
    def test_usd_term_uses_act360(self):
        result = option_discount("EURUSD", EXPIRY, VALUATION, RATE)
        assert "360" in result.day_count
        assert result.year_fraction == pytest.approx(186 / 360.0, rel=1e-12)

    def test_gbp_term_uses_act365(self):
        """EURGBP has the same delivery date but a 365 basis."""
        eurusd = option_discount("EURUSD", EXPIRY, VALUATION, RATE)
        eurgbp = option_discount("EURGBP", EXPIRY, VALUATION, RATE)
        assert eurgbp.delivery_date == eurusd.delivery_date
        assert "365" in eurgbp.day_count
        assert eurgbp.year_fraction == pytest.approx(186 / 365.0, rel=1e-12)
        assert eurgbp.year_fraction < eurusd.year_fraction

    def test_year_fraction_helper_matches_full_result(self):
        result = option_discount("EURUSD", EXPIRY, VALUATION, RATE)
        assert discount_year_fraction(
            "EURUSD", VALUATION, result.delivery_date) == pytest.approx(
            result.year_fraction, rel=1e-15)

    def test_year_fraction_runs_to_delivery_not_expiry(self):
        """ACT/360 to expiry would be 180/360; to delivery it is 186/360."""
        to_expiry = discount_year_fraction("EURUSD", VALUATION, EXPIRY)
        to_delivery = discount_year_fraction(
            "EURUSD", VALUATION, delivery_date("EURUSD", EXPIRY))
        assert to_expiry == pytest.approx(180 / 360.0, rel=1e-12)
        assert to_delivery == pytest.approx(186 / 360.0, rel=1e-12)


# --------------------------------------------------------------------------
# Simple discounting maths
# --------------------------------------------------------------------------
class TestSimpleDiscount:
    def test_matches_closed_form(self):
        assert simple_discount_factor(RATE, 0.5) == pytest.approx(
            1.0 / (1.0 + RATE * 0.5), rel=1e-15)

    def test_zero_rate_is_unit(self):
        assert simple_discount_factor(0.0, 0.5) == 1.0

    def test_zero_time_is_unit(self):
        assert simple_discount_factor(RATE, 0.0) == 1.0

    def test_simple_exceeds_continuous_for_same_tau(self):
        tau = 0.5
        assert simple_discount_factor(RATE, tau) > math.exp(-RATE * tau)

    def test_negative_rate_discount_exceeds_one(self):
        assert simple_discount_factor(-0.01, 0.5) > 1.0

    def test_carry_rate_is_rate_times_df(self):
        r = option_discount("EURUSD", EXPIRY, VALUATION, RATE)
        assert r.carry_rate == pytest.approx(r.rate * r.discount_factor, rel=1e-15)

    def test_carry_rate_is_bounded_by_the_quoted_rate(self):
        """carry = r/(1+r*tau) <= r for every positive tenor.

        This is precisely why the carry rate is passed instead of an
        equivalent continuously compounded rate.
        """
        for expiry in ("2026-09-28", "2026-10-02", "2026-12-24", "2027-03-24", "2027-09-24"):
            result = option_discount("EURUSD", expiry, VALUATION, RATE)
            assert 0.0 < result.carry_rate <= RATE

    def test_equivalent_rate_blows_up_at_short_tenors(self):
        """The rejected shortcut: delivery lag dominates a near-dated option.

        Three calendar days to expiry but seven to delivery, so backing a
        continuously compounded rate out of the delivery discount factor over
        the *expiry* year fraction inflates it far above the quoted rate.
        """
        near = option_discount("EURUSD", "2026-09-28", VALUATION, RATE)
        r_eq = equivalent_continuous_rate(near.discount_factor, 3 / 365.0)
        assert r_eq > 1.5 * RATE
        assert near.carry_rate < RATE

    def test_equivalent_rate_round_trips(self):
        r_eq = equivalent_continuous_rate(0.97, 0.5)
        assert math.exp(-r_eq * 0.5) == pytest.approx(0.97, rel=1e-14)


# --------------------------------------------------------------------------
# Golden values
# --------------------------------------------------------------------------
class TestGoldenValues:
    def test_six_month_eurusd_discount_factor(self):
        result = option_discount("EURUSD", EXPIRY, VALUATION, RATE)
        assert result.discount_factor == pytest.approx(0.9784145445, abs=1e-9)

    def test_convention_costs_about_seven_basis_points(self):
        legacy = math.exp(-RATE * (180 / 365.0))
        conv = option_discount("EURUSD", EXPIRY, VALUATION, RATE).discount_factor
        assert (conv - legacy) * 1e4 == pytest.approx(-7.48, abs=0.1)

    def test_one_year_sign_flips(self):
        """ACT/360 and simple compounding partly offset; at 1Y the net is positive."""
        expiry = "2027-09-27"
        days = (fxd._to_date(expiry) - fxd._to_date(VALUATION)).days
        legacy = math.exp(-RATE * days / 365.0)
        conv = option_discount("EURUSD", expiry, VALUATION, RATE).discount_factor
        assert conv > legacy

    def test_premium_moves_by_fourteen_dollars(self, overrides):
        old = BlackScholes(term_discount=RATE, **OPTION).price()
        new = BlackScholes(term_discount=RATE, **dict(OPTION, **overrides)).price()
        assert new - old == pytest.approx(-14.04, abs=0.05)


# --------------------------------------------------------------------------
# The seam is inert unless explicitly switched on
# --------------------------------------------------------------------------
class TestOptIn:
    def test_default_market_emits_no_overrides(self, legacy_market):
        assert discount_kwargs(legacy_market, EXPIRY, VALUATION) == {}

    def test_pair_without_explicit_flag_still_applies(self):
        """Carrying a pair is enough; the flag is an opt-out, not an opt-in."""
        market = MarketData(currency_pair="EURUSD", **BASE_MARKET)
        assert set(discount_kwargs(market, EXPIRY, VALUATION)) == {
            "term_discount_factor", "carry_rate"}

    def test_flag_false_forces_legacy(self):
        market = MarketData(currency_pair="EURUSD", settlement_discounting=False,
                            **BASE_MARKET)
        assert discount_kwargs(market, EXPIRY, VALUATION) == {}

    def test_flag_without_pair_emits_nothing(self):
        market = MarketData(settlement_discounting=True, **BASE_MARKET)
        assert discount_kwargs(market, EXPIRY, VALUATION) == {}

    def test_both_set_emits_both_keys(self, overrides):
        assert set(overrides) == {"term_discount_factor", "carry_rate"}

    def test_none_overrides_reproduce_legacy_exactly(self):
        explicit = BlackScholes(term_discount=RATE, term_discount_factor=None,
                                carry_rate=None, **OPTION)
        implicit = BlackScholes(term_discount=RATE, **OPTION)
        assert explicit.price() == implicit.price()
        assert explicit.theta() == implicit.theta()

    def test_legacy_discount_factor_is_continuous_to_expiry(self):
        bs = BlackScholes(term_discount=RATE, **OPTION)
        assert bs.term_discount_factor == pytest.approx(
            math.exp(-RATE * bs.t), rel=1e-14)


# --------------------------------------------------------------------------
# Failsafes: never raise, always degrade to legacy
# --------------------------------------------------------------------------
class TestFailsafes:
    @pytest.mark.parametrize("pair", ["XAUUSD", "USDEUR", "EURUS", "", "12345Z"])
    def test_safe_lookup_returns_default_for_bad_pairs(self, pair):
        assert safe_option_discount(pair, EXPIRY, VALUATION, RATE) is None

    def test_safe_lookup_honours_a_custom_default(self):
        sentinel = object()
        assert safe_option_discount("XAUUSD", EXPIRY, VALUATION, RATE,
                                    default=sentinel) is sentinel

    @pytest.mark.parametrize("expiry", ["not-a-date", None, "2020-01-01"])
    def test_safe_lookup_handles_bad_expiry(self, expiry):
        assert safe_option_discount("EURUSD", expiry, VALUATION, RATE) is None

    @pytest.mark.parametrize("rate", [float("nan"), float("inf"), None, "x"])
    def test_safe_lookup_handles_bad_rate(self, rate):
        assert safe_option_discount("EURUSD", EXPIRY, VALUATION, rate) is None

    def test_absurd_rate_rejected_by_sanity_band(self):
        """A rate that drives the DF outside (0, 10) is refused."""
        assert safe_option_discount("EURUSD", EXPIRY, VALUATION, -50.0) is None

    def test_discount_kwargs_never_raises(self):
        for market in (None, object(), "not a market"):
            assert discount_kwargs(market, EXPIRY, VALUATION) == {}

    def test_discount_kwargs_survives_bad_dates(self, conv_market):
        assert discount_kwargs(conv_market, "garbage", VALUATION) == {}
        assert discount_kwargs(conv_market, EXPIRY, "garbage") == {}

    def test_expiry_before_valuation_falls_back(self, conv_market):
        assert discount_kwargs(conv_market, "2020-01-01", VALUATION) == {}

    @pytest.mark.parametrize("pair", ["EUR", "EURUSDX", "EUR1SD", 42])
    def test_market_data_rejects_malformed_pairs(self, pair):
        with pytest.raises(ValueError):
            MarketData(currency_pair=pair, **BASE_MARKET)

    @pytest.mark.parametrize("pair", ["EUR/USD", "eur usd", " EURUSD "])
    def test_market_data_normalises_separators(self, pair):
        assert MarketData(currency_pair=pair, **BASE_MARKET).currency_pair == "EURUSD"

    def test_market_data_normalises_case(self):
        assert MarketData(currency_pair="eurusd", **BASE_MARKET).currency_pair == "EURUSD"

    def test_market_data_rejects_non_boolean_flag(self):
        with pytest.raises(ValueError):
            MarketData(settlement_discounting="yes", **BASE_MARKET)

    def test_unavailable_conventions_degrade(self, conv_market, monkeypatch):
        """Simulate a deployment without QuantLib installed."""
        monkeypatch.setattr(fxd, "CONVENTIONS_AVAILABLE", False)
        assert discount_kwargs(conv_market, EXPIRY, VALUATION) == {}
        assert safe_option_discount("EURUSD", EXPIRY, VALUATION, RATE) is None
        with pytest.raises(Exception):
            option_discount("EURUSD", EXPIRY, VALUATION, RATE)


# --------------------------------------------------------------------------
# Pricing invariants: a discount factor swap is a pure rescale
# --------------------------------------------------------------------------
class TestPricingInvariants:
    def test_price_is_undiscounted_times_the_new_factor(self, overrides):
        undiscounted = BlackScholes(term_discount=0.0, **OPTION).price()
        new = BlackScholes(term_discount=RATE, **dict(OPTION, **overrides)).price()
        assert new == pytest.approx(
            undiscounted * overrides["term_discount_factor"], rel=1e-13)

    def test_delta_is_untouched(self, overrides):
        """Forward delta is a pure probability and carries no discounting."""
        old = BlackScholes(term_discount=RATE, delta_convention="forward", **OPTION)
        new = BlackScholes(term_discount=RATE, delta_convention="forward",
                           **dict(OPTION, **overrides))
        assert new.delta() == pytest.approx(old.delta(), rel=1e-14)

    def test_spot_delta_scales_with_the_base_discount_factor(self, overrides):
        """Spot delta *does* move, because the base factor is derived.

        ``base_DF = term_DF * F / S``, so changing the term discounting
        convention changes the base factor by the same ratio and rescales the
        spot delta with it. This is the intended behaviour, not a leak: both
        legs settle on the same delivery date.
        """
        old = BlackScholes(term_discount=RATE, **OPTION)
        new = BlackScholes(term_discount=RATE, **dict(OPTION, **overrides))
        assert new.delta() / old.delta() == pytest.approx(
            new.base_discount_factor / old.base_discount_factor, rel=1e-12)

    def test_gamma_is_untouched(self, overrides):
        old = BlackScholes(term_discount=RATE, delta_convention="forward", **OPTION)
        new = BlackScholes(term_discount=RATE, delta_convention="forward",
                           **dict(OPTION, **overrides))
        assert new.gamma() == pytest.approx(old.gamma(), rel=1e-14)

    def test_vega_scales_with_the_discount_factor(self, overrides):
        old = BlackScholes(term_discount=RATE, **OPTION)
        new = BlackScholes(term_discount=RATE, **dict(OPTION, **overrides))
        assert new.vega() / old.vega() == pytest.approx(
            new.term_discount_factor / old.term_discount_factor, rel=1e-12)

    def test_theta_uses_the_carry_rate(self, overrides):
        """Theta consumes a rate, not a factor, so it gets the true carry."""
        new = BlackScholes(term_discount=RATE, **dict(OPTION, **overrides))
        assert new.term_carry_rate == pytest.approx(overrides["carry_rate"], rel=1e-15)
        spoofed = BlackScholes(
            term_discount=RATE,
            **dict(OPTION, term_discount_factor=overrides["term_discount_factor"],
                   carry_rate=RATE))
        assert new.theta() != spoofed.theta()

    def test_put_call_parity_holds_under_the_new_factor(self, overrides):
        call = BlackScholes(term_discount=RATE, **dict(OPTION, **overrides))
        put = BlackScholes(term_discount=RATE,
                           **dict(OPTION, CallPut="PUT", **overrides))
        parity = (call.f - call.K) * OPTION["Notional"] * overrides["term_discount_factor"]
        assert call.price() - put.price() == pytest.approx(parity, rel=1e-10)


# --------------------------------------------------------------------------
# Both engines must agree
# --------------------------------------------------------------------------
class TestEngineAgreement:
    @staticmethod
    def _numba(**extra):
        return black_scholes_values(
            spot=OPTION["S"], strike=OPTION["K"], forward_points=OPTION["fwd_pts"],
            volatility_pct=OPTION["v"], expiry_date=EXPIRY, valuation_date=VALUATION,
            call_put="CALL", buy_sell="BUY", notional=OPTION["Notional"],
            delta_base=True, term_discount=RATE, **extra)

    def test_legacy_engines_agree(self):
        assert self._numba()[0] == pytest.approx(
            BlackScholes(term_discount=RATE, **OPTION).price(), rel=1e-10)

    def test_convention_engines_agree(self, overrides):
        expected = BlackScholes(term_discount=RATE, **dict(OPTION, **overrides)).price()
        assert self._numba(**overrides)[0] == pytest.approx(expected, rel=1e-10)

    def test_explicit_none_matches_omitted(self):
        assert self._numba(term_discount_factor=None, carry_rate=None)[0] == \
            pytest.approx(self._numba()[0], rel=1e-14)

    def test_engines_agree_when_given_only_a_pair(self):
        """A pair is a label: both engines stay on the legacy factor."""
        expected = BlackScholes(term_discount=RATE, **OPTION).price()
        assert self._numba(currency_pair="EURUSD")[0] == pytest.approx(expected, rel=1e-10)
        assert self._numba(currency_pair="EURUSD")[0] == self._numba()[0]


# --------------------------------------------------------------------------
# A currency pair is a label: the core resolves no calendar (Atlas). The
# convention inputs are resolved outside it (apeiron.hyperion.resolve) and
# passed as numbers.
# --------------------------------------------------------------------------
class TestCurrencyPairResolution:
    def test_pair_alone_stays_on_the_legacy_factor(self):
        bs = BlackScholes(term_discount=RATE, currency_pair="EURUSD", **OPTION)
        legacy = BlackScholes(term_discount=RATE, **OPTION)
        assert bs.settlement_discounting is False
        assert float(bs.term_discount_factor) == float(legacy.term_discount_factor)
        assert bs.price() == legacy.price()

    def test_resolved_inputs_give_the_convention_discounting(self, overrides):
        from apeiron.atlas.resolved import simple_discount_factor
        from apeiron.hyperion.resolve import resolve_inputs
        resolved = resolve_inputs("EURUSD", EXPIRY, VALUATION)
        factor = simple_discount_factor(RATE, resolved.year_fraction)
        assert factor == pytest.approx(overrides["term_discount_factor"], rel=1e-15)
        bs = BlackScholes(term_discount=RATE, currency_pair="EURUSD",
                          term_discount_factor=factor, carry_rate=RATE * factor, **OPTION)
        assert bs.settlement_discounting is True
        assert float(bs.term_carry_rate) == pytest.approx(overrides["carry_rate"], rel=1e-15)

    def test_resolved_inputs_carry_the_delivery_metadata(self):
        from apeiron.hyperion.resolve import resolve_inputs
        resolved = resolve_inputs("EURUSD", EXPIRY, VALUATION)
        assert resolved.delivery_date.isoformat() == "2027-03-30"
        assert "360" in resolved.day_count
        assert resolved.year_fraction == pytest.approx(186 / 360.0, rel=1e-12)

    def test_no_pair_stays_on_the_legacy_factor(self):
        bs = BlackScholes(term_discount=RATE, **OPTION)
        assert bs.settlement_discounting is False
        assert float(bs.term_discount_factor) == pytest.approx(
            math.exp(-RATE * float(bs.t)), rel=1e-14)

    @pytest.mark.parametrize("pair", ["XAUUSD", "USDEUR", "nonsense"])
    def test_unresolvable_pair_degrades_silently(self, pair):
        bs = BlackScholes(term_discount=RATE, currency_pair=pair, **OPTION)
        assert bs.settlement_discounting is False
        assert float(bs.term_discount_factor) == pytest.approx(
            math.exp(-RATE * float(bs.t)), rel=1e-14)

    def test_explicit_factor_wins_over_the_pair(self):
        bs = BlackScholes(term_discount=RATE, currency_pair="EURUSD",
                          term_discount_factor=0.5, **OPTION)
        assert float(bs.term_discount_factor) == 0.5

    def test_t_plus_one_pair_discounts_less(self):
        """USDCAD settles T+1, so its delivery date is a day nearer."""
        from apeiron.atlas.resolved import simple_discount_factor
        from apeiron.hyperion.resolve import resolve_inputs
        two = resolve_inputs("EURUSD", EXPIRY, VALUATION)
        one = resolve_inputs("USDCAD", EXPIRY, VALUATION)
        assert one.delivery_date < two.delivery_date
        assert simple_discount_factor(RATE, one.year_fraction) >             simple_discount_factor(RATE, two.year_fraction)

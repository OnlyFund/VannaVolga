"""Spot vs forward delta, the second of the two orthogonal delta choices.

There are two independent switches on an FX delta and they are routinely
conflated:

* ``deltaBase`` -- premium adjustment. Unadjusted ``N(d1)`` versus
  premium-adjusted ``K/F * N(d2)``. This answers "is the option premium paid
  in the base currency already part of my base-currency exposure?".
* ``delta_convention`` -- spot versus forward. ``spot = base_DF * forward``.
  This answers "am I hedging with a spot trade settling in two days, or with
  an outright forward settling on the option's delivery date?".

The interbank market quotes a *spot* delta out to roughly one year and a
forward delta beyond, so ``"spot"`` is the reporting default. Every internal
consumer -- the pillar strike solvers and the Vanna-Volga calibration -- is
pinned to ``"forward"`` so that a reporting choice can never move the
calibrated surface.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from helper_functions.VannaVolga_function import get_strike_from_delta
from market_data import MarketData
from pricing_utils.BlackScholes import BlackScholes
from pricing_utils.black_scholes_numba import black_scholes_values
from VannaVolga_ImpliedVol import VannaVolga


OPTION = dict(
    S=1.1380,
    K=1.145265,
    fwd_pts=88.75,
    v=5.92,
    expiry_date='2027-03-24',
    valuation_date='2026-09-25',
    CallPut='CALL',
    BuySell='BUY',
    Notional=100_000_000,
    fwd_divisor=10000,
)


def _bs(**overrides):
    return BlackScholes(**dict(OPTION, **overrides))


class TestConventionContract:
    """The switch itself: default, validation, and the defining identity."""

    def test_default_is_the_spot_delta(self):
        """Market standard at this tenor, and what Bloomberg shows."""
        assert _bs().delta_convention == "spot"
        assert _bs().delta() == pytest.approx(
            _bs(delta_convention="spot").delta(), rel=1e-15)

    @pytest.mark.parametrize("delta_base", [True, False])
    def test_spot_delta_is_the_base_factor_times_the_forward_delta(
            self, delta_base):
        fwd = _bs(deltaBase=delta_base, delta_convention="forward")
        spot = _bs(deltaBase=delta_base, delta_convention="spot")
        assert spot.delta() == pytest.approx(
            fwd.delta() * fwd.base_discount_factor, rel=1e-14)

    def test_the_two_switches_are_independent(self):
        """Premium adjustment and spot/forward must not interact.

        The ratio between the conventions is the same base discount factor
        whether or not the premium adjustment is on, which is what makes the
        two switches genuinely orthogonal rather than two spellings of one
        adjustment.
        """
        unadj = (_bs(deltaBase=False, delta_convention="spot").delta()
                 / _bs(deltaBase=False, delta_convention="forward").delta())
        adj = (_bs(deltaBase=True, delta_convention="spot").delta()
               / _bs(deltaBase=True, delta_convention="forward").delta())
        assert unadj == pytest.approx(adj, rel=1e-13)

    def test_spot_delta_is_strictly_smaller_for_a_positive_base_rate(self):
        """A discount factor below one, so the spot hedge is smaller.

        This is the whole economic point: discounting to delivery means you
        need slightly less spot today than the forward delta suggests.
        """
        fwd = _bs(delta_convention="forward").delta()
        spot = _bs(delta_convention="spot").delta()
        assert 0.0 < spot < fwd

    @pytest.mark.parametrize("bad", ["Spot", "fwd", "", None, 0])
    def test_an_unknown_convention_is_rejected(self, bad):
        with pytest.raises(ValueError):
            _bs(delta_convention=bad)
        with pytest.raises(ValueError):
            black_scholes_values(
                OPTION["S"], OPTION["K"], OPTION["fwd_pts"], OPTION["v"],
                OPTION["expiry_date"], OPTION["valuation_date"],
                'CALL', 'BUY', OPTION["Notional"], delta_convention=bad,
            )

    def test_an_option_at_expiry_is_unaffected(self):
        """At expiry the delta collapses to the payoff indicator.

        There is no time left to discount over, so both conventions must agree
        rather than the spot branch silently scaling a settled position.
        """
        expiring = dict(OPTION, valuation_date=OPTION["expiry_date"])
        assert (BlackScholes(**dict(expiring, delta_convention="spot")).delta()
                == pytest.approx(
                    BlackScholes(
                        **dict(expiring, delta_convention="forward")).delta(),
                    rel=1e-15))


class TestGammaFollowsDelta:
    """Gamma *is* the change in delta, so it must track whatever delta reports."""

    @pytest.mark.parametrize("convention", ["spot", "forward"])
    @pytest.mark.parametrize("delta_base", [True, False])
    def test_gamma_is_the_finite_difference_of_the_reported_delta(
            self, convention, delta_base):
        """Exact, including the derivative of the base factor itself.

        ``base_DF = term_DF * (S + pts) / S`` with the forward points held
        fixed across the bump, as a spot ladder holds them. The base factor is
        therefore spot-dependent and gamma has to carry the product-rule term
        ``-term_DF * pts / S**2``; without it the spot gamma is wrong at the
        fourth significant figure.
        """
        h = 1e-6
        kw = dict(deltaBase=delta_base, delta_convention=convention)
        up = _bs(S=OPTION["S"] + h, **kw).delta()
        down = _bs(S=OPTION["S"] - h, **kw).delta()
        fd = (up - down) / (2 * h) * OPTION["S"] * 0.01
        assert _bs(**kw).gamma() == pytest.approx(fd, rel=1e-5)

    def test_spot_and_forward_gamma_differ(self):
        """Guards against gamma being accidentally pinned to one convention."""
        assert (_bs(delta_convention="spot").gamma()
                != _bs(delta_convention="forward").gamma())

    @pytest.mark.parametrize("greek", ["vega", "vanna", "volga", "theta"])
    def test_the_vega_side_greeks_are_untouched(self, greek):
        """Deliberate asymmetry.

        Vega, vanna and volga feed the Vanna-Volga weights. Moving them with a
        delta *reporting* convention would ripple into the smile construction
        and change quoted volatilities, so only delta and gamma follow.
        """
        spot = getattr(_bs(delta_convention="spot"), greek)()
        fwd = getattr(_bs(delta_convention="forward"), greek)()
        assert spot == pytest.approx(fwd, rel=1e-15)


class TestEnginesAgree:
    """The numba kernel is a twin of the analytic engine and must stay in step."""

    @staticmethod
    def _fast(**overrides):
        kw = dict(
            delta_base=True, forward_divisor=OPTION["fwd_divisor"],
            delta_convention="spot")
        kw.update(overrides)
        price, delta, gamma, vega, vanna, volga, theta = black_scholes_values(
            OPTION["S"], OPTION["K"], OPTION["fwd_pts"], OPTION["v"],
            OPTION["expiry_date"], OPTION["valuation_date"],
            OPTION["CallPut"], OPTION["BuySell"], OPTION["Notional"], **kw)
        return {"price": price, "delta": delta, "gamma": gamma}

    @pytest.mark.parametrize("convention", ["spot", "forward"])
    @pytest.mark.parametrize("delta_base", [True, False])
    def test_delta_and_gamma_match_the_analytic_engine(
            self, convention, delta_base):
        bs = _bs(deltaBase=delta_base, delta_convention=convention)
        fast = self._fast(delta_base=delta_base, delta_convention=convention)
        assert fast["delta"] == pytest.approx(bs.delta(), rel=1e-10)
        assert fast["gamma"] == pytest.approx(bs.gamma(), rel=1e-10)

    def test_the_kernel_defaults_to_spot_like_the_analytic_engine(self):
        default = black_scholes_values(
            OPTION["S"], OPTION["K"], OPTION["fwd_pts"], OPTION["v"],
            OPTION["expiry_date"], OPTION["valuation_date"],
            OPTION["CallPut"], OPTION["BuySell"], OPTION["Notional"],
            delta_base=True, forward_divisor=OPTION["fwd_divisor"],
        )[1]
        assert default == pytest.approx(
            self._fast(delta_convention="spot")["delta"], rel=1e-15)

    def test_the_price_is_the_same_under_both_conventions(self):
        """Delta reporting must never touch the premium."""
        assert (self._fast(delta_convention="forward")["price"]
                == pytest.approx(
                    self._fast(delta_convention="spot")["price"], rel=1e-15))


class TestCalibrationIsInsulated:
    """A reporting default must never move the surface."""

    MARKET = MarketData(
        s=1.1380, fwd_pts=88.75, v_atm=5.92,
        rr_25=-0.518, rr_10=-0.9650, bf_25=0.235, bf_10=0.8530,
    )

    def test_the_solved_strike_is_a_forward_delta_strike(self):
        """The pillars are quoted in the forward convention internally.

        Switching the solver would move all five pillar strikes and therefore
        the whole smile, so it is pinned and this test says so out loud. Read
        back in the forward convention the solved strike is exact; read back
        in the reporting convention it is short by the base discount factor.
        """
        kw = dict(S=1.1380, fwd=88.75, v=5.92,
                  expiry_date='2027-03-24', valuation_date='2026-09-25')
        strike = get_strike_from_delta(
            CallPut='CALL', delta=0.25, delta_base=True, **kw)

        def read_back(convention):
            return BlackScholes(
                S=kw["S"], K=strike, fwd_pts=kw["fwd"], v=kw["v"],
                expiry_date=kw["expiry_date"],
                valuation_date=kw["valuation_date"],
                CallPut='CALL', BuySell='BUY', Notional=1, deltaBase=True,
                fwd_divisor=10000, delta_convention=convention,
            )

        assert read_back("forward").delta() == pytest.approx(0.25, abs=1e-8)
        assert read_back("spot").delta() == pytest.approx(
            0.25 * read_back("spot").base_discount_factor, abs=1e-8)

    def test_the_implied_volatility_is_unchanged_by_the_new_default(self):
        """A golden value recorded from the forward-pinned calibration.

        If a future change lets the reporting convention leak into the
        calibration, this moves and the surface silently reprices.
        """
        assert self._vol() == pytest.approx(5.903298, abs=5e-6)

    def test_the_calibration_ignores_the_reporting_convention(self, monkeypatch):
        """Force every pricer in the calibration onto the spot convention.

        The pillar solvers and the target-delta objective pin themselves to
        ``"forward"``, so overriding the argument globally must not change the
        answer at all. This is the property the golden value above only
        indirectly guards.
        """
        original = BlackScholes.__init__

        def forced(self, *args, **kwargs):
            kwargs["delta_convention"] = "forward"
            return original(self, *args, **kwargs)

        monkeypatch.setattr(BlackScholes, "__init__", forced)
        assert self._vol() == pytest.approx(5.903298, abs=5e-6)

    def _vol(self):
        return VannaVolga(
            market=self.MARKET, K=1.148, expiry_date='2027-03-24',
            valuation_date='2026-09-25', CallPut='CALL',
            deltaBase=True, atm_conv='dn').get_vol_for_strike()


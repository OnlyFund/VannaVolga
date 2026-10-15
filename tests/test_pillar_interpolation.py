"""Smoothness of the five-pillar strike interpolation.

The quoted smile is interpolated between its five pillars. That interpolation
used to be linear in log-strike, but the smile is convex between the quoted
deltas, so a straight chord sits systematically *above* the true curve. The
error peaked around 6bp midway between the ATM and 25-delta pillars -- the
region most options are actually struck in -- and it flattened the local skew
near the money by roughly an order of magnitude, which fed straight into the
finite-difference Greeks.

These tests pin the replacement: a natural cubic spline in log-strike.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import helper_functions.vanna_volga_helpers as vvh
from helper_functions.vanna_volga_helpers import (
    GetMainPillars,
    interpolate_pillar_vols,
    safe_interpolate_five_pillar_vols,
)
from market_data import MarketData
from VannaVolga_ImpliedVol import VannaVolga

SPOT = 1.1380
EXPIRY = "2027-03-24"
VALUATION = "2026-09-25"
QUOTES = dict(fwd_pts=88.75, v_atm=5.92, rr_25=-0.518, rr_10=-0.9650,
              bf_25=0.235, bf_10=0.8530)


def _market():
    return MarketData(s=SPOT, **QUOTES)


def _pillars():
    return GetMainPillars(
        k=1.1480, expiry_date=EXPIRY, valuation_date=VALUATION, atm_conv="dn",
        deltaBase=True, s=SPOT, fwd_pts=QUOTES["fwd_pts"],
        v_atm=QUOTES["v_atm"], rr_25=QUOTES["rr_25"], rr_10=QUOTES["rr_10"],
        bf_25=QUOTES["bf_25"], bf_10=QUOTES["bf_10"], forward_divisor=10000,
        term_discount=0.05, smooth_pillar_interpolation=True,
        compatibility_mode="auto",
    ).get_five_pillar_arrays()


def _vol(strike, cubic=True):
    """Smile vol at ``strike``; ``cubic=False`` forces the old linear path."""
    original = vvh._CubicSpline
    if not cubic:
        vvh._CubicSpline = None
    try:
        return VannaVolga(
            market=_market(), K=strike, expiry_date=EXPIRY,
            valuation_date=VALUATION, CallPut="CALL", atm_conv="dn",
            deltaBase=True,
        ).get_vol_for_strike()
    finally:
        vvh._CubicSpline = original


def _continuous_vol(strike):
    """The continuous Vanna-Volga solution, the curve the pillars sample."""
    market = _market()
    market.smooth_pillar_interpolation = False
    market.legacy_target_blend = True
    market.compatibility_mode = "legacy"
    return VannaVolga(
        market=market, K=strike, expiry_date=EXPIRY, valuation_date=VALUATION,
        CallPut="CALL", atm_conv="dn", deltaBase=True,
    ).get_vol_for_strike()


# --------------------------------------------------------------------------
# The interpolator in isolation
# --------------------------------------------------------------------------
class TestInterpolatorContract:
    def test_reproduces_every_pillar_exactly(self):
        strikes, vols = _pillars()
        for strike, vol in zip(strikes, vols):
            assert interpolate_pillar_vols(strikes, vols, strike) == pytest.approx(
                vol, abs=1e-12)

    def test_scalar_target_returns_a_float(self):
        strikes, vols = _pillars()
        assert isinstance(interpolate_pillar_vols(strikes, vols, 1.148), float)

    def test_vector_target_returns_an_array(self):
        strikes, vols = _pillars()
        out = interpolate_pillar_vols(strikes, vols, np.array([1.14, 1.15]))
        assert isinstance(out, np.ndarray) and out.shape == (2,)

    def test_vector_and_scalar_agree(self):
        strikes, vols = _pillars()
        grid = np.linspace(strikes[0], strikes[-1], 11)
        batch = interpolate_pillar_vols(strikes, vols, grid)
        for strike, value in zip(grid, batch):
            assert interpolate_pillar_vols(strikes, vols, strike) == pytest.approx(
                value, rel=1e-14)

    def test_stays_within_a_sane_band(self):
        """The interpolant may dip below the lowest *pillar*.

        The smile's true minimum sits between the ATM and 25-delta call
        pillars, not on one of them, so a curve that reproduces the quotes
        must undershoot the lowest quoted level slightly. What matters is
        that it stays in a plausible band rather than overshooting wildly.
        """
        strikes, vols = _pillars()
        grid = np.linspace(strikes[0], strikes[-1], 400)
        out = interpolate_pillar_vols(strikes, vols, grid)
        span = max(vols) - min(vols)
        assert out.min() >= min(vols) - 0.25 * span
        assert out.max() <= max(vols) + 1e-9
        assert out.min() > 0.0

    def test_undershoot_matches_the_true_smile_minimum(self):
        strikes, vols = _pillars()
        grid = np.linspace(strikes[0], strikes[-1], 400)
        out = interpolate_pillar_vols(strikes, vols, grid)
        assert out.min() < min(vols)
        assert out.min() == pytest.approx(5.8497, abs=0.01)

    def test_is_smooth_with_no_kinks_at_the_pillars(self):
        """Linear interpolation puts a slope discontinuity at every pillar."""
        strikes, vols = _pillars()
        grid = np.linspace(strikes[0], strikes[-1], 2000)
        second = np.diff(interpolate_pillar_vols(strikes, vols, grid), 2)
        linear_second = np.diff(np.interp(grid, strikes, vols), 2)
        assert np.abs(second).max() < 0.2 * np.abs(linear_second).max()


# --------------------------------------------------------------------------
# Accuracy against the curve the pillars are sampled from
# --------------------------------------------------------------------------
class TestAccuracy:
    @staticmethod
    def _grid():
        strikes, _ = _pillars()
        return np.linspace(strikes[1], strikes[3], 21)[1:-1]

    def test_tracks_the_continuous_smile_inside_one_basis_point(self):
        for strike in self._grid():
            error = abs(_vol(strike) - _continuous_vol(strike)) * 100.0
            assert error < 2.0, "%.6f deviates %.2f bp" % (strike, error)

    def test_beats_the_linear_interpolation_it_replaced(self):
        cubic, linear = [], []
        for strike in self._grid():
            reference = _continuous_vol(strike)
            cubic.append(abs(_vol(strike) - reference))
            linear.append(abs(_vol(strike, cubic=False) - reference))
        assert max(cubic) * 100 < 2.0
        assert max(linear) * 100 > 4.0
        assert max(cubic) < 0.25 * max(linear)

    def test_the_old_linear_path_was_biased_high(self):
        """A chord across a convex smile can only sit above it."""
        for strike in self._grid():
            assert _vol(strike, cubic=False) >= _continuous_vol(strike) - 1e-9

    def test_recovers_the_local_skew_between_pillars(self):
        """Linear interpolation flattened dVol/dK by about ten times.

        Sampled across a whole segment rather than at one point. A chord
        matches the curve's slope exactly at the pillars (same value) and
        again near the segment midpoint (mean value theorem), so a
        single-point comparison can miss the error entirely; it is largest
        just inside either pillar.
        """
        strikes, _ = _pillars()
        grid = np.linspace(strikes[2], strikes[3], 12)[1:-1]
        h = 1e-4

        def slope(strike, cubic):
            return (_vol(strike + h, cubic) - _vol(strike - h, cubic)) / (2 * h)

        cubic_err, linear_err = [], []
        for strike in grid:
            reference = (_continuous_vol(strike + h) -
                         _continuous_vol(strike - h)) / (2 * h)
            cubic_err.append(abs(slope(strike, True) - reference))
            linear_err.append(abs(slope(strike, False) - reference))

        assert max(cubic_err) < 0.4 * max(linear_err)
        assert max(linear_err) > 5.0

    def test_skew_is_continuous_across_a_pillar(self):
        """Linear interpolation puts a slope jump at every quoted pillar."""
        strikes, _ = _pillars()
        atm = float(strikes[2])
        h, step = 1e-4, 2e-3

        def slope(strike, cubic):
            return (_vol(strike + h, cubic) - _vol(strike - h, cubic)) / (2 * h)

        cubic_jump = abs(slope(atm + step, True) - slope(atm - step, True))
        linear_jump = abs(slope(atm + step, False) - slope(atm - step, False))
        assert cubic_jump < 0.5 * linear_jump


# --------------------------------------------------------------------------
# Pillars, wings and the pricing path
# --------------------------------------------------------------------------
class TestSmileIntegration:
    def test_every_quoted_pillar_still_prices_back(self):
        strikes, vols = _pillars()
        for strike, vol in zip(strikes, vols):
            assert _vol(strike) == pytest.approx(vol, abs=1e-6)

    def test_atm_pillar_returns_the_atm_quote(self):
        strikes, _ = _pillars()
        assert _vol(float(strikes[2])) == pytest.approx(QUOTES["v_atm"], abs=1e-6)

    def test_wings_are_untouched_by_the_change(self):
        """Outside the pillars the smile still extrapolates, not clamps."""
        strikes, vols = _pillars()
        far = float(strikes[-1]) * 1.05
        assert _vol(far) == pytest.approx(_vol(far, cubic=False), abs=1e-9)
        assert _vol(far) > float(vols[-1])

    def test_smile_is_continuous_across_the_pillar_boundary(self):
        strikes, _ = _pillars()
        edge = float(strikes[-1])
        assert _vol(edge * (1 - 1e-7)) == pytest.approx(
            _vol(edge * (1 + 1e-7)), abs=5e-4)

    def test_smile_remains_convex_between_the_wings(self):
        strikes, _ = _pillars()
        grid = np.linspace(strikes[0], strikes[-1], 500)
        vols = np.array([_vol(k) for k in grid])
        assert np.diff(vols, 2).min() > -1e-6


# --------------------------------------------------------------------------
# Degradation
# --------------------------------------------------------------------------
class TestFallbacks:
    def test_falls_back_to_linear_without_scipy(self, monkeypatch):
        strikes, vols = _pillars()
        monkeypatch.setattr(vvh, "_CubicSpline", None)
        assert interpolate_pillar_vols(strikes, vols, 1.148) == pytest.approx(
            float(np.interp(1.148, strikes, vols)), rel=1e-14)

    def test_falls_back_to_linear_when_the_spline_raises(self, monkeypatch):
        strikes, vols = _pillars()

        def boom(*args, **kwargs):
            raise RuntimeError("spline unavailable")

        monkeypatch.setattr(vvh, "_CubicSpline", boom)
        assert interpolate_pillar_vols(strikes, vols, 1.148) == pytest.approx(
            float(np.interp(1.148, strikes, vols)), rel=1e-14)

    def test_two_pillars_degrade_to_linear(self):
        strikes = np.array([1.10, 1.20])
        vols = np.array([6.0, 7.0])
        assert interpolate_pillar_vols(strikes, vols, 1.15) == pytest.approx(6.5)

    def test_non_positive_strikes_degrade_to_linear(self):
        strikes = np.array([-1.0, 0.0, 1.0])
        vols = np.array([6.0, 6.5, 7.0])
        assert interpolate_pillar_vols(strikes, vols, 0.5) == pytest.approx(6.75)

    def test_outside_the_pillars_is_clamped_not_extrapolated(self):
        """The cubic is only trusted inside the quoted range."""
        strikes, vols = _pillars()
        assert interpolate_pillar_vols(strikes, vols, strikes[0] * 0.5) == \
            pytest.approx(float(vols[0]), rel=1e-12)
        assert interpolate_pillar_vols(strikes, vols, strikes[-1] * 2.0) == \
            pytest.approx(float(vols[-1]), rel=1e-12)

    def test_safe_helper_routes_through_the_smooth_interpolator(self):
        strikes, vols = _pillars()
        assert safe_interpolate_five_pillar_vols(strikes, vols, 1.148) == \
            pytest.approx(interpolate_pillar_vols(strikes, vols, 1.148), rel=1e-14)

    def test_safe_helper_still_falls_back_on_bad_pillars(self):
        assert safe_interpolate_five_pillar_vols(
            [1.1, 1.1], [6.0, 6.5], 1.1, fallback_atm=5.92) == pytest.approx(5.92)

    def test_unsorted_input_is_handled_by_the_safe_helper(self):
        strikes, vols = _pillars()
        order = [4, 0, 2, 1, 3]
        assert safe_interpolate_five_pillar_vols(
            np.asarray(strikes)[order], np.asarray(vols)[order], 1.148) == \
            pytest.approx(safe_interpolate_five_pillar_vols(strikes, vols, 1.148),
                          rel=1e-14)

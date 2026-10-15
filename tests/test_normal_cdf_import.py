"""The normal CDF import and its fallbacks.

``scipy.special.ndtr`` is a compiled ufunc re-exported dynamically from
``scipy.special._ufuncs``. Static analysers cannot see through that and report
an unresolved import, so the module imports it from the private location
first, then the public name, then falls back to an ``erf``-based
implementation. These tests pin that the fallback is a genuine drop-in rather
than a placeholder that would silently misprice every option.
"""

import os
import sys
from math import erf, sqrt

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pricing_utils.BlackScholes import BlackScholes, ndtr


_INV_SQRT_TWO = 1.0 / sqrt(2.0)
_erf = np.vectorize(erf, otypes=[float])


def _fallback(x):
    """A copy of the module's last-resort implementation."""
    x = np.asarray(x, dtype=float)
    result = 0.5 * (1.0 + _erf(x * _INV_SQRT_TWO))
    return result if result.ndim else result[()]


class TestTheImportResolves:
    def test_ndtr_is_available_and_callable(self):
        assert callable(ndtr)

    def test_it_is_the_standard_normal_cdf(self):
        assert float(ndtr(0.0)) == pytest.approx(0.5, abs=1e-15)
        assert float(ndtr(-40.0)) == pytest.approx(0.0, abs=1e-15)
        assert float(ndtr(40.0)) == pytest.approx(1.0, abs=1e-15)

    def test_it_is_symmetric(self):
        for x in (0.25, 1.0, 2.5, 6.0):
            assert float(ndtr(x)) + float(ndtr(-x)) == pytest.approx(
                1.0, abs=1e-14)


class TestTheFallbackIsADropIn:
    """If SciPy ever disappears the prices must not move."""

    @pytest.mark.parametrize(
        "x", [-8.0, -3.5, -1.0, -0.25, 0.0, 0.25, 1.0, 3.5, 8.0])
    def test_it_matches_scipy_pointwise(self, x):
        assert _fallback(x) == pytest.approx(float(ndtr(x)), rel=1e-12,
                                             abs=1e-15)

    def test_it_matches_scipy_across_an_array(self):
        grid = np.linspace(-10.0, 10.0, 2001)
        assert np.allclose(_fallback(grid), ndtr(grid), rtol=1e-12, atol=1e-15)

    def test_it_preserves_scalar_and_array_shapes(self):
        """Shape discipline matters: BlackScholes broadcasts over strikes."""
        assert np.asarray(_fallback(0.3)).ndim == 0
        assert _fallback([0.1, 0.2]).shape == (2,)
        assert _fallback(np.zeros((2, 3))).shape == (2, 3)

    def test_an_empty_array_does_not_raise(self):
        """``np.vectorize`` needs an explicit dtype or empty input raises."""
        assert _fallback(np.array([])).shape == (0,)

    def test_the_deep_tails_stay_in_range(self):
        extreme = _fallback(np.array([-300.0, 300.0]))
        assert extreme[0] == pytest.approx(0.0, abs=1e-15)
        assert extreme[1] == pytest.approx(1.0, abs=1e-15)


class TestPricingIsUnaffected:
    OPTION = dict(
        S=1.1380, K=1.145265, fwd_pts=88.75, v=5.92,
        expiry_date='2027-03-24', valuation_date='2026-09-25',
        CallPut='CALL', BuySell='BUY', Notional=100_000_000,
        fwd_divisor=10000,
    )

    def test_swapping_in_the_fallback_reprices_identically(self, monkeypatch):
        """Patch the name the module actually calls and reprice end to end.

        A tolerance of 1e-12 relative on a 100m notional is a fraction of a
        cent, so this would catch any real numerical regression in the
        fallback while tolerating last-bit differences between ``erf`` and
        SciPy's own series.
        """
        import pricing_utils.BlackScholes as module

        expected = BlackScholes(**self.OPTION)
        reference = (expected.price(), expected.delta(), expected.vega())

        monkeypatch.setattr(module, "ndtr", _fallback)
        patched = BlackScholes(**self.OPTION)
        actual = (patched.price(), patched.delta(), patched.vega())

        for got, want in zip(actual, reference):
            assert got == pytest.approx(want, rel=1e-12)

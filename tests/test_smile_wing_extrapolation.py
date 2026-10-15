"""Regression tests for smile behaviour outside the quoted pillar range.

The five quoted pillars (10d put, 25d put, ATM, 25d call, 10d call) were
previously interpolated with ``np.interp``, which *clamps* to the endpoint
value outside the quoted range.  Every strike beyond the 10-delta pillars
therefore returned exactly the 10-delta quote, producing a perfectly flat
wing and materially under-pricing out-of-the-money options.
"""

import numpy as np
import pytest

from market_data import MarketData
from VannaVolga_ImpliedVol import VannaVolga
from vanna_volga_sabr import VannaVolga_SABR


MARKET = dict(
    s=1.1386,
    fwd_pts=88.71,
    v_atm=5.92,
    rr_25=-0.518,
    rr_10=-0.9650,
    bf_25=0.235,
    bf_10=0.8530,
)

CONTRACT = dict(
    expiry_date="2027-03-24",
    valuation_date="2026-09-25",
    CallPut="CALL",
    deltaBase=True,
    atm_conv="dn",
)

PUT_PILLAR = 1.0755803960019266
CALL_PILLAR = 1.214852208293309

PILLARS = [
    (PUT_PILLAR, 7.2555),
    (1.113210, 6.4140),
    (1.146480, 5.9200),
    (1.180025, 5.8960),
    (CALL_PILLAR, 6.2905),
]


def _market():
    return MarketData(**MARKET)


def _vol(strike, engine=VannaVolga):
    return engine(market=_market(), K=strike, **CONTRACT).get_vol_for_strike()


@pytest.mark.parametrize("strike,quote", PILLARS)
def test_quoted_pillars_are_reproduced(strike, quote):
    assert _vol(strike) == pytest.approx(quote, abs=1e-4)


def test_call_wing_is_not_flat_beyond_the_ten_delta_pillar():
    """The defect returned the 10d call quote for every strike beyond it."""
    beyond = [1.2200, 1.2385, 1.2500, 1.3000]
    vols = [_vol(k) for k in beyond]

    call_quote = 6.2905
    assert all(v > call_quote + 1e-3 for v in vols), (
        "call wing is clamped to the 10-delta quote: %s" % vols
    )
    assert vols == sorted(vols), "call wing must rise away from the money"


def test_put_wing_is_not_flat_beyond_the_ten_delta_pillar():
    beyond = [1.0700, 1.0600, 1.0500]
    vols = [_vol(k) for k in beyond]

    put_quote = 7.2555
    assert all(v > put_quote + 1e-3 for v in vols), (
        "put wing is clamped to the 10-delta quote: %s" % vols
    )
    assert vols == sorted(vols), "put wing must rise away from the money"


@pytest.mark.parametrize("pillar", [PUT_PILLAR, CALL_PILLAR])
def test_smile_is_continuous_across_the_boundary_pillars(pillar):
    """Extrapolation is anchored to the boundary quote, so no step appears.

    The Vanna-Volga put anchor does not coincide with the 10-delta put
    pillar, so an unanchored wing left a ~0.1 vol discontinuity here.
    """
    step = pillar * 1e-5
    inside = _vol(pillar - step)
    at = _vol(pillar)
    outside = _vol(pillar + step)

    assert abs(inside - at) < 5e-3
    assert abs(outside - at) < 5e-3


def test_deep_wings_stay_finite_and_positive():
    """Far outside the quotes the price inversion is ill posed.

    It must degrade to the boundary quote rather than returning zero.
    """
    for strike in (0.70, 0.80, 0.90, 1.40, 1.60, 2.00):
        vol = _vol(strike)
        assert np.isfinite(vol)
        assert 1.0 < vol < 50.0, "implausible deep-wing vol at %s: %s" % (strike, vol)


def test_out_of_range_strike_matches_market_quote():
    """User-reported case: K=1.2385 quoted at 6.577, previously 6.2905."""
    vol = _vol(1.2385)
    assert vol == pytest.approx(6.577, abs=0.15)


class TestSabrSmile:
    """The SABR variant was reported as *more* compressed than plain VV."""

    @pytest.mark.parametrize("strike,quote", PILLARS)
    def test_calibration_reproduces_quoted_pillars(self, strike, quote):
        """A single-seed descent stalled in a local minimum.

        It undershot both 10-delta wings by ~0.25 vol and missed ATM.
        """
        assert _vol(strike, engine=VannaVolga_SABR) == pytest.approx(quote, abs=0.05)

    def test_calibration_beats_the_single_seed_local_minimum(self):
        obj = VannaVolga_SABR(market=_market(), K=1.2385, **CONTRACT)
        alpha, rho, nu = obj.calibrate_sabr()

        assert alpha is not None and rho is not None and nu is not None

        strikes = np.array(sorted(obj.smile), dtype=float)
        quotes = np.array([obj.smile[k] for k in strikes], dtype=float)
        fitted = obj.sabr_smile(strikes)

        assert np.max(np.abs(fitted - quotes)) < 0.05

    def test_wing_is_not_compressed(self):
        vol = _vol(1.2385, engine=VannaVolga_SABR)
        assert vol == pytest.approx(6.577, abs=0.15)

    def test_calibration_is_cached(self):
        obj = VannaVolga_SABR(market=_market(), K=1.2385, **CONTRACT)
        first = obj.calibrate_sabr()
        second = obj.calibrate_sabr()
        assert first == second

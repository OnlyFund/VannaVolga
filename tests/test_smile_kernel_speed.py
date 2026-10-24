"""Smile kernel speed-ups (performance step 1, the Greeks) and their fail-safes:
the pillar spline built once per smile core, scalars passed through market
snapshots, the wrapped smile's local skews computed once per market."""
import logging
import pickle

import numpy as np
import pandas as pd
import pytest

from golden.app_harness import load_markets
from helper_functions import vanna_volga_helpers as vvh
from helper_functions.vanna_volga_helpers import PillarSpline, interpolate_pillar_vols
from resolved_smile_inputs import _freeze, _thaw
from surface_models import smile_extension
from surface_models.smile_extension import SmileExtension

pytestmark = pytest.mark.filterwarnings("ignore")

STRIKES = np.array([1.05, 1.08, 1.10, 1.12, 1.16])
VOLS = np.array([8.9, 7.6, 7.1, 7.3, 8.4])
TARGETS = np.array([1.00, 1.05, 1.07, 1.10, 1.111, 1.16, 1.25])


@pytest.mark.parametrize("curvatures", [None, (0.4, 0.9)])
def test_the_spline_built_once_gives_what_the_function_gives(curvatures):
    spline = PillarSpline(STRIKES, VOLS, curvatures)
    for target in TARGETS:
        assert spline(target) == interpolate_pillar_vols(STRIKES, VOLS, target, curvatures)
    assert np.array_equal(spline(TARGETS), interpolate_pillar_vols(STRIKES, VOLS, TARGETS, curvatures))


def test_too_few_pillars_interpolate_linearly():
    spline = PillarSpline(STRIKES[:2], VOLS[:2])
    assert spline(1.06) == pytest.approx(np.interp(1.06, STRIKES[:2], VOLS[:2]))


def test_a_spline_that_cannot_be_built_falls_back_to_linear_every_time(caplog):
    spline = PillarSpline(STRIKES, VOLS, ("not a number", 0.0))
    with caplog.at_level(logging.WARNING, logger=vvh.logger.name):
        for _ in range(2):
            assert spline(1.09) == pytest.approx(np.interp(1.09, STRIKES, VOLS))
    assert sum("using linear" in record.message for record in caplog.records) == 2


def test_a_core_pickled_before_the_spline_still_prices():
    from resolved_smile_cores import VannaVolgaCore
    core = VannaVolgaCore.__new__(VannaVolgaCore)    # as an old pickle restores it
    core.context, core.strikes, core.vols = None, STRIKES, VOLS
    core.end_curvatures, core._wings = vvh.end_segment_curvatures(STRIKES, VOLS), {}
    old = pickle.loads(pickle.dumps(core))
    assert "_pillar_spline" not in old.__dict__
    assert old._interior_vol(1.09) == interpolate_pillar_vols(STRIKES, VOLS, 1.09, core.end_curvatures)


def test_snapshot_scalars_pass_through_and_arrays_are_still_copied():
    for value in (1.5, 3, "EURUSD", True, None):
        assert _freeze(value) is value and _thaw(value) is value
    array = np.array([1.0, 2.0])
    frozen = _freeze(array)
    assert frozen is not array and not frozen.flags.writeable
    numpy_float = np.float64(1.25)               # a float subclass: copied as before
    assert _freeze(numpy_float) == numpy_float


@pytest.fixture(scope="module")
def market():
    from app.market_context import MarketContext
    context = MarketContext.from_payload(load_markets("MARKETS_20261015.json")["EURUSD"])
    day = pd.Timestamp(context.valuation_date) + pd.DateOffset(months=3)
    return context, context.market_for_expiry(day), day


def extension(market, strike):
    context, quotes, day = market
    return SmileExtension(market=quotes, K=strike, expiry_date=day.date().isoformat(),
                          valuation_date=pd.Timestamp(context.valuation_date).date().isoformat())


def test_local_skews_are_computed_once_per_market_and_each_caller_gets_its_own(market, monkeypatch):
    calls = []
    original = SmileExtension._wrapped_local_skews

    def counted(self, *args):
        calls.append(self.K)
        return original(self, *args)
    monkeypatch.setattr(SmileExtension, "_wrapped_local_skews", counted)
    forward = market[1].s + market[1].fwd_pts / market[1].forward_divisor
    first = extension(market, forward * 1.01).get_local_skews()
    second = extension(market, forward * 0.97).get_local_skews()   # another strike
    assert len(calls) <= 1                       # at most once (it may be cached already)
    assert first["local_skew"] == second["local_skew"]
    first["main_vols"][0] = -1.0                 # one caller's change stays its own
    assert extension(market, forward).get_local_skews()["main_vols"][0] > 0


def test_local_skews_off_switch_computes_every_time_and_agrees(market, monkeypatch):
    forward = market[1].s + market[1].fwd_pts / market[1].forward_divisor
    cached = extension(market, forward).get_local_skews()
    monkeypatch.setenv("VV_MARKET_CACHE", "0")
    calls = []
    original = SmileExtension._wrapped_local_skews
    monkeypatch.setattr(SmileExtension, "_wrapped_local_skews",
                        lambda self, *args: calls.append(1) or original(self, *args))
    plain = extension(market, forward).get_local_skews()
    extension(market, forward).get_local_skews()
    assert len(calls) == 2
    assert plain["local_skew"] == cached["local_skew"]
    assert np.array_equal(plain["main_vols"], cached["main_vols"])

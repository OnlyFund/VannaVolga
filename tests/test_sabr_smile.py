from types import SimpleNamespace

import numpy as np
import pytest

from pricing_utils.sabr_numba import hagan_sabr_log_normal_vol_numba
from resolved_smile_cores import make_core
from resolved_smile_inputs import Pillars, PreparationError


def _synthetic_smile():
    context = SimpleNamespace(forward=1.1, time=1.2)
    strikes = np.array([0.82, 0.96, 1.1, 1.24, 1.42])
    vols = 100.0 * hagan_sabr_log_normal_vol_numba(
        0.18, 0.5, -0.35, 0.65, context.forward, context.time, strikes)
    return context, Pillars(tuple(strikes), tuple(vols))


def test_pure_sabr_core_recovers_synthetic_smile_at_and_between_pillars():
    context, pillars = _synthetic_smile()
    core = make_core("sabr", context, pillars, beta=0.5)

    for strike in (*pillars.strikes, 1.03, 1.31):
        expected = hagan_sabr_log_normal_vol_numba(
            *core.parameters[:1], core.beta, *core.parameters[1:],
            context.forward, context.time, np.array([strike]))[0] * 100.0
        assert core.get_vol_for_strike(strike) == pytest.approx(expected, abs=1e-7)
    assert core.calibration_rms < 1e-8


@pytest.mark.parametrize("strike", [0.0, -1.0, float("nan"), float("inf"), True])
def test_pure_sabr_core_rejects_invalid_strikes(strike):
    context, pillars = _synthetic_smile()
    core = make_core("sabr", context, pillars, beta=0.5)

    with pytest.raises(ValueError, match="strike"):
        core.get_vol_for_strike(strike)


def test_pure_sabr_core_rejects_a_fit_that_cannot_reproduce_pillars():
    context, pillars = _synthetic_smile()
    bad_vols = tuple(vol + offset for vol, offset in zip(
        pillars.vols, (0.0, 3.0, 0.0, -2.0, 0.0)))
    with pytest.raises(PreparationError, match="SABR fit error"):
        make_core("sabr", context, Pillars(pillars.strikes, bad_vols), beta=0.5)

import numpy as np
import pytest

from pricing_utils.sabr_numba import hagan_sabr_log_normal_vol_numba


def _hagan_reference(alpha, beta, rho, nu, f, t, strike):
    one_minus_beta = 1.0 - beta
    fk_beta = (f * strike) ** (one_minus_beta / 2.0)
    log_fk = np.log(f / strike)
    z = (nu / alpha) * fk_beta * log_fk
    sqrt_term = np.sqrt(1.0 - 2.0 * rho * z + z ** 2)
    x_z = np.log((sqrt_term + z - rho) / (1.0 - rho))
    z_over_x_z = 1.0 if abs(x_z) <= 1e-14 else z / x_z
    log_correction = (
        1.0
        + one_minus_beta ** 2 * log_fk ** 2 / 24.0
        + one_minus_beta ** 4 * log_fk ** 4 / 1920.0
    )
    time_correction = 1.0 + t * (
        one_minus_beta ** 2 * alpha ** 2 / (24.0 * fk_beta ** 2)
        + rho * beta * nu * alpha / (4.0 * fk_beta)
        + (2.0 - 3.0 * rho ** 2) * nu ** 2 / 24.0
    )
    return alpha / fk_beta * z_over_x_z * time_correction / log_correction


@pytest.mark.parametrize("nu", [0.0, 0.65])
@pytest.mark.parametrize("strike", [0.82, 1.1, 1.34])
def test_hagan_sabr_matches_full_log_normal_formula(strike, nu):
    alpha, beta, rho, forward, expiry = 0.18, 0.6, -0.35, 1.1, 1.4
    actual = hagan_sabr_log_normal_vol_numba(
        alpha, beta, rho, nu, forward, expiry, np.array([strike])
    )[0]
    expected = _hagan_reference(alpha, beta, rho, nu, forward, expiry, strike)

    assert actual == pytest.approx(expected, rel=1e-13)

import numpy as np
from numba import njit

@njit
def hagan_sabr_log_normal_vol_numba(alpha, beta, rho, nu, f, t, K_array):
    """
    Compute SABR log-normal implied vol for an array of strikes (Numba-compatible).
    K_array must always be a 1D np.ndarray of floats.
    """
    n = K_array.size
    vols = np.empty(n, dtype=np.float64)
    if alpha <= 0.0 or f <= 0.0 or t <= 0.0 or beta < 0.0 or beta > 1.0 or abs(rho) >= 1.0 or nu < 0.0:
        for i in range(n):
            vols[i] = np.nan
        return vols

    for i in range(n):
        K = K_array[i]
        if K <= 0.0:
            vols[i] = np.nan
        elif abs(f - K) <= 1e-10 * max(1.0, abs(f), abs(K)):
            one_minus_beta = 1.0 - beta
            f_power = f ** one_minus_beta
            term2 = 1.0 + t * (
                (one_minus_beta ** 2 * alpha ** 2) / (24.0 * f ** (2.0 * one_minus_beta))
                + rho * beta * nu * alpha / (4.0 * f_power)
                + (2.0 - 3.0 * rho ** 2) * nu ** 2 / 24.0
            )
            vols[i] = alpha / f_power * term2
        else:
            FK_beta = (f * K) ** ((1 - beta) / 2.0)
            log_fk = np.log(f / K)
            z = (nu / alpha) * FK_beta * log_fk
            sqrt_term = np.sqrt(1 - 2 * rho * z + z ** 2)
            x_z = np.log((sqrt_term + z - rho) / (1 - rho))
            log_correction = (
                1.0
                + (1 - beta) ** 2 * log_fk ** 2 / 24.0
                + (1 - beta) ** 4 * log_fk ** 4 / 1920.0
            )
            if abs(x_z) <= 1e-14:
                z_over_x_z = 1.0
            else:
                z_over_x_z = z / x_z
            vols[i] = (alpha / FK_beta) * (z_over_x_z / log_correction) * (
                1
                + t
                * (
                    ((1 - beta) ** 2 * alpha ** 2) / (24 * FK_beta ** 2)
                    + 0.25 * rho * beta * nu * alpha / FK_beta
                    + ((2 - 3 * rho ** 2) * nu ** 2) / 24
                    )
            )
    return vols
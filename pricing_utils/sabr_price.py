import numpy as np


def hagan_sabr_log_normal_vol(alpha, beta, rho, nu, f, t, K):
    """
    Hagan SABR log-normal implied volatility formula.

    All vols returned in decimal (e.g., 0.12 for 12%).
    """
    if f <= 0 or K <= 0 or t <= 0:
        return np.nan

    ONE_MINUS_BETA = 1 - beta
    FK_beta = (f * K) ** (ONE_MINUS_BETA / 2)
    log_FK = np.log(f / K) if f != K else 0.0

    if abs(f - K) < 1e-07:
        term1 = alpha / (f ** ONE_MINUS_BETA)
        term2 = 1 + (
            ((ONE_MINUS_BETA) ** 2) * (alpha ** 2) / (24 * (f ** (2 * ONE_MINUS_BETA))) +
            (rho * beta * nu * alpha) / (4 * (f ** ONE_MINUS_BETA)) +
            (nu ** 2) * (2 - 3 * rho ** 2) / 24
        ) * t
        return term1 * term2

    z = (nu / alpha) * FK_beta * log_FK
    x_z = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))
    term1 = alpha * z / (FK_beta * x_z)
    term2 = 1 + (
        ((ONE_MINUS_BETA) ** 2) * (alpha ** 2) / (24 * FK_beta ** 2) +
        (rho * beta * nu * alpha) / (4 * FK_beta) +
        ((2 - 3 * rho ** 2) * (nu ** 2) / 24)
    ) * t
    return term1 * term2
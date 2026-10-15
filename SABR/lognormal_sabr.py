import numpy as np
from SABR import BaseLogNormalSABR
from black_price import Black76
from scipy.optimize import minimize


class Hagan2002_LogNormal(BaseLogNormalSABR):

    def alpha(self):
        f, s, t, v_atm_n = self.f, self.shift, self.t, self.v_atm_n
        beta, rho, volvol = self.beta, self.rho, self.volvol

        # Convert ATM normal vol to shifted lognormal vol at forward
        v_atm_sln = Black76(f=f, t=t).normal_to_shifted_lognormal(k=f, v_n=v_atm_n)
        return alpha(v_atm_sln, f + s, t, beta, rho, volvol)

    def lognormal_vol(self, k):
        f, s, t = self.f, self.shift, self.t
        beta, rho, volvol = self.beta, self.rho, self.volvol
        alpha_ = self.alpha()
        return lognormal_vol(k + s, f + s, t, alpha_, beta, rho, volvol)

    def fit(self, k, v_sln):
        f, s, t, beta = self.f, self.shift, self.t, self.beta

        def vol_square_error(x):
            # x = [alpha, rho, volvol]
            model_vols = [
                lognormal_vol(k_i + s, f + s, t, x[0], beta, x[1], x[2]) * 100
                for k_i in k
            ]
            return np.sum((np.array(model_vols) - np.array(v_sln)) ** 2)

        x0 = np.array([0.01, 0.0, 0.1])
        bounds = [(1e-4, None), (-0.9999, 0.9999), (1e-4, None)]
        res = minimize(vol_square_error, x0, method='L-BFGS-B', bounds=bounds)
        alpha_, rho_, volvol_ = res.x
        self.rho = rho_
        self.volvol = volvol_
        return [alpha_, rho_, volvol_]


def alpha(v_atm_ln, f, t, beta, rho, volvol):
    """
    Calculate SABR parameter alpha to an ATM lognormal volatility.

    Alpha is determined as the root of a cubic polynomial. Returns a scalar alpha.
    """
    f_ = f ** (beta - 1)
    p = [
        t * f_**3 * (1 - beta)**2 / 24,
        t * f_**2 * rho * beta * volvol / 4,
        (1 + t * volvol**2 * (2 - 3 * rho**2) / 24) * f_,
        -v_atm_ln
    ]
    roots = np.roots(p)
    roots_real = roots[np.isreal(roots)].real
    alpha_guess = v_atm_ln * f**(1 - beta)
    i_min = np.argmin(np.abs(roots_real - alpha_guess))
    return roots_real[i_min]


def lognormal_vol(k, f, t, alpha, beta, rho, volvol):
    if k <= 0 or f <= 0:
        return 0.0

    eps = 1e-7
    logfk = np.log(f / k)
    fkbeta = (f * k) ** (1 - beta)
    a = (1 - beta) ** 2 * alpha ** 2 / (24 * fkbeta)
    b = 0.25 * rho * beta * volvol * alpha / np.sqrt(fkbeta)
    c = (2 - 3 * rho ** 2) * volvol ** 2 / 24
    d = np.sqrt(fkbeta)
    v = (1 - beta) ** 2 * logfk ** 2 / 24
    w = (1 - beta) ** 4 * logfk ** 4 / 1920
    z = volvol * d * logfk / alpha

    def _x(rho, z):
        numerator = np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho
        denominator = 1 - rho
        return np.log(numerator / denominator)

    if abs(z) > eps:
        return alpha * z * (1 + (a + b + c) * t) / (d * (1 + v + w) * _x(rho, z))
    else:
        return alpha * (1 + (a + b + c) * t) / (d * (1 + v + w))

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq, minimize_scalar

class Black76:
    """
    FX option pricing and volatility conversion between
    Normal, LogNormal, and Shifted LogNormal models.
    """

    def __init__(self, f, t, r=0.0, shift=0.0):
        """
        Parameters
        ----------
        f : float
            Forward price
        t : float
            Time to maturity (in years)
        r : float, default 0.0
            Discount rate
        shift : float, default 0.0
            Shift for shifted lognormal model
        """
        self.f = f
        self.t = t
        self.r = r
        self.shift = shift

    # Pricing functions
    def lognormal_call(self, k, v, cp='call'):
        if k <= 0 or self.f <= 0:
            raise ValueError("Strikes and forwards must be > 0 for lognormal model")

        if self.t <= 0 or v <= 0:
            intrinsic = max(self.f - k, 0.0) if cp == 'call' else max(k - self.f, 0.0)
            return np.exp(-self.r * max(self.t, 0.0)) * intrinsic

        d1 = (np.log(self.f / k) + 0.5 * v**2 * self.t) / (v * np.sqrt(self.t))
        d2 = d1 - v * np.sqrt(self.t)

        if cp == 'call':
            return np.exp(-self.r * self.t) * (self.f * norm.cdf(d1) - k * norm.cdf(d2))
        elif cp == 'put':
            return np.exp(-self.r * self.t) * (k * norm.cdf(-d2) - self.f * norm.cdf(-d1))
        else:
            raise ValueError("cp must be either 'call' or 'put'")

    def shifted_lognormal_call(self, k, v, cp='call'):
        shifted = Black76(self.f + self.shift, self.t, self.r, 0.0)
        return shifted.lognormal_call(k + self.shift, v, cp)

    def normal_call(self, k, v, cp='call'):
        if self.t <= 0 or v <= 0:
            intrinsic = max(self.f - k, 0.0) if cp == 'call' else max(k - self.f, 0.0)
            return np.exp(-self.r * max(self.t, 0.0)) * intrinsic

        cp_sign = 1.0 if cp == 'call' else -1.0
        d = (self.f - k) / (v * np.sqrt(self.t))
        return np.exp(-self.r * self.t) * (
            cp_sign * (self.f - k) * norm.cdf(cp_sign * d) +
            v * np.sqrt(self.t) * norm.pdf(d)
        )

    def bachelier_call(self, k, v, cp='call'):
        return self.normal_call(k, v, cp)

    # Analytic Conversions (Hagan)
    @staticmethod
    def hagan_normal_to_lognormal(k, f, shift, t, v_n):
        k += shift
        f += shift
        if abs(np.log(f / k)) <= 1e-8:
            factor = k
        else:
            factor = (f - k) / np.log(f / k)

        p = [factor * (-1 / 24) * t, factor, -v_n]
        roots = np.roots(p)
        roots_real = np.extract(np.isreal(roots), np.real(roots))
        v_sln_0 = v_n / f

        if len(roots_real) == 0:
            return v_sln_0

        i_min = np.argmin(np.abs(roots_real - v_sln_0))
        return roots_real[i_min]

    @staticmethod
    def hagan_lognormal_to_normal(k, f, shift, t, v_sln):
        k += shift
        f += shift
        logFK = np.log(f / k)
        a = v_sln * np.sqrt(f / k)
        b = (1 / 24) * logFK**2
        c = (1 / 1920) * logFK**4
        d = (1 / 24) * (1 - (1 / 120) * logFK**2) * v_sln**2 * t
        e = (1 / 5760) * v_sln**4 * t**2
        return a * (1 + b + c) / (1 + d + e)

    # Generic solver for implied vol conversion
    def _solve_for_vol(self, diff, v_guess):
        v_lo, v_hi = 1e-12, max(1.0, v_guess * 10)
        for _ in range(50):
            if diff(v_lo) * diff(v_hi) < 0:
                return brentq(diff, v_lo, v_hi)
            v_hi *= 2
        res = minimize_scalar(lambda x: abs(diff(x)),
                              bounds=(1e-12, v_hi),
                              method='bounded')
        return float(res.x)

    # Conversions
    def normal_to_shifted_lognormal(self, k, v_n, cp='call'):
        return self._solve_for_vol(
            lambda v_sln: self.normal_call(k, v_n, cp) -
                          self.shifted_lognormal_call(k, v_sln, cp),
            v_n
        )

    def shifted_lognormal_to_normal(self, k, v_sln, cp='call'):
        return self._solve_for_vol(
            lambda v_n: self.normal_call(k, v_n, cp) -
                        self.shifted_lognormal_call(k, v_sln, cp),
            v_sln
        )

    def lognormal_to_lognormal(self, k, v_u_sln, u, cp='call'):
        shifted_u = Black76(self.f, self.t, self.r, u)
        return self._solve_for_vol(
            lambda v_sln: shifted_u.shifted_lognormal_call(k, v_u_sln, cp) -
                          self.shifted_lognormal_call(k, v_sln, cp),
            v_u_sln
        )

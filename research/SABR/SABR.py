import numpy as np
from abc import ABCMeta, abstractmethod
from black_price import Black76


def alpha(v_atm_ln, f, t, beta, rho, volvol):
    f_ = f**(beta - 1)
    p = [
        t * f_**3 * (1 - beta)**2 / 24,
        t * f_**2 * rho * beta * volvol / 4,
        1 + t * volvol**2 * (2 - 3 * rho**2) / 24 * f_,
        -v_atm_ln
    ]
    roots = np.roots(p)
    roots_real = np.extract(np.isreal(roots), np.real(roots))
    alpha_1st_guess = v_atm_ln * f**(1 - beta)
    i_min = np.argmin(np.abs(roots_real - alpha_1st_guess))
    return roots_real[i_min]


def _pprint(params):
    params_list = []
    for k, v in params.items():
        if isinstance(v, float):
            this_repr = f'{k}={v:.4f}'
        else:
            this_repr = f'{k}={v}'
        params_list.append(this_repr)
    return params_list


class BaseSABR(metaclass=ABCMeta):

    def __init__(self, f=0.01, shift=0, t=1, v_atm_n=0.0010, beta=1, rho=0, volvol=0):
        self.f = f
        self.shift = shift
        self.t = t
        self.v_atm_n = v_atm_n
        self.beta = beta
        self.rho = rho
        self.volvol = volvol
        self.params = dict()

    @abstractmethod
    def alpha(self):
        """Implied alpha parameter from the ATM normal vol"""
        pass

    @abstractmethod
    def fit(self, k, v):
        """Best fit the model to a discrete vol smile"""
        pass

    @abstractmethod
    def lognormal_vol(self, k):
        """Return the lognormal vol from a given strike"""
        pass

    @abstractmethod
    def normal_vol(self, k):
        """Returns the normal vol from a given strike"""
        pass

    @abstractmethod
    def call(self, k, cp='call'):
        """Abstract method for call prices"""
        pass

    def density(self, k):
        std_dev = self.v_atm_n * np.sqrt(self.t)
        dk = 1e-4 * std_dev
        d2call = self.call(k + dk) - 2 * self.call(k) + self.call(k - dk)
        return d2call / dk**2

    def get_params(self):
        return self.__dict__

    def __repr__(self):
        class_name = self.__class__.__name__
        return f"<{class_name}: {', '.join(_pprint(self.__dict__))}>"


class BaseLogNormalSABR(BaseSABR):

    def alpha(self):
        # Implement your alpha calculation for lognormal here, if needed.
        # For example, use ATM lognormal vol in self.v_atm_n or pass explicitly.
        # Placeholder return (should be replaced with proper calculation)
        return 0.02

    def lognormal_vol(self, k):
        return lognormal_vol(k, self.f, self.t, self.alpha(), self.beta, self.rho, self.volvol)

    def normal_vol(self, k):
        black = Black76(f=self.f, t=self.t, r=0, shift=self.shift)
        v_sln = self.lognormal_vol(k)
        # You need a method to convert shifted lognormal vol to normal vol in Black76
        return black.shifted_lognormal_to_normal(k=k, v_sln=v_sln)

    def call(self, k, cp='call'):
        black = Black76(f=self.f, t=self.t, r=0, shift=self.shift)
        v_sln = self.lognormal_vol(k)
        return black.shifted_lognormal_call(k=k, v=v_sln, cp=cp)


class BaseNormalSABR(BaseSABR):

    def alpha(self):
        return alpha(self.v_atm_n, self.f, self.t, self.beta, self.rho, self.volvol)

    def normal_vol(self, k):
        # Implement your normal_vol formula here (e.g., SABR normal vol formula)
        # Placeholder: return ATM normal vol for now
        return self.v_atm_n

    def lognormal_vol(self, k):
        black = Black76(f=self.f, t=self.t, r=0, shift=self.shift)
        v_n = self.normal_vol(k)
        return black.normal_to_shifted_lognormal(k=k, v_n=v_n)

    def call(self, k, cp='call'):
        black = Black76(f=self.f, t=self.t, r=0, shift=self.shift)
        v_n = self.normal_vol(k)
        return black.normal_call(k=k, v=v_n, cp=cp)


def lognormal_vol(k, f, t, alpha, beta, rho, volvol):
    if k <= 0 or f <= 0:
        return 0.
    eps = 1e-7
    logfk = np.log(f / k)
    fkbeta = (f * k) ** (1 - beta)
    a = (1 - beta) ** 2 * alpha ** 2 / (24 * fkbeta)
    b = 0.25 * rho * beta * volvol * alpha / np.sqrt(fkbeta)
    c = (2 - 3 * rho ** 2) * volvol ** 2 / 24
    d = np.sqrt(fkbeta)
    v = (1 - beta) ** 2 * logfk ** 2 / 24
    w = (1 - beta) ** 4 * logfk ** 4 / 1920
    z = volvol * np.sqrt(fkbeta) * logfk / alpha

    def _x(rho, z):
        numerator = np.sqrt(1 - 2 * rho * z + z**2) + z - rho
        denominator = 1 - rho
        return np.log(numerator / denominator)

    if abs(z) > eps:
        return alpha * z * (1 + (a + b + c) * t) / (d * (1 + v + w) * _x(rho, z))
    else:
        return alpha * (1 + (a + b + c) * t) / (d * (1 + v + w))

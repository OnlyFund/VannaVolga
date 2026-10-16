import numpy as np
import matplotlib.pyplot as plt
import math

class Hagan2002_LogNormal:
    def __init__(self, alpha, beta, rho, volvol, f, t, shift=0):
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.volvol = volvol
        self.f = f + shift
        self.t = t
        self.shift = shift

    def lognormal_vol(self, K):
        # Avoid division by zero or negative strike
        K = max(K + self.shift, 1e-12)
        f = self.f

        if abs(f - K) < 1e-12:
            # ATM formula
            term1 = ( (1 - self.beta)**2 / 24 ) * (self.alpha**2) / (f**(2 - 2*self.beta))
            term2 = 0.25 * self.rho * self.beta * self.volvol * self.alpha / (f**(1 - self.beta))
            term3 = ( (2 - 3*self.rho**2) / 24 ) * (self.volvol**2)
            vol = self.alpha / (f**(1 - self.beta)) * (1 + (term1 + term2 + term3)*self.t)
            return vol

        # Non-ATM formula
        FK_beta = (f*K)**((1 - self.beta)/2)
        log_FK = math.log(f / K)
        z = (self.volvol / self.alpha) * FK_beta * log_FK
        x_z = math.log((math.sqrt(1 - 2*self.rho*z + z**2) + z - self.rho) / (1 - self.rho))

        # Prevent division by zero
        if abs(x_z) < 1e-12:
            return self.alpha / (f**(1 - self.beta))

        A = self.alpha / (FK_beta * (1 + ((1 - self.beta)**2 / 24) * (log_FK**2) + ((1 - self.beta)**4 / 1920) * (log_FK**4)))
        B = z / x_z

        term1 = ( (1 - self.beta)**2 / 24 ) * (self.alpha**2) / ( (f*K)**(1 - self.beta) )
        term2 = 0.25 * self.rho * self.beta * self.volvol * self.alpha / ( (f*K)**((1 - self.beta)/2) )
        term3 = ( (2 - 3*self.rho**2) / 24 ) * self.volvol**2
        vol = A * B * (1 + (term1 + term2 + term3) * self.t)
        return vol

def generate_sabr_smile(fwd, t, atm_vol, beta=0.5, rho=-0.3, nu=0.5, strikes=None):
    # Compute SABR alpha to match ATM vol:
    # ATM vol from Hagan SABR approx:
    # vol_atm = alpha / (fwd^(1 - beta)) * (1 + terms)
    # Solve for alpha numerically or approximate:

    # Approximate alpha from atm_vol:
    # Rearrange ATM formula ignoring small terms for simplicity:
    alpha_approx = atm_vol * (fwd**(1 - beta))

    # Instantiate SABR model
    sabr = Hagan2002_LogNormal(alpha=alpha_approx, beta=beta, rho=rho, volvol=nu, f=fwd, t=t)

    if strikes is None:
        # generate strikes +/- 20% around forward
        strikes = np.linspace(fwd * 0.95, fwd * 1.10, 50)

    vols = np.array([sabr.lognormal_vol(K) for K in strikes])

    return strikes, vols

if __name__ == "__main__":
    # Parameters:
    forward = 1.16
    time_to_expiry = 0.5  # 6 months
    atm_vol_decimal = 0.0755  # 7.55% ATM vol in decimals

    beta = -0.2
    rho = 0.22
    nu = 0.65

    strikes, vols = generate_sabr_smile(forward, time_to_expiry, atm_vol_decimal, beta, rho, nu)

    plt.plot(strikes, vols * 100, label="SABR Smile")
    plt.axhline(atm_vol_decimal * 100, color="red", linestyle="--", label="ATM Vol")
    plt.xlabel("Strike")
    plt.ylabel("Implied Volatility (%)")
    plt.title("SABR Lognormal Volatility Smile")
    plt.legend()
    plt.show()

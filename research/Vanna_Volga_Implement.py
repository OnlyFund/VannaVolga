import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from helper_functions.helper_functions import year_fraction
from helper_functions.vanna_volga_helpers import get_strike_from_delta_pa
from pricing_utils.BlackScholes import BlackScholes
from vanna_volga_sabr import VannaVolga_SABR,VannaVolga
from market_data import MarketData
from market_data.vol_data import VolData

sns.set(style="whitegrid")

# --- Black-Scholes Call ----------------------------------------------------
def bs_call_price(S, K, T, r, sigma):
    d1 = (np.log(S/K) + 0.5*sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

# --- Your original smile plotting function ---------------------------------
def plot_smile(market, expiry_date, valuation_date, deltaBase=True, atm_conv='DN'):
    strikes, vols = [], []

    for delta, cp in delta_call_put:
        K = get_strike_from_delta_pa(
            S=market.s, fwd=market.fwd_pts, v=market.v_atm,
            expiry_date=expiry_date, valuation_date=valuation_date,
            CallPut=cp, delta=delta, atm_type=atm_conv
        )
        strikes.append(K)
        vol = VannaVolga_SABR(
            market=market,
            K=K,
            expiry_date=expiry_date,
            valuation_date=valuation_date,
            CallPut='CALL',
            deltaBase=deltaBase,
            atm_conv=atm_conv
        ).get_vol_for_strike()
        vols.append(vol)

    df = pd.DataFrame({'Delta': delta_names, 'Volatility': vols})
    plt.figure(figsize=(10,4))
    sns.lineplot(x='Delta', y='Volatility', data=df)
    plt.title("Vanna-Volga SABR Smile")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return np.array(strikes), np.array(vols)

# --- New function to compute smile & RND on dense fixed strike grid --------
def compute_smile_and_rnd_dense_grid(market, expiry_date, valuation_date):
    S = market.s
    T = year_fraction(expiry_date, valuation_date)
    r = 0.0

    # Dense strike grid 70%-130% of spot
    K_min = S * 0.95
    K_max = S * 1.05
    K_dense = np.linspace(K_min, K_max,200)

    # Compute vols on dense grid using VV-SABR
    vols_dense = []
    for K in K_dense:
        vol = VannaVolga_SABR(
            market=market,
            K=K,
            expiry_date=expiry_date,
            valuation_date=valuation_date,
            CallPut='CALL',
            deltaBase=True,
            atm_conv='DN'
        ).get_vol_for_strike()
        vols_dense.append(vol)
    vols_dense = np.array(vols_dense)

    # Smooth vols for stability
    vols_dense = gaussian_filter1d(vols_dense, sigma=5)

    # Compute call prices
    call_prices = np.array([bs_call_price(S, K, T, r, sigma) for K, sigma in zip(K_dense, vols_dense)])

    # Compute risk-neutral density (RND)
    d1 = np.gradient(call_prices, K_dense)
    d2 = np.gradient(d1, K_dense)
    density = np.maximum(d2, 0)

    # Also compute flat vol and flat skew densities for comparison
    flat_vol = market.v_atm
    call_flat_vol = np.array([bs_call_price(S, K, T, r, flat_vol) for K in K_dense])
    d1_flat = np.gradient(call_flat_vol, K_dense)
    d2_flat = np.gradient(d1_flat, K_dense)
    density_flat = np.maximum(d2_flat, 0)

    # Flat skew: linear vol skew from ATM to ends (simple linear interpolation)
    vol_low = vols_dense[0]
    vol_high = vols_dense[-1]
    vol_skew = np.linspace(vol_low, vol_high, len(K_dense))
    call_skew_vol = np.array([bs_call_price(S, K, T, r, v) for K, v in zip(K_dense, vol_skew)])
    d1_skew = np.gradient(call_skew_vol, K_dense)
    d2_skew = np.gradient(d1_skew, K_dense)
    density_skew = np.maximum(d2_skew, 0)

    # Plot RNDs
    plt.figure(figsize=(10,4))
    plt.plot(K_dense, density, label='VV-SABR RND')
    plt.plot(K_dense, density_flat, label='Flat Vol RND')
    plt.plot(K_dense, density_skew, label='Linear Skew RND')
    plt.xlabel('Strike')
    plt.ylabel('Density')
    plt.title('Risk-Neutral Density Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return K_dense, vols_dense, density

# --- Your delta grid unchanged ---------------------------------------------
delta_range = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.49,
               0.49,0.45,0.4,0.35,0.3,0.25,0.2,0.15,0.1,0.05]
call_put = ['PUT']*10 + ['CALL']*10
delta_call_put = list(zip(delta_range, call_put))
delta_names = ['5d Put','10d Put','15d Put','20d Put','25d Put','30d Put','35d Put','40d Put','45d Put','50d Put',
               '50d Call','45d Call','40d Call','35d Call','30d Call','25d Call','20d Call','15d Call','10d Call','5d Call']

if __name__ == "__main__":
    # --- Running your code -----------------------------------------------------
    vd = VolData('vol_data.csv', valuation_date='2025-09-12')
    market = MarketData.from_vol_data(vol_data=vd, target_date='2025-09-19',
                                      spot=1.1696, fwd_pts=11)

    # 1) Plot original smile (your original function)
    strikes, vols = plot_smile(market, expiry_date='2026-02-20', valuation_date='2025-11-23')

    # 2) Compute smile and RND on dense strike grid, compare to flat and flat skew
    K_dense, vols_dense, rnd_density = compute_smile_and_rnd_dense_grid(market,
                                                                       expiry_date='2026-02-20',
                                                                       valuation_date='2025-11-23')
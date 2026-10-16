from Greeks.smile_greeks import SmileGreeks
from Greeks.fixed_greeks import FixedDelta , FixedVega , FixedGamma
from pricing_utils.BlackScholes import BlackScholes
from market_data import MarketData, Modifications
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
import time

sns.set()

# Parameters
ref_spot = 1.175
k = 1.22
expiry_date = '2026-04-06'
valuation_date = '2025-10-05'
increment = 0.005
steps =40
low_bound = ref_spot * (1 - increment * steps / 2)
upper_bound = ref_spot * (1 + increment * steps / 2)
spot_range = ref_spot + increment * np.arange(-steps // 2, steps // 2)


mods = Modifications(beta_atm=0.5, beta_rr=0.5, beta_bf=0.5)

# Function to compute gammas for a single spot
def compute_gammas(s):
    market = MarketData(
        s=s,
        fwd_pts=128,
        v_atm=6.65,
        rr_25=0.48,
        rr_10=0.87,
        bf_25=0.21,
        bf_10=0.80
    )
    fixed_gamma = FixedGamma(
        market=market,
        k=k,
        expiry_date=expiry_date,
        valuation_date=valuation_date,
        CallPut='CALL',
        BuySell='BUY',
        pricing_model='vv'
    ).calculate()

    smile_gamma = SmileGreeks(
        market=market,
        mods=mods,
        k=k,
        expiry_date=expiry_date,
        valuation_date=valuation_date
    ).gamma()

    # Vectorized Black-Scholes (assuming it can handle scalar inputs here)
    bs_delta = BlackScholes(
        S=s,
        K=k,
        fwd_pts=128,
        v=6.65,
        expiry_date=expiry_date,
        valuation_date=valuation_date,
        CallPut='CALL',
        BuySell='BUY',
        Notional=1
    ).delta()

    return fixed_gamma, smile_gamma, bs_delta

def main():
    # Compile the Numba pricing kernels once before starting worker threads.
    # Concurrent first-use compilation otherwise serializes behind Numba's
    # compiler lock and makes the first run appear disproportionately slow.
    warmup_market = MarketData(
        s=ref_spot,
        fwd_pts=128,
        v_atm=6.65,
        rr_25=0.48,
        rr_10=0.87,
        bf_25=0.21,
        bf_10=0.80
    )
    FixedGamma(
        market=warmup_market,
        k=k,
        expiry_date=expiry_date,
        valuation_date=valuation_date,
        CallPut='CALL',
        BuySell='BUY',
        pricing_model='vv'
    ).calculate()

    started = time.perf_counter()
    # Parallel computation
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(compute_gammas, spot_range))
    print("Computed {} spot points in {:.3f}s".format(
        len(spot_range), time.perf_counter() - started
    ))

    # Unpack results
    fixed_gamma, smile_gamma, bs_delta = zip(*results)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(spot_range, fixed_gamma, label='Fixed-smile gamma', linewidth=2)
    plt.plot(spot_range, smile_gamma, label='Smile-recalibrated gamma', linewidth=2)
    plt.xlabel('Spot')
    plt.ylabel('Gamma per 1% spot move')
    plt.title('Fixed-smile versus smile-recalibrated gamma')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()

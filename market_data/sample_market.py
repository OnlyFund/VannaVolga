"""An illustrative sample market for the whole universe: made-up, not market data.

The app opens on it when there is no market data store (a fresh checkout,
the Streamlit Cloud deploy). Users can save it from the Markets tab ("Save
all markets"), edit the JSON and load it back.

Every number is invented: round spots, smooth rate curves, and vols built
from one vol per currency against USD with a simple correlation for the
crosses, a skew that favours the riskier currency and modest butterflies.
The forward points follow from the rate curves by covered interest parity
(``implied_rates`` turns them back into the same curves), so forwards,
discount rates and the smile are consistent. Nothing comes from Bloomberg.

``python -m market_data.sample_market [YYYY-MM-DD] [--store PATH]`` writes
a sample market into a store (the app's sample store by default).
"""
import argparse
import math
import tempfile
from pathlib import Path

import pandas as pd

from helper_functions.fx_conventions import CCY, get_pair, spot_date
from market_data.depo_tickers import DEPO_ALIASES
from market_data.discount_curves import NOMINAL_DAYS, TENORS as RATE_TENORS
from market_data.market_tables import _ql_date, forward_delivery
from market_data.universe import CCY_LIST, CURRENCIES, FX_UNIVERSE
from market_data.vol_tickers import TENORS as VOL_TENORS

SAMPLE_SOURCE = "sample"
SAMPLE_NOTE = "Illustrative sample market: made-up values, not market data"

#: Illustrative spots of the USD pairs (crosses are derived through USD).
SPOTS = {"EURUSD": 1.10, "GBPUSD": 1.30, "AUDUSD": 0.66, "NZDUSD": 0.60,
         "USDCAD": 1.36, "USDCHF": 0.88, "USDNOK": 10.50, "USDSEK": 10.20,
         "USDDKK": 6.78, "USDSGD": 1.33, "USDHKD": 7.80, "USDZAR": 18.00,
         "USDMXN": 18.50, "USDTRY": 35.00, "USDCNH": 7.20, "USDPLN": 3.95,
         "USDHUF": 360.0, "USDILS": 3.70, "USDCZK": 22.80, "USDJPY": 150.0}

#: Illustrative 6M rate per currency, in percent; curves tilt gently with tenor.
RATES = {"USD": 4.0, "EUR": 2.0, "GBP": 4.0, "AUD": 3.8, "NZD": 3.5, "CAD": 3.0,
         "CHF": 0.3, "NOK": 4.2, "SEK": 2.5, "DKK": 2.1, "JPY": 0.6, "SGD": 2.8,
         "HKD": 3.8, "ZAR": 7.5, "MXN": 9.5, "TRY": 40.0, "CNH": 1.6, "PLN": 5.0,
         "HUF": 6.5, "ILS": 4.3, "CZK": 3.5}
RATE_TILT = -0.2                     # percentage points per year, around 6M

#: Illustrative 1M ATM vol of each currency against USD, in percent.
USD_VOLS = {"EUR": 7.5, "GBP": 8.0, "AUD": 10.0, "NZD": 10.5, "CAD": 6.0,
            "CHF": 8.0, "NOK": 10.0, "SEK": 9.5, "DKK": 7.4, "JPY": 10.0,
            "SGD": 4.5, "HKD": 1.5, "ZAR": 15.0, "MXN": 13.0, "TRY": 18.0,
            "CNH": 4.5, "PLN": 9.0, "HUF": 11.0, "ILS": 8.0, "CZK": 9.0}
#: Correlation of two currencies' moves against USD, for cross vols.
BLOCS = ({"EUR", "DKK", "CHF", "SEK", "NOK", "GBP"}, {"EUR", "PLN", "HUF", "CZK"},
         {"AUD", "NZD", "CAD"}, {"SGD", "CNH", "HKD"})
BLOC_CORRELATION, DEFAULT_CORRELATION = 0.75, 0.45
MIN_VOL = 3.0

#: How much the market pays for protection against a currency falling
#: (safe havens negative): risk reversals favour calls on the safer one.
RISK = {"JPY": -1.0, "CHF": -0.8, "USD": 0.0, "EUR": 0.0, "DKK": 0.0, "HKD": 0.0,
        "GBP": 0.3, "SEK": 0.3, "CAD": 0.3, "SGD": 0.3, "CNH": 0.4, "NOK": 0.5,
        "CZK": 0.5, "ILS": 0.5, "AUD": 0.6, "NZD": 0.6, "PLN": 0.6, "HUF": 0.8,
        "ZAR": 1.2, "MXN": 1.2, "TRY": 1.5}


def _years(tenor):
    return NOMINAL_DAYS[tenor] / 365.0


def rate_curve(ccy):
    """``{tenor: decimal rate}`` for ``ccy`` at the depo tenors."""
    level = RATES[ccy]
    return {tenor: round((level + RATE_TILT * (_years(tenor) - 0.5)) / 100.0, 6)
            for tenor in RATE_TENORS}


def _correlation(a, b):
    return BLOC_CORRELATION if any(a in bloc and b in bloc for bloc in BLOCS) \
        else DEFAULT_CORRELATION


def atm_1m(pair):
    """The pair's 1M ATM vol: its USD vol, or two USD vols combined for a cross."""
    base, term = pair[:3], pair[3:]
    if "USD" in (base, term):
        return USD_VOLS[term if base == "USD" else base]
    a, b = USD_VOLS[base], USD_VOLS[term]
    rho = _correlation(base, term)
    return max(MIN_VOL, math.sqrt(a * a + b * b - 2.0 * rho * a * b))


def vol_surface(pair):
    """``{tenor: {atm, rr_25, rr_10, bf_25, bf_10}}`` in vol points, ON to 1Y."""
    base, term = pair[:3], pair[3:]
    level = atm_1m(pair)
    skew = RISK[term] - RISK[base]           # > 0: calls on the base are bid
    surface = {}
    for tenor in VOL_TENORS:
        shape = math.log(_years(tenor) * 12.0)          # 0 at 1M
        atm = level * (1.0 + 0.04 * shape)
        rr_25 = 0.12 * atm * skew * (1.0 + 0.08 * shape)
        bf_25 = 0.03 * atm * (1.0 + 0.3 * abs(skew)) * (1.0 + 0.05 * shape)
        surface[tenor] = {"atm": round(atm, 3), "rr_25": round(rr_25, 3),
                          "rr_10": round(1.85 * rr_25, 3), "bf_25": round(bf_25, 3),
                          "bf_10": round(3.2 * bf_25, 3)}
    return surface


def usd_spot(ccy):
    """USD per one unit of ``ccy``."""
    if ccy == "USD":
        return 1.0
    if f"{ccy}USD" in SPOTS:
        return SPOTS[f"{ccy}USD"]
    return 1.0 / SPOTS[f"USD{ccy}"]


def forward_points(pair, valuation_date):
    """``{tenor: pips}`` from the two rate curves (covered interest parity)."""
    base, term = pair[:3], pair[3:]
    spot = usd_spot(base) / usd_spot(term)
    divisor = float(get_pair(pair).fwd_divisor)
    spot_day = spot_date(pair, _ql_date(valuation_date))
    r_base, r_term = rate_curve(base), rate_curve(term)
    points = {}
    for tenor in RATE_TENORS:
        delivery = forward_delivery(pair, valuation_date, tenor)
        delivery = _ql_date(delivery)
        tau_base = CCY[base].day_count.yearFraction(spot_day, delivery)
        tau_term = CCY[term].day_count.yearFraction(spot_day, delivery)
        forward = spot * (1 + r_term[tenor] * tau_term) / (1 + r_base[tenor] * tau_base)
        points[tenor] = round((forward - spot) * divisor, 4)
    return points


def sample_quotes(valuation_date, pairs=FX_UNIVERSE):
    """The keyword arguments of ``MarketDataStore.save`` for a sample market."""
    depo = {DEPO_ALIASES.get(ccy, ccy): rate_curve(ccy) for ccy in CURRENCIES}
    return {"spots": {pair: SPOTS[pair] for pair in CCY_LIST},
            "depo_curves": depo,
            "vols": {pair: vol_surface(pair) for pair in pairs},
            "forward_points": {pair: forward_points(pair, valuation_date) for pair in pairs}}


def sample_store_path():
    """Where the app keeps its sample store: a temporary folder, never the repo."""
    return Path(tempfile.gettempdir()) / "vannavolga" / "sample_market_data.sqlite"


def ensure_sample_market(store, valuation_date):
    """The ``market_id`` of ``store``'s sample market dated ``valuation_date``,
    saved first when there is none (so any valuation date has one)."""
    day = pd.Timestamp(valuation_date).strftime("%Y-%m-%d")
    existing = store.snapshots(date=day)
    existing = existing[existing["market_source"] == SAMPLE_SOURCE]
    if not existing.empty:
        return int(existing["market_id"].iloc[0])
    return store.save(f"{day}T00:00:00", SAMPLE_SOURCE, note=SAMPLE_NOTE,
                      **sample_quotes(day))


def main(argv=None):
    from market_data.store import MarketDataStore
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("date", nargs="?", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--store", default=str(sample_store_path()))
    args = parser.parse_args(argv)
    market_id = ensure_sample_market(MarketDataStore(args.store), args.date)
    print(f"sample market {market_id} for {args.date} in {args.store}")


if __name__ == "__main__":
    main()

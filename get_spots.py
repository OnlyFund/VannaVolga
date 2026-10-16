"""Fetch USD spots and depo curves from Bloomberg into the market data store."""
from datetime import datetime

import pandas as pd
from xbbg.blp import bdh

from app.market_data_store import open_market_data_store
from helper_functions.depo_tickers import TENORS, depo_tickers

COMMON_CCY = 'USD'
BASE_EXCEPTIONS = ['EUR', 'GBP', 'AUD', 'NZD']
TERMS = ['CAD', 'CHF', 'NOK', 'SEK', 'DKK', 'SGD', 'HKD', 'ZAR', 'MXN', 'TRY', 'CNH', 'PLN', 'HUF', 'ILS', 'CZK', 'JPY']
CCY_LIST = [f'{base}{COMMON_CCY}' for base in BASE_EXCEPTIONS] + [f'{COMMON_CCY}{term}' for term in TERMS]
tickers = [f'{ccy} Curncy' for ccy in CCY_LIST]

CURRENCIES = sorted({COMMON_CCY, *BASE_EXCEPTIONS, *TERMS})

#: Calendar days searched back for a depo quote: fixings such as WIBOR and
#: TIIE publish a day late.
DEPO_LOOKBACK_DAYS = 7


def fetch_spots(valuation_date):
    """Return ``{pair: last price}`` for ``valuation_date``; pairs with no price are left out."""
    df = bdh(tickers=tickers,
             flds=['last price'],
             start_date=valuation_date,
             end_date=valuation_date)
    spots = {}
    for _pair in CCY_LIST:
        column = (f'{_pair} Curncy', 'last price')
        if column not in df.columns or df[column].dropna().empty:
            print(f'No spot for {_pair} on {valuation_date}')
            continue
        spots[_pair] = float(df[column].dropna().values[-1])
    return spots


def build_depo_curve(quotes):
    """``{tenor: decimal rate}`` from ``{tenor: pct rate or None}``, in ``TENORS`` order.

    A missing tenor takes the next quoted tenor's rate (bfill); tenors after
    the last quote stay missing and are left out.
    """
    curve = pd.Series([quotes.get(tenor) for tenor in TENORS], index=list(TENORS), dtype=float)
    curve = (curve / 100.0).bfill().dropna()
    return {tenor: round(float(rate), 10) for tenor, rate in curve.items()}


def fetch_depo_curves(valuation_date, currencies=CURRENCIES):
    """``{ccy: {tenor: decimal rate}}`` from the last quote on or before ``valuation_date``.

    Currencies with no ticker or no quote are left out, with a printed note.
    """
    tickers = {ccy: depo_tickers(ccy) for ccy in currencies}
    flat = sorted({ticker for by_tenor in tickers.values() for ticker in by_tenor.values()})
    start = pd.Timestamp(valuation_date) - pd.Timedelta(days=DEPO_LOOKBACK_DAYS)
    df = bdh(tickers=flat,
             flds=['last price'],
             start_date=start.strftime('%Y-%m-%d'),
             end_date=valuation_date)

    def last(ticker):
        column = (ticker, 'last price')
        if column not in df.columns or df[column].dropna().empty:
            return None
        return float(df[column].dropna().values[-1])

    curves = {}
    for ccy, by_tenor in tickers.items():
        if not by_tenor:
            print(f'No depo tickers for {ccy}')
            continue
        quotes = {tenor: last(ticker) for tenor, ticker in by_tenor.items()}
        curve = build_depo_curve(quotes)
        missing = [tenor for tenor in TENORS if quotes.get(tenor) is None]
        if not curve:
            print(f'No depo quotes for {ccy}')
            continue
        if missing:
            dropped = [tenor for tenor in TENORS if tenor not in curve]
            print(f'{ccy}: no quote for {", ".join(missing)}'
                  + (f'; {", ".join(dropped)} left out (nothing after them to bfill from)'
                     if dropped else '; filled from the next tenor'))
        curves[ccy] = curve
    return curves


def fetch_and_store(valuation_date=None, store=None):
    """Fetch spots and depo curves for ``valuation_date`` (default today) as one snapshot."""
    valuation_date = valuation_date or pd.Timestamp.today().strftime('%Y-%m-%d')
    as_of = datetime.now()
    spots = fetch_spots(valuation_date)
    curves = fetch_depo_curves(valuation_date)
    store = store or open_market_data_store()
    snapshot_id = store.save(valuation_date, as_of, spots=spots, depo_curves=curves,
                             source='bloomberg')
    print(f'Saved snapshot {snapshot_id} ({as_of:%Y-%m-%d %H:%M:%S}): {len(spots)} spots, '
          f'{len(curves)} depo curves to {store.path}')
    return snapshot_id


if __name__ == '__main__':
    fetch_and_store()

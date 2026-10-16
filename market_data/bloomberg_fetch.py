"""Fetch spots, depo curves, vol surfaces and forward points from Bloomberg into the store.

Everything is fetched for one valuation date (``bdh`` with a one-day or
short window), not as a historical bulk download.
"""
from datetime import datetime

import pandas as pd
from xbbg.blp import bdh

from market_data.depo_tickers import TENORS, depo_tickers
from market_data.fwd_tickers import TENORS as FWD_TENORS, fwd_ticker
from market_data.store import open_market_data_store
from market_data.universe import (  # noqa: F401  (re-exported for callers)
    BASE_EXCEPTIONS, CCY_LIST, COMMON_CCY, CURRENCIES, EM_BASE, EM_CROSSES, FX_UNIVERSE, G10,
    G10_CROSSES, TERMS,
)
from market_data.vol_tickers import FIELDS as VOL_FIELDS, PARAMS as VOL_PARAMS
from market_data.vol_tickers import TENORS as VOL_TENORS, vol_ticker

tickers = [f'{ccy} Curncy' for ccy in CCY_LIST]

vol_tickers = [vol_ticker(_pair, _vol, _tenor)
               for _pair in FX_UNIVERSE for _vol in VOL_PARAMS for _tenor in VOL_TENORS]


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
    """``{tenor: decimal rate}`` from ``{tenor: pct rate or None}``: the quoted
    tenors only, in tenor order (``ON`` first when quoted).

    Missing tenors are not filled here: markets discount at forward-implied
    rates, and a quoted curve is only their fallback
    (``market_data.discount_curves``), which fills gaps itself.
    """
    tenors = (['ON'] if quotes.get('ON') is not None else []) + list(TENORS)
    return {tenor: round(float(quotes[tenor]) / 100.0, 10)
            for tenor in tenors if quotes.get(tenor) is not None}


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
        if not curve:
            print(f'No depo quotes for {ccy}')
            continue
        missing = [tenor for tenor in TENORS if quotes.get(tenor) is None]
        if missing:
            print(f'{ccy}: no depo quote for {", ".join(missing)}')
        curves[ccy] = curve
    return curves


def build_vol_surface(quotes):
    """``{tenor: {field: vol}}`` from ``{(param, tenor): vol or None}``, in tenor order.

    A tenor is kept only when all five quotes (ATM, 25/10 RR, 25/10 BF) are
    there; nothing is filled in, since a made-up wing would move the smile.
    """
    surface = {}
    for tenor in VOL_TENORS:
        values = {VOL_FIELDS[param]: quotes.get((param, tenor)) for param in VOL_PARAMS}
        if all(value is not None for value in values.values()):
            surface[tenor] = {field: round(float(value), 10) for field, value in values.items()}
    return surface


def fetch_vols(valuation_date, pairs=FX_UNIVERSE):
    """``{pair: {tenor: {field: vol}}}`` quoted on ``valuation_date``.

    One request for the whole universe on that single date. Tenors missing a
    quote and pairs with no complete tenor are left out, with a printed note.
    """
    tickers = [vol_ticker(pair, param, tenor)
               for pair in pairs for param in VOL_PARAMS for tenor in VOL_TENORS]
    df = bdh(tickers=tickers,
             flds=['last price'],
             start_date=valuation_date,
             end_date=valuation_date)

    def last(ticker):
        column = (ticker, 'last price')
        if column not in df.columns or df[column].dropna().empty:
            return None
        return float(df[column].dropna().values[-1])

    vols = {}
    for pair in pairs:
        quotes = {(param, tenor): last(vol_ticker(pair, param, tenor))
                  for param in VOL_PARAMS for tenor in VOL_TENORS}
        surface = build_vol_surface(quotes)
        if not surface:
            print(f'No vol surface for {pair} on {valuation_date}')
            continue
        dropped = [tenor for tenor in VOL_TENORS if tenor not in surface]
        if dropped:
            print(f'{pair}: incomplete quotes for {", ".join(dropped)}; left out')
        vols[pair] = surface
    return vols


def fetch_forward_points(valuation_date, pairs=FX_UNIVERSE):
    """``{pair: {tenor: points}}`` quoted on ``valuation_date``, tenors ``1W`` ... ``1Y``.

    One request for the whole universe on that single date. Missing tenors
    are left out (not filled), with a printed note.
    """
    tickers = [fwd_ticker(pair, tenor) for pair in pairs for tenor in FWD_TENORS]
    df = bdh(tickers=tickers,
             flds=['last price'],
             start_date=valuation_date,
             end_date=valuation_date)

    def last(ticker):
        column = (ticker, 'last price')
        if column not in df.columns or df[column].dropna().empty:
            return None
        return float(df[column].dropna().values[-1])

    forwards = {}
    for pair in pairs:
        points = {tenor: last(fwd_ticker(pair, tenor)) for tenor in FWD_TENORS}
        quoted = {tenor: value for tenor, value in points.items() if value is not None}
        if not quoted:
            print(f'No forward points for {pair} on {valuation_date}')
            continue
        if len(quoted) < len(points):
            missing = [tenor for tenor in FWD_TENORS if tenor not in quoted]
            print(f'{pair}: no forward points for {", ".join(missing)}; left out')
        forwards[pair] = quoted
    return forwards


def fetch_and_store(valuation_date=None, store=None, vols=True):
    """Fetch spots, depo curves and (with ``vols``) vol surfaces and forward
    points for ``valuation_date`` (default today) and save them as one snapshot."""
    valuation_date = valuation_date or pd.Timestamp.today().strftime('%Y-%m-%d')
    as_of = datetime.now()
    spots = fetch_spots(valuation_date)
    curves = fetch_depo_curves(valuation_date)
    surfaces = fetch_vols(valuation_date) if vols else {}
    forwards = fetch_forward_points(valuation_date) if vols else {}
    store = store or open_market_data_store()
    snapshot_id = store.save(valuation_date, as_of, spots=spots, depo_curves=curves,
                             vols=surfaces, forward_points=forwards, source='bloomberg')
    print(f'Saved snapshot {snapshot_id} ({as_of:%Y-%m-%d %H:%M:%S}) for {valuation_date}: '
          f'{len(spots)} spots, {len(curves)} depo curves, {len(surfaces)} vol surfaces, '
          f'{len(forwards)} forward curves to {store.path}')
    from market_data.discount_curves import DiscountCurves
    for alert in DiscountCurves(store.load(snapshot_id=snapshot_id), store).basis_alerts():
        print('Basis alert:', alert)
    return snapshot_id


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('valuation_date', nargs='?', help='YYYY-MM-DD; default today')
    parser.add_argument('--no-vols', action='store_true',
                        help='skip the vol surfaces and forward points')
    args = parser.parse_args()
    fetch_and_store(args.valuation_date, vols=not args.no_vols)

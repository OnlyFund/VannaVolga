"""Fetch spots, depo curves, vol surfaces and forward points from Bloomberg into the store.

Everything is fetched for one valuation date (``bdh`` with a one-day or
short window), not as a historical bulk download.

Run for today, the command also stores today's live pillar dates
(``store_live_dates``: ``python -m market_data.live_dates --store`` in its own
process, so the levels stay apart and a failure there never touches the saved
market). ``--no-dates`` skips it; a past valuation date has no live dates.
"""
from datetime import datetime

import pandas as pd
from xbbg.blp import bdh

from market_data.xbbg_compat import patch_xbbg

from market_data.depo_tickers import TENORS, depo_tickers
from market_data.fwd_tickers import SHORT_TENORS, TENORS as FWD_TENORS, fwd_ticker
from market_data.store import open_market_data_store
from market_data.universe import (  # noqa: F401  (re-exported for callers)
    BASE_EXCEPTIONS, CCY_LIST, COMMON_CCY, CURRENCIES, EM_BASE, EM_CROSSES, FX_UNIVERSE, G10,
    G10_CROSSES, TERMS,
)
from market_data.vol_tickers import FIELDS as VOL_FIELDS, PARAMS as VOL_PARAMS
from market_data.vol_tickers import TENORS as VOL_TENORS, vol_ticker

patch_xbbg()

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
    """``{pair: {tenor: points}}`` quoted on ``valuation_date``, tenors ``1W`` ... ``1Y``
    plus the short swaps ``ON``/``TN``/``SN`` where quoted.

    One request for the whole universe on that single date. Missing tenors
    are left out (not filled), with a printed note; a missing short swap
    only means deals inside spot are valued at spot.
    """
    tenors = list(FWD_TENORS) + list(SHORT_TENORS)
    tickers = [fwd_ticker(pair, tenor) for pair in pairs for tenor in tenors]
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
        points = {tenor: last(fwd_ticker(pair, tenor)) for tenor in tenors}
        quoted = {tenor: value for tenor, value in points.items() if value is not None}
        if not any(tenor in quoted for tenor in FWD_TENORS):
            print(f'No forward points for {pair} on {valuation_date}')
            continue
        missing = [tenor for tenor in FWD_TENORS if tenor not in quoted]
        if missing:
            print(f'{pair}: no forward points for {", ".join(missing)}; left out')
        short = [tenor for tenor in SHORT_TENORS if tenor not in quoted]
        if short:
            print(f'{pair}: no {"/".join(short)} points; deals inside spot value at spot')
        forwards[pair] = quoted
    return forwards


def unusable_fetch(spots, surfaces, forwards, vols=True):
    """Why a fetch must not be saved as a market, or ``None`` when it may be.

    No spots, or (with ``vols``) no vol surfaces or no forward points: the
    market could not price (a weekend or holiday fetch, Bloomberg not
    returning data). Some pairs missing is fine; they are reported as fetched.
    """
    missing = [what for what, got in (('spots', spots), ('vol surfaces', surfaces),
                                      ('forward points', forwards))
               if not got and (vols or what == 'spots')]
    return f"no {' and no '.join(missing)}" if missing else None


def fetch_and_store(valuation_date=None, store=None, vols=True, force=False):
    """Fetch spots, depo curves and (with ``vols``) vol surfaces and forward
    points for ``valuation_date`` (default today) and save them as one market.

    Its ``date_time`` is the fetch time for today, else ``valuation_date``'s
    close (``store.data_date_time``). Returns the ``market_id``, or ``None``
    when the fetch cannot price (``unusable_fetch``) and is not saved;
    ``force`` saves it anyway, noted on the market.
    """
    from market_data.store import data_date_time
    fetched_at = datetime.now()
    valuation_date = valuation_date or fetched_at.strftime('%Y-%m-%d')
    spots = fetch_spots(valuation_date)
    curves = fetch_depo_curves(valuation_date)
    surfaces = fetch_vols(valuation_date) if vols else {}
    forwards = fetch_forward_points(valuation_date) if vols else {}
    problem = unusable_fetch(spots, surfaces, forwards, vols)
    if problem and not force:
        print(f'Not saved: the fetch for {valuation_date} has {problem} '
              f'({len(spots)} spots, {len(curves)} depo curves, {len(surfaces)} vol surfaces, '
              f'{len(forwards)} forward curves), so it cannot price; a weekend or holiday, '
              'or Bloomberg not returning data. --force saves it anyway.')
        return None
    store = store or open_market_data_store()
    date_time = data_date_time(valuation_date, fetched_at)
    # Pin the event calendar current at the fetch, so the market always
    # prices with the same events.
    event_set_id = store.latest_event_set_id(at_or_before=fetched_at)
    note = f'saved with --force: {problem}' if problem else ''
    market_id = store.save(date_time, 'bloomberg', spots=spots, depo_curves=curves,
                           vols=surfaces, forward_points=forwards, fetched_at=fetched_at,
                           note=note, event_set_id=event_set_id)
    events = f'event set {event_set_id}' if event_set_id is not None else 'no event set'
    print(f'Saved market {market_id} (bloomberg, {date_time}): {len(spots)} spots, '
          f'{len(curves)} depo curves, {len(surfaces)} vol surfaces, '
          f'{len(forwards)} forward curves, {events} to {store.path}')
    from market_data.discount_curves import DiscountCurves
    for alert in DiscountCurves(store.load(market_id=market_id), store).basis_alerts():
        print('Basis alert:', alert)
    return market_id


def store_live_dates(run=None):
    """Store today's live pillar dates (``market_data.live_dates --store``).

    Runs as its own process; returns its exit code (0: stored, 1: refused or
    failed). Never raises: the market is already saved either way.
    """
    import subprocess
    import sys
    from pathlib import Path
    run = run or subprocess.run
    try:
        done = run([sys.executable, '-m', 'market_data.live_dates', '--store'],
                   cwd=str(Path(__file__).resolve().parents[1]))
        code = done.returncode
    except Exception as error:                 # the market is saved; say so and go on
        print(f'Live pillar dates not stored: {type(error).__name__}: {error}')
        return 1
    if code:
        print('Live pillar dates not stored (see above); tenors are solved instead.')
    return code


def prewarm_app(run=None):
    """Pre-warm the app's caches for the new market (``app.prewarm``).

    Runs as its own process; returns its exit code (0: done, 1: failed).
    Never raises: the market is already saved, and the app only starts colder.
    """
    import subprocess
    import sys
    from pathlib import Path
    run = run or subprocess.run
    try:
        done = run([sys.executable, '-m', 'app.prewarm'],
                   cwd=str(Path(__file__).resolve().parents[1]))
        code = done.returncode
    except Exception as error:                 # the market is saved; say so and go on
        print(f'App not pre-warmed: {type(error).__name__}: {error}')
        return 1
    if code:
        print('App not pre-warmed (see above); it prices on first use instead.')
    return code


def main(argv=None, fetch=None, store_dates=None, today=None, warm=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('valuation_date', nargs='?', help='YYYY-MM-DD; default today')
    parser.add_argument('--no-vols', action='store_true',
                        help='skip the vol surfaces and forward points')
    parser.add_argument('--no-dates', action='store_true',
                        help="skip today's live pillar dates")
    parser.add_argument('--no-prewarm', action='store_true',
                        help="skip pre-warming the app's caches for the new market")
    parser.add_argument('--force', action='store_true',
                        help='save the market even when it cannot price (no spots, vols '
                             'or forward points)')
    args = parser.parse_args(argv)
    market_id = (fetch or fetch_and_store)(args.valuation_date, vols=not args.no_vols,
                                           force=args.force)
    if market_id is None:
        # Nothing saved: the live dates and the pre-warm would describe the
        # previous market, so they wait for the next good fetch.
        print('No market saved: live pillar dates and the app pre-warm skipped.')
        return 1
    today = today or datetime.now().strftime('%Y-%m-%d')
    if args.no_dates:
        pass
    elif args.valuation_date and args.valuation_date != today:
        print(f'No live pillar dates for {args.valuation_date}: Bloomberg has them for today only.')
    else:
        (store_dates or store_live_dates)()
    # Last, so the pre-warm sees the live dates the app will price with.
    if not (args.no_prewarm or args.no_vols):
        (warm or prewarm_app)()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

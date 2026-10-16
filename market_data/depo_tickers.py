"""Bloomberg deposit-curve tickers per currency.

A currency's ticker for a tenor is its ``DEPO_TICKERS`` root followed by the
tenor's ``TENORS`` suffix (``USOSFR`` + ``1Z Curncy`` -> ``USOSFR1Z Curncy``),
unless ``SPECIAL_TICKERS`` names that tenor's ticker outright (which may
add an ``ON`` point). ``DEPO_ALIASES`` lets a currency use another's curve
(CNH discounts on the CNY curve).

Quoted depos are stored as quoted. Markets discount at forward-implied rates
and fall back to these curves (``market_data.discount_curves``); they also
monitor the cross-currency basis.
"""

TENORS = {'1W': '1Z Curncy', '2W': '2Z Curncy', '1M': 'A Curncy', '2M': 'B Curncy',
          '3M': 'C Curncy', '6M': 'F Curncy', '9M': 'I Curncy', '1Y': '1 Curncy'}
DEPO_TICKERS = {'USD': 'USOSFR',
                'EUR': 'EESWE',
                'JPY': 'JYSO',
                'GBP': 'BPSWS',
                'CHF': 'SFSNT',
                'CAD': 'CDSO',
                'SEK': 'SKSWTN',
                'CNY': 'CCSOO',
                'AUD': 'ADSO',
                'NZD': 'NDSO',
                'SGD': 'SDSOA',
                'HKD': 'HDSQO',
                'ZAR': 'SAOIAA',
                'TRY': 'TYSO',
                'ILS': 'ISOIS'}
# MXN: the MPSW swap tickers return no data; the quoted curve is the Banxico
# overnight rate and 28-day TIIE.

SPECIAL_TICKERS = {'NOK': {_tenor: f'NKS{_tenor} Curncy' for _tenor in TENORS},
                   'AUD': {'1Y': 'ADSWAP1Q Curncy'},
                   'PLN': {'1M': 'WIBO1M Index', '3M': 'WIBO3M Index', '6M': 'PZFR0CF curncy',
                           '9M': 'PZFR0FI curncy', '1Y': 'PZFR0I1 curncy'},
                   'MXN': {'ON': 'MXONBR Index', '1M': 'MXIBTIIE Index'},
                   'HUF':{'1W':'BUBOR01W Index',
                          '2W':'BUBOR02W Index',
                          '1M':'BUBOR01M Index',
                          '2M':'BUBOR02M Index',
                          '3M':'BUBOR03M Index',
                          '6M':'HFFR0CF Curncy',
                          '9M':'HFFR0FI Curncy',
                          '1Y':'HFFR0I1 Curncy'}}

#: Currency -> the currency whose depo curve it uses.
DEPO_ALIASES = {'CNH': 'CNY'}


def depo_tickers(ccy):
    """``{tenor: ticker}`` for ``ccy`` (``ON`` first when it has one); tenors
    with no ticker are left out."""
    ccy = DEPO_ALIASES.get(ccy, ccy)
    special = SPECIAL_TICKERS.get(ccy, {})
    root = DEPO_TICKERS.get(ccy)
    tickers = {'ON': special['ON']} if 'ON' in special else {}
    for tenor, suffix in TENORS.items():
        if tenor in special:
            tickers[tenor] = special[tenor]
        elif root:
            tickers[tenor] = f'{root}{suffix}'
    return tickers

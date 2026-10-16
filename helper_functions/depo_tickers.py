"""Bloomberg deposit-curve tickers per currency.

A currency's ticker for a tenor is its ``DEPO_TICKERS`` root followed by the
tenor's ``TENORS`` suffix (``USOSFR`` + ``1Z Curncy`` -> ``USOSFR1Z Curncy``),
unless ``SPECIAL_TICKERS`` names that tenor's ticker outright.
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
                'ILS': 'ISOIS',
                'MXN': 'MPSW'}

SPECIAL_TICKERS = {'NOK': {_tenor: f'NKS{_tenor} Curncy' for _tenor in TENORS},
                   'AUD': {'1Y': 'ADSWAP1Q Curncy'},
                   'PLN': {'1M': 'WIBO1M Index', '3M': 'WIBO3M Index', '6M': 'PZFR0CF curncy',
                           '9M': 'PZFR0FI curncy', '1Y': 'PZFR0I1 curncy'},
                   'MXN': {'1M': 'MXIBTIIE Index'}}


def depo_tickers(ccy):
    """``{tenor: ticker}`` for ``ccy``; tenors with no ticker are left out."""
    special = SPECIAL_TICKERS.get(ccy, {})
    root = DEPO_TICKERS.get(ccy)
    tickers = {}
    for tenor, suffix in TENORS.items():
        if tenor in special:
            tickers[tenor] = special[tenor]
        elif root:
            tickers[tenor] = f'{root}{suffix}'
    return tickers

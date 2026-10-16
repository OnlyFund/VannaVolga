"""The FX universe the market data covers, and its quoting conventions.

Kept free of Bloomberg imports so the app and the market builders can use it.
"""
from itertools import combinations, product

COMMON_CCY = 'USD'
BASE_EXCEPTIONS = ['EUR', 'GBP', 'AUD', 'NZD']
TERMS = ['CAD', 'CHF', 'NOK', 'SEK', 'DKK', 'SGD', 'HKD', 'ZAR', 'MXN', 'TRY', 'CNH', 'PLN', 'HUF', 'ILS', 'CZK', 'JPY']
#: Every USD pair in market quoting order (EURUSD, USDJPY ...).
CCY_LIST = [f'{base}{COMMON_CCY}' for base in BASE_EXCEPTIONS] + [f'{COMMON_CCY}{term}' for term in TERMS]
CURRENCIES = sorted({COMMON_CCY, *BASE_EXCEPTIONS, *TERMS})

#: G10 in market priority order, so the first currency of each pair is the base.
G10 = [*BASE_EXCEPTIONS, COMMON_CCY, 'CAD', 'CHF', 'NOK', 'SEK', 'JPY']
EM_BASE = ['USD', 'EUR']
#: The 45 G10 crosses in market quoting order (EURGBP ... NOKSEK, SEKJPY).
G10_CROSSES = [f'{_ccy1}{_ccy2}' for _ccy1, _ccy2 in combinations(G10, 2)]
EM_CROSSES = [f'{_ccy1}{_ccy2}' for _ccy1, _ccy2 in product(EM_BASE, TERMS[5:]) if _ccy2 != 'JPY']
FX_UNIVERSE = G10_CROSSES + EM_CROSSES


def broker_strangle(pair):
    """Whether the pair's quoted butterflies are broker (market) strangles.

    G10 flies are broker strangles; the EM pairs' flies are smile strangles.
    """
    return pair not in EM_CROSSES

"""Bloomberg FX forward point tickers: ``<CODE><TENOR> Curncy``.

``CODE`` is the non-USD currency for a USD pair (``EUR1M Curncy`` for
EURUSD, ``JPY1M Curncy`` for USDJPY) and the pair itself for a cross
(``EURGBP1M Curncy``). Points are in the pair's pips: forward = spot +
points / forward divisor (``helper_functions.fx_conventions``).
"""
#: App tenor -> Bloomberg tenor (Bloomberg quotes the year as 12M).
TENORS = {'1W': '1W', '2W': '2W', '1M': '1M', '2M': '2M', '3M': '3M',
          '6M': '6M', '9M': '9M', '1Y': '12M'}


def fwd_code(pair):
    if pair[:3] == 'USD':
        return pair[3:]
    if pair[3:] == 'USD':
        return pair[:3]
    return pair


def fwd_ticker(pair, tenor):
    return f'{fwd_code(pair)}{TENORS[tenor]} Curncy'

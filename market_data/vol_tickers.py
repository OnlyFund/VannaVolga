"""Bloomberg FX vol surface tickers: ``<PAIR><PARAM><TENOR> Curncy``.

``EURUSDV1M Curncy`` is the 1M ATM vol, ``EURUSD25R1M Curncy`` the 25-delta
risk reversal, ``EURUSD25B1M Curncy`` the 25-delta butterfly; all in vol
percentage points, as the app's vol tables are.
"""
TENORS = ['ON', '1W', '2W', '1M', '2M', '3M', '6M', '9M', '1Y']
PARAMS = ['V', '25R', '10R', '25B', '10B']
#: Bloomberg parameter -> the app's vol table column.
FIELDS = {'V': 'atm', '25R': 'rr_25', '10R': 'rr_10', '25B': 'bf_25', '10B': 'bf_10'}


def vol_ticker(pair, param, tenor):
    return f'{pair}{param}{tenor} Curncy'

"""Depo rates implied by FX forward points (covered interest parity).

For a pair BASE/TERM with simple rates on each currency's day count, from
the spot date to the tenor's delivery date::

    forward / spot = (1 + r_term * tau_term) / (1 + r_base * tau_base)

so one currency's rate follows from the other's and the forward points.
Used where a currency has no depo quote for a tenor (MXN beyond its quoted
O/N and 1M points) but its USD pair has forward points.
"""

from helper_functions.fx_conventions import CCY, get_pair, spot_date
from market_data.fwd_tickers import SHORT_TENORS
from market_data.market_tables import _ql_date, forward_delivery


def implied_rates(ccy, pair, valuation_date, spot, forward_points, other_curve):
    """``{tenor: decimal rate}`` for ``ccy`` (either leg of ``pair``) implied by
    the pair's ``forward_points`` (``{tenor: pips}``) and the other leg's
    depo curve (``{tenor: decimal rate}``).

    Tenors missing either the forward points or the other leg's rate are left
    out, and so are the short swaps (``ON``/``TN``/``SN``): they are not
    spot-to-tenor points.
    """
    base, term = pair[:3], pair[3:]
    if ccy not in (base, term):
        raise ValueError(f"{ccy} is not a leg of {pair}")
    divisor = float(get_pair(pair).fwd_divisor)
    spot_day = spot_date(pair, _ql_date(valuation_date))
    rates = {}
    for tenor, points in forward_points.items():
        other = other_curve.get(tenor)
        if other is None or tenor in SHORT_TENORS:
            continue
        delivery = forward_delivery(pair, valuation_date, tenor)
        delivery = _ql_date(delivery)
        tau = {c: CCY[c].day_count.yearFraction(spot_day, delivery) for c in (base, term)}
        ratio = (spot + points / divisor) / spot
        if ccy == term:
            rate = (ratio * (1 + other * tau[base]) - 1) / tau[term]
        else:
            rate = ((1 + other * tau[term]) / ratio - 1) / tau[base]
        rates[tenor] = round(rate, 10)
    return rates

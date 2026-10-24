"""Conversion to USD with the USD spots from the market data store.

``market_data.bloomberg_fetch`` stores the spots of every USD pair (``EURUSD``, ``USDJPY``
...), so any currency converts to USD with one spot, and a cross such as
EURJPY is the product or ratio of two USD spots.
"""
import pandas as pd

USD = "USD"
#: Base-currency amounts the Portfolio reports again in USD.
#: The decay of a converted Greek, and weighted vega, convert the same way.
USD_COLUMNS = (("mtm_base", "mtm_usd"), ("pnl_base", "pnl_usd"),
               ("vega", "vega_usd"), ("vanna", "vanna_usd"), ("volga", "volga_usd"),
               ("rega", "rega_usd"), ("sega", "sega_usd"),
               *((f"{greek}_decay", f"{greek}_decay_usd")
                 for greek in ("vega", "vanna", "volga", "rega", "sega")),
               ("weighted_vega", "weighted_vega_usd"))
#: Theta is in the pair's premium currency (USD when USD is the term currency,
#: else the base currency; ``pricing.greeks.theta_currency``), so these
#: convert from that currency, not from base.
THETA_USD_COLUMNS = tuple((column, f"{column}_usd")
                          for name in ("theta", "theta_static", "theta_vol_roll", "theta_carry")
                          for column in (name, f"fixed_{name}"))
ALL_USD_COLUMNS = USD_COLUMNS + THETA_USD_COLUMNS


def theta_ccy(pair):
    """The currency theta is reported in for ``pair`` (its premium currency)."""
    from helper_functions.premium_quotes import conventional_premium_ccy
    return conventional_premium_ccy(pair) or str(pair)[:3]


def latest_snapshot(valuation_date=None, store=None, market_id=None):
    """The stored market ``market_id`` when given (the one the app prices on),
    else the latest with data on or before ``valuation_date`` (the latest of
    all without one), or ``None``. A ``market_id`` that is gone falls back to
    the latest. The latest is one that can price (a weekend fetch with no vols
    has no spots either), else the latest of all."""
    from market_data.store import open_market_data_store
    store = store or open_market_data_store()
    if market_id is not None:
        snapshot = store.load(market_id=market_id)
        if snapshot is not None:
            return snapshot
    return (store.load(on_or_before=valuation_date, priceable=True)
            or store.load(on_or_before=valuation_date))


def usd_rate(ccy, spots):
    """USD per one unit of ``ccy``, or ``nan`` when no snapshot spot covers it."""
    ccy = str(ccy).upper()
    if ccy == USD:
        return 1.0
    if f"{ccy}{USD}" in spots:
        return float(spots[f"{ccy}{USD}"])
    if f"{USD}{ccy}" in spots:
        return 1.0 / float(spots[f"{USD}{ccy}"])
    return float("nan")


def pair_spot(pair, spots):
    """Spot of ``pair`` from the snapshot, crossed through USD when needed."""
    pair = str(pair).upper()
    if pair in spots:
        return float(spots[pair])
    return usd_rate(pair[:3], spots) / usd_rate(pair[3:], spots)


def add_usd_columns(totals, spots):
    """``totals`` with the pair's ``spot``, its base-currency amounts in USD,
    and theta in USD from the pair's premium currency.

    Pairs the snapshot cannot convert get ``nan`` in the new columns.
    """
    frame = totals.copy()
    rates = pd.Series([usd_rate(pair[:3], spots) for pair in frame["currency_pair"]],
                      index=frame.index)
    frame.insert(frame.columns.get_loc("currency_pair") + 1, "spot",
                 [pair_spot(pair, spots) for pair in frame["currency_pair"]])
    for source, target in USD_COLUMNS:
        if source in frame:
            frame[target] = frame[source].astype(float) * rates
    theta_rates = pd.Series([usd_rate(theta_ccy(pair), spots) for pair in frame["currency_pair"]],
                            index=frame.index)
    for source, target in THETA_USD_COLUMNS:
        if source in frame:
            frame[target] = frame[source].astype(float) * theta_rates
    return frame

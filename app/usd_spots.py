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


def latest_snapshot(valuation_date=None, store=None):
    """The latest stored market with data on or before ``valuation_date``
    (the latest of all without one), or ``None``."""
    from market_data.store import open_market_data_store
    store = store or open_market_data_store()
    return store.load(on_or_before=valuation_date)


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
    """``totals`` with the pair's ``spot`` and its base-currency amounts in USD.

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
    return frame

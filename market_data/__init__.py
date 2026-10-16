"""Market data: the pricing input model, the stored snapshots and their sources.

* ``model``: ``MarketData`` (one expiry's quotes for the pricers) and
  ``Modifications`` (smile dynamics), re-exported here so
  ``from market_data import MarketData`` keeps working;
* ``store``: snapshots of fetched quotes by datetime (``market_data.sqlite``);
* ``bloomberg_fetch``: fetches spots, depo curves, vol surfaces and forward points
  (``python -m market_data.bloomberg_fetch``); the only module that loads
  Bloomberg, and nothing here imports it;
* ``universe``: the pairs covered and their butterfly convention;
* ``depo_tickers``, ``vol_tickers``: Bloomberg depo curve and vol surface tickers;
* ``fwd_tickers``: Bloomberg forward point tickers;
* ``market_tables``: the app's vol and forward tables, and whole markets,
  built from a stored snapshot;
* ``vol_data``, ``event_calendar``: CSV vol curves and event weights;
* ``samples/``: example vol, event calendar and market files.
"""
from market_data.model import MarketData, Modifications

__all__ = ["MarketData", "Modifications"]

"""The markets of several currency pairs, valued together.

``MarketSet`` maps each pair to its ``MarketContext`` and is the only place
the pricer asks for a market: ``market(pair)``. Each pair is still its own
quoted market (spot, forward points, term discount, vol surface, events);
a later currency layer can build these contexts from per-currency curves
without changing the pricer.

Fallbacks and failsafes, so one pair never takes down another:

* A pair whose market cannot be built becomes an ``UnavailableMarket``
  carrying the reason; only lines on that pair report it.
* A stored market valued on a different date than the set is not used: its
  quotes belong to another day. It becomes unavailable with both dates.
* A market stored under a pair key it does not contain is rejected.
* A pair with no market at all gets a clear "no market loaded" error, with
  a hint when the inverse pair is loaded (USDEUR vs EURUSD).
"""
from collections import OrderedDict

from app.market_context import (
    MarketContext, MarketContextError, UnavailableMarket, naive_date,
)
from app.market_store import normalize_pair


class MarketSet:
    """Pair -> market for one valuation date."""

    def __init__(self, valuation_date, contexts=()):
        self.valuation_date = valuation_date
        self._markets = OrderedDict()
        for context in contexts:
            self.add(context)

    @classmethod
    def from_payloads(cls, payloads, valuation_date, settings=None, market_service=None,
                      diagnostics=None, live=None):
        """Build from saved market dicts ``{pair: payload}``.

        ``live`` is a context for the pair being edited; it replaces that
        pair's stored payload, so pricing uses the tables as currently edited.
        """
        markets = cls(valuation_date)
        for key, payload in (payloads or {}).items():
            pair = normalize_pair(key)
            if live is not None and pair == live.currency_pair:
                continue
            try:
                context = MarketContext.from_payload(
                    payload, settings, market_service=market_service, diagnostics=diagnostics,
                )
                if context.currency_pair != pair:
                    raise MarketContextError(
                        f"market stored under {pair} contains {context.currency_pair}"
                    )
            except MarketContextError as exc:
                # Name the pair: in a mixed book the line alone may not say.
                reason = str(exc) if pair in str(exc) else f"{pair} {exc}"
                context = UnavailableMarket(pair, reason, valuation_date)
            markets.add(context)
        if live is not None:
            markets.add(live)
        return markets

    def add(self, context):
        """Add or replace a pair's market; a stale-dated one becomes unavailable."""
        pair = context.currency_pair
        if not isinstance(context, UnavailableMarket):
            market_date = naive_date(context.valuation_date)
            if market_date != naive_date(self.valuation_date):
                context = UnavailableMarket(
                    pair,
                    f"{pair} market is valued on {market_date:%Y-%m-%d}, not on the pricing "
                    f"date {naive_date(self.valuation_date):%Y-%m-%d}; reload or re-save it",
                    self.valuation_date,
                )
        self._markets[pair] = context
        return context

    @property
    def pairs(self):
        return list(self._markets)

    def __contains__(self, pair):
        return normalize_pair(pair) in self._markets

    def __len__(self):
        return len(self._markets)

    def market(self, pair):
        """The pair's market; raises ``MarketContextError`` if none is loaded."""
        pair = normalize_pair(pair)
        try:
            return self._markets[pair]
        except KeyError:
            inverse = pair[3:] + pair[:3]
            hint = f"; {inverse} is loaded, enter the pair as {inverse}" if inverse in self._markets else ""
            raise MarketContextError(f"No market loaded for {pair}{hint}") from None

    def available(self):
        """Pairs that can price."""
        return [pair for pair, context in self._markets.items()
                if not isinstance(context, UnavailableMarket)]

    def problems(self):
        """``{pair: reason}`` for pairs that cannot price."""
        return {pair: context.reason for pair, context in self._markets.items()
                if isinstance(context, UnavailableMarket)}

    def __repr__(self):
        return f"MarketSet({naive_date(self.valuation_date):%Y-%m-%d}: {', '.join(self.pairs)})"

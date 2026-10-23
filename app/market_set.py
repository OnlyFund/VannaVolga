"""The markets of several currency pairs, valued together.

``MarketSet`` maps each pair to its ``MarketContext`` and is the only place
the pricer asks for a market: ``market(pair)``. Each pair is still its own
quoted market (spot, forward points, term discount, vol surface, events);
a later currency layer can build these contexts from per-currency curves
without changing the pricer.

Fallbacks and failsafes, so one pair never takes down another:

* A pair whose market cannot be built becomes an ``UnavailableMarket``
  carrying the reason; only lines on that pair report it.
* Every pair is valued as of the set's date. A market saved for another
  day is valued as of the set's date too (as the app does for the market
  being edited when the valuation date changes), and ``notes()`` says so,
  since its quotes are not that day's.
* A context added directly for another date is not silently re-dated: it
  becomes unavailable with both dates.
* A market stored under a pair key it does not contain is rejected.
* A pair with no market at all gets a clear "no market loaded" error, with
  a hint when the inverse pair is loaded (USDEUR vs EURUSD).
"""
import re
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
        self._notes = OrderedDict()
        for context in contexts:
            self.add(context)

    @classmethod
    def from_payloads(cls, payloads, valuation_date, settings=None, market_service=None,
                      diagnostics=None, live=None):
        """Build from saved market dicts ``{pair: payload}``, all valued as of
        ``valuation_date``.

        A market saved for an earlier date is **rolled** to ``valuation_date``
        (``app.rolled_market``: the new date's pillar dates, forward vols
        held, forward points sticky tenor). If it cannot be rolled, or rolling
        is off (``VV_ROLL_MARKET=0``), its quotes are valued as of
        ``valuation_date`` with their own dates, as before, and the note says
        why.

        ``live`` is a context for the pair being edited; it replaces that
        pair's stored payload, so pricing uses the tables as currently edited.
        """
        markets = cls(valuation_date)
        for key, payload in (payloads or {}).items():
            pair = normalize_pair(key)
            if live is not None and pair == live.currency_pair:
                continue
            try:
                saved = payload.get("valuation_date") if isinstance(payload, dict) else None
                context = markets._rolled(payload, saved, pair, settings, market_service,
                                          diagnostics)
                if context is None:
                    context = MarketContext.from_payload(
                        payload, settings, market_service=market_service,
                        diagnostics=diagnostics, valuation_date=valuation_date,
                    )
                if context.currency_pair != pair:
                    raise MarketContextError(
                        f"market stored under {pair} contains {context.currency_pair}"
                    )
                if pair not in markets._notes:
                    markets._note_saved_date(pair, saved)
            except MarketContextError as exc:
                # Name the pair: in a mixed book the line alone may not say.
                reason = str(exc) if pair in str(exc) else f"{pair} {exc}"
                context = UnavailableMarket(pair, reason, valuation_date)
            markets.add(context)
        if live is not None and re.fullmatch(r"[A-Z]{6}", live.currency_pair):
            markets.add(live)
        return markets

    def _rolled(self, payload, saved, pair, settings, market_service, diagnostics):
        """The market rolled from its saved date to this set's date, or ``None``
        (not earlier, rolling off, or it cannot roll: noted)."""
        from app.rolled_market import RollError, roll_context, rolling_enabled
        try:
            saved_date = naive_date(saved)
        except (TypeError, ValueError):
            return None
        valuation = naive_date(self.valuation_date)
        if saved_date >= valuation:
            return None
        as_of = (f"{pair} quotes were saved for {saved_date:%Y-%m-%d} and are valued as of "
                 f"{valuation:%Y-%m-%d}")
        if not rolling_enabled():
            self._notes[pair] = as_of + " (rolling is off: VV_ROLL_MARKET=0)"
            return None
        try:
            saved_context = MarketContext.from_payload(
                payload, settings, market_service=market_service, diagnostics=diagnostics)
            rolled, notes = roll_context(saved_context, self.valuation_date)
        except (RollError, MarketContextError) as exc:
            self._notes[pair] = f"{as_of}, not rolled: {exc}"
            return None
        self._notes[pair] = "; ".join(notes)
        return rolled

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

    def with_spots(self, spots):
        """A copy whose pairs in ``spots`` (``{pair: spot}``) price at those spots.

        Only the spot moves; vol quotes and forward points stay as quoted.
        Unavailable pairs and pairs not in the set are left as they are.
        """
        moved = MarketSet(self.valuation_date)
        moved._notes = OrderedDict(self._notes)
        spots = {normalize_pair(pair): value for pair, value in (spots or {}).items()}
        for pair, context in self._markets.items():
            if pair in spots and not isinstance(context, UnavailableMarket):
                context = context.with_spot(spots[pair])
            moved._markets[pair] = context
        return moved

    def _note_saved_date(self, pair, saved):
        try:
            saved_date = naive_date(saved)
        except (TypeError, ValueError):
            return
        if saved_date != naive_date(self.valuation_date):
            self._notes[pair] = (
                f"{pair} quotes were saved for {saved_date:%Y-%m-%d} and are valued as of "
                f"{naive_date(self.valuation_date):%Y-%m-%d}"
            )

    def notes(self):
        """``{pair: note}`` for pairs that price, but on another day's quotes."""
        return {pair: note for pair, note in self._notes.items()
                if pair in self._markets
                and not isinstance(self._markets[pair], UnavailableMarket)}

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

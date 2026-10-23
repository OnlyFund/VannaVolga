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
from app.rolled_market import RollError, rolled, rolling_enabled


class MarketSet:
    """Pair -> market for one valuation date."""

    def __init__(self, valuation_date, contexts=()):
        self.valuation_date = valuation_date
        self._markets = OrderedDict()
        self._notes = OrderedDict()
        #: Pairs to roll when first priced: ``{pair: (payload, settings,
        #: market_service, diagnostics, spot or None)}`` (``_plan_roll``).
        self._pending = {}
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
                context = MarketContext.from_payload(
                    payload, settings, market_service=market_service,
                    diagnostics=diagnostics, valuation_date=valuation_date,
                )
                markets._plan_roll(payload, saved, pair, settings, market_service, diagnostics)
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

    def _plan_roll(self, payload, saved, pair, settings, market_service, diagnostics):
        """A market saved before this set's date is rolled, lazily: when the pair
        is first asked for (``market``), since a roll builds every pillar's smile.
        Until then it holds its quotes valued as of the set's date."""
        try:
            saved_date = naive_date(saved)
        except (TypeError, ValueError):
            return
        valuation = naive_date(self.valuation_date)
        if saved_date >= valuation:
            return
        as_of = (f"{pair} quotes were saved for {saved_date:%Y-%m-%d} and are valued as of "
                 f"{valuation:%Y-%m-%d}")
        if not rolling_enabled():
            self._notes[pair] = as_of + " (rolling is off: VV_ROLL_MARKET=0)"
            return
        self._notes[pair] = (f"{pair} quotes of {saved_date:%Y-%m-%d}, rolled to "
                             f"{valuation:%Y-%m-%d} when priced")
        self._pending[pair] = (payload, settings, market_service, diagnostics, None)

    def _resolve(self, pair):
        """Roll a pending pair now; on failure it keeps its quotes valued as of
        the set's date, and the note says why."""
        payload, settings, market_service, diagnostics, spot = self._pending.pop(pair)
        saved_date = naive_date(payload.get("valuation_date"))
        try:
            saved_context = MarketContext.from_payload(
                payload, settings, market_service=market_service, diagnostics=diagnostics)
            if spot is not None:
                saved_context = saved_context.with_spot(spot)
            context, notes = rolled(saved_context, self.valuation_date)
        except (RollError, MarketContextError) as exc:
            self._notes[pair] = (f"{pair} quotes were saved for {saved_date:%Y-%m-%d} and are "
                                 f"valued as of {naive_date(self.valuation_date):%Y-%m-%d}, "
                                 f"not rolled: {exc}")
            return
        self._markets[pair] = context
        self._notes[pair] = "; ".join(notes)

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
        # A pair still to roll rolls at its moved spot.
        moved._pending = {pair: entry[:4] + (spots.get(pair, entry[4]),)
                          for pair, entry in self._pending.items()}
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
        if pair in self._pending:
            self._resolve(pair)
        try:
            return self._markets[pair]
        except KeyError:
            inverse = pair[3:] + pair[:3]
            hint = f"; {inverse} is loaded, enter the pair as {inverse}" if inverse in self._markets else ""
            raise MarketContextError(f"No market loaded for {pair}{hint}") from None

    def spot(self, pair):
        """The pair's spot, without rolling it (a roll never moves spot)."""
        pair = normalize_pair(pair)
        if pair not in self._markets:
            return self.market(pair).spot                  # raises with the reason
        return self._markets[pair].spot

    def rolls_when_priced(self, pair):
        """Whether the pair is still to be rolled (``_plan_roll``)."""
        return normalize_pair(pair) in self._pending

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

"""Test helpers: resolve hand-built markets the way the app does.

The core reads no calendar: a market carries its dates and conventions
(``atlas.resolved``), resolved by Hyperion (``apeiron.hyperion.resolve``).
Tests that build ``MarketData`` by hand resolve it here, so they exercise the
same inputs the app's markets carry.
"""
from apeiron.hyperion.resolve import attach
from resolved_smile_inputs import MarketSnapshot
from resolved_smile_inputs import resolve_conventions as core_resolve_conventions

#: The dates the smile tests use by default.
EXPIRY, VALUATION = "2026-12-02", "2026-10-02"


def resolved(market, expiry=EXPIRY, valuation=VALUATION):
    """A copy of ``market`` carrying its inputs for ``expiry`` seen from ``valuation``."""
    return attach(market.copy_with(resolved=None), expiry, valuation)


def resolve_conventions(raw, expiry, valuation):
    """As the app: resolve the market's inputs, then the core's conventions."""
    market = raw.to_market() if isinstance(raw, MarketSnapshot) else raw
    return core_resolve_conventions(MarketSnapshot.capture(resolved(market, expiry, valuation)),
                                    expiry, valuation)

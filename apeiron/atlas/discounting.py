"""Convention discounting from a market's resolved inputs (no calendars here).

``discount_kwargs(market, expiry, valuation, delivery=None)`` returns the
``BlackScholes`` keyword arguments for discounting to delivery on the term
currency's basis - the same dict ``helper_functions.fx_discounting.
discount_kwargs`` returns - computed with plain arithmetic from
``market.resolved`` (``atlas.resolved.ResolvedInputs``) and the market's rate:

* ``term_discount_factor`` = ``1 / (1 + rate * year_fraction)``;
* ``carry_rate`` = ``rate * term_discount_factor``;
* ``delta_convention`` for a wrapped smile's option (as resolved).

An empty dict keeps the library's legacy ``exp(-r * t)``, exactly as before:
discounting switched off, no pair, or a factor out of range.

A request the carried inputs do not describe (another expiry, valuation or
delivery, or a market built without them, as a standalone user's) gets the
legacy discounting too: Atlas reads no calendar. Such requests are counted in
``FALLBACKS`` by reason, so a pricing path that lost its inputs shows up.
"""
from collections import Counter

from apeiron.atlas.resolved import _day as _date_of, simple_discount_factor

#: ``{reason: count}`` of requests not served from ``market.resolved``.
FALLBACKS = Counter()


def _fallback(reason, market, expiry, valuation, delivery):
    FALLBACKS[reason] += 1
    return {}


def discount_kwargs(market, expiry, valuation, delivery=None):
    if market is None:
        return {}
    if not getattr(market, "settlement_discounting", True):
        return {}
    if not getattr(market, "currency_pair", None):
        return {}
    resolved = getattr(market, "resolved", None)
    if resolved is None:
        return _fallback("market without resolved inputs", market, expiry, valuation, delivery)
    if not resolved.matches(expiry, valuation, delivery):
        return _fallback("request differs from the resolved dates", market, expiry, valuation,
                         delivery)
    rate = float(getattr(market, "term_discount", 0.0))
    try:
        factor = simple_discount_factor(rate, resolved.year_fraction_from(valuation))
    except ValueError:
        return {}
    if not (0.0 < factor < 10.0):                # as fx_discounting._sane
        return {}
    result = {"term_discount_factor": factor, "carry_rate": rate * factor}
    if delivery is not None:
        return result
    if (getattr(market, "wrapped_smile", False) and not market.force_legacy_smile
            and resolved.expiry_date > _date_of(valuation)):
        result["delta_convention"] = resolved.delta_convention_from(valuation)
    return result

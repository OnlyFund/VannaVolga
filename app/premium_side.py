"""Premium currency as a side of the pair: ``BASE`` or ``TERM``.

The Pricer grid and the Blotter offer only these two, so a premium can never
be in a currency outside its pair, and changing a line's pair carries the
choice over (Term on EURUSD is USD; on USDJPY it is JPY). Pricing and
booking still work in currency codes: ``premium_currency`` turns a side into
the code in one place, and also passes a code through, so lines and files
written before keep working.

Fallbacks: a code that is not in the pair becomes ``TERM`` with a note
(``premium_side``); ``premium_currency`` returns such a code unchanged, so
the pricer's and the trade model's checks still reject it.
"""
BASE, TERM = "BASE", "TERM"
SIDES = (BASE, TERM)


def _pair(pair):
    pair = "" if pair is None else str(pair).strip().upper()
    return pair if len(pair) == 6 and pair.isalpha() else ""


def _text(value):
    if value is None:
        return ""
    try:
        if value != value:                   # NaN
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip().upper()


def premium_side(pair, value):
    """``(side, note)``: ``value`` (a side, a currency code or blank) as a side.

    Blank is ``TERM`` (the market default). A code outside ``pair`` is
    ``TERM`` with a note saying so; ``note`` is ``None`` otherwise.
    """
    value, pair = _text(value), _pair(pair)
    if value in SIDES:
        return value, None
    if not value:
        return TERM, None
    if pair and value == pair[:3]:
        return BASE, None
    if pair and value == pair[3:]:
        return TERM, None
    shown = pair or "the pair"
    return TERM, f"premium currency {value} is not in {shown}; set to Term"


def premium_currency(pair, value):
    """The currency code for ``value`` (a side or a code) on ``pair``.

    A side needs a valid pair (else ``""``); blank is the term currency; a
    code is returned as given, so a wrong one is still caught downstream.
    """
    value, pair = _text(value), _pair(pair)
    if value == BASE:
        return pair[:3]
    if value == TERM or not value:
        return pair[3:]
    return value

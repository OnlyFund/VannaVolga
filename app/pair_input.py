"""Safeguard for the Markets tab's typed currency pair.

A typed pair is accepted only when it is six letters (after removing "/" and
spaces and upper-casing) and the app has conventions for it. Anything else
is rejected and the input falls back to a pair that works:

1. the last valid active pair;
2. EURUSD, when the market set has it;
3. the first pair in the market set;
4. EURUSD.

A rejected input never becomes the active pair, so it cannot copy or replace
any market.
"""
import re

from app.market_store import normalize_pair

DEFAULT_PAIR = "EURUSD"


def pair_problem(value):
    """``None`` for a usable pair, else why it is not one."""
    pair = normalize_pair(value)
    if not re.fullmatch(r"[A-Z]{6}", pair):
        return "a currency pair is six letters, e.g. EURUSD"
    try:
        from helper_functions.fx_conventions import get_pair
        get_pair(pair)
    except KeyError as exc:
        # "USDEUR is quoted as EURUSD in the market; invert ..." -> first clause.
        return str(exc).strip("'\"").split(";")[0]
    except Exception as exc:        # conventions unavailable: judge by shape only
        return None if isinstance(exc, ImportError) else str(exc)
    return None


def fallback_pair(active=None, library=()):
    """The pair to fall back to; see the module docstring for the order."""
    pairs = list(library)
    for candidate in (active, DEFAULT_PAIR if DEFAULT_PAIR in pairs else None,
                      pairs[0] if pairs else None):
        if candidate and pair_problem(candidate) is None:
            return normalize_pair(candidate)
    return DEFAULT_PAIR


def resolve_pair_input(value, active=None, library=()):
    """``(pair, message)``: the typed pair, normalized, or the fallback.

    ``message`` is ``None`` when the input is accepted and says what was
    rejected and where the input went back to otherwise.
    """
    problem = pair_problem(value)
    if problem is None:
        return normalize_pair(value), None
    fallback = fallback_pair(active, library)
    typed = str(value).strip() if value is not None else ""
    return fallback, f"'{typed}' is not a supported currency pair ({problem}); kept {fallback}."

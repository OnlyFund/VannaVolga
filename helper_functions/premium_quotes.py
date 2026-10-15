"""FX premium quotation conventions.

Option values are computed internally as a **cash amount in the term
currency** for a ``notional`` expressed in the **base** currency.  Traders
quote that same value in several different units, so this module owns the one
set of conversions used by the Black-Scholes reference pricer, the portfolio
valuation layer and the Streamlit UI.

Two conventions are easy to get wrong and are centralised here deliberately:

* the term notional is ``notional * strike``, **not** ``notional * spot`` --
  percent-of-term is struck-notional based, and using spot overstates it by
  ``strike / spot``;
* cash-to-cash conversion uses **spot**, whereas the pips and percent forms
  are struck-notional based and therefore involve **strike**.
"""

import numpy as np

#: Supported premium quotation styles.
PREMIUM_QUOTES = (
    "term_cash",
    "base_cash",
    "term_pips",
    "base_pips",
    "pct_term",
    "pct_base",
)

#: Quotes that are per-unit-of-notional and so do not scale with trade size.
RELATIVE_QUOTES = ("term_pips", "base_pips", "pct_term", "pct_base")

#: Quotes whose value is an absolute cash amount.
CASH_QUOTES = ("term_cash", "base_cash")


def premium_quote_label(quote, base_ccy="BASE", term_ccy="TERM"):
    """Return a short human-readable label for ``quote``."""
    base = str(base_ccy).strip().upper() or "BASE"
    term = str(term_ccy).strip().upper() or "TERM"
    labels = {
        "term_cash": "{} cash".format(term),
        "base_cash": "{} cash".format(base),
        "term_pips": "{} pips".format(term),
        "base_pips": "{} pips".format(base),
        "pct_term": "% {}".format(term),
        "pct_base": "% {}".format(base),
    }
    return labels.get(quote, str(quote))


def premium_quote_currency(quote, base_ccy, term_ccy):
    """Return the currency leg a quote is denominated against.

    Every convention has a well-defined leg: the ``*_cash`` quotes are amounts
    in that currency, ``term_pips``/``base_pips`` are pips of it, and
    ``pct_term``/``pct_base`` are percentages of that leg's notional.  Unknown
    quotes return ``None``.
    """
    if quote in ("term_cash", "term_pips", "pct_term"):
        return str(term_ccy).strip().upper()
    if quote in ("base_cash", "base_pips", "pct_base"):
        return str(base_ccy).strip().upper()
    return None


def conventional_premium_ccy(pair):
    """Return the pair's market-conventional premium currency.

    Premium is paid in USD when USD is the term currency (EURUSD, GBPUSD) and
    in the base currency otherwise (USDJPY, EURGBP, AUDJPY). Returns ``None``
    for anything that is not a six-letter pair.
    """
    code = str(pair or "").replace("/", "").replace(" ", "").strip().upper()
    if len(code) != 6 or not code.isalpha():
        return None
    return code[3:] if code[3:] == "USD" else code[:3]


def validate_premium_quote(quote):
    """Return ``quote`` normalized, raising ``ValueError`` if unsupported."""
    normalized = str(quote).strip().lower()
    if normalized not in PREMIUM_QUOTES:
        raise ValueError(
            "Unknown premium quote {!r}; expected one of {}".format(
                quote, ", ".join(PREMIUM_QUOTES)
            )
        )
    return normalized


def _unwrap(value):
    """Return a plain ``float`` for scalar results, preserving array inputs.

    ``convert_premium_quote`` works in numpy so it supports vectorised
    portfolio columns, but the overwhelmingly common case is a single trade.
    Handing a 0-d ``ndarray`` back to the UI would format as ``array(1.23)``,
    so scalars are unwrapped here.
    """
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return float(array)
    return array


def convert_premium_quote(value_term, quote, spot, strike=None, notional=None):
    """Re-express a term-currency cash amount in a market quotation convention.

    ``value_term`` is cash in the term currency for ``notional`` units of the
    base currency.  Only the reporting units change; the economic value of the
    trade is untouched, and the sign of ``value_term`` is preserved.

    ``strike`` is required for the struck-notional conventions (``pct_term``,
    ``base_pips``) and ``notional`` for every relative convention.  Raises
    ``ValueError`` when a conversion is not well defined; callers that must not
    fail should use :func:`safe_convert_premium_quote`.
    """
    quote = validate_premium_quote(quote)
    value = np.asarray(value_term, dtype=float)

    if quote == "term_cash":
        return _unwrap(value)

    spot_arr = np.asarray(spot, dtype=float)
    if not np.all(np.isfinite(spot_arr)) or np.any(spot_arr <= 0.0):
        raise ValueError("Spot must be positive and finite to convert the premium")
    base_cash = value / spot_arr
    if quote == "base_cash":
        return _unwrap(base_cash)

    if notional is None:
        raise ValueError("Notional is required to quote a premium per notional")
    notional_arr = np.asarray(notional, dtype=float)
    if not np.all(np.isfinite(notional_arr)) or np.any(notional_arr == 0.0):
        raise ValueError("Notional must be non-zero and finite to quote per notional")

    if quote == "term_pips":
        return _unwrap(value / notional_arr * 1e4)
    if quote == "pct_base":
        return _unwrap(base_cash / notional_arr * 100.0)

    if strike is None:
        raise ValueError("Strike is required to quote against the term notional")
    strike_arr = np.asarray(strike, dtype=float)
    if not np.all(np.isfinite(strike_arr)) or np.any(strike_arr <= 0.0):
        raise ValueError("Strike must be positive and finite to quote against the term notional")

    # The term notional is struck: notional * strike, never notional * spot.
    term_notional = notional_arr * strike_arr
    if quote == "pct_term":
        return _unwrap(value / term_notional * 100.0)
    return _unwrap(base_cash / term_notional * 1e4)


def safe_convert_premium_quote(value_term, quote, spot, strike=None,
                               notional=None, default=None):
    """Best-effort :func:`convert_premium_quote` that never raises.

    Returns ``default`` when the inputs do not define the requested quote, so
    a single malformed row cannot break a whole portfolio report.
    """
    if value_term is None:
        return default
    try:
        converted = convert_premium_quote(
            value_term, quote, spot, strike=strike, notional=notional
        )
    except (ValueError, TypeError, ZeroDivisionError, FloatingPointError):
        return default
    if converted is None:
        return default
    if np.isscalar(converted) and not np.isfinite(converted):
        return default
    return converted


def format_premium_quote(value, quote, default="—"):
    """Format a converted premium for display, never raising on bad input.

    Cash amounts are shown with thousands separators, pips to two decimals and
    percentages to four, matching how each convention is quoted in the market.
    """
    if value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(number):
        return default
    if quote in CASH_QUOTES:
        return "{:,.0f}".format(number)
    if quote in ("term_pips", "base_pips"):
        return "{:,.2f}".format(number)
    return "{:,.4f}".format(number)



    """Return a plain float for scalar results, preserving arrays otherwise."""
    array = np.asarray(value)
    if array.ndim == 0:
        return float(array)
    return array

"""Tests for the premium quotation helpers and their failsafe layers.

``tests/test_premium_quotes.py`` covers the numerical conversions through
``BlackScholes.premium``.  This module targets the three-tier failure policy
used by the app: strict conversion raises, the safe wrapper degrades to a
default, and the formatter degrades to a placeholder.
"""

import numpy as np
import pytest

from helper_functions.premium_quotes import (
    CASH_QUOTES,
    PREMIUM_QUOTES,
    RELATIVE_QUOTES,
    convert_premium_quote,
    format_premium_quote,
    premium_quote_currency,
    premium_quote_label,
    safe_convert_premium_quote,
    validate_premium_quote,
)

VALUE = 18131.9420
SPOT = 1.1386
STRIKE = 1.1480
NOTIONAL = 1_000_000.0


def _convert(quote, **overrides):
    kwargs = {"spot": SPOT, "strike": STRIKE, "notional": NOTIONAL}
    kwargs.update(overrides)
    return convert_premium_quote(VALUE, quote, **kwargs)


def _safe(quote, **overrides):
    kwargs = {"spot": SPOT, "strike": STRIKE, "notional": NOTIONAL}
    kwargs.update(overrides)
    return safe_convert_premium_quote(VALUE, quote, **kwargs)


# --------------------------------------------------------------------------
# quote registry
# --------------------------------------------------------------------------

def test_quote_groups_partition_the_registry():
    assert set(CASH_QUOTES) | set(RELATIVE_QUOTES) == set(PREMIUM_QUOTES)
    assert not set(CASH_QUOTES) & set(RELATIVE_QUOTES)


@pytest.mark.parametrize("quote", PREMIUM_QUOTES)
def test_validate_accepts_every_supported_quote(quote):
    assert validate_premium_quote(quote) == quote


@pytest.mark.parametrize("raw", ["  TERM_CASH  ", "Base_Cash", "PCT_Term"])
def test_validate_normalizes_case_and_whitespace(raw):
    assert validate_premium_quote(raw) == raw.strip().lower()


@pytest.mark.parametrize("bad", ["", "cash", "term cash", None, 3])
def test_validate_rejects_unsupported_quotes(bad):
    with pytest.raises(ValueError):
        validate_premium_quote(bad)


@pytest.mark.parametrize("quote", PREMIUM_QUOTES)
def test_labels_are_distinct_and_mention_the_pair(quote):
    label = premium_quote_label(quote, "EUR", "USD")
    assert label and ("EUR" in label or "USD" in label)


def test_labels_are_unique_across_quotes():
    labels = {premium_quote_label(q, "EUR", "USD") for q in PREMIUM_QUOTES}
    assert len(labels) == len(PREMIUM_QUOTES)


@pytest.mark.parametrize("quote,expected", [
    ("term_cash", "USD"),
    ("term_pips", "USD"),
    ("pct_term", "USD"),
    ("base_cash", "EUR"),
    ("base_pips", "EUR"),
    ("pct_base", "EUR"),
])
def test_quote_currency_follows_the_quote_leg(quote, expected):
    assert premium_quote_currency(quote, "EUR", "USD") == expected


# --------------------------------------------------------------------------
# strict conversion
# --------------------------------------------------------------------------

def test_term_cash_is_the_identity():
    assert _convert("term_cash") == pytest.approx(VALUE, abs=1e-12)


def test_term_cash_needs_neither_spot_strike_nor_notional():
    assert convert_premium_quote(VALUE, "term_cash", None) == pytest.approx(VALUE)


def test_cash_conversion_uses_spot():
    assert _convert("base_cash") == pytest.approx(VALUE / SPOT, rel=1e-12)


def test_pct_term_uses_struck_notional_not_spot_notional():
    assert _convert("pct_term") == pytest.approx(
        VALUE / (NOTIONAL * STRIKE) * 100.0, rel=1e-12)
    assert _convert("pct_term") != pytest.approx(
        VALUE / (NOTIONAL * SPOT) * 100.0, rel=1e-9)


def test_base_pips_uses_struck_notional():
    assert _convert("base_pips") == pytest.approx(
        VALUE / SPOT / (NOTIONAL * STRIKE) * 1e4, rel=1e-12)


@pytest.mark.parametrize("quote", PREMIUM_QUOTES)
def test_conversions_return_plain_floats(quote):
    assert type(_convert(quote)) is float


@pytest.mark.parametrize("quote", PREMIUM_QUOTES)
def test_sign_is_preserved(quote):
    kwargs = {"spot": SPOT, "strike": STRIKE, "notional": NOTIONAL}
    assert convert_premium_quote(-VALUE, quote, **kwargs) == pytest.approx(
        -convert_premium_quote(VALUE, quote, **kwargs), rel=1e-12)


@pytest.mark.parametrize("quote", PREMIUM_QUOTES)
def test_vector_input_is_preserved_as_an_array(quote):
    values = np.array([VALUE, 2.0 * VALUE])
    result = convert_premium_quote(
        values, quote, SPOT, strike=STRIKE, notional=NOTIONAL)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2,)
    assert result[1] == pytest.approx(2.0 * result[0], rel=1e-12)


@pytest.mark.parametrize("quote", [q for q in PREMIUM_QUOTES if q != "term_cash"])
@pytest.mark.parametrize("bad_spot", [0.0, -1.1, float("nan"), float("inf")])
def test_non_positive_spot_raises(quote, bad_spot):
    with pytest.raises(ValueError):
        _convert(quote, spot=bad_spot)


@pytest.mark.parametrize("quote", RELATIVE_QUOTES)
def test_missing_notional_raises(quote):
    with pytest.raises(ValueError):
        _convert(quote, notional=None)


@pytest.mark.parametrize("quote", RELATIVE_QUOTES)
@pytest.mark.parametrize("bad_notional", [0.0, float("nan"), float("inf")])
def test_degenerate_notional_raises(quote, bad_notional):
    with pytest.raises(ValueError):
        _convert(quote, notional=bad_notional)


@pytest.mark.parametrize("quote", ["pct_term", "base_pips"])
def test_missing_strike_raises_for_struck_notional_quotes(quote):
    with pytest.raises(ValueError):
        _convert(quote, strike=None)


@pytest.mark.parametrize("quote", ["pct_term", "base_pips"])
@pytest.mark.parametrize("bad_strike", [0.0, -1.0, float("nan")])
def test_degenerate_strike_raises_for_struck_notional_quotes(quote, bad_strike):
    with pytest.raises(ValueError):
        _convert(quote, strike=bad_strike)


@pytest.mark.parametrize("quote", ["term_pips", "pct_base"])
def test_spot_notional_quotes_do_not_need_a_strike(quote):
    assert _convert(quote, strike=None) == pytest.approx(_convert(quote))


# --------------------------------------------------------------------------
# safe conversion -- must never raise
# --------------------------------------------------------------------------

@pytest.mark.parametrize("quote", PREMIUM_QUOTES)
def test_safe_matches_strict_on_valid_input(quote):
    assert _safe(quote) == pytest.approx(_convert(quote), rel=1e-12)


@pytest.mark.parametrize("quote", [q for q in PREMIUM_QUOTES if q != "term_cash"])
@pytest.mark.parametrize("bad_spot", [0.0, -1.1, float("nan"), float("inf"), None])
def test_safe_returns_default_on_bad_spot(quote, bad_spot):
    assert _safe(quote, spot=bad_spot) is None


@pytest.mark.parametrize("quote", RELATIVE_QUOTES)
@pytest.mark.parametrize("bad_notional", [0.0, None, float("nan")])
def test_safe_returns_default_on_bad_notional(quote, bad_notional):
    assert _safe(quote, notional=bad_notional) is None


@pytest.mark.parametrize("quote", ["pct_term", "base_pips"])
@pytest.mark.parametrize("bad_strike", [0.0, -1.0, None, float("nan")])
def test_safe_returns_default_on_bad_strike(quote, bad_strike):
    assert _safe(quote, strike=bad_strike) is None


@pytest.mark.parametrize("bad_value", [None, float("nan"), float("inf"), "abc", object()])
def test_safe_returns_default_on_bad_value(bad_value):
    assert safe_convert_premium_quote(
        bad_value, "term_cash", SPOT, strike=STRIKE, notional=NOTIONAL) is None


def test_safe_returns_default_on_unknown_quote():
    assert _safe("not_a_quote") is None


def test_safe_honours_a_custom_default():
    sentinel = "n/a"
    assert safe_convert_premium_quote(
        VALUE, "base_cash", 0.0, default=sentinel) == sentinel


# --------------------------------------------------------------------------
# formatting -- must never raise
# --------------------------------------------------------------------------

@pytest.mark.parametrize("quote", CASH_QUOTES)
def test_cash_is_formatted_with_thousands_separators(quote):
    assert format_premium_quote(18131.942, quote) == "18,132"


@pytest.mark.parametrize("quote", ["term_pips", "base_pips"])
def test_pips_are_formatted_to_two_decimals(quote):
    assert format_premium_quote(181.3194, quote) == "181.32"


@pytest.mark.parametrize("quote", ["pct_term", "pct_base"])
def test_percent_is_formatted_to_four_decimals(quote):
    assert format_premium_quote(1.579421, quote) == "1.5794"


@pytest.mark.parametrize("quote", PREMIUM_QUOTES)
@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), "abc", object()])
def test_format_degrades_to_the_placeholder(quote, bad):
    assert format_premium_quote(bad, quote) == "—"


def test_format_honours_a_custom_default():
    assert format_premium_quote(None, "term_cash", default="") == ""


def test_format_tolerates_an_unknown_quote():
    # Unknown quotes must still render rather than break a table row.
    assert format_premium_quote(1.5, "not_a_quote") not in (None,)


def test_format_preserves_negative_signs():
    assert format_premium_quote(-18131.942, "term_cash") == "-18,132"

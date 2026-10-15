"""Premium quotation conventions on :class:`BlackScholes`.

``price()`` returns a term-currency cash amount for a base-currency notional.
``premium(quote=...)`` only changes the reporting units, so these tests pin
both the absolute values and the identities that relate the conventions.
"""
import pytest

from pricing_utils.BlackScholes import BlackScholes

VALUATION = "2026-09-25"
EXPIRY = "2027-03-25"
SPOT = 1.1386
STRIKE = 1.1480
NOTIONAL = 1_000_000.0


def _bs(**overrides):
    kwargs = dict(
        S=SPOT, K=STRIKE, fwd_pts=84.0, v=5.92, expiry_date=EXPIRY,
        valuation_date=VALUATION, CallPut="CALL", BuySell="BUY",
        Notional=NOTIONAL, deltaBase=True, term_discount=0.05,
    )
    kwargs.update(overrides)
    return BlackScholes(**kwargs)


@pytest.mark.parametrize("quote,expected", [
    ("term_cash", 18131.9420),
    ("base_cash", 15924.7690),
    ("term_pips", 181.3194),
    ("base_pips", 138.7175),
    ("pct_term", 1.5794),
    ("pct_base", 1.5925),
])
def test_premium_quote_values(quote, expected):
    assert _bs().premium(quote) == pytest.approx(expected, abs=1e-3)


def test_premium_defaults_to_price():
    bs = _bs()
    assert bs.premium() == bs.price()
    assert bs.premium("term_cash") == bs.price()


def test_base_cash_is_term_cash_at_spot():
    bs = _bs()
    assert bs.premium("base_cash") == pytest.approx(bs.price() / SPOT, rel=1e-15)


def test_pct_term_uses_struck_notional_not_spot():
    """Percent-of-term divides by N*K.  Using N*S instead overstates it by K/S."""
    bs = _bs()
    assert bs.premium("pct_term") == pytest.approx(
        100.0 * bs.price() / (NOTIONAL * STRIKE), rel=1e-15)
    wrong = 100.0 * bs.price() / (NOTIONAL * SPOT)
    assert bs.premium("pct_term") != pytest.approx(wrong, rel=1e-6)
    assert wrong / bs.premium("pct_term") == pytest.approx(STRIKE / SPOT, rel=1e-12)


def test_conversion_identities():
    bs = _bs()
    pct_base = bs.premium("pct_base")
    assert bs.premium("pct_term") == pytest.approx(pct_base * SPOT / STRIKE, rel=1e-12)
    assert pct_base == pytest.approx(bs.premium("term_pips") / (SPOT * 100.0), rel=1e-12)
    assert bs.premium("base_pips") == pytest.approx(
        pct_base * 1e4 / STRIKE / 100.0, rel=1e-12)


@pytest.mark.parametrize("quote", BlackScholes.PREMIUM_QUOTES)
def test_sell_flips_premium_sign(quote):
    assert _bs(BuySell="SELL").premium(quote) == pytest.approx(
        -_bs().premium(quote), rel=1e-15)


@pytest.mark.parametrize("quote", ["term_pips", "base_pips", "pct_term", "pct_base"])
def test_relative_quotes_are_notional_invariant(quote):
    """Pips and percent are per-notional, so they must not scale with size."""
    assert _bs(Notional=7.5e6).premium(quote) == pytest.approx(
        _bs().premium(quote), rel=1e-12)


@pytest.mark.parametrize("quote", ["term_cash", "base_cash"])
def test_cash_quotes_scale_with_notional(quote):
    assert _bs(Notional=2 * NOTIONAL).premium(quote) == pytest.approx(
        2 * _bs().premium(quote), rel=1e-12)


def test_unknown_quote_is_rejected():
    with pytest.raises(ValueError, match="Unknown premium quote"):
        _bs().premium("usd_pips")


@pytest.mark.parametrize("quote", BlackScholes.PREMIUM_QUOTES)
def test_put_call_parity_holds_in_every_quote(quote):
    """Parity is linear in price, so it survives any units rescaling."""
    call_term = _bs().price()
    put_term = _bs(CallPut="PUT").price()
    # Each quote is a fixed positive multiple of the term-cash amount.
    scale = _bs().premium(quote) / call_term
    parity = _bs().premium(quote) - _bs(CallPut="PUT").premium(quote)
    assert parity == pytest.approx((call_term - put_term) * scale, rel=1e-12)


def test_zero_notional_rejected_for_relative_quotes():
    bs = _bs(Notional=0.0)
    assert bs.premium("term_cash") == 0.0
    with pytest.raises(ValueError, match="Notional must be non-zero"):
        bs.premium("term_pips")

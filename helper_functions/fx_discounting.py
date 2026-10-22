"""Market-convention discounting for FX options.

An FX option does not settle on its expiry date.  The premium and the payout
both settle on the **delivery date**, which is the expiry date advanced by the
pair's settlement lag over the joint holiday calendar.  Discounting therefore
runs to delivery, not to expiry, and uses a **simple** rate on the term
currency's own day-count basis rather than a continuously compounded ACT/365
rate.

This module is deliberately standalone: it imports nothing from the pricing
stack, so it can be tested and reasoned about in isolation, and a missing
QuantLib install degrades to the library's legacy behaviour rather than
breaking pricing.

Why a discount *factor* and not a rate
-------------------------------------
``BlackScholes.price()`` is exactly ``DF_term * undiscounted Black`` -- the
base discount factor cancels through FX forward parity.  So overriding the
discount factor is a pure multiplicative rescale that cannot disturb the
smile, the strike solvers or delta.

It is tempting to skip the plumbing and smuggle the convention in as an
"equivalent" continuously compounded rate ``-ln(DF) / t_expiry``.  That
reproduces the price exactly but is numerically unusable at short tenors: with
delivery two days past expiry, a 4.27% rate implies an equivalent 6.80% at 1W
and 7.21% at 1D.  ``theta`` consumes the raw rate, so that shortcut would
badly distort short-dated decay.  The discount factor and the carry rate are
therefore reported separately.

The carry rate
--------------
``theta`` needs the instantaneous decay rate of the discount factor, not the
quoted rate.  For simple discounting ``DF = 1 / (1 + r * tau)``, so

    d(DF)/d(tau) = -r / (1 + r * tau)**2 = -DF * (r / (1 + r * tau))

giving a carry rate of ``r / (1 + r * tau) = r * DF``.  This is bounded by
``r`` and tends to ``r`` as ``tau -> 0``, so unlike the equivalent-rate
shortcut it stays well conditioned at every tenor.
"""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

try:  # pragma: no cover - exercised by the import guard tests
    import QuantLib as ql
except Exception:  # pragma: no cover
    ql = None

try:  # pragma: no cover
    from helper_functions.fx_conventions import CCY, get_pair
except Exception:  # pragma: no cover
    CCY = None
    get_pair = None


#: True when both QuantLib and the conventions table imported cleanly.
CONVENTIONS_AVAILABLE = ql is not None and get_pair is not None

_DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%b-%Y", "%d%b%Y", "%d%b%y")


@dataclass(frozen=True)
class OptionDiscount:
    """Convention-based discounting for one option expiry.

    ``discount_factor`` replaces the library's ``exp(-r * t)`` and
    ``carry_rate`` replaces the raw rate inside ``theta``.  ``year_fraction``
    is measured to *delivery* on the term currency's basis, which is why it
    does not match the ACT/365 expiry time used for volatility.
    """

    pair: str
    term_ccy: str
    valuation_date: date
    expiry_date: date
    delivery_date: date
    day_count: str
    year_fraction: float
    rate: float
    discount_factor: float
    carry_rate: float

    @property
    def settlement_days(self) -> int:
        """Calendar days between expiry and delivery."""
        return (self.delivery_date - self.expiry_date).days


def _to_date(value) -> date:
    """Coerce a date-like value to ``datetime.date``."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if ql is not None and isinstance(value, ql.Date):
        return date(value.year(), value.month(), value.dayOfMonth())
    text = str(value).strip()
    if not text:
        raise ValueError("A date is required")
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValueError("Unrecognised date {!r}".format(value))


def _to_ql(value: date):
    return ql.Date(value.day, value.month, value.year)


def _require_conventions():
    if not CONVENTIONS_AVAILABLE:
        raise ValueError(
            "QuantLib and helper_functions.fx_conventions are required for "
            "convention-based discounting"
        )


def delivery_date(pair: str, expiry, valuation=None) -> date:
    """Return the delivery (settlement) date for an option expiry.

    The expiry is first rolled back to a good business day on the joint
    calendar, then advanced by the pair's settlement lag (the pillar date
    solver, ``apeiron.gaia.time.pillars``).  ``valuation`` is
    accepted for API symmetry but does not affect the result.
    """
    _require_conventions()
    from apeiron.gaia.time.pillars import delivery_for_expiry
    from helper_functions.fx_conventions import settlement
    return delivery_for_expiry(settlement(pair), _to_date(expiry))


def discount_year_fraction(pair: str, valuation, end) -> float:
    """Year fraction from valuation to ``end`` on the term currency's basis."""
    _require_conventions()
    conv = get_pair(pair)
    basis = CCY[conv.quote].day_count
    return float(basis.yearFraction(_to_ql(_to_date(valuation)), _to_ql(_to_date(end))))


def simple_discount_factor(rate: float, year_fraction: float) -> float:
    """Return ``1 / (1 + rate * year_fraction)`` with validation.

    Money-market FX discounting is simple, not continuously compounded.
    """
    rate = float(rate)
    year_fraction = float(year_fraction)
    if year_fraction < 0.0:
        raise ValueError("Discount year fraction must not be negative")
    growth = 1.0 + rate * year_fraction
    if growth <= 0.0:
        raise ValueError("Rate and tenor imply a non-positive discount factor")
    return 1.0 / growth


def option_discount(pair: str, expiry, valuation, rate) -> OptionDiscount:
    """Build convention-based discounting for one expiry. Raises on bad input.

    ``rate`` is the simple term-currency deposit rate quoted on that
    currency's own basis.  Callers that must not fail should use
    :func:`safe_option_discount`.
    """
    _require_conventions()
    conv = get_pair(pair)
    expiry_date = _to_date(expiry)
    valuation_date = _to_date(valuation)
    if expiry_date < valuation_date:
        raise ValueError("Expiry date must not precede the valuation date")

    rate = float(rate)
    if rate != rate or rate in (float("inf"), float("-inf")):
        raise ValueError("Discount rate must be finite")

    return _discount_to(conv, pair, expiry_date, delivery_date(pair, expiry_date),
                        valuation_date, rate)


def delivery_discount(pair: str, delivery, valuation, rate) -> OptionDiscount:
    """Convention discounting to a given delivery date (a value-dated forward).

    As :func:`option_discount`, but the delivery date is the deal's own value
    date instead of one derived from an option expiry. Raises on bad input.
    """
    _require_conventions()
    conv = get_pair(pair)
    delivery = _to_date(delivery)
    rate = float(rate)
    if rate != rate or rate in (float("inf"), float("-inf")):
        raise ValueError("Discount rate must be finite")
    return _discount_to(conv, pair, delivery, delivery, _to_date(valuation), rate)


def _discount_to(conv, pair, expiry_date, delivery, valuation_date, rate):
    if delivery < valuation_date:
        raise ValueError("Delivery date must not precede the valuation date")

    year_fraction = discount_year_fraction(pair, valuation_date, delivery)
    factor = simple_discount_factor(rate, year_fraction)
    # Instantaneous decay rate of a simple discount factor; see module docs.
    carry_rate = rate * factor

    return OptionDiscount(
        pair=conv.name,
        term_ccy=conv.quote,
        valuation_date=valuation_date,
        expiry_date=expiry_date,
        delivery_date=delivery,
        day_count=CCY[conv.quote].day_count.name(),
        year_fraction=year_fraction,
        rate=rate,
        discount_factor=factor,
        carry_rate=carry_rate,
    )


def safe_option_discount(pair, expiry, valuation, rate,
                         default=None) -> Optional[OptionDiscount]:
    """Best-effort :func:`option_discount` that never raises.

    Returns ``default`` whenever the conventions are unavailable or the inputs
    do not define a discount, so the caller transparently keeps the library's
    legacy ``exp(-r * t)`` behaviour instead of failing to price.
    """
    if not CONVENTIONS_AVAILABLE or pair is None:
        return default
    try:
        result = option_discount(pair, expiry, valuation, rate)
    except Exception:
        return default
    return _sane(result, default)


def safe_delivery_discount(pair, delivery, valuation, rate, default=None):
    """Best-effort :func:`delivery_discount` that never raises."""
    if not CONVENTIONS_AVAILABLE or pair is None:
        return default
    try:
        result = delivery_discount(pair, delivery, valuation, rate)
    except Exception:
        return default
    return _sane(result, default)


def _sane(result, default):
    factor = result.discount_factor
    if not (0.0 < factor < 10.0):
        return default
    return result


def equivalent_continuous_rate(discount_factor: float,
                               year_fraction: float) -> float:
    """Return the continuously compounded rate matching a discount factor.

    Provided for reporting and reconciliation only. It is **not** suitable for
    driving ``theta`` at short tenors -- see the module docstring.
    """
    import math

    discount_factor = float(discount_factor)
    year_fraction = float(year_fraction)
    if discount_factor <= 0.0:
        raise ValueError("Discount factor must be positive")
    if year_fraction <= 0.0:
        raise ValueError("Year fraction must be positive")
    return -math.log(discount_factor) / year_fraction


def discount_kwargs(market, expiry, valuation, delivery=None):
    """Return ``BlackScholes`` keyword arguments for convention discounting.

    This is the single seam between market data and the pricers. It is
    duck-typed on ``market`` so this module never imports the pricing stack.

    Convention discounting applies whenever the market carries a currency pair
    that resolves against the convention tables. Setting
    ``settlement_discounting=False`` is an explicit opt-out that forces the
    legacy behaviour back on.

    Returns an **empty dict** -- meaning "keep the library's legacy
    ``exp(-r * t)`` behaviour" -- whenever settlement discounting is switched
    off, the currency pair is unknown, QuantLib is missing, or the dates do
    not define a discount. Pricing therefore degrades rather than fails.
    """
    if market is None:
        return {}
    if not getattr(market, "settlement_discounting", True):
        return {}
    pair = getattr(market, "currency_pair", None)
    if not pair:
        return {}
    if delivery is not None:
        # A value-dated forward: discount to its own delivery date. It has no
        # smile, so no delta convention either.
        discount = safe_delivery_discount(
            pair, delivery, valuation, getattr(market, "term_discount", 0.0))
        if discount is None:
            return {}
        return {"term_discount_factor": discount.discount_factor,
                "carry_rate": discount.carry_rate}
    discount = safe_option_discount(
        pair, expiry, valuation, getattr(market, "term_discount", 0.0)
    )
    if discount is None:
        return {}
    result = {
        "term_discount_factor": discount.discount_factor,
        "carry_rate": discount.carry_rate,
    }
    if getattr(market, "wrapped_smile", False) and not market.force_legacy_smile and _to_date(expiry) > _to_date(valuation):
        result["delta_convention"] = option_delta_convention(pair, expiry, valuation)
    return result


def option_delta_convention(pair, expiry, valuation):
    """``"forward"`` or ``"spot"``: the delta ``pair`` quotes at ``expiry``.

    The table's ``0D`` cutoff means all-forward (EM pairs); otherwise the
    QuantLib delta type at this expiry decides. The same rule Hyperion
    resolves onto markets (``apeiron.hyperion.resolve``).
    """
    from helper_functions.fx_conventions import delta_type
    _require_conventions()
    convention = get_pair(pair)
    kind = delta_type(convention.name, _to_ql(_to_date(expiry)), _to_ql(_to_date(valuation)))
    forward = (convention.fwd_delta_beyond == "0D"
               or kind in (ql.DeltaVolQuote.Fwd, ql.DeltaVolQuote.PaFwd))
    return "forward" if forward else "spot"

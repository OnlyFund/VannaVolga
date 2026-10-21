"""Inputs the core needs that come from dates and conventions, resolved outside it.

Atlas prices with plain numbers. The dates an option settles on, the year
fraction its discounting runs over, and the pair conventions that decide its
delta and ATM are resolved by Hyperion (from the calendars in Gaia) and
carried on the market object as ``ResolvedInputs``, so the core never reads a
calendar. Rates are NOT part of it: discounting is arithmetic on the rate and
``year_fraction`` (``simple_discount_factor``), so a rate bump needs no new
dates.

A ``ResolvedInputs`` validates itself when built and is immutable; an
instance describes one (valuation, expiry, delivery) and ``matches`` says
whether it applies to a request.
"""
import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

DELTA_CONVENTIONS = ("spot", "forward")
ATM_CONVENTIONS = ("dn", "spot", "atmf")


def _day(value):
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if hasattr(value, "date") and callable(value.date):          # pandas Timestamp
        return value.date()
    if isinstance(value, str):                                   # ISO, as the pricers pass it
        try:
            return datetime.strptime(value.strip()[:10], "%Y-%m-%d").date()
        except ValueError:
            pass
    raise ValueError(f"{value!r} is not a date")


@dataclass(frozen=True)
class ResolvedInputs:
    #: The dates the inputs were resolved for.
    valuation_date: date
    expiry_date: date
    delivery_date: date
    #: Valuation to delivery on the term currency's day count, and that count
    #: ("Actual/360", "Actual/365 (Fixed)" ...).
    year_fraction: float
    day_count: str
    #: Options only (``None`` for a value-dated forward, or an expiry on or
    #: before valuation): the delta the pair quotes at this expiry.
    delta_convention: Optional[str]
    premium_adjusted: bool
    atm_convention: str
    #: The same expiry and delivery seen from other valuation dates (sticky
    #: theta's next weekday): ``((date, year_fraction, delta_convention), ...)``.
    others: tuple = ()
    #: Valuation to delivery on the BASE currency's day count (base-rate risk).
    base_year_fraction: Optional[float] = None

    def __post_init__(self):
        for name in ("valuation_date", "expiry_date", "delivery_date"):
            object.__setattr__(self, name, _day(getattr(self, name)))
        if self.delivery_date < self.valuation_date:
            raise ValueError("delivery must not precede valuation")
        if self.expiry_date > self.delivery_date:
            raise ValueError("expiry must not follow delivery")
        fraction = float(self.year_fraction)
        if not math.isfinite(fraction) or fraction < 0:
            raise ValueError("year_fraction must be finite and non-negative")
        object.__setattr__(self, "year_fraction", fraction)
        if not str(self.day_count).strip():
            raise ValueError("day_count is required")
        if self.delta_convention not in (None, *DELTA_CONVENTIONS):
            raise ValueError(f"delta_convention must be one of {DELTA_CONVENTIONS} or None")
        if not isinstance(self.premium_adjusted, bool):
            raise ValueError("premium_adjusted must be a bool")
        if self.atm_convention not in ATM_CONVENTIONS:
            raise ValueError(f"atm_convention must be one of {ATM_CONVENTIONS}")
        others = []
        for entry in self.others:
            day, fraction, convention = entry
            day, fraction = _day(day), float(fraction)
            if day > self.delivery_date or not math.isfinite(fraction) or fraction < 0:
                raise ValueError("other valuation dates must precede delivery with valid fractions")
            if convention not in (None, *DELTA_CONVENTIONS):
                raise ValueError(f"delta_convention must be one of {DELTA_CONVENTIONS} or None")
            others.append((day, fraction, convention))
        object.__setattr__(self, "others", tuple(others))
        if self.base_year_fraction is not None:
            base = float(self.base_year_fraction)
            if not math.isfinite(base) or base < 0:
                raise ValueError("base_year_fraction must be finite and non-negative")
            object.__setattr__(self, "base_year_fraction", base)

    def _entry(self, valuation):
        """``(year_fraction, delta_convention)`` seen from ``valuation``, or ``None``."""
        day = _day(valuation)
        if day == self.valuation_date:
            return self.year_fraction, self.delta_convention
        for other, fraction, convention in self.others:
            if other == day:
                return fraction, convention
        return None

    def matches(self, expiry, valuation, delivery=None):
        """True when these inputs cover ``(expiry, valuation[, delivery])``."""
        try:
            if _day(expiry) != self.expiry_date or self._entry(valuation) is None:
                return False
            return delivery is None or _day(delivery) == self.delivery_date
        except (TypeError, ValueError):
            return False

    def year_fraction_from(self, valuation):
        """Valuation to delivery, from ``valuation`` (one ``matches`` accepts)."""
        return self._entry(valuation)[0]

    def delta_convention_from(self, valuation):
        return self._entry(valuation)[1]


def simple_discount_factor(rate, year_fraction):
    """``1 / (1 + rate * year_fraction)``: money-market FX discounting is simple."""
    rate, year_fraction = float(rate), float(year_fraction)
    if year_fraction < 0.0:
        raise ValueError("Discount year fraction must not be negative")
    denominator = 1.0 + rate * year_fraction
    if denominator <= 0.0:
        raise ValueError("Discount rate implies a non-positive discount factor")
    return 1.0 / denominator

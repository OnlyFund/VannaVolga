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

    def matches(self, expiry, valuation, delivery=None):
        """True when these inputs are the ones for ``(expiry, valuation[, delivery])``."""
        try:
            same = (_day(expiry) == self.expiry_date and _day(valuation) == self.valuation_date)
            return same and (delivery is None or _day(delivery) == self.delivery_date)
        except (TypeError, ValueError):
            return False


def simple_discount_factor(rate, year_fraction):
    """``1 / (1 + rate * year_fraction)``: money-market FX discounting is simple."""
    rate, year_fraction = float(rate), float(year_fraction)
    if year_fraction < 0.0:
        raise ValueError("Discount year fraction must not be negative")
    denominator = 1.0 + rate * year_fraction
    if denominator <= 0.0:
        raise ValueError("Discount rate implies a non-positive discount factor")
    return 1.0 / denominator

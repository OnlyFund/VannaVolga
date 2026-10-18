"""Booked trades: what a trade is, independent of storage and UI.

A ``Trade`` holds booked values only — a numeric strike, a dated expiry —
never aliases such as ``25d`` or ``1M``, so its terms cannot drift with the
market. Every trade has a ``trade_id``; identical economics booked twice
are two trades with two ids.

Two products for now:

* ``fx_option`` — an FX option: a vanilla unless its ``option_type`` says
  otherwise (``app.option_types``; trades booked before types existed read
  as vanillas). Binary-payout types carry a ``payout_ccy``;
* ``fx_forward`` — an FX forward, e.g. a delta hedge. ``HedgeAllocation``
  links a hedge forward to the options it hedges, with the share of its
  base amount assigned to each, so one forward can hedge several options
  at the same pair and expiry.

Versions: a trade is never edited in place. Booking writes version 1; an
amendment or cancellation writes the next version. ``status`` is "live" or
"cancelled"; matured is derived from the valuation date, never stored.
"""
import math
import re
from dataclasses import dataclass, field, fields, replace
from datetime import date, datetime
from typing import Optional, Tuple, Union

import pandas as pd

from app.option_types import PAYOUT_TYPES, VANILLA, option_type_of

FX_OPTION = "fx_option"
FX_FORWARD = "fx_forward"
PRODUCT_TYPES = (FX_OPTION, FX_FORWARD)

LIVE = "live"
CANCELLED = "cancelled"
STATUSES = (LIVE, CANCELLED)

NEW, AMEND, CANCEL = "new", "amend", "cancel"
ACTIONS = (NEW, AMEND, CANCEL)

#: Fields an amendment may change. Identity, product and version history
#: are not amendable; a different product is a new trade.
AMENDABLE = (
    "currency_pair", "trade_date", "book", "counterparty", "strike", "expiry",
    "call_put", "buy_sell", "notional", "premium", "premium_ccy", "contract_rate",
)


class TradeValidationError(ValueError):
    """A trade's terms are incomplete or inconsistent; lists every problem."""


def as_date(value):
    """A ``datetime.date`` from a date, timestamp or ISO-like string."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        raise ValueError("date is missing")
    return pd.Timestamp(str(value).strip()).date()


@dataclass(frozen=True)
class Trade:
    """One version of a booked trade."""

    product_type: str
    currency_pair: str
    trade_date: date
    expiry: date
    buy_sell: str
    notional: float
    book: str = ""
    counterparty: str = ""
    strike: Optional[float] = None
    call_put: Optional[str] = None
    premium: float = 0.0
    premium_ccy: Optional[str] = None
    contract_rate: Optional[float] = None
    trade_id: Optional[str] = None
    version: int = 0
    status: str = LIVE
    package_id: Optional[str] = None
    #: Options only: ``app.option_types`` key; None reads as vanilla.
    option_type: Optional[str] = None
    #: Binary-payout options only: the currency the payout is paid in.
    payout_ccy: Optional[str] = None

    @property
    def base(self):
        return self.currency_pair[:3]

    @property
    def term(self):
        return self.currency_pair[3:]

    @property
    def is_option(self):
        return self.product_type == FX_OPTION

    @property
    def is_forward(self):
        return self.product_type == FX_FORWARD

    @property
    def is_vanilla(self):
        """A vanilla option (an option without a type is one)."""
        return self.is_option and (self.option_type or VANILLA) == VANILLA

    def is_live_on(self, valuation_date):
        """Live and not yet expired on ``valuation_date``."""
        return self.status == LIVE and self.expiry > as_date(valuation_date)

    def normalized(self):
        """Canonical types and spellings; raises ``TradeValidationError``."""
        problems = []

        def text(value):
            return "" if value is None else str(value).strip()

        def number(name, value, required=True, positive=True):
            if value is None or (isinstance(value, str) and not value.strip()):
                if required:
                    problems.append(f"{name} is missing")
                return None
            try:
                result = float(str(value).replace(",", "")) if isinstance(value, str) else float(value)
            except (TypeError, ValueError):
                problems.append(f"{name} must be a number, not {value!r}")
                return None
            if not math.isfinite(result) or (result <= 0 if positive else result < 0):
                problems.append(f"{name} must be a finite {'positive' if positive else 'non-negative'} "
                                f"number, not {value!r}")
                return None
            return result

        def day(name, value):
            try:
                return as_date(value)
            except (TypeError, ValueError):
                problems.append(f"{name} must be a date, not {value!r}")
                return None

        product = text(self.product_type).lower()
        if product not in PRODUCT_TYPES:
            problems.append(f"product type must be one of {', '.join(PRODUCT_TYPES)}, not {self.product_type!r}")
        pair = text(self.currency_pair).replace("/", "").replace(" ", "").upper()
        if not re.fullmatch(r"[A-Z]{6}", pair):
            problems.append(f"currency pair must be two 3-letter ISO codes, not {self.currency_pair!r}")
        trade_date = day("trade date", self.trade_date)
        expiry = day("expiry", self.expiry)
        if trade_date and expiry and expiry <= trade_date:
            problems.append(f"expiry {expiry} must be after the trade date {trade_date}")
        buy_sell = text(self.buy_sell).upper()
        if buy_sell not in ("BUY", "SELL"):
            problems.append(f"buy/sell must be BUY or SELL, not {self.buy_sell!r}")
        notional = number("notional", self.notional)
        premium = number("premium", self.premium or 0.0, positive=False)
        premium_ccy = text(self.premium_ccy).upper() or (pair[3:] if len(pair) == 6 else "")
        strike = call_put = contract_rate = option_type = payout_ccy = None
        if product == FX_OPTION:
            try:
                option_type = option_type_of(self.option_type)
            except ValueError as exc:
                problems.append(str(exc).lower())
            payout_ccy = text(self.payout_ccy).upper() or None
            if option_type in PAYOUT_TYPES:
                payout_ccy = payout_ccy or (pair[:3] if len(pair) == 6 else None)
                if len(pair) == 6 and payout_ccy not in (pair[:3], pair[3:]):
                    problems.append(f"payout currency must be {pair[:3]} or {pair[3:]}, "
                                    f"not {payout_ccy!r}")
            elif payout_ccy is not None:
                problems.append("only a binary-payout option has a payout currency")
            strike = number("strike", self.strike)
            call_put = text(self.call_put).upper()
            if call_put not in ("CALL", "PUT"):
                problems.append(f"call/put must be CALL or PUT, not {self.call_put!r}")
            if len(pair) == 6 and premium_ccy not in (pair[:3], pair[3:]):
                problems.append(f"premium currency must be {pair[:3]} or {pair[3:]}, not {premium_ccy!r}")
            if self.contract_rate not in (None, ""):
                problems.append("an option has no contract rate")
        elif product == FX_FORWARD:
            contract_rate = number("contract rate", self.contract_rate)
            if self.strike not in (None, "") or self.call_put not in (None, ""):
                problems.append("a forward has no strike or call/put")
            if premium:
                problems.append("a forward has no premium")
            if self.option_type not in (None, "") or self.payout_ccy not in (None, ""):
                problems.append("a forward has no option type or payout currency")
            premium, premium_ccy = 0.0, None
        status = text(self.status).lower() or LIVE
        if status not in STATUSES:
            problems.append(f"status must be live or cancelled, not {self.status!r}")
        if problems:
            label = self.trade_id or "new trade"
            raise TradeValidationError(f"{label}: " + "; ".join(problems))
        return replace(
            self, product_type=product, currency_pair=pair, trade_date=trade_date,
            expiry=expiry, buy_sell=buy_sell, notional=notional, premium=premium,
            premium_ccy=premium_ccy, strike=strike, call_put=call_put or None,
            contract_rate=contract_rate, book=text(self.book), counterparty=text(self.counterparty),
            status=status, option_type=option_type, payout_ccy=payout_ccy,
        )

    def amended(self, **changes):
        """A copy with ``changes`` applied, checked and normalized."""
        unknown = set(changes).difference(AMENDABLE)
        if unknown:
            raise TradeValidationError(
                f"{self.trade_id}: cannot amend {', '.join(sorted(unknown))}")
        return replace(self, **changes).normalized()

    def signed_base_amount(self):
        """Base amount received (+) or paid (-); forwards only."""
        return self.notional if self.buy_sell == "BUY" else -self.notional

    def to_pricer_row(self):
        """The trade as a Pricer line (options only)."""
        if not self.is_option:
            raise TradeValidationError(f"{self.trade_id}: only options are Pricer lines")
        return {
            "trade_id": self.trade_id, "currency_pair": self.currency_pair,
            "strike": self.strike, "expiry": self.expiry.isoformat(),
            "call_put": self.call_put, "buy_sell": self.buy_sell,
            "notional": self.notional, "premium": self.premium,
            "premium_ccy": self.premium_ccy,
            "option_type": self.option_type or VANILLA, "payout_ccy": self.payout_ccy,
        }

    def to_position(self):
        """The trade as an ``FxForwardPosition`` (forwards only)."""
        from pricing.fx_forward import FxForwardPosition
        if not self.is_forward:
            raise TradeValidationError(f"{self.trade_id}: only forwards are positions")
        return FxForwardPosition(
            currency_pair=self.currency_pair, expiry=self.expiry,
            base_amount=self.signed_base_amount(), contract_rate=self.contract_rate,
        )

    def terms(self):
        """Field values for storage."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


#: A reference to an option inside ``book``: an existing trade id, or the
#: position of a not-yet-booked option in the same booking.
OptionRef = Union[str, int]


@dataclass(frozen=True)
class HedgeAllocation:
    """The share of a hedge forward's signed base amount for one option."""

    option: OptionRef
    base_amount: float

    def checked(self):
        amount = float(self.base_amount)
        if not math.isfinite(amount) or amount == 0.0:
            raise TradeValidationError(f"hedge allocation must be finite and non-zero, not {self.base_amount!r}")
        return replace(self, base_amount=amount)


@dataclass(frozen=True)
class Booking:
    """Trades to book together, with each hedge's option allocations.

    ``hedges`` maps a forward's position in ``trades`` to its allocations.
    """

    trades: Tuple[Trade, ...]
    hedges: dict = field(default_factory=dict)

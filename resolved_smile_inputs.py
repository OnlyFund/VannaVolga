"""Immutable one-expiry inputs and shared convention-aware FX pillars.

Volatility quotes are percentage points; rates are decimal simple deposit
rates. Dates accept the formats supported by ``fx_discounting`` (ISO preferred).
No broker conversion or pair lookup occurs in ``build_pillars``.
"""
import math
import copy
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Tuple

import numpy as np

from market_data import MarketData
from helper_functions.fx_discounting import _to_date, _to_ql, option_discount
from helper_functions.fx_strangle_converter import _strike_from_delta


class PreparationError(RuntimeError):
    """Convention resolution or numerical smile preparation failed."""


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, np.ndarray):
        frozen = copy.deepcopy(value)
        frozen.setflags(write=False)
        return frozen
    return copy.deepcopy(value)


def _thaw(value):
    if isinstance(value, MappingProxyType):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_thaw(item) for item in value)
    return copy.deepcopy(value)


@dataclass(frozen=True)
class MarketSnapshot:
    """Detached original data, including legacy options used by the benchmark."""
    values: Tuple[tuple, ...]

    @classmethod
    def capture(cls, market):
        if not isinstance(market, MarketData):
            raise TypeError("market must be MarketData")
        # Revalidation matters because MarketData is intentionally mutable.
        validated = MarketData(**{field.name: _thaw(_freeze(getattr(market, field.name)))
                                  for field in fields(MarketData)})
        return cls(tuple((field.name, _freeze(getattr(validated, field.name)))
                         for field in fields(MarketData)))

    def to_market(self):
        return MarketData(**{key: _thaw(value) for key, value in self.values})

    def __getattr__(self, name):
        for key, value in self.values:
            if key == name:
                return value
        raise AttributeError(name)


@dataclass(frozen=True)
class ResolvedConventions:
    pair: str
    spot: float
    valuation_date: object
    expiry_date: object
    delivery_date: object
    time: float
    forward: float
    forward_divisor: float
    domestic_discount: float
    foreign_discount: float
    delta_convention: str
    premium_adjusted: bool
    atm_convention: str
    deposit_basis: int

    @property
    def delta_factor(self):
        return self.foreign_discount if self.delta_convention == "spot" else 1.0


@dataclass(frozen=True)
class ResolvedQuotes:
    atm: float
    rr_25: float
    rr_10: float
    bf_25: float
    bf_10: float


@dataclass(frozen=True)
class Pillars:
    """Ordered put10, put25, ATM, call25, call10; vols in percent."""
    strikes: Tuple[float, ...]
    vols: Tuple[float, ...]


def resolve_conventions(raw, expiry_date, valuation_date):
    valuation, expiry = _to_date(valuation_date), _to_date(expiry_date)
    if expiry <= valuation:
        raise ValueError("expiry must be after valuation")
    try:
        import QuantLib as ql
        from helper_functions import fx_conventions as conventions

        if not raw.currency_pair:
            raise PreparationError("currency_pair is required")
        pair = conventions.get_pair(raw.currency_pair)
        discount = option_discount(pair.name, expiry, valuation, raw.term_discount)
        kind = conventions.delta_type(pair.name, _to_ql(expiry), _to_ql(valuation))
        # The table specifies all-forward EM; its 0D cutoff otherwise leaves
        # expiries on/before spot in the spot branch.
        forward_delta = (pair.fwd_delta_beyond == "0D" or
                         kind in (ql.DeltaVolQuote.Fwd, ql.DeltaVolQuote.PaFwd))
        forward = raw.s + raw.fwd_pts / pair.fwd_divisor
        if forward <= 0:
            raise ValueError("pair divisor implies a non-positive forward")
        return ResolvedConventions(
            pair.name, raw.s, valuation, expiry, discount.delivery_date,
            (expiry - valuation).days / 365.0, forward, pair.fwd_divisor,
            discount.discount_factor, discount.discount_factor * forward / raw.s,
            "forward" if forward_delta else "spot", pair.premium_adjusted,
            pair.atm.lower(), 360 if conventions.CCY[pair.quote].day_count.name()
            == "Actual/360" else 365)
    except (ImportError, KeyError, RuntimeError) as exc:
        raise PreparationError("convention resolution: {}".format(exc)) from exc


def build_pillars(context, quotes, spot):
    """Build explicit pillars from already resolved two-vol quotes."""
    vols = (quotes.atm + quotes.bf_10 - quotes.rr_10 / 2,
            quotes.atm + quotes.bf_25 - quotes.rr_25 / 2,
            quotes.atm,
            quotes.atm + quotes.bf_25 + quotes.rr_25 / 2,
            quotes.atm + quotes.bf_10 + quotes.rr_10 / 2)
    if any(not math.isfinite(v) or v <= 0 for v in vols):
        raise PreparationError("all two-vol pillar volatilities must be positive")
    if context.atm_convention == "dn":
        atm = context.forward * math.exp(
            (-0.5 if context.premium_adjusted else 0.5) *
            (quotes.atm / 100.0) ** 2 * context.time)
    elif context.atm_convention in ("atmf", "fwd"):
        atm = context.forward
    elif context.atm_convention == "spot":
        atm = spot
    else:
        raise PreparationError("unsupported ATM convention")
    try:
        strikes = tuple(atm if i == 2 else _strike_from_delta(
            context.forward, vol / 100.0, context.time, context.delta_factor,
            delta, phi, context.premium_adjusted)
            for i, (vol, delta, phi) in enumerate(zip(
                vols, (-.10, -.25, 0, .25, .10), (-1, -1, 1, 1, 1))))
    except RuntimeError as exc:
        raise PreparationError("pillar strike solve: {}".format(exc)) from exc
    if any(not math.isfinite(k) or k <= 0 for k in strikes) or any(
            right <= left for left, right in zip(strikes, strikes[1:])):
        raise PreparationError("delta pillars are not strictly ordered")
    return Pillars(strikes, vols)

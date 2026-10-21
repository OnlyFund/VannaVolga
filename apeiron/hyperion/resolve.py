"""Resolve a market's dates and conventions for the core (``atlas.resolved``).

Hyperion knows the pair and the dates when it builds a market, so it resolves
what the core would otherwise read from the calendars, with the very same
functions the core uses today (``fx_discounting``, ``fx_conventions``):

* ``resolve_inputs(pair, expiry, valuation, delivery=None)`` - an option's
  inputs (delivery on the pair's settlement calendar), or a value-dated
  deal's (``delivery`` given, no delta convention);
* ``attach(market, expiry, valuation, delivery=None)`` - sets
  ``market.resolved``; never raises: on any failure ``resolved`` stays
  ``None`` (the core keeps resolving for itself, unchanged) and the reason is
  kept on ``market.resolution_note``.
"""
from apeiron.atlas.resolved import ResolvedInputs


def _theta_date(valuation):
    """Sticky theta's shifted valuation (``Greeks.fixed_greeks.FixedTheta``):
    the next weekday."""
    import pandas as pd
    return (pd.Timestamp(valuation) + pd.offsets.BDay(1)).date()


def _delta_convention(conventions, convention, expiry, valuation):
    import QuantLib as ql
    from helper_functions.fx_discounting import _to_ql
    if expiry <= valuation:
        return None
    # As resolved_smile_inputs.resolve_conventions: the table's 0D cutoff
    # means all-forward (EM), else the delta type at this expiry.
    kind = conventions.delta_type(convention.name, _to_ql(expiry), _to_ql(valuation))
    forward = (convention.fwd_delta_beyond == "0D"
               or kind in (ql.DeltaVolQuote.Fwd, ql.DeltaVolQuote.PaFwd))
    return "forward" if forward else "spot"


def resolve_inputs(pair, expiry, valuation, delivery=None):
    import QuantLib as ql
    from helper_functions import fx_conventions as conventions
    from helper_functions.fx_discounting import (
        _to_date, _to_ql, delivery_discount, option_discount,
    )
    convention = conventions.get_pair(str(pair).replace("/", "").strip().upper())
    valuation, expiry = _to_date(valuation), _to_date(expiry)
    # The rate does not change dates or year fractions; any finite one will do.
    if delivery is None:
        discount = option_discount(convention.name, expiry, valuation, 0.0)
    else:
        discount = delivery_discount(convention.name, _to_date(delivery), valuation, 0.0)
    delta_convention = (None if delivery is not None
                        else _delta_convention(conventions, convention, expiry, valuation))
    others = ()
    shifted = _theta_date(valuation)
    if shifted <= discount.delivery_date:
        from helper_functions.fx_discounting import discount_year_fraction
        others = ((shifted, discount_year_fraction(convention.name, shifted, discount.delivery_date),
                   None if delivery is not None
                   else _delta_convention(conventions, convention, expiry, shifted)),)
    return ResolvedInputs(
        valuation_date=valuation,
        expiry_date=discount.expiry_date,
        delivery_date=discount.delivery_date,
        year_fraction=discount.year_fraction,
        day_count=discount.day_count,
        delta_convention=delta_convention,
        premium_adjusted=bool(convention.premium_adjusted),
        atm_convention=str(convention.atm).lower(),
        others=others,
        base_year_fraction=float(conventions.CCY[convention.base].day_count.yearFraction(
            _to_ql(valuation), _to_ql(discount.delivery_date))),
        forward_divisor=float(convention.fwd_divisor),
    )


def attach(market, expiry, valuation, delivery=None):
    """``market`` with ``resolved`` set (or ``None`` and a note); returns it."""
    pair = getattr(market, "currency_pair", None)
    try:
        if not pair:
            raise ValueError("no currency pair")
        market.resolved = resolve_inputs(pair, expiry, valuation, delivery)
        market.resolution_note = None
    except Exception as exc:                  # the core resolves for itself, as before
        market.resolved = None
        market.resolution_note = f"inputs not resolved: {exc}"
    return market

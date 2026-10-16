"""Term-currency discount curves for markets built from a stored snapshot.

The rate a market discounts its term currency at (``term_discount``) is,
for each currency, the first of:

1. ``sofr`` (USD): the quoted SOFR OIS curve;
   ``implied`` (any other currency): implied by its USD pair's forward points
   and SOFR (``implied_rates``): the USD-collateralised rate, consistent with
   the forwards. A tenor without a forward point is interpolated in time
   between its implied neighbours, flat beyond the ends;
2. ``quoted``: the currency's own quoted depo curve, when its USD pair has
   no spot or forward points; it leaves out the cross-currency basis;
3. ``stale``: the curve steps 1-2 give from the latest earlier snapshot
   within ``MAX_STALE_BDAYS`` business days;
4. ``unavailable``: none of these; the market cannot be built, so nothing
   prices on a made-up rate.

``DiscountCurves.basis_alerts`` flags implied-minus-quoted bases that jumped
since the previous snapshot by more than ``BASIS_JUMP_BP`` (a bad tick
usually shows up this way); it only warns, since a basis can genuinely move.
"""
from collections import namedtuple

import pandas as pd
from pandas.tseries.offsets import BDay

from market_data.implied_rates import implied_rates
from market_data.universe import CCY_LIST, COMMON_CCY, G10

TENORS = ("1W", "2W", "1M", "2M", "3M", "6M", "9M", "1Y")
#: Nominal days per tenor, for interpolating a curve's missing tenors in time.
NOMINAL_DAYS = {"ON": 1, "1W": 7, "2W": 14, "1M": 30, "2M": 61, "3M": 91,
                "6M": 182, "9M": 273, "1Y": 365}
MAX_STALE_BDAYS = 5
#: Basis jump since the previous snapshot that is flagged, G10 and other currencies.
BASIS_JUMP_BP = {"G10": 50.0, "EM": 200.0}

SOFR, IMPLIED, QUOTED, STALE, UNAVAILABLE = "sofr", "implied", "quoted", "stale", "unavailable"

#: ``rates`` is ``{tenor: decimal rate}`` (empty when unavailable); ``as_of`` is
#: the valuation date of the snapshot the rates come from.
Curve = namedtuple("Curve", "ccy rates source as_of notes")


def fill_tenors(points, tenors=TENORS):
    """``(rates, filled)``: ``points`` at every tenor, interpolated linearly in
    nominal days between known tenors and held flat beyond them."""
    known = sorted((NOMINAL_DAYS[tenor], rate) for tenor, rate in points.items()
                   if tenor in NOMINAL_DAYS and rate is not None)
    if not known:
        return {}, []
    days, values = zip(*known)
    series = pd.Series(values, index=days, dtype=float)
    rates, filled = {}, []
    for tenor in tenors:
        if tenor in points and points[tenor] is not None:
            rates[tenor] = float(points[tenor])
            continue
        d = NOMINAL_DAYS[tenor]
        if d <= days[0]:
            rates[tenor] = values[0]
        elif d >= days[-1]:
            rates[tenor] = values[-1]
        else:
            rates[tenor] = float(series.reindex(sorted(set(days) | {d}))
                                 .interpolate("index")[d])
        filled.append(tenor)
    return {tenor: round(rate, 10) for tenor, rate in rates.items()}, filled


def usd_pair(ccy):
    """The currency's USD pair in market quoting order (EURUSD, USDJPY ...)."""
    return next((pair for pair in CCY_LIST if ccy in (pair[:3], pair[3:])), None)


def live_curve(ccy, snapshot):
    """Steps 1-2 on one snapshot: a ``Curve``, or ``None`` when neither works."""
    usd = snapshot.depo_curves.get(COMMON_CCY)
    if ccy == COMMON_CCY:
        if not usd:
            return None
        rates, filled = fill_tenors(usd)
        notes = [f"USD: SOFR {', '.join(filled)} interpolated"] if filled else []
        return Curve(ccy, rates, SOFR, snapshot.valuation_date, notes)
    pair = usd_pair(ccy)
    points = snapshot.forward_points.get(pair) if pair else None
    if usd and points and pair in snapshot.spots:
        implied = implied_rates(ccy, pair, snapshot.valuation_date, snapshot.spots[pair],
                                points, fill_tenors(usd)[0])
        if implied:
            rates, filled = fill_tenors(implied)
            notes = ([f"{ccy}: {', '.join(filled)} interpolated (no {pair} forward points)"]
                     if filled else [])
            return Curve(ccy, rates, IMPLIED, snapshot.valuation_date, notes)
    quoted = snapshot.depo_curves.get(ccy)
    if quoted:
        rates, _ = fill_tenors(quoted)
        why = "no USD curve" if not usd else f"no {pair} spot or forward points"
        return Curve(ccy, rates, QUOTED, snapshot.valuation_date,
                     [f"{ccy}: quoted depo curve, basis not included ({why})"])
    return None


class DiscountCurves:
    """Each currency's discount curve for one snapshot, worked out once."""

    def __init__(self, snapshot, store=None, max_stale_bdays=MAX_STALE_BDAYS):
        self.snapshot, self.store, self.max_stale_bdays = snapshot, store, max_stale_bdays
        self._curves = {}

    def curve(self, ccy):
        if ccy not in self._curves:
            self._curves[ccy] = self._build(ccy)
        return self._curves[ccy]

    def _build(self, ccy):
        live = live_curve(ccy, self.snapshot)
        if live is not None:
            return live
        if self.store is not None:
            since = pd.Timestamp(self.snapshot.valuation_date) - BDay(self.max_stale_bdays)
            for earlier in self.store.earlier(self.snapshot, since):
                found = live_curve(ccy, earlier)
                if found is not None:
                    return found._replace(source=STALE, notes=found.notes + [
                        f"{ccy}: no live rate; using the {found.source} curve from snapshot "
                        f"{earlier.id} ({earlier.valuation_date})"])
        return Curve(ccy, {}, UNAVAILABLE, None, [
            f"{ccy}: no discount rate (no {usd_pair(ccy) or ccy} forward points, no quoted "
            f"depo, no snapshot within {self.max_stale_bdays} business days)"])

    def basis_alerts(self, previous=None):
        """Notes for implied-minus-quoted bases that moved more than
        ``BASIS_JUMP_BP`` since ``previous`` (default: the store's previous
        snapshot)."""
        if previous is None and self.store is not None:
            previous = self.store.previous(self.snapshot)
        if previous is None:
            return []
        alerts = []
        for ccy in sorted(set(self.snapshot.depo_curves) - {COMMON_CCY}):
            now, before = _basis(ccy, self.snapshot), _basis(ccy, previous)
            limit = BASIS_JUMP_BP["G10" if ccy in G10 else "EM"]
            for tenor in TENORS:
                if tenor in now and tenor in before:
                    jump = (now[tenor] - before[tenor]) * 1e4
                    if abs(jump) > limit:
                        alerts.append(f"{ccy} {tenor}: basis moved {jump:+.0f}bp since snapshot "
                                      f"{previous.id} (limit {limit:.0f}bp); check the quotes")
        return alerts


def _basis(ccy, snapshot):
    """``{tenor: implied - quoted}`` where both exist."""
    implied = live_curve(ccy, snapshot)
    quoted = snapshot.depo_curves.get(ccy, {})
    if implied is None or implied.source != IMPLIED:
        return {}
    return {tenor: implied.rates[tenor] - quoted[tenor]
            for tenor in TENORS if tenor in quoted and tenor in implied.rates}

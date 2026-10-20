"""Strike grid and Greek bucket reports for one currency pair.

Axes:

* **Strike (delta) buckets** in fixed delta steps (5 by default), ordered by
  strike: ``5P, 10P, ..., 45P, ATM, 45C, ..., 5C``. A strike's position is
  its call delta (forward delta at the smile vol of that strike), so a put
  and a call at the same strike share a bucket; the ``nP`` bucket sits at
  call delta ``1 - n/100``.
* **Tenor buckets** at expiry dates (pillar tenors resolved on the pair's
  market).

Allocation is proportional on both axes: an option between two buckets is
split between them linearly in call delta (strike axis) and in calendar
days (tenor axis); beyond the outermost bucket it goes to that bucket.
Every line's full amount lands somewhere, so grid totals equal the
portfolio totals.

Hedge forwards are optional; when included they go to the last call bucket
(the highest strike) at their own expiry.
"""
import math
from dataclasses import dataclass
from datetime import date
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from pricing.greek_catalog import CATALOG

ATM = "ATM"
TOTAL = "Total"
DEFAULT_TENORS = ("1W", "1M", "2M", "3M", "6M", "9M", "1Y", "2Y")
#: Measures a grid cell can show, by result field.
#: The original measures first (Vega stays the default), then every other Greek.
MEASURES = {
    "notional": "Notional (base, signed)",
    "vega": "Vega", "delta": "Delta", "gamma": "Gamma", "vanna": "Vanna",
    "volga": "Volga", "theta": "Theta", "mtm_base": "MTM (base)",
}
MEASURES.update({name: greek.label for name, greek in CATALOG.items() if name not in MEASURES})
BUCKET_GREEKS = ("delta", "gamma", "vega", "vanna", "volga", "theta")


def delta_buckets(step=5):
    """``[(label, call delta)]`` from the lowest strike to the highest."""
    step = int(step)
    if step <= 0 or 50 % step:
        raise ValueError("the delta step must divide 50, e.g. 5, 10 or 25")
    puts = [(f"{d}P", 1.0 - d / 100.0) for d in range(step, 50, step)]
    calls = [(f"{d}C", d / 100.0) for d in range(50 - step, 0, -step)]
    return puts + [(ATM, 0.5)] + calls


def linear_weights(x, nodes):
    """``[(index, weight)]`` splitting ``x`` between its neighbouring nodes.

    ``nodes`` must be strictly monotonic (either direction). Outside the
    nodes everything goes to the nearest end node.
    """
    nodes = np.asarray(nodes, dtype=float)
    if len(nodes) == 1:
        return [(0, 1.0)]
    if not math.isfinite(x):
        raise ValueError("cannot bucket a non-finite position")
    ascending = nodes[-1] > nodes[0]
    ordered = nodes if ascending else nodes[::-1]
    if x <= ordered[0]:
        index = 0
        return [(index if ascending else len(nodes) - 1, 1.0)]
    if x >= ordered[-1]:
        return [(len(nodes) - 1 if ascending else 0, 1.0)]
    upper = int(np.searchsorted(ordered, x, side="right"))
    lower = upper - 1
    weight_upper = (x - ordered[lower]) / (ordered[upper] - ordered[lower])
    pairs = [(lower, 1.0 - weight_upper), (upper, weight_upper)]
    if not ascending:
        pairs = [(len(nodes) - 1 - index, weight) for index, weight in pairs]
    return [(index, weight) for index, weight in pairs if weight > 0.0]


def strike_call_delta(forward, strike, vol, years):
    """Forward call delta ``N(d1)`` of ``strike`` (vol as a decimal)."""
    from scipy.stats import norm
    if min(forward, strike, vol, years) <= 0:
        raise ValueError("forward, strike, vol and time must be positive")
    d1 = (math.log(forward / strike) + 0.5 * vol * vol * years) / (vol * math.sqrt(years))
    return float(norm.cdf(d1))


@dataclass(frozen=True)
class RiskLine:
    """One position on the grid axes with the values a cell can show."""

    trade_id: str
    expiry: date
    call_delta: Optional[float]          # None for a forward (goes to the last call bucket)
    values: dict


def _tenor_weights(expiry, tenor_dates):
    ordinals = [pd.Timestamp(d).toordinal() for d in tenor_dates]
    return linear_weights(pd.Timestamp(expiry).toordinal(), ordinals)


def strike_grid(lines: Sequence[RiskLine], tenor_labels, tenor_dates, measure, step=5,
                totals=False):
    """Tenor x strike-bucket table of ``measure``; ``totals`` adds a Total row and column.

    Forwards (``call_delta is None``) go to the last call bucket.
    """
    buckets = delta_buckets(step)
    labels = [label for label, _ in buckets]
    nodes = [call_delta for _, call_delta in buckets]
    grid = np.zeros((len(tenor_labels), len(labels)))
    for line in lines:
        amount = float(line.values.get(measure, 0.0) or 0.0)
        if amount == 0.0:
            continue
        strikes = ([(len(labels) - 1, 1.0)] if line.call_delta is None
                   else linear_weights(line.call_delta, nodes))
        for tenor, tenor_weight in _tenor_weights(line.expiry, tenor_dates):
            for column, strike_weight in strikes:
                grid[tenor, column] += amount * tenor_weight * strike_weight
    frame = pd.DataFrame(grid, index=list(tenor_labels), columns=labels)
    if totals:
        frame[TOTAL] = frame.sum(axis=1)
        frame.loc[TOTAL] = frame.sum(axis=0)
    return frame


def greek_buckets(lines: Sequence[RiskLine], tenor_labels, tenor_dates, greeks=BUCKET_GREEKS):
    """Greek x tenor-bucket table, with totals."""
    table = np.zeros((len(greeks), len(tenor_labels)))
    for line in lines:
        weights = _tenor_weights(line.expiry, tenor_dates)
        for row, greek in enumerate(greeks):
            amount = float(line.values.get(greek, 0.0) or 0.0)
            for tenor, weight in weights:
                table[row, tenor] += amount * weight
    frame = pd.DataFrame(table, index=list(greeks), columns=list(tenor_labels))
    frame[TOTAL] = frame.sum(axis=1)
    return frame


def risk_lines(markets, pair, trades, results, forwards, include_hedges=False, fixed=False,
               greeks=BUCKET_GREEKS):
    """``RiskLine`` for each priced option of ``pair`` (and its forwards).

    ``results`` / ``forwards`` come from ``service.price_trades``; ``fixed``
    uses the sticky-delta (``fixed_``) Greeks instead of the smile ones. ``greeks`` are the
    Greeks each line carries; one the results lack (or that failed) is 0.
    Forwards carry delta and theta only.
    """
    valuation = pd.Timestamp(markets.valuation_date)
    context = markets.market(pair)
    by_id = {trade.trade_id: trade for trade in trades}
    lines, errors = [], []
    for result in results:
        if result["currency_pair"] != pair:
            continue
        trade = by_id[result["trade_id"]]
        try:
            market = context.market_for_expiry(result["expiry"])
            forward = market.s + market.fwd_pts / market.forward_divisor
            years = (pd.Timestamp(result["expiry"]) - valuation).days / 365.0
            call_delta = strike_call_delta(forward, float(result["strike"]),
                                           float(result["vol_pct"]) / 100.0, years)
        except Exception as exc:
            errors.append(f"{trade.trade_id}: {exc}")
            continue
        sign = 1.0 if trade.buy_sell == "BUY" else -1.0
        # An exotic's notional is a payout, not a vanilla notional: it is
        # left out of the Notional measure (its Greeks and MTM count).
        values = {"notional": sign * trade.notional if trade.is_vanilla else 0.0,
                  "mtm_base": result["mtm_base"]}
        for greek in greeks:
            values[greek] = result.get(f"fixed_{greek}" if fixed else greek) or 0.0
        lines.append(RiskLine(trade.trade_id, pd.Timestamp(result["expiry"]).date(),
                              call_delta, values))
    if include_hedges:
        for row in forwards:
            if row["currency_pair"] != pair:
                continue
            values = {"notional": row["base_amount"], "delta": row["delta_offset"],
                      "mtm_base": row["mtm_base"], "theta": row["theta"]}
            for greek in ("spot_delta", "fwd_delta", "rho", "phi", "bs_delta",
                          "bs_delta_prem", "bs_gamma"):
                if row.get(greek) is not None:
                    values[greek] = row[greek]
            lines.append(RiskLine(row["trade_id"], pd.Timestamp(row["expiry"]).date(), None,
                                  values))
    return lines, errors


def market_tenors(context):
    """The pair's quoted pillar tenors, in market order (the default buckets)."""
    try:
        tenors = [str(t).strip().upper() for t in context.vol_rows["tenor"] if str(t).strip()]
    except Exception:
        tenors = []
    return list(dict.fromkeys(tenors)) or list(DEFAULT_TENORS)


def tenor_axis(context, tenors=DEFAULT_TENORS):
    """``(labels, dates)``: the tenors resolved on the pair's market, ascending.

    Tenors that cannot be resolved are dropped and reported.
    """
    resolved, errors = {}, []
    for tenor in tenors:
        try:
            resolved[str(tenor).upper()] = pd.Timestamp(context.resolve_market_expiry(tenor)).date()
        except Exception as exc:
            errors.append(f"tenor {tenor}: {exc}")
    labels, dates = [], []
    for label, expiry in sorted(resolved.items(), key=lambda item: item[1]):
        if dates and expiry == dates[-1]:
            errors.append(f"tenor {label} expires with {labels[-1]} on {expiry}; merged into it")
            continue
        labels.append(label)
        dates.append(expiry)
    return labels, dates, errors

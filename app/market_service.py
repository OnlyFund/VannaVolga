"""Convert broker quotes at their own expiries before variance interpolation."""
from collections import OrderedDict
import warnings

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from market_data import MarketData
from helper_functions.fx_discounting import delivery_date
from app.smile_service import FALLBACK_PREFIX, cache_key, is_fallback, prepared_smile
from wrapped_vol_resolver import DegradedSmileWarning


def pair_delivery(pair, expiry, collector=None):
    try:
        return pd.Timestamp(delivery_date(pair, pd.Timestamp(expiry).date()))
    except (ValueError, KeyError, RuntimeError) as exc:
        reason = "Delivery conventions unavailable for {}: {}; using expiry + 2 business days".format(pair, exc)
        if collector is not None:
            collector.append(("conventions", str(expiry), reason))
        warnings.warn("Delivery conventions unavailable; using T+2 fallback.",
                      DegradedSmileWarning, stacklevel=2)
        return pd.Timestamp(expiry) + BDay(2)


def pair_divisor(pair, legacy=10000.0):
    try:
        from helper_functions.fx_conventions import get_pair
        return float(get_pair(pair).fwd_divisor)
    except (ImportError, KeyError, ValueError, RuntimeError):
        return float(legacy)


class MarketService:
    def __init__(self, collector):
        self.collector = collector
        self.cache = OrderedDict()
        self.progress = None

    def convert_curve(self, rows, valuation, spot, pair, divisor, model,
                      broker, forward_for_expiry, rate_for_expiry):
        markets = [
            MarketData(
                s=float(spot), fwd_pts=forward_for_expiry(row.expiry),
                v_atm=float(row.atm), rr_25=float(row.rr_25), rr_10=float(row.rr_10),
                bf_25=float(row.bf_25), bf_10=float(row.bf_10),
                term_discount=rate_for_expiry(row.expiry),
                currency_pair=pair, forward_divisor=divisor,
                wrapped_smile=True, broker_strangle=broker, smile_model=model,
                diagnostics_token=self.collector.token,
            ) for row in rows.itertuples()
        ]
        # Each pillar's dates and conventions, resolved here (atlas.resolved).
        from apeiron.hyperion.resolve import attach
        for market, row in zip(markets, rows.itertuples()):
            attach(market, row.expiry, valuation)
        key = (str(valuation), model, broker,
               tuple(cache_key(m, row.expiry, valuation, model, broker, .5)
                     for m, row in zip(markets, rows.itertuples())))
        if key in self.cache:
            self.cache.move_to_end(key)
            converted, reasons, prepared, notes = self.cache[key]
            for reason in reasons + notes:
                self.collector.append((model, "curve", reason))
            return converted.copy(), reasons, prepared
        prepared = []
        for index, (m, row) in enumerate(zip(markets, rows.itertuples())):
            if self.progress is not None:
                self.progress(index, len(markets), getattr(row, "tenor", str(row.expiry)[:10]))
            prepared.append(prepared_smile(m, row.expiry, valuation, model, broker))
        prepared = tuple(prepared)
        converted, reasons, notes = convert_pillars(rows, prepared)
        for reason in reasons + notes:
            self.collector.append((model, "curve", reason))
        if reasons:
            warnings.warn("Smile curve preparation failed; using original quotes.",
                          DegradedSmileWarning, stacklevel=2)
        self.cache[key] = (converted.copy(), reasons, prepared, notes)
        if len(self.cache) > 64:
            self.cache.popitem(last=False)
        return converted, reasons, prepared


_QUOTE_COLUMNS = ("atm", "rr_25", "rr_10", "bf_25", "bf_10")


def convert_pillars(rows, prepared):
    """Replace broker quotes with converted two-vol quotes, pillar by pillar.

    Returns ``(converted, reasons, notes)``. A pillar whose conversion failed
    keeps its ATM and risk reversals and takes the strangle adjustment
    (two-vol minus broker butterfly) interpolated by expiry from the pillars
    that converted, flat beyond the first and last. ``notes`` describe those
    and other fallbacks. Only when no pillar converted are ``reasons``
    returned: the original quotes are kept and the caller uses the legacy
    smile.
    """
    converted = rows.copy()
    good = [not p.diagnostics.degraded for p in prepared]
    if not any(good):
        return converted, tuple(dict.fromkeys(p.diagnostics.reason for p in prepared)), ()
    labels = [str(getattr(row, "tenor", str(row.expiry)[:10])) for row in rows.itertuples()]
    ordinals = np.array([pd.Timestamp(e).toordinal() for e in rows["expiry"]], dtype=float)
    adjustments = {}
    for column in ("bf_25", "bf_10"):
        adjustments[column] = np.array([
            getattr(p.resolved, column) - float(rows.at[index, column]) if ok else np.nan
            for index, p, ok in zip(rows.index, prepared, good)])
    order = np.argsort(ordinals)
    known = [i for i in order if good[i]]
    for position, (index, p) in enumerate(zip(converted.index, prepared)):
        if good[position]:
            for column in _QUOTE_COLUMNS:
                converted.at[index, column] = getattr(p.resolved, column)
            continue
        for column in ("bf_25", "bf_10"):
            converted.at[index, column] = float(rows.at[index, column]) + float(np.interp(
                ordinals[position], ordinals[known], adjustments[column][known]))
    notes = []
    failed = [label for label, ok in zip(labels, good) if not ok]
    if failed:
        notes.append(FALLBACK_PREFIX + "broker strangle conversion failed at {}; strangle "
                     "adjustment interpolated from converted pillars".format(", ".join(failed)))
    substituted = [label for label, p in zip(labels, prepared) if is_fallback(p.diagnostics)]
    if substituted:
        notes.append(FALLBACK_PREFIX + "conversion fallback at {} (see pillar diagnostics)"
                     .format(", ".join(substituted)))
    return converted, (), tuple(notes)

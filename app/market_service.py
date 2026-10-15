"""Convert broker quotes at their own expiries before variance interpolation."""
from collections import OrderedDict
import warnings

import pandas as pd
from pandas.tseries.offsets import BDay

from market_data import MarketData
from helper_functions.fx_discounting import delivery_date
from app.smile_service import prepared_smile, cache_key
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
        key = (str(valuation), model, broker,
               tuple(cache_key(m, row.expiry, valuation, model, broker, .5)
                     for m, row in zip(markets, rows.itertuples())))
        if key in self.cache:
            self.cache.move_to_end(key)
            converted, reasons, prepared = self.cache[key]
            for reason in reasons:
                self.collector.append((model, "curve", reason))
            return converted.copy(), reasons, prepared
        prepared = []
        for index, (m, row) in enumerate(zip(markets, rows.itertuples())):
            if self.progress is not None:
                self.progress(index, len(markets), getattr(row, "tenor", str(row.expiry)[:10]))
            prepared.append(prepared_smile(m, row.expiry, valuation, model, broker))
        prepared = tuple(prepared)
        reasons = tuple(dict.fromkeys(
            p.diagnostics.reason for p in prepared if p.diagnostics.degraded))
        converted = rows.copy()
        if not reasons:
            for index, p in zip(converted.index, prepared):
                for column, attr in (("atm", "atm"), ("rr_25", "rr_25"),
                                     ("rr_10", "rr_10"), ("bf_25", "bf_25"),
                                     ("bf_10", "bf_10")):
                    converted.at[index, column] = getattr(p.resolved, attr)
        else:
            for reason in reasons:
                self.collector.append((model, "curve", reason))
            warnings.warn("Smile curve preparation failed; using original quotes.",
                          DegradedSmileWarning, stacklevel=2)
        self.cache[key] = (converted.copy(), reasons, prepared)
        if len(self.cache) > 64:
            self.cache.popitem(last=False)
        return converted, reasons, prepared

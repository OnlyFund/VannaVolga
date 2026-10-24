"""Work budgets (performance step 0): pricing must not redo work it has done.

Timing limits depend on the machine, so these count work instead. Each test
prices a small book the way the app does and checks an invariant whose
breaking made the app slow before (2026-10-24): a market built twice for the
same request, a forward curve read per pillar, a pillar spline rebuilt per
strike, the vol table parsed per build. A failure names the repeated work.
"""
import collections

import pandas as pd
import pytest

from app import market_context as mc
from app import pricer
from app.market_context import DeliveryCurve, MarketContext
from golden.app_harness import load_markets
from helper_functions import vanna_volga_helpers as vvh
from market_data import Modifications

pytestmark = pytest.mark.filterwarnings("ignore")

MODS = Modifications(beta_atm=0.7, beta_rr=1.2, beta_bf=0.0)
#: Two expiries met twice each, calls and puts, a delta strike, a broken date:
#: repeats on purpose, so repeated work would show.
LINES = [dict(currency_pair="EURUSD", strike=strike, expiry=expiry, call_put=call_put,
              buy_sell="BUY", notional="1m", premium=0.0, premium_ccy="USD")
         for strike, expiry, call_put in [("atm", "1M", "CALL"), ("25d", "1M", "PUT"),
                                          ("atm", "3M", "CALL"), ("25d", "3M", "CALL"),
                                          ("atm", "2026-12-01", "PUT")]]


@pytest.fixture
def context():
    return MarketContext.from_payload(load_markets("MARKETS_20261015.json")["EURUSD"])


def count(monkeypatch, owner, name, key=lambda *args, **kwargs: None):
    """Record ``key(...)`` of every call to ``owner.name``."""
    seen = []
    original = getattr(owner, name)

    def counted(*args, **kwargs):
        seen.append(key(*args, **kwargs))
        return original(*args, **kwargs)
    monkeypatch.setattr(owner, name, counted)
    return seen


def repeats(keys):
    return {key: n for key, n in collections.Counter(keys).items() if n > 1}


def price(context):
    results, errors = pricer.price_lines(context, pd.DataFrame(LINES), MODS)
    assert not errors and len(results) == len(LINES)
    return results


def test_no_market_is_built_twice_for_the_same_request(context, monkeypatch):
    builds = count(monkeypatch, MarketContext, "_build_market_for_expiry",
                   lambda self, expiry, as_of=None, allow_short_end=False, spot_value=None:
                   (id(self.__dict__.get("_market_cache")), mc.naive_date(expiry),
                    as_of and mc.naive_date(as_of), bool(allow_short_end),
                    float(self.spot if spot_value is None else spot_value)))
    price(context)
    assert builds and not repeats(builds), f"markets built again: {repeats(builds)}"


def test_each_forward_curve_is_read_once_per_market_and_curve_date(context, monkeypatch):
    reads = count(monkeypatch, DeliveryCurve, "_prepare", lambda self: id(self))
    curves = count(monkeypatch, MarketContext, "_market_for_expiry")
    price(context)
    assert not repeats(reads), "a curve object was read twice"
    # Two curves (points and rates) per context and curve date, not per build.
    assert len(reads) < len(curves), (f"{len(reads)} curve reads for {len(curves)} market "
                                      "builds: curves are read per build again")


def test_the_pillar_curve_is_converted_once_per_curve_date_and_spot(context, monkeypatch):
    conversions = count(monkeypatch, context.market_service, "convert_curve",
                        lambda rows, valuation, spot, *rest: (valuation, spot))
    price(context)
    assert conversions and not repeats(conversions), f"converted again: {repeats(conversions)}"


def test_a_smile_core_builds_its_pillar_spline_once(context, monkeypatch):
    from resolved_smile_cores import VannaVolgaCore
    splines = count(monkeypatch, vvh.PillarSpline, "__init__")
    cores = count(monkeypatch, VannaVolgaCore, "__init__")
    price(context)
    # Not per strike or per Greek bump: at most one per core (a calibration
    # may build a core twice at the same pillars; that is the optimizer's).
    assert cores and len(splines) <= len(cores), (
        f"{len(splines)} pillar splines for {len(cores)} smile cores")


def test_the_vol_table_is_parsed_once_per_market(context, monkeypatch):
    parses = count(monkeypatch, MarketContext, "_with_interpolation_nodes",
                   lambda self, rows: id(self.vol_rows))     # shared by with_spot copies
    price(context)
    assert not repeats(parses), f"vol table parsed again: {repeats(parses)}"


def test_variance_days_are_counted_once_per_date_pair(context, monkeypatch):
    weighted = count(monkeypatch, MarketContext, "_weighted_days",
                     lambda self, start, end: (id(self.__dict__.get("_variance_between_cache")),
                                               start, end))
    price(context)
    assert weighted and not repeats(weighted), f"counted again: {repeats(weighted)}"

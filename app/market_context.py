"""One currency pair's market on a valuation date, independent of the UI.

``MarketContext`` holds a snapshot of a pair's market tables (vol pillars,
forward points, event calendar) together with the pricing conventions, and
builds the per-expiry ``MarketData`` the pricers consume. It is the backend
form of what the Streamlit app used to define inline in ``render_app``; the
numbers it produces are pinned by the golden tests.

Failsafes:

* Construction checks the structure of the inputs (pair code, spot, forward
  divisor, valuation date, required table columns, pricing settings) and
  raises ``MarketContextError`` naming every problem at once.
* The tables are deep-copied, so later edits to the caller's frames cannot
  change a context that is already pricing.
* Content problems that affect only some expiries (a missing quote, an
  expiry outside the quoted range) stay per-expiry errors, raised by
  ``market_for_expiry`` with the same messages as before, so one bad pillar
  never stops the rest of a book from pricing.

Fallback: ``UnavailableMarket`` stands in for a pair whose market could not
be built. It answers every request with the construction error, so callers
report that error per line instead of failing as a whole.
"""
import copy
import math
import os
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from market_data import MarketData
from market_data.variance_calendar import DEFAULT_WEIGHTS, pair_holidays, weighted_days
from app.curve_roll import (
    OVERNIGHT_TENORS, anchor_forward_rows, overnight_expiry, roll_forward_rows, with_short_end,
    roll_vol_rows,
)
from apeiron.hyperion.resolve import attach as attach_resolved
from app.market_service import MarketService, pair_delivery
from app.market_store import validate_market
from app.smile_service import DiagnosticsCollector, SurfaceCache, bind_cache, reuse_prepared
from helper_functions.input_resolvers import resolve_expiry

PRICING_MODELS = ("vv", "vv-sabr")
DELTA_REPORTING = (None, "forward", "spot")
VOL_COLUMNS = ("tenor", "expiry", "atm", "rr_25", "bf_25", "rr_10", "bf_10")
FORWARD_COLUMNS = ("delivery_date", "fwd_pts", "term_discount")
EVENT_COLUMNS = ("event_date", "variance_days")
#: The event table as the app shows and saves it: each event its own row,
#: named (``event`` is optional on input: older files have no names). Same-day
#: events add up in the variance days.
EVENT_TABLE_COLUMNS = ("event_date", "event", "variance_days")


class MarketContextError(ValueError):
    """The inputs cannot form a market for any expiry."""


def naive_date(value):
    """Midnight, timezone-naive ``pd.Timestamp`` of ``value``."""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp.normalize()


class DeliveryCurve:
    """A curve read from ``rows`` once, then interpolated linearly at any date
    within its supplied delivery range (``interpolate_delivery_curve``).

    The rows are parsed on the first call, not here, so a curve that is never
    used raises nothing; a bad curve raises on every call, as before.
    """

    def __init__(self, rows, date_column, value_column, name):
        self.rows, self.date_column, self.value_column, self.name = (
            rows, date_column, value_column, name)
        self._prepared = None

    def _prepare(self):
        rows, name = self.rows, self.name
        dates = pd.to_datetime(rows[self.date_column], errors="coerce")
        values = pd.to_numeric(rows[self.value_column], errors="coerce")
        valid = dates.notna() & values.notna() & np.isfinite(values)
        curve = rows.loc[valid].copy()
        if curve.empty:
            raise ValueError(f"{name} rows must have valid delivery dates and finite values")
        curve["_date"] = dates.loc[valid].map(pd.Timestamp.toordinal)
        curve["_value"] = values.loc[valid]
        curve.sort_values("_date", inplace=True)
        if curve["_date"].duplicated().any():
            raise ValueError("Forward delivery dates must be unique")
        return curve["_date"].to_numpy(), curve["_value"].to_numpy()

    def __call__(self, target_date):
        if self._prepared is None:
            self._prepared = self._prepare()
        ordinals, values = self._prepared
        target = pd.Timestamp(target_date).toordinal()
        if target < ordinals[0] or target > ordinals[-1]:
            raise ValueError(f"{self.name} delivery date is outside the quoted forward range")
        return float(np.interp(target, ordinals, values))


def interpolate_delivery_curve(rows, date_column, value_column, target_date, name):
    """Linearly interpolate a curve only within its supplied delivery range."""
    return DeliveryCurve(rows, date_column, value_column, name)(target_date)


#: ``market_for_expiry``'s cache: off with ``VV_MARKET_CACHE=0``; with
#: ``VV_MARKET_CACHE_CHECK=1`` every reuse is checked against a fresh build.
MARKET_CACHE_ENV, MARKET_CACHE_CHECK_ENV = "VV_MARKET_CACHE", "VV_MARKET_CACHE_CHECK"
#: Markets kept per context (the spot ladder adds one per spot and expiry).
MARKET_CACHE_SIZE = 1024


def market_difference(cached, fresh):
    """The first field (or resolved input) on which two markets differ, or ``""``."""
    for name in sorted(set(vars(cached)) | set(vars(fresh))):
        if name == "diagnostics_token":
            continue
        a, b = getattr(cached, name, None), getattr(fresh, name, None)
        try:
            same = bool(a == b) or (isinstance(a, float) and isinstance(b, float)
                                    and math.isnan(a) and math.isnan(b))
        except Exception:                  # arrays and the like: compare their text
            same = repr(a) == repr(b)
        if not same:
            return f"{name}: {a!r} vs {b!r}"[:200]
    return ""


THETA_VOL_ROLLS = ("rolled", "forward", "tenor")
#: The rolled market's builder, ``roller(context, to_date) -> (context, notes)``:
#: registered by ``app.rolled_market`` when it loads (it imports this module,
#: so this one does not import it). Unset, theta takes the forward-vol roll.
ROLLER = None
_WING_QUOTES = ("PUT_10", "PUT_25", "ATM", "CALL_25", "CALL_10")


def wing_vols(market):
    """The five smile vols (10D/25D put, ATM, 25D/10D call) from a market's quotes."""
    atm, rr25, bf25, rr10, bf10 = (market.v_atm, market.rr_25, market.bf_25,
                                   market.rr_10, market.bf_10)
    return {"PUT_10": atm + bf10 - rr10 / 2, "PUT_25": atm + bf25 - rr25 / 2, "ATM": atm,
            "CALL_25": atm + bf25 + rr25 / 2, "CALL_10": atm + bf10 + rr10 / 2}


def quotes_from_wings(wings):
    """ATM / RR / BF quotes back from the five smile vols."""
    atm = wings["ATM"]
    return {"v_atm": atm,
            "rr_25": wings["CALL_25"] - wings["PUT_25"],
            "bf_25": (wings["CALL_25"] + wings["PUT_25"]) / 2.0 - atm,
            "rr_10": wings["CALL_10"] - wings["PUT_10"],
            "bf_10": (wings["CALL_10"] + wings["PUT_10"]) / 2.0 - atm}


@dataclass(frozen=True)
class PricingSettings:
    """Model and convention choices shared by every expiry of a market."""

    pricing_model: str = "vv"
    atm_type: str = "dn"
    delta_base: bool = True
    smooth_pillar_interpolation: bool = True
    #: Spot-equivalent: every delta and gamma adds up across tenors and pairs.
    #: Line tables convert each option's delta to its pair's convention
    #: (``pricing.greeks.convention_deltas``).
    delta_reporting: Optional[str] = "spot"
    #: Weights of a Saturday, Sunday, holiday and Dec 25 / Jan 1 on the vol
    #: time clock (``market_data.variance_calendar``); ``CALENDAR`` weights
    #: every day 1, the calendar-day clock used before.
    variance_weights: tuple = DEFAULT_WEIGHTS
    #: The next-day market theta reprices on: ``"rolled"`` is the rolled
    #: market (``app.rolled_market``: the next day's pillar dates, today's
    #: pillar dates kept as nodes, forward variance held at all of them,
    #: forward points sticky tenor), the market a later valuation date uses,
    #: so theta is exactly the date roll; ``"forward"`` holds today's forward
    #: vols at each expiry on trading-day-shifted pillars (theta before
    #: 2026-10-23); ``"tenor"`` keeps each tenor's quotes on its rolled date.
    #: The last two remain for comparison.
    theta_vol_roll: str = "rolled"

    def problems(self):
        found = []
        if self.theta_vol_roll not in THETA_VOL_ROLLS:
            found.append(f"theta vol roll must be one of {', '.join(THETA_VOL_ROLLS)}, "
                         f"not {self.theta_vol_roll!r}")
        weights = tuple(self.variance_weights)
        if len(weights) != 4 or not all(isinstance(w, (int, float)) and math.isfinite(w)
                                        and 0.0 <= w <= 1.0 for w in weights):
            found.append(f"variance weights must be four numbers between 0 and 1, "
                         f"not {self.variance_weights!r}")
        if self.pricing_model not in PRICING_MODELS:
            found.append(f"pricing model must be one of {', '.join(PRICING_MODELS)}, "
                         f"not {self.pricing_model!r}")
        if self.delta_reporting not in DELTA_REPORTING:
            found.append(f"delta reporting must be forward or spot, not {self.delta_reporting!r}")
        return found


def _table(name, frame, columns, problems):
    if not isinstance(frame, pd.DataFrame):
        problems.append(f"{name} must be a table")
        return pd.DataFrame(columns=list(columns))
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        problems.append(f"{name} is missing columns: {', '.join(missing)}")
    return frame.copy(deep=True)


class MarketContext:
    """A pair's market on one valuation date, ready to build per-expiry markets.

    ``market_service`` and ``diagnostics`` let a caller share calibration
    caches and diagnostics across contexts (the app keeps them for the whole
    session). Without them the context gets its own, cached for its lifetime.
    """

    def __init__(self, currency_pair, valuation_date, spot, forward_divisor,
                 broker_strangle, vol_rows, forward_rows, event_rows=None,
                 settings=None, market_service=None, diagnostics=None, short_points=None,
                 interpolation_nodes=None):
        problems = []
        pair = str(currency_pair or "").replace("/", "").replace(" ", "").strip().upper()
        if not re.fullmatch(r"[A-Z]{6}", pair):
            problems.append(f"currency pair must be two 3-letter ISO codes, not {currency_pair!r}")
        try:
            naive_date(valuation_date)
        except (TypeError, ValueError):
            problems.append(f"valuation date {valuation_date!r} is not a date")
        for name, value in (("spot", spot), ("forward divisor", forward_divisor)):
            try:
                number = float(value)
            except (TypeError, ValueError):
                number = math.nan
            if not math.isfinite(number) or number <= 0:
                problems.append(f"{name} must be a finite positive number, not {value!r}")
        settings = PricingSettings() if settings is None else settings
        problems += settings.problems()
        vol_rows = _table("vol data", vol_rows, VOL_COLUMNS, problems)
        forward_rows = _table("forward points", forward_rows, FORWARD_COLUMNS, problems)
        if event_rows is None:
            event_rows = pd.DataFrame(columns=list(EVENT_COLUMNS))
        event_rows = _table("event calendar", event_rows, EVENT_COLUMNS, problems)
        if problems:
            raise MarketContextError(f"{pair or currency_pair} market: " + "; ".join(problems))

        self.currency_pair = pair
        self.valuation_date = valuation_date
        self.spot = spot
        self.forward_divisor = forward_divisor
        self.broker_strangle = bool(broker_strangle)
        self.vol_rows, self.forward_rows, self.event_rows = vol_rows, forward_rows, event_rows
        #: Quoted ON/TN/SN swap points (``app.market_store.SHORT_TENORS``);
        #: only value-dated deals read them (``market_for_delivery``).
        self.short_points = dict(short_points or {})
        #: Extra vol nodes (``expiry`` and the five ATM/RR/BF quotes) the surface
        #: interpolates through but that are no tenor: a rolled market's
        #: forward-variance nodes at the earlier day's pillar dates
        #: (``app.rolled_market``). Tenor lookups, calibration and the risk
        #: buckets see the tenor pillars (``vol_rows``) only.
        self.interpolation_nodes = (pd.DataFrame(columns=list(VOL_COLUMNS))
                                    if interpolation_nodes is None else interpolation_nodes.copy())
        self.settings = settings
        if diagnostics is None:
            diagnostics = DiagnosticsCollector()
            # Held here: the binding is a weak reference, so an unheld cache
            # is dropped at once and smiles fall back to a small shared one.
            self._surface_cache = SurfaceCache()
            bind_cache(diagnostics.token, self._surface_cache)
        self.diagnostics = diagnostics
        self.market_service = market_service or MarketService(diagnostics)

    @classmethod
    def from_payload(cls, payload, settings=None, market_service=None, diagnostics=None,
                     valuation_date=None):
        """Build from a saved market dict (see ``app.market_store``).

        ``valuation_date`` values the saved quotes as of another date than
        the one they were saved for, as the app does for its active market
        when the valuation date changes.
        """
        try:
            loaded = validate_market(payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise MarketContextError(f"Invalid market: {exc}") from exc
        if valuation_date is None:
            valuation_date = loaded["valuation_date"]
        return cls(
            loaded["currency_pair"], valuation_date, loaded["spot"],
            loaded["forward_divisor"], loaded["broker_strangle"], loaded["vol_rows"],
            loaded["forward_rows"], loaded["event_rows"], settings=settings,
            market_service=market_service, diagnostics=diagnostics,
            short_points=loaded.get("short_points"),
        )

    def with_spot(self, spot):
        """This market at another spot; quotes, curves and caches are shared."""
        number = float(spot)
        if not math.isfinite(number) or number <= 0:
            raise MarketContextError(
                f"{self.currency_pair} spot must be a finite positive number, not {spot!r}")
        moved = copy.copy(self)
        # The market-data spot stays the anchor of the smile dynamics, however
        # often the spot is moved (``app.pricer``: the Portfolio's spot edit).
        moved.spot_anchor = self.__dict__.get("spot_anchor", self.spot)
        moved.spot = number
        return moved

    def __repr__(self):
        return (f"MarketContext({self.currency_pair}, {naive_date(self.valuation_date):%Y-%m-%d}, "
                f"spot={self.spot}, model={self.settings.pricing_model})")

    # -- time and curves -------------------------------------------------

    def variance_days_between(self, start, end):
        """Variance days in ``(start, end]``: each day weighted as a weekday,
        weekend day, holiday of either currency or year-end day
        (``market_data.variance_calendar``, weights from the settings), plus
        the event weights."""
        start = naive_date(start)
        end = naive_date(end)
        if end < start:
            raise ValueError("End date must not precede start date")
        # Kept per context and date pair (the events are fixed, so the answer
        # is too); ``VV_MARKET_CACHE=0`` computes every time.
        cache = self.__dict__.setdefault("_variance_between_cache", {})
        use_cache = os.environ.get(MARKET_CACHE_ENV, "1").strip() != "0"
        if use_cache and (start, end) in cache:
            return cache[(start, end)]
        total = self._weighted_days(start, end)
        events = self.event_rows
        if not events.empty:
            event_dates = pd.to_datetime(
                events["event_date"], errors="coerce", utc=True
            ).dt.tz_localize(None)
            weights = pd.to_numeric(events["variance_days"], errors="coerce").fillna(0.0)
            if not np.isfinite(weights.to_numpy(dtype=float)).all():
                raise ValueError("Event variance days must be finite")
            total += float(weights.loc[(event_dates > start) & (event_dates <= end)].sum())
        if use_cache:
            cache[(start, end)] = float(total)
        return float(total)

    def _weighted_days(self, start, end):
        """Weighted days, cached per context; the pair's holidays load once."""
        cache = self.__dict__.setdefault("_variance_day_cache", {})
        key = (start, end)
        if key not in cache:
            if "_holidays" not in self.__dict__:
                self._holidays, self.calendar_notes = pair_holidays(self.currency_pair)
            cache[key] = weighted_days(start, end, self._holidays,
                                       type(DEFAULT_WEIGHTS)(*self.settings.variance_weights))
        return cache[key]

    def delivery(self, expiry):
        return pair_delivery(self.currency_pair, expiry, self.diagnostics)

    def forward_curve(self, curve_date=None, rows=None):
        """Forward rows for ``curve_date``, anchored at its spot date."""
        rows = self.forward_rows if rows is None else rows
        return anchor_forward_rows(
            rows, self.currency_pair,
            self.valuation_date if curve_date is None else curve_date,
        )

    def forward_points_for_expiry(self, expiry, curve_rows=None):
        return interpolate_delivery_curve(
            self.forward_curve() if curve_rows is None else curve_rows,
            "delivery_date", "fwd_pts", self.delivery(expiry), "Forward points",
        )

    def term_discount_for_expiry(self, expiry, curve_rows=None):
        """Interpolate decimal SIMPLE term deposit rates by delivery date."""
        return interpolate_delivery_curve(
            self.forward_curve() if curve_rows is None else curve_rows,
            "delivery_date", "term_discount", self.delivery(expiry), "Term discount rate",
        )

    def market_for_delivery(self, delivery, as_of=None, spot_value=None):
        """The market for a value-dated forward delivering on ``delivery``.

        Forward points and the term rate are read off the forward curve at the
        delivery date itself (linear between quoted deliveries, so a broken
        date takes the linearly interpolated outright). No smile is built: the
        vol fields hold the first pillar's ATM only to make a valid
        ``MarketData``, and a forward never reads them. With ``as_of`` after
        the valuation date (theta) the curve rolls as in ``market_for_expiry``
        while the delivery stays fixed.

        The market's quoted short swaps (``short_points``) are nodes too: a
        delivery before spot takes spot less TN (tomorrow) or less TN and ON
        (today), the day after spot spot plus SN. A delivery inside spot
        without the points it needs (before the curve's first node) takes
        zero points (forward = spot) and the first term rate, and the
        market's ``forward_note`` says so.
        """
        valuation_date = naive_date(self.valuation_date)
        curve_date = valuation_date if as_of is None else naive_date(as_of)
        delivery = naive_date(delivery)
        if delivery < curve_date:
            raise ValueError("Value date must not be before the valuation date")
        rolled = self._rolled_for(delivery, as_of)
        if rolled is not None:
            return rolled.market_for_delivery(
                delivery, spot_value=self.spot if spot_value is None else spot_value)
        rows = with_short_end(self.forward_curve(valuation_date), self.currency_pair,
                              valuation_date, self.short_points)
        if curve_date == valuation_date:
            curve_rows = rows
        else:
            curve_rows = roll_forward_rows(rows, None, self.currency_pair, valuation_date,
                                           curve_date, self.delivery)
        first = pd.to_datetime(curve_rows["delivery_date"], errors="coerce").min()
        note = None
        if pd.notna(first) and delivery < naive_date(first):
            note = (f"value date {delivery:%Y-%m-%d} is inside spot "
                    f"({naive_date(first):%Y-%m-%d}) and its ON/TN points are not "
                    f"quoted: valued at spot")
            fwd_pts = 0.0
            term_rate = interpolate_delivery_curve(curve_rows, "delivery_date",
                                                   "term_discount", first, "Term discount rate")
        else:
            fwd_pts = interpolate_delivery_curve(curve_rows, "delivery_date", "fwd_pts",
                                                 delivery, "Forward points")
            term_rate = interpolate_delivery_curve(curve_rows, "delivery_date",
                                                   "term_discount", delivery,
                                                   "Term discount rate")
        atm = pd.to_numeric(self.vol_rows.get("atm"), errors="coerce").dropna()
        market = MarketData(
            s=self.spot if spot_value is None else spot_value, fwd_pts=fwd_pts,
            v_atm=float(atm.iloc[0]) if len(atm) and atm.iloc[0] > 0 else 10.0,
            rr_25=0.0, rr_10=0.0, bf_25=0.0, bf_10=0.0,
            forward_divisor=self.forward_divisor, term_discount=term_rate,
            currency_pair=self.currency_pair, broker_strangle=False,
            delta_reporting=self.settings.delta_reporting,
        )
        market.forward_note = note
        # A value-dated deal's dates, resolved here (atlas.resolved).
        attach_resolved(market, delivery, curve_date, delivery=delivery)
        return market

    def resolve_market_expiry(self, value):
        """Resolve pillar tenor strings to their configured expiry dates."""
        token = str(value).strip().upper()
        if token in OVERNIGHT_TENORS:
            return overnight_expiry(self.currency_pair, self.valuation_date)
        rows = self.vol_rows
        matches = rows[rows["tenor"].astype(str).str.strip().str.upper() == token]
        if len(matches) > 1:
            raise ValueError(f"Tenor {token} appears more than once in Vol data")
        if len(matches) == 1:
            return pd.Timestamp(matches.iloc[0]["expiry"]).date()
        return resolve_expiry(value, self.valuation_date, rows["tenor"])

    # -- calibration -----------------------------------------------------

    def pillar_expiries(self):
        """Quoted pillar expiries after the valuation date, ascending."""
        expiries = pd.to_datetime(
            self.vol_rows["expiry"], errors="coerce", utc=True
        ).dt.tz_localize(None)
        valuation = naive_date(self.valuation_date)
        return sorted({naive_date(e) for e in expiries.dropna() if naive_date(e) > valuation})

    def calibrate_pillar(self, expiry):
        """Build and calibrate the pricing smile at ``expiry``.

        The calibrated surface lands in the shared cache, so pricing at
        this pillar afterwards reuses it instead of calibrating again.
        """
        from helper_functions.vanna_volga_helpers import get_model_map
        market = self.market_for_expiry(expiry)
        forward = market.s + market.fwd_pts / market.forward_divisor
        get_model_map(market)[self.settings.pricing_model](
            market=market, K=forward, expiry_date=naive_date(expiry).date().isoformat(),
            valuation_date=naive_date(self.valuation_date).date().isoformat(),
            CallPut="CALL", atm_conv=market.atm_conv, deltaBase=market.deltaBase,
        ).get_vol_for_strike()

    # -- per-expiry market -----------------------------------------------

    def market_for_expiry(self, expiry, as_of=None, allow_short_end=False, spot_value=None):
        """Build a MarketData object for ``expiry`` from the market tables.

        With ``as_of`` after the valuation date (theta's next day), by default
        (``PricingSettings.theta_vol_roll`` = ``"rolled"``) the market is the
        rolled market (``app.rolled_market``), the one a later valuation date
        prices on, so theta is exactly the date roll. If the market cannot
        roll, the reason is recorded and the forward-vol roll below is used.

        With ``"forward"`` the forward points and rates roll as described in
        ``_market_for_expiry`` (carry) and the vol quotes hold today's forward
        vols. Each of the five smile vols
        keeps today's total variance to the expiry minus the variance of the
        days that pass (today's ``as_of`` node), so on an upward-sloping curve
        the passing low-vol day raises the remaining vol. A weekend passes with
        almost no variance on the variance-day clock. If that cannot be formed
        (non-positive remaining variance, or a failure), the sticky-tenor quotes
        are kept and the reason is recorded in the diagnostics.

        Each market is built once per (expiry, ``as_of``, short end, spot) and
        reused (``_cached_market``); every caller gets its own copy.
        """
        return self._cached_market(expiry, as_of, allow_short_end, spot_value)

    def _cached_market(self, expiry, as_of, allow_short_end, spot_value):
        """``_build_market_for_expiry``, built once per request and reused.

        Fail-safes: the key carries the effective spot (``with_spot`` copies
        share this cache); a failed build is not cached, so it fails the same
        way again; the diagnostics a build recorded are recorded again on each
        reuse; each caller gets its own shallow copy, so no caller sees
        another's changes; the cache is bounded per context.
        ``VV_MARKET_CACHE=0`` turns it off, and ``VV_MARKET_CACHE_CHECK=1``
        rebuilds on every reuse and uses the fresh market (recording the
        difference) if they disagree.
        """
        if os.environ.get(MARKET_CACHE_ENV, "1").strip() == "0":
            return self._build_market_for_expiry(expiry, as_of, allow_short_end, spot_value)
        try:
            key = (naive_date(expiry), None if as_of is None else naive_date(as_of),
                   bool(allow_short_end), float(self.spot if spot_value is None else spot_value))
        except (TypeError, ValueError):
            return self._build_market_for_expiry(expiry, as_of, allow_short_end, spot_value)
        lock = self.__dict__.setdefault("_market_cache_lock", threading.RLock())
        with lock:
            cache = self.__dict__.setdefault("_market_cache", OrderedDict())
            found = cache.get(key)
            if found is not None:
                cache.move_to_end(key)
        collectors = self._collectors()
        if found is not None:
            market, records = found
            for collector in collectors:
                for record in records:
                    collector.append(record)
            if os.environ.get(MARKET_CACHE_CHECK_ENV) == "1":
                fresh = self._build_market_for_expiry(expiry, as_of, allow_short_end, spot_value)
                difference = market_difference(market, fresh)
                if difference:
                    self.diagnostics.append((self.settings.pricing_model, "market cache",
                                             f"{self.currency_pair} {key[0]:%Y-%m-%d}: cached "
                                             f"market differs ({difference}); rebuilt"))
                    with lock:
                        cache.pop(key, None)
                    return fresh
            reused = copy.copy(market)
            # The build registered its pillar smile for reuse; so does a reuse,
            # so smiles never depend on what the surface cache has dropped.
            prepared = getattr(market, "_prepared_reuse", None)
            if prepared is not None:
                reuse_prepared(reused, prepared)
            return reused
        before = [len(collector.records()) for collector in collectors]
        market = self._build_market_for_expiry(expiry, as_of, allow_short_end, spot_value)
        records = tuple(record for collector, start in zip(collectors, before)
                        for record in collector.records()[start:])
        with lock:
            cache[key] = (market, records)
            while len(cache) > MARKET_CACHE_SIZE:
                cache.popitem(last=False)
        return copy.copy(market)

    def _converted_curve(self, curve_date, rows, forward_points, term_discount):
        """``market_service.convert_curve`` for ``curve_date``'s pillar rows, kept
        per curve date and spot: its inputs depend on nothing else, so later
        expiries skip rebuilding every pillar's market just to find its cache.
        A reuse records the build's diagnostics again (as ``convert_curve``'s
        own cache does) and returns a copy of the rows; a failure is not kept;
        ``VV_MARKET_CACHE=0`` turns it off."""
        def convert():
            return self.market_service.convert_curve(
                rows, curve_date.date(), self.spot, self.currency_pair, self.forward_divisor,
                self.settings.pricing_model, self.broker_strangle, forward_points,
                term_discount)
        if os.environ.get(MARKET_CACHE_ENV, "1").strip() == "0":
            return convert()
        key = (curve_date, float(self.spot))
        lock = self.__dict__.setdefault("_market_cache_lock", threading.RLock())
        with lock:
            found = self.__dict__.setdefault("_curve_cache", {}).get(key)
        collectors = self._collectors()
        if found is None:
            before = [len(collector.records()) for collector in collectors]
            converted, reasons, prepared = convert()
            records = tuple(record for collector, start in zip(collectors, before)
                            for record in collector.records()[start:])
            found = (converted.copy(), reasons, prepared, records)
            with lock:
                self._curve_cache[key] = found
            return converted, reasons, prepared
        converted, reasons, prepared, records = found
        for collector in collectors:
            for record in records:
                collector.append(record)
        return converted.copy(), reasons, prepared

    def _collectors(self):
        """The diagnostics collectors a build records into: this market's and
        its market service's (often the same one)."""
        found = [self.diagnostics]
        service = getattr(getattr(self, "market_service", None), "collector", None)
        if service is not None and service is not self.diagnostics:
            found.append(service)
        return [collector for collector in found if hasattr(collector, "records")]

    def _build_market_for_expiry(self, expiry, as_of=None, allow_short_end=False,
                                 spot_value=None):
        """``market_for_expiry`` without the cache."""
        rolled = self._rolled_for(expiry, as_of)
        if rolled is not None:
            return rolled.market_for_expiry(
                expiry, spot_value=self.spot if spot_value is None else spot_value)
        market = self._market_for_expiry(expiry, as_of, allow_short_end, spot_value)
        if (as_of is None or self.settings.theta_vol_roll not in ("forward", "rolled")
                or naive_date(as_of) <= naive_date(self.valuation_date)
                or naive_date(expiry) <= naive_date(as_of)):
            return market
        try:
            return market.copy_with(**self._forward_vol_quotes(expiry, as_of, spot_value))
        except Exception as exc:
            self.diagnostics.append((self.settings.pricing_model, "theta",
                                     f"{self.currency_pair} {naive_date(expiry):%Y-%m-%d}: "
                                     f"theta on tenor-rolled quotes ({exc})"))
            return market

    def _pillar_rows(self):
        """The vol table by expiry, with the interpolation nodes: parsed once per
        context (a market build used to parse it every time) and copied to each
        caller, who may change its copy. A table that cannot be parsed raises
        every time; ``VV_MARKET_CACHE=0`` parses on every call."""
        cached = self.__dict__.get("_pillar_rows_cache")
        if cached is None or os.environ.get(MARKET_CACHE_ENV, "1").strip() == "0":
            rows = self.vol_rows.copy()
            if rows.empty:
                raise ValueError("Enter at least one volatility expiry")
            rows["expiry"] = pd.to_datetime(
                rows["expiry"], errors="coerce", utc=True
            ).dt.tz_localize(None)
            rows = rows.dropna(subset=["expiry"]).sort_values("expiry").reset_index(drop=True)
            if rows.empty:
                raise ValueError("Volatility expiries must be valid dates")
            cached = self._pillar_rows_cache = self._with_interpolation_nodes(rows)
        return cached.copy()

    def interpolation_rows(self):
        """The vol curve the surface interpolates: the tenor pillars and any
        extra nodes, by expiry (naive timestamps)."""
        try:
            return self._pillar_rows()          # parsed once per context
        except ValueError:                      # an empty table: no error here
            pass
        rows = self.vol_rows.copy()
        rows["expiry"] = pd.to_datetime(rows["expiry"], errors="coerce", utc=True).dt.tz_localize(None)
        rows = rows.dropna(subset=["expiry"]).sort_values("expiry").reset_index(drop=True)
        return self._with_interpolation_nodes(rows)

    def _with_interpolation_nodes(self, rows):
        nodes = self.interpolation_nodes
        if nodes is None or nodes.empty:
            return rows
        extra = nodes.copy()
        extra["expiry"] = pd.to_datetime(extra["expiry"], errors="coerce", utc=True).dt.tz_localize(None)
        extra = extra.dropna(subset=["expiry"])
        extra = extra.loc[~extra["expiry"].isin(set(rows["expiry"]))]
        if extra.empty:
            return rows
        if "tenor" not in extra.columns:
            extra["tenor"] = ""
        merged = pd.concat([rows, extra[list(rows.columns)]], ignore_index=True)
        return merged.sort_values("expiry").reset_index(drop=True)

    def _rolled_for(self, end, as_of):
        """The market rolled to ``as_of`` that prices ``end`` (an expiry or a
        delivery after it), or ``None``: not the rolled theta, ``as_of`` not
        later, ``end`` not after it, or the roll failed (recorded; the
        forward-vol roll is used instead)."""
        if (as_of is None or self.settings.theta_vol_roll != "rolled"
                or naive_date(as_of) <= naive_date(self.valuation_date)
                or naive_date(end) <= naive_date(as_of)):
            return None
        # Keyed by spot too: ``with_spot`` copies share this cache.
        key = (naive_date(as_of), float(self.spot))
        cache = self.__dict__.setdefault("_rolled_cache", {})
        if key not in cache:
            try:
                if ROLLER is None:
                    raise RuntimeError("the rolled market is not loaded (app.rolled_market)")
                cache[key] = ROLLER(self, as_of)[0]
            except Exception as exc:          # a RollError, or a failure while rolling
                cache[key] = None
                self.diagnostics.append((self.settings.pricing_model, "theta",
                                         f"{self.currency_pair}: theta on the forward-vol roll, "
                                         f"the market does not roll to {naive_date(as_of):%Y-%m-%d} "
                                         f"({exc})"))
        return cache[key]

    def _forward_vol_quotes(self, expiry, as_of, spot_value=None):
        """Quotes at ``expiry`` seen from ``as_of`` with today's forward vols held."""
        valuation = naive_date(self.valuation_date)
        # Today's markets at the expiry and at ``as_of``: the same requests
        # pricing makes, so they come from the cache.
        whole = wing_vols(self._cached_market(expiry, None, False, spot_value))
        passed = wing_vols(self._cached_market(as_of, None, True, spot_value))
        total_days = (naive_date(expiry) - valuation).days
        passed_days = (naive_date(as_of) - valuation).days
        remaining = total_days - passed_days
        wings = {}
        for name in _WING_QUOTES:
            variance = whole[name] ** 2 * total_days - passed[name] ** 2 * passed_days
            if not (variance > 0 and np.isfinite(variance)):
                raise ValueError(f"no forward variance left in the {name} vol")
            wings[name] = math.sqrt(variance / remaining)
        return quotes_from_wings(wings)

    def _market_for_expiry(self, expiry, as_of=None, allow_short_end=False, spot_value=None):
        """Build a MarketData object for ``expiry`` from the market tables.

        With ``as_of`` after the valuation date the curves are rolled to it
        (sticky tenor): pillar expiries and forward deliveries move with their
        tenors while every tenor keeps its quotes, so a fixed expiry picks up
        the vol and forward-point roll-down. Pillar total variance is fixed by
        the quotes; events only distribute it between pillar dates, so a
        passed event's share moves to the remaining days.

        Expiries before the first pillar (O/N up to 1W) are priced from the
        first pillar: total variance runs linearly in event-weighted variance
        days from zero at the curve date, which is the same as deriving an
        O/N node from the 1W quotes with the event weights and interpolating
        between that node and 1W. Each of the five wing vols is scaled this
        way, so RR and BF follow as a by-product.
        """
        settings = self.settings
        pair, spot, valuation_date = self.currency_pair, self.spot, self.valuation_date
        curve_date = naive_date(valuation_date if as_of is None else as_of)
        requested_date = naive_date(expiry)
        if requested_date == curve_date and allow_short_end and as_of is not None:
            return self._market_for_expiry(expiry, spot_value=spot_value)
        rows = self._pillar_rows()
        forward_rows = self.forward_rows
        roll_days = (curve_date - naive_date(valuation_date)).days
        # A rolled pillar (moved by trading days) keeps its tenor's
        # settlement: its delivery comes from the calendar-day roll.
        calendar_rolled = {}
        # The forward curve for a curve date depends only on this market's
        # tables, so it is built and read once per date and reused (shared
        # with ``with_spot`` copies: it does not depend on spot).
        curves = self.__dict__.setdefault("_delivery_curve_cache", {})
        cached = curves.get(curve_date)
        if curve_date != naive_date(valuation_date):
            if cached is None:
                forward_rows = self.forward_curve(valuation_date, forward_rows)
            unadjusted = rows["expiry"] + pd.Timedelta(days=roll_days)
            rows = roll_vol_rows(rows, pair, valuation_date, curve_date)
            calendar_rolled = {naive_date(a): naive_date(u)
                               for a, u in zip(rows["expiry"], unadjusted)}
            if cached is None:
                forward_rows = roll_forward_rows(
                    forward_rows, rows, pair, valuation_date, curve_date, self.delivery,
                )
        if cached is None:
            curve_rows = (forward_rows if roll_days
                          else self.forward_curve(curve_date, forward_rows))
            cached = curves[curve_date] = (
                curve_rows,
                DeliveryCurve(curve_rows, "delivery_date", "fwd_pts", "Forward points"),
                DeliveryCurve(curve_rows, "delivery_date", "term_discount",
                              "Term discount rate"))
        curve_rows, forward_points, discount_curve = cached

        def pillar_delivery(value):
            # Synthetic risk nodes retain their original settlement offsets.
            value = calendar_rolled.get(naive_date(value), naive_date(value))
            original_expiry = naive_date(value) - pd.Timedelta(days=roll_days)
            return self.delivery(original_expiry) + pd.Timedelta(days=roll_days)

        def pillar_forward_points(value):
            return forward_points(pillar_delivery(value))

        def pillar_term_discount(value):
            return discount_curve(pillar_delivery(value))
        if requested_date < curve_date or (requested_date == curve_date and not allow_short_end):
            raise ValueError("Expiry must be after the valuation date")
        target_date = requested_date
        # Before the first pillar the variance interpolation below runs from
        # zero at the curve date, so the short end carries the first pillar's
        # variance scaled by its own variance-day weights.
        if target_date > rows["expiry"].iloc[-1]:
            raise ValueError("Expiry is outside the quoted volatility range")
        for column in ("atm", "rr_25", "bf_25", "rr_10", "bf_10"):
            rows[column] = pd.to_numeric(rows[column], errors="coerce")
        rows, curve_reasons, curve_prepared = self._converted_curve(
            curve_date, rows, pillar_forward_points, pillar_term_discount)

        quotes = pd.DataFrame({
            "PUT_10": rows["atm"] + rows["bf_10"] - rows["rr_10"] / 2,
            "PUT_25": rows["atm"] + rows["bf_25"] - rows["rr_25"] / 2,
            "ATM": rows["atm"],
            "CALL_25": rows["atm"] + rows["bf_25"] + rows["rr_25"] / 2,
            "CALL_10": rows["atm"] + rows["bf_10"] + rows["rr_10"] / 2,
        })
        values = quotes.to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("Vol quotes must reconstruct finite positive wing vols")
        pillar_vdays = np.array([
            self.variance_days_between(curve_date, pillar_date)
            for pillar_date in rows["expiry"]
        ], dtype=float)
        if (pillar_vdays <= 0).any() or np.any(np.diff(pillar_vdays) <= 0):
            raise ValueError("Volatility pillars must have increasing positive variance days")
        pillar_calendar_days = np.array([
            (pillar_date - curve_date).days for pillar_date in rows["expiry"]
        ], dtype=float)
        if (pillar_calendar_days <= 0).any():
            raise ValueError("Volatility pillars must have positive calendar days")
        target_vdays = self.variance_days_between(curve_date, target_date)
        calendar_days = (target_date - curve_date).days
        if target_vdays <= 0 or calendar_days <= 0:
            raise ValueError("Expiry must have positive calendar and variance days")
        x = np.concatenate(([0.0], pillar_vdays))
        interpolated = {}
        for name in quotes:
            # Quotes are annualized over calendar time, so total variance must be
            # accumulated in calendar-annualized units. Only the interpolation
            # clock is the event-weighted variance-day measure; mixing the two
            # measures here would rescale every quote by sqrt(vdays/caldays).
            total_variance = quotes[name].to_numpy(dtype=float) ** 2 * pillar_calendar_days
            y = np.concatenate(([0.0], total_variance))
            interpolated[name] = float(np.sqrt(np.interp(target_vdays, x, y) / calendar_days))
        atm = interpolated["ATM"]
        rr25 = interpolated["CALL_25"] - interpolated["PUT_25"]
        bf25 = (interpolated["CALL_25"] + interpolated["PUT_25"]) / 2.0 - atm
        rr10 = interpolated["CALL_10"] - interpolated["PUT_10"]
        bf10 = (interpolated["CALL_10"] + interpolated["PUT_10"]) / 2.0 - atm
        market_args = dict(
            s=spot if spot_value is None else spot_value,
            fwd_pts=forward_points(self.delivery(requested_date)),
            v_atm=atm, rr_25=rr25, rr_10=rr10, bf_25=bf25, bf_10=bf10,
            forward_divisor=self.forward_divisor, atm_conv=settings.atm_type,
            deltaBase=settings.delta_base,
        )
        term_discount = discount_curve(self.delivery(requested_date))
        try:
            market = MarketData(**market_args, term_discount=term_discount)
        except TypeError as exc:
            # Streamlit can retain this imported class until its process
            # restarts, even after the app module has reloaded.
            if "term_discount" not in str(exc) or "unexpected keyword" not in str(exc):
                raise
            market = MarketData(**market_args)
            market.term_discount = term_discount
        smooth = settings.smooth_pillar_interpolation
        market.smooth_pillar_interpolation = smooth
        market.delta_reporting = settings.delta_reporting
        # Lets the pricers discount to the option's delivery date on the pair's
        # own day-count basis. Assigned after construction so a stale imported
        # MarketData class retained across a Streamlit reload cannot raise.
        market.currency_pair = pair
        market.legacy_target_blend = not smooth
        market.compatibility_mode = "auto" if smooth else "legacy"
        market.wrapped_smile = True
        market.broker_strangle = False
        market.force_legacy_smile = bool(curve_reasons)
        market.smile_reasons = curve_reasons
        market.smile_model = settings.pricing_model
        market.diagnostics_token = self.diagnostics.token
        # The core's dates and conventions, resolved here (atlas.resolved),
        # before the smile reads them.
        attach_resolved(market, requested_date, curve_date)
        if not curve_reasons and requested_date > curve_date:
            from resolved_smile_inputs import resolve_conventions, MarketSnapshot
            context = resolve_conventions(MarketSnapshot.capture(market),
                                          requested_date.date(), curve_date.date())
            market.deltaBase, market.atm_conv = context.premium_adjusted, context.atm_convention
            for pillar_date, prepared in zip(rows.expiry, curve_prepared):
                # A pillar whose conversion failed carries interpolated
                # quotes; its smile is built from them, not reused.
                if (naive_date(pillar_date) == requested_date and market.s == spot and
                        curve_date == naive_date(valuation_date) and
                        not prepared.diagnostics.degraded):
                    reuse_prepared(market, prepared)
                    # Replayed when the market is reused (``_cached_market``).
                    market._prepared_reuse = prepared
                    break
        return market


class UnavailableMarket:
    """Stands in for a pair whose market could not be built.

    Every pricing request raises the construction error, so each line that
    needs this market reports why instead of the whole run failing.
    """

    def __init__(self, currency_pair, reason, valuation_date=None):
        self.currency_pair = str(currency_pair or "").replace("/", "").replace(" ", "").upper()
        self.reason = str(reason)
        self.valuation_date = valuation_date
        self.settings = PricingSettings()
        self.diagnostics = DiagnosticsCollector()

    def _fail(self, *args, **kwargs):
        raise MarketContextError(self.reason)

    market_for_expiry = resolve_market_expiry = variance_days_between = _fail
    forward_points_for_expiry = term_discount_for_expiry = forward_curve = _fail

    def __repr__(self):
        return f"UnavailableMarket({self.currency_pair}: {self.reason})"


def build_context(*args, **kwargs):
    """``MarketContext(...)``, or an ``UnavailableMarket`` carrying the reason."""
    try:
        return MarketContext(*args, **kwargs)
    except MarketContextError as exc:
        pair = kwargs.get("currency_pair", args[0] if args else "")
        valuation = kwargs.get("valuation_date", args[1] if len(args) > 1 else None)
        return UnavailableMarket(pair, exc, valuation)

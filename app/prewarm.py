"""Pre-warm after the daily fetch (performance step 2): the app opens warm.

Run after a new market is stored (``market_data.bloomberg_fetch`` starts it in
its own process), or by hand::

    python -m app.prewarm                 # the latest market, as the app opens it
    python -m app.prewarm --date 2026-10-23 --market-id 52

It reproduces the app's opening state (the latest usable snapshot and its FX
date, the default pricing settings, betas and By-pair columns) and fills the
two disk caches the app reads:

* every pair's calibrated pillar smiles (``app.calibration_cache``);
* every live trade's valuation (``apeiron.hyperion.cache``'s result cache),
  so the Portfolio's first pricing reads them instead of pricing.

Cache keys are the full inputs, so a pre-warm can only make the app faster,
never change a number: if the app's state differs (another date, betas,
columns, an edited market), its keys differ and it prices as before.

Fail-safes: it never raises (the fetch has stored the market whatever
happens here); a pair or trade that fails is reported and skipped; a lock
file keeps two pre-warms from running at once (a stale lock, older than
``LOCK_STALE``, is taken over); the self-check rebuilds the market set as
the app does and counts how many trades would be read from the cache.
"""
import argparse
import os
import time
from collections import namedtuple
from datetime import datetime, timedelta
from pathlib import Path

LOCK = Path(__file__).resolve().parents[1] / "data" / "local" / "prewarm.lock"
LOCK_STALE = timedelta(minutes=30)

Report = namedtuple("Report", "market_id valuation_date pairs calibrations trades cached "
                              "seconds problems")


def opening_date(today=None):
    """The valuation date the app opens on: the latest usable snapshot's FX
    date (``app.ui.market_picker.init_dates``, linked to the market date), or
    today's FX date when there is none."""
    import pandas as pd
    from apeiron.gaia.market_dates.rollover import fx_date, fx_today
    from app.store_markets import app_market_store
    from market_data.snapshot_filter import data_day, usable_snapshots
    day = pd.Timestamp(today or fx_today()).normalize()
    try:
        frame = app_market_store(day).snapshots().copy()
        zones = frame["timezone"] if "timezone" in frame.columns else [None] * len(frame)
        frame["fx_date"] = [str(fx_date(stamp, zone))
                            for stamp, zone in zip(frame["date_time"], zones)]
        usable = usable_snapshots(frame, day)
        if not usable.empty:
            return pd.Timestamp(data_day(usable.iloc[0])).date()
    except Exception:
        pass
    return day.date()


def app_settings():
    """The pricing settings, betas and extras the app prices the Portfolio with
    by default (the Model tab's defaults and the default By-pair columns)."""
    from app.market_context import PricingSettings
    from app.ui.model_block import DEFAULTS
    from app.ui.trades_block import DEFAULT_COLUMNS, chosen_extras
    from market_data import Modifications
    settings = PricingSettings(pricing_model=DEFAULTS["pricing_model"], atm_type="dn",
                               delta_base=True, smooth_pillar_interpolation=True,
                               delta_reporting="spot")
    mods = Modifications(beta_atm=DEFAULTS["beta_atm"], beta_rr=DEFAULTS["beta_rr"],
                         beta_bf=DEFAULTS["beta_bf"])
    return settings, mods, chosen_extras(DEFAULT_COLUMNS)


def _session(payloads, valuation_date, settings):
    """A market set built as the app builds one: its own surface cache (held)
    and market service."""
    from app.market_service import MarketService
    from app.market_set import MarketSet
    from app.smile_service import DiagnosticsCollector, SurfaceCache, bind_cache
    diagnostics = DiagnosticsCollector()
    surface_cache = SurfaceCache()
    bind_cache(diagnostics.token, surface_cache)
    service = MarketService(diagnostics)
    markets = MarketSet.from_payloads(payloads, valuation_date, settings,
                                      market_service=service, diagnostics=diagnostics)
    return markets, surface_cache, service


def _live_book(valuation_date, trade_store=None):
    from app.trades import service
    from app.trades.model import LIVE
    from app.trades.store import open_trade_store
    from app.ui.trades_block import trade_store_path
    store, problem = open_trade_store(trade_store or trade_store_path(), read_only=True)
    if store is None:
        raise RuntimeError(f"trade store unavailable: {problem}")
    live, _ = service.book_on(store.trades(status=LIVE), valuation_date)
    return live


def _take_lock(lock, now):
    try:
        if lock.exists() and now - datetime.fromtimestamp(lock.stat().st_mtime) < LOCK_STALE:
            return False
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text(f"{os.getpid()} {now.isoformat(timespec='seconds')}")
        return True
    except Exception:
        return True                       # no lock possible: run anyway


def prewarm(valuation_date=None, market_id=None, trade_store=None, result_cache=None,
            calibration_cache=None, lock=LOCK, now=datetime.now, out=print):
    """Fill the disk caches for the app's opening state; returns a ``Report``.
    Never raises."""
    started = time.perf_counter()
    problems = []
    if not _take_lock(lock, now()):
        out(f"pre-warm skipped: another is running ({lock})")
        return Report(None, valuation_date, 0, {}, 0, 0, 0.0, ["another pre-warm is running"])
    try:
        return _prewarm(valuation_date, market_id, trade_store, result_cache,
                        calibration_cache, started, problems, out)
    except Exception as exc:              # never fail the caller
        problems.append(f"pre-warm failed: {type(exc).__name__}: {exc}")
        out(problems[-1])
        return Report(market_id, valuation_date, 0, {}, 0, 0,
                      time.perf_counter() - started, problems)
    finally:
        try:
            lock.unlink()
        except Exception:
            pass


def _prewarm(valuation_date, market_id, trade_store, result_cache, calibration_cache,
             started, problems, out):
    from apeiron.hyperion.cache import pricing as cached_pricing
    from apeiron.hyperion.cache.keys import market_fingerprint
    from apeiron.hyperion.cache.store import default_cache
    from app import pricer
    from app.calibration_plan import calibrate_pairs
    from app.store_markets import load_store_markets

    valuation_date = valuation_date or opening_date()
    loaded = load_store_markets(valuation_date, market_id=market_id)
    if loaded.snapshot is None or not loaded.payloads:
        raise RuntimeError(f"no stored market for {valuation_date}: {loaded.problem}")
    market_id = int(loaded.snapshot.market_id)
    settings, mods, extras = app_settings()
    markets, surface_cache, service = _session(loaded.payloads, valuation_date, settings)
    pairs = markets.available()
    out(f"pre-warm: market {market_id}, valuation {valuation_date}, {len(pairs)} pairs")

    fingerprints = {pair: market_fingerprint(markets.market(pair)) for pair in pairs}
    outcomes = calibrate_pairs(markets, pairs, fingerprints, {}, settings, valuation_date,
                               surface_cache, service,
                               origins={pair: market_id for pair in pairs},
                               payloads=loaded.payloads, cache=calibration_cache)
    calibrations = {}
    for outcome in outcomes:
        calibrations[outcome.source] = calibrations.get(outcome.source, 0) + 1
        problems += [note for note in outcome.notes if outcome.source == "failed"]
    out(f"  calibrated: {calibrations}")

    trades = []
    try:
        trades = _live_book(valuation_date, trade_store)
    except Exception as exc:
        problems.append(f"book not priced: {exc}")
    cache = result_cache or default_cache()
    if trades:
        _, _, errors = cached_pricing.price_trades(markets, trades, mods, extras=extras,
                                                   spot_path=pricer.SMILE_PATH, cache=cache)
        problems += errors
        out(f"  priced {len(trades)} live trades into the result cache"
            + (f" ({len(errors)} errors)" if errors else ""))

    # Self-check: a market set built afresh, as the app's next session builds
    # it, must find the trades in the cache.
    fresh, _, _ = _session(load_store_markets(valuation_date, market_id=market_id).payloads,
                           valuation_date, settings)
    keys = cached_pricing._keys(fresh, trades, mods, extras, pricer.SMILE_PATH)
    cached = sum(cache.get(keys[t.trade_id]) is not None for t in trades)
    if trades and cached < len(trades):
        problems.append(f"only {cached} of {len(trades)} trades found in the cache by a "
                        "fresh session")
    seconds = time.perf_counter() - started
    out(f"  self-check: {cached} of {len(trades)} trades read from the cache; {seconds:.1f} s")
    return Report(market_id, valuation_date, len(pairs), calibrations, len(trades), cached,
                  seconds, problems)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--date", help="valuation date YYYY-MM-DD; default the app's opening date")
    parser.add_argument("--market-id", type=int, help="a stored market id; default the latest")
    args = parser.parse_args(argv)
    import warnings
    warnings.filterwarnings("ignore")
    report = prewarm(args.date and datetime.strptime(args.date, "%Y-%m-%d").date(),
                     args.market_id)
    for problem in report.problems[:20]:
        print("  problem:", problem)
    return 0 if report.market_id is not None and not any(
        p.startswith("pre-warm failed") for p in report.problems) else 1


if __name__ == "__main__":
    raise SystemExit(main())

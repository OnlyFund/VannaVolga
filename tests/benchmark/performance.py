"""Performance benchmark: the app's slow paths, timed the way the app runs them.

Item 14, step 0 ("measure first"). Each workload runs in its own Python
process, so every one starts cold, as the app does when it opens:

* ``book``: the frozen book priced on market 52 through the Portfolio's path
  (``cached_pricing.price_trades``, the default By-pair columns' extras,
  theta on the rolled market). Phases: imports, market load, the book cold,
  the book again with the surfaces prepared but no result cache (pricing
  alone), a cached rerun, and one extra line on the busiest pair;
* ``book_theta_forward``: the book cold with theta on today's forward vols
  (``theta_vol_roll="forward"``) instead of the rolled market: the difference
  with ``book`` is what the rolled theta costs;
* ``roll``: every pair of market 52 rolled to the next business day;
* ``ladder``: the spot ladder (11 spots, the app's default) of the busiest
  pair's options and hedges, in the app's default "optimized" mode.

The book is frozen once from the trade store into ``data/local/benchmark/
book.json`` (personal data, not committed), so later runs time the same book;
``--freeze`` refreezes it. Results are appended to ``performance_history.csv``
(next to this file) with the commit and Python version, so a change is
compared against the runs before it.

Run from the project root (market 52 must be in the local market store):

    python tests/benchmark/performance.py                  # every workload
    python tests/benchmark/performance.py book roll        # some of them
    python tests/benchmark/performance.py --profile book   # cProfile one workload
"""
import argparse
import csv
import dataclasses
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import warnings
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MARKET_ID, FX_DATE, NEXT_DAY = 52, "2026-10-23", "2026-10-26"
LOCAL = ROOT / "data" / "local" / "benchmark"
BOOK = LOCAL / "book.json"
HISTORY = Path(__file__).resolve().parent / "performance_history.csv"
WORKLOADS = ("book", "book_theta_forward", "roll", "ladder")
LADDER_STEPS, LADDER_INCREMENT = 5, 0.5          # the Risk tab's defaults


class Timer:
    """Named phases, in order, in seconds."""

    def __init__(self):
        self.phases = []

    def __call__(self, name):
        timer = self

        class _Phase:
            def __enter__(self):
                self.start = time.perf_counter()

            def __exit__(self, *exc):
                timer.phases.append((name, time.perf_counter() - self.start))
        return _Phase()


def freeze_book(path=BOOK):
    """Export the trade store's book to ``path`` once, so every run times it."""
    from app.trades import service
    from app.ui.trades_block import trade_store_path
    from app.trades.store import open_trade_store
    store, problem = open_trade_store(trade_store_path(), read_only=True)
    if store is None:
        raise RuntimeError(f"trade store unavailable: {problem}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(service.export_book(store, include_cancelled=False)),
                    encoding="utf-8")
    return path


def book_trades(valuation_date=FX_DATE, path=BOOK):
    """The frozen book's live trades on ``valuation_date``, from an in-memory store."""
    from app.trades import service
    from app.trades.model import LIVE
    from app.trades.store import MEMORY, open_trade_store
    store, problem = open_trade_store(MEMORY)
    if store is None:
        raise RuntimeError(f"in-memory trade store unavailable: {problem}")
    service.import_book(store, path.read_text(encoding="utf-8"), user="benchmark")
    live, _ = service.book_on(store.trades(status=LIVE), date.fromisoformat(valuation_date))
    return live


def _market_set(valuation_date, theta_vol_roll="rolled"):
    """Market 52 as the app holds it: its own surface cache and market service."""
    from app.market_context import PricingSettings
    from app.market_service import MarketService
    from app.market_set import MarketSet
    from app.smile_service import DiagnosticsCollector, SurfaceCache, bind_cache
    from app.store_markets import load_store_markets
    diagnostics = DiagnosticsCollector()
    bind_cache(diagnostics.token, SurfaceCache())
    service = MarketService(diagnostics)
    settings = PricingSettings(theta_vol_roll=theta_vol_roll)
    loaded = load_store_markets(FX_DATE, market_id=MARKET_ID)
    # A date, as the sidebar's date input gives the app.
    return MarketSet.from_payloads(loaded.payloads, date.fromisoformat(valuation_date), settings,
                                   market_service=service, diagnostics=diagnostics)


def _mods():
    from market_data import Modifications
    return Modifications(beta_atm=0.7, beta_rr=0.5, beta_bf=0.0)


def _busiest_pair(trades):
    counts = {}
    for trade in trades:
        counts[trade.currency_pair] = counts.get(trade.currency_pair, 0) + 1
    return max(sorted(counts), key=counts.get)


def _price_book(markets, trades, cache):
    from apeiron.hyperion.cache import pricing as cached_pricing
    from app.ui.trades_block import DEFAULT_COLUMNS, chosen_extras
    return cached_pricing.price_trades(markets, trades, _mods(),
                                       extras=chosen_extras(DEFAULT_COLUMNS), cache=cache)


def _fresh_cache():
    from apeiron.hyperion.cache.store import ResultCache
    return ResultCache(path=Path(tempfile.gettempdir()) / "vv_benchmark_unused.sqlite",
                       disk=False)


def run_book(timer, theta_vol_roll="rolled", full=True):
    with timer("imports"):
        import app.pricer  # noqa: F401
        from apeiron.hyperion.cache import pricing  # noqa: F401
        import app.ui.trades_block  # noqa: F401
    with timer("market load"):
        markets = _market_set(FX_DATE, theta_vol_roll)
        trades = book_trades()
    with timer("book cold"):
        results, forwards, errors = _price_book(markets, trades, _fresh_cache())
    counts = dict(trades=len(trades), options=len(results), forwards=len(forwards),
                  errors=len(errors), pairs=len({t.currency_pair for t in trades}))
    if not full:
        return counts
    with timer("book prepared, no result cache"):
        _price_book(markets, trades, _fresh_cache())
    cache = _fresh_cache()
    _price_book(markets, trades, cache)
    with timer("book cached rerun"):
        _price_book(markets, trades, cache)
    pair = _busiest_pair(trades)
    option = next(t for t in trades if t.currency_pair == pair and t.is_option)
    extra = dataclasses.replace(option, trade_id="BENCH-EXTRA",
                                notional=float(option.notional) * 1.5)
    with timer("one extra line"):
        _price_book(markets, [*trades, extra], cache)
    counts["busiest_pair"] = pair
    return counts


def run_roll(timer):
    with timer("imports"):
        import app.rolled_market  # noqa: F401
        import app.market_set  # noqa: F401
    with timer("market load"):
        markets = _market_set(NEXT_DAY)
    pairs = markets.available()
    with timer("roll every pair"):
        for pair in pairs:
            markets.market(pair)
    rolled = sum("rolled to" in note for note in markets.notes().values()) \
        if isinstance(markets.notes(), dict) else None
    return dict(pairs=len(pairs), rolled=rolled)


def run_ladder(timer):
    with timer("imports"):
        import pandas as pd
        from app.reports.spot_ladder import calculate_spot_ladder
        from app.ui.risk_block import ladder_rows, risk_options
    with timer("market load"):
        markets = _market_set(FX_DATE)
        trades = book_trades()
    pair = _busiest_pair(trades)
    options, _ = risk_options(trades, pair)
    hedges = [t for t in trades if t.currency_pair == pair and t.is_forward]
    context = markets.market(pair)
    spot = float(context.spot)
    spots = [spot * (1.0 + step * LADDER_INCREMENT / 100.0)
             for step in range(-LADDER_STEPS, LADDER_STEPS + 1)]
    rows = pd.DataFrame(ladder_rows(options))
    positions = [{"currency_pair": h.currency_pair, "expiry": h.expiry,
                  "base_amount": h.signed_base_amount(), "contract_rate": h.contract_rate,
                  "value_date": h.value_date} for h in hedges]
    settings = context.settings
    with timer("spot ladder"):
        _, errors = calculate_spot_ladder(
            list(rows.iterrows()), spots, markets.valuation_date, context.market_for_expiry,
            context.resolve_market_expiry, settings.pricing_model, settings.atm_type,
            settings.delta_base, _mods(), mode="optimized", forward_positions=positions,
            market_for_delivery=getattr(context, "market_for_delivery", None))
    return dict(pair=pair, options=len(options), hedges=len(hedges), spots=len(spots),
                errors=len(errors))


def run(workload):
    """``(phases, counts)`` of one workload, in this process."""
    timer = Timer()
    if workload == "book":
        counts = run_book(timer)
    elif workload == "book_theta_forward":
        counts = run_book(timer, theta_vol_roll="forward", full=False)
    elif workload == "roll":
        counts = run_roll(timer)
    elif workload == "ladder":
        counts = run_ladder(timer)
    else:
        raise ValueError(f"unknown workload {workload!r}; one of {', '.join(WORKLOADS)}")
    return timer.phases, counts


def _child(workload):
    warnings.filterwarnings("ignore")
    import logging
    logging.disable(logging.CRITICAL)
    os.environ.setdefault("VV_LIVE_DATES", "0")    # time the stored market, not today's fetch
    phases, counts = run(workload)
    print("RESULT " + json.dumps(dict(phases=phases, counts=counts)))


def _in_process(workload, profile=None):
    """Run one workload in a fresh process; ``(phases, counts)``."""
    command = [sys.executable, str(Path(__file__).resolve()), "--child", workload]
    if profile:
        command = [sys.executable, "-m", "cProfile", "-o", str(profile),
                   str(Path(__file__).resolve()), "--child", workload]
    done = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True)
    lines = [line for line in done.stdout.splitlines() if line.startswith("RESULT ")]
    if done.returncode or not lines:
        raise RuntimeError(f"{workload} failed:\n{done.stderr[-3000:]}")
    data = json.loads(lines[-1][len("RESULT "):])
    return [tuple(phase) for phase in data["phases"]], data["counts"]


def _commit():
    """The short commit, with ``+dirty`` when tracked code has uncommitted changes
    (the history file itself aside), so a run is never filed under the wrong code."""
    try:
        def git(*args):
            return subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True,
                                  text=True).stdout
        changed = [line for line in git("status", "--porcelain", "--untracked-files=no").splitlines()
                   if not line.endswith(HISTORY.name)]
        return git("rev-parse", "--short", "HEAD").strip() + ("+dirty" if changed else "")
    except Exception:
        return ""


def record(rows, path=HISTORY):
    """Append ``rows`` (dicts) to the history file."""
    columns = ["run_at", "commit", "python", "machine", "workload", "phase", "seconds"]
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, columns)
        if new:
            writer.writeheader()
        writer.writerows(rows)


def print_profile(path, out=print, top=30):
    import io
    import pstats
    for order in ("cumulative", "tottime"):
        buffer = io.StringIO()
        pstats.Stats(str(path), stream=buffer).sort_stats(order).print_stats(top)
        out(f"--- top {top} by {order} ---")
        out(buffer.getvalue().split("\n", 6)[-1] if order else buffer.getvalue())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("workloads", nargs="*", default=list(WORKLOADS))
    parser.add_argument("--child", help=argparse.SUPPRESS)
    parser.add_argument("--freeze", action="store_true", help="refreeze the book from the trade store")
    parser.add_argument("--profile", metavar="WORKLOAD", help="cProfile one workload")
    parser.add_argument("--no-record", action="store_true", help="do not append to the history")
    args = parser.parse_args(argv)
    if args.child:
        _child(args.child)
        return 0
    if args.freeze or not BOOK.exists():
        print(f"book frozen to {freeze_book()}")
    if args.profile:
        LOCAL.mkdir(parents=True, exist_ok=True)
        profile = LOCAL / f"{args.profile}.prof"
        phases, counts = _in_process(args.profile, profile)
        for name, seconds in phases:
            print(f"  {name:32s} {seconds:8.2f} s")
        print_profile(profile)
        print(f"profile saved to {profile} (snakeviz or pstats to explore)")
        return 0
    stamp = datetime.now().isoformat(timespec="seconds")
    common = dict(run_at=stamp, commit=_commit(), python=platform.python_version(),
                  machine=platform.node())
    rows = []
    for workload in args.workloads:
        started = time.perf_counter()
        phases, counts = _in_process(workload)
        total = time.perf_counter() - started
        print(f"{workload}  ({', '.join(f'{k} {v}' for k, v in counts.items())})")
        for name, seconds in [*phases, ("process total", total)]:
            print(f"  {name:32s} {seconds:8.2f} s")
            rows.append(dict(common, workload=workload, phase=name, seconds=f"{seconds:.3f}"))
    if not args.no_record:
        record(rows)
        print(f"appended to {HISTORY.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

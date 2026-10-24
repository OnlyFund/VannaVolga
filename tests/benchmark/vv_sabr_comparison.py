"""Compare VV and standalone SABR vols on the fixed Bloomberg option sample."""
import argparse
import csv
from datetime import date
import importlib.util
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BENCHMARK_DIR = Path(__file__).resolve().parent
FILE = BENCHMARK_DIR / "vv_sabr_comparison.csv"
SOURCE_FILE = BENCHMARK_DIR / "bloomberg_vols.py"
MARKET_ID = 55
BBG_COLUMN = "bbg vols"
EDGE_CASES = {
    "B08": ("EURUSD", "PUT", "5d", "6M"),
    "B09": ("EURUSD", "CALL", "5d", "6M"),
    "B10": ("EURUSD", "PUT", "5d", "1Y"),
    "B11": ("EURUSD", "CALL", "5d", "1Y"),
    "B12": ("USDJPY", "PUT", "5d", "6M"),
    "B13": ("USDJPY", "CALL", "5d", "6M"),
    "B14": ("USDJPY", "PUT", "5d", "1Y"),
    "B15": ("USDJPY", "CALL", "5d", "1Y"),
    "B16": ("GBPUSD", "PUT", "5d", "6M"),
    "B17": ("GBPUSD", "CALL", "5d", "6M"),
    "B18": ("GBPUSD", "PUT", "5d", "1Y"),
    "B19": ("GBPUSD", "CALL", "5d", "1Y"),
}
OUTPUT_COLUMNS = (
    "market_id", "market_datetime", "valuation_date", "case_id", "pair",
    "call_put", "buy_sell", "notional", "strike_input", "expiry_input",
    "strike", "expiry", "delivery", "forward", "spot", "vv_vol_pct",
    "sabr_vol_pct", "sabr_minus_vv_vol_pts", "vv_status", "sabr_status",
    BBG_COLUMN,
)


def _benchmark_module():
    spec = importlib.util.spec_from_file_location("bloomberg_vols", SOURCE_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the Bloomberg benchmark helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _model_vol(market, line, model):
    from helper_functions.vanna_volga_helpers import get_model_map

    options = dict(
        market=market,
        K=float(line["strike"]),
        expiry_date=str(line["expiry"])[:10],
        valuation_date=str(line["valuation"])[:10],
        CallPut=line["call_put"],
        BuySell=line["buy_sell"],
        Notional=float(line["notional"]),
        atm_conv=market.atm_conv,
        deltaBase=market.deltaBase,
    )
    if model == "sabr":
        options["beta"] = 0.5
    instance = get_model_map(market)[model](**options)
    vol = float(instance.get_vol_for_strike())
    resolver = getattr(instance, "resolver", None)
    diagnostics = getattr(resolver, "diagnostics", None)
    status = getattr(diagnostics, "status", "resolved")
    reason = getattr(diagnostics, "reason", None)
    if reason:
        status += ": " + str(reason)
    if not math.isfinite(vol) or vol <= 0:
        raise ValueError("{} returned invalid volatility {}".format(model, vol))
    return vol, status


def generate(path=FILE, seed=None, count=None):
    """Price the seeded option sample under both models."""
    benchmark = _benchmark_module()
    seed = benchmark.SEED if seed is None else seed
    count = benchmark.COUNT if count is None else count
    existing = {}
    if path.exists():
        with path.open(newline="", encoding="utf-8-sig") as handle:
            existing = {
                row["case_id"]: row
                for row in csv.DictReader(handle)
                if row.get("case_id")
            }

    from app.market_set import MarketSet
    from app.store_markets import load_store_markets

    loaded = load_store_markets(benchmark.FX_DATE, market_id=MARKET_ID)
    if loaded.problem or loaded.snapshot is None or loaded.snapshot.market_id != MARKET_ID:
        raise RuntimeError("market {} cannot be loaded: {}".format(
            MARKET_ID, loaded.problem))
    valuation_date = date.fromisoformat(loaded.snapshot.fx_date)
    markets = MarketSet.from_payloads(loaded.payloads, valuation_date)

    output = []
    failures = []
    trades = benchmark.generate(seed=seed, count=count, today=valuation_date)
    for trade in trades:
        edge_case = EDGE_CASES.get(trade["case_id"])
        if edge_case:
            trade["pair"], trade["call_put"], trade["strike_input"], trade["expiry_input"] = edge_case
            trade["buy_sell"] = "BUY"

    for trade in trades:
        row = dict(
            market_id=MARKET_ID,
            market_datetime=loaded.snapshot.date_time,
            valuation_date=valuation_date.isoformat(),
            **trade
        )
        try:
            line = benchmark._line(trade, markets)
            line["valuation"] = valuation_date.isoformat()
            pair_context = markets.market(trade["pair"])
            market = pair_context.market_for_expiry(line["expiry"])
            vv, vv_status = _model_vol(market, line, "vv")
            sabr, sabr_status = _model_vol(market, line, "sabr")
            strike = float(line["strike"])
            expiry = str(line["expiry"])[:10]
            forward = market.s + market.fwd_pts / market.forward_divisor
            row.update(
                strike="{:.8f}".format(strike),
                expiry=expiry,
                delivery=str(pair_context.delivery(line["expiry"]))[:10],
                forward="{:.8f}".format(forward),
                spot="{:.8f}".format(market.s),
                vv_vol_pct="{:.6f}".format(vv),
                sabr_vol_pct="{:.6f}".format(sabr),
                sabr_minus_vv_vol_pts="{:+.6f}".format(sabr - vv),
                vv_status=vv_status,
                sabr_status=sabr_status,
            )
        except (KeyError, TypeError, ValueError, RuntimeError, ArithmeticError, OverflowError) as exc:
            message = "failed: {}".format(exc)
            row.update(strike="", expiry="", delivery="", forward="", spot="",
                       vv_vol_pct="", sabr_vol_pct="", sabr_minus_vv_vol_pts="",
                       vv_status=message, sabr_status=message)
            failures.append(trade["case_id"])

        previous = existing.get(trade["case_id"], {})
        same_option = (
            str(previous.get("pair", "")).upper() == trade["pair"]
            and str(previous.get("call_put", "")).upper() == trade["call_put"]
            and str(previous.get("strike_input", "")).lower() == trade["strike_input"].lower()
            and benchmark._iso(previous.get("expiry_input", "")) == trade["expiry_input"]
        )
        row[BBG_COLUMN] = previous.get(BBG_COLUMN, "") if same_option else ""
        output.append(row)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows({column: row.get(column, "") for column in OUTPUT_COLUMNS}
                         for row in output)
    if failures:
        raise RuntimeError("wrote {} rows to {}; failed cases: {}".format(
            len(output), path, ", ".join(failures)))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=FILE)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--count", type=int, default=None)
    args = parser.parse_args(argv)
    rows = generate(args.output, args.seed, args.count)
    print("wrote {} VV/SABR comparisons to {}".format(len(rows), args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""30 random trades priced on stored market 52: a benchmark against Bloomberg.

Market 52 is the Bloomberg snapshot of Fri 23 Oct 2026, 12:53:32 London
(07:53 New York), FX date 23 Oct. Price the same trades on Bloomberg's option
pricing page (OVML) as of that snapshot, so both sides use the same market
data, and fill in the ``bbg_`` columns of ``market52_trades.csv``.

The trades are drawn with a fixed seed (``SEED``), so they are the same every
run: a pair mix of majors, crosses and EM; pillar tenors and broken dates; ATM,
delta and fixed strikes; calls and puts. Each is priced as the app prices it:
model ``vv``, betas 0.7 / 0.5 / 0.0 (the Model tab's defaults; they move the
smile Greeks, not the vol), each pair's own ATM, delta and premium conventions.

Run from the project root (a Bloomberg terminal is not needed; market 52 must
be in the local market store):

    python tests/benchmark/bloomberg_vols.py            # (re)price; keeps bbg_ columns
    python tests/benchmark/bloomberg_vols.py --compare  # our numbers against Bloomberg's

Columns: the trade as entered (``strike_input``, ``expiry_input``) and as
resolved (``strike``, ``expiry``, ``delivery``); our ``vol``, ``premium_pct``
(% of the base notional), ``premium_term`` (term currency per unit of base),
``spot_delta_pct`` and ``fwd_delta_pct`` (% of the base notional, the pair's
premium adjustment included), ``forward`` and ``spot``; premium and deltas
are signed by ``buy_sell``, as the app shows them. The ``bbg_`` columns take
Bloomberg's vol, premium (% of base) and spot delta (%) for one BOUGHT option,
whatever ``buy_sell`` says: ``--compare`` sets our numbers on the same footing.
``checked_on`` and ``note`` record the check. Excel may rewrite the dates when the file is saved:
keep them as YYYY-MM-DD.
"""
import argparse
import csv
import random
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MARKET_ID = 52
FX_DATE = date(2026, 10, 23)
SEED = 52
COUNT = 30
FILE = Path(__file__).resolve().parent / "market52_trades.csv"

MAJORS = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"]
CROSSES = ["EURJPY", "EURGBP", "GBPJPY", "AUDJPY", "EURCHF", "EURAUD", "CADJPY", "EURNOK"]
EM = ["USDMXN", "USDZAR", "USDTRY", "USDCNH", "EURPLN", "USDHUF", "USDSGD", "USDILS", "USDCZK"]
TENORS = ["1W", "2W", "1M", "2M", "3M", "6M", "9M", "1Y"]
DELTAS = ["10d", "15d", "25d", "35d"]

OURS = ("strike", "expiry", "delivery", "vol", "premium_pct", "premium_term", "spot_delta_pct",
        "fwd_delta_pct", "forward", "spot", "smile_status")
BBG = ("bbg_vol", "bbg_premium_pct", "bbg_delta_pct", "checked_on", "note")
COLUMNS = ("case_id", "pair", "call_put", "buy_sell", "notional", "strike_input", "expiry_input") \
    + OURS + BBG


def generate(seed=SEED, count=COUNT, today=FX_DATE):
    """``count`` trades, the same for the same seed."""
    rng = random.Random(seed)
    trades = []
    for number in range(1, count + 1):
        group = rng.choices([MAJORS, CROSSES, EM], weights=[0.45, 0.25, 0.30])[0]
        pair = rng.choice(group)
        if rng.random() < 0.55:
            expiry = rng.choice(TENORS)
        else:                                   # a broken date, a weekday 3 days to 1 year out
            day = today + timedelta(days=rng.randint(3, 365))
            while day.weekday() >= 5:
                day += timedelta(days=1)
            expiry = day.isoformat()
        kind = rng.random()
        if kind < 0.3:
            strike = "atm"
        elif kind < 0.75:
            strike = rng.choice(DELTAS)
        else:                                   # fixed: forward times (1 +/- up to 6%)
            strike = f"fwd{rng.uniform(-0.06, 0.06):+.4f}"
        trades.append({"case_id": f"B{number:02d}", "pair": pair,
                       "call_put": rng.choice(["CALL", "PUT"]), "buy_sell": rng.choice(["BUY", "SELL"]),
                       "notional": 1_000_000, "strike_input": strike, "expiry_input": expiry})
    return trades


def _line(trade, markets):
    """The pricer's line for ``trade``; a ``fwd+x`` strike becomes the forward times (1 + x)."""
    pair = trade["pair"]
    expiry = trade["expiry_input"]
    if len(expiry) == 10:                       # ISO date -> the app's mm/dd/yyyy
        expiry = f"{expiry[5:7]}/{expiry[8:10]}/{expiry[:4]}"
    strike = trade["strike_input"]
    if strike.startswith("fwd"):
        context = markets.market(pair)
        market = context.market_for_expiry(context.resolve_market_expiry(expiry)
                                           if len(expiry) != 10 else _as_date(expiry))
        forward = market.s + market.fwd_pts / market.forward_divisor
        strike = f"{forward * (1 + float(strike[3:])):.6g}"
    return dict(currency_pair=pair, strike=strike, expiry=expiry, call_put=trade["call_put"],
                buy_sell=trade["buy_sell"], notional=str(trade["notional"]), premium=0.0,
                premium_ccy=pair[3:])


def _as_date(text):
    month, day, year = text.split("/")
    return date(int(year), int(month), int(day))


def price(trades):
    """``trades`` with our columns filled in, priced on market ``MARKET_ID``."""
    warnings.filterwarnings("ignore")
    import logging
    logging.disable(logging.CRITICAL)
    from app import pricer
    from app.market_set import MarketSet
    from app.store_markets import load_store_markets
    from market_data import Modifications

    loaded = load_store_markets(FX_DATE, market_id=MARKET_ID)
    if loaded.problem or loaded.snapshot is None or loaded.snapshot.market_id != MARKET_ID:
        raise SystemExit(f"market {MARKET_ID} cannot be loaded: {loaded.problem}")
    markets = MarketSet.from_payloads(loaded.payloads, FX_DATE)
    mods = Modifications(beta_atm=0.7, beta_rr=0.5, beta_bf=0.0)
    priced = []
    for trade in trades:
        row = dict(trade)
        try:
            (result,), errors = pricer.price_lines(markets, [_line(trade, markets)], mods)
            if errors:
                raise ValueError(errors[0])
        except Exception as exc:
            row.update({column: "" for column in OURS}, smile_status=f"failed: {exc}")
            priced.append(row)
            continue
        notional = float(trade["notional"])
        context = markets.market(trade["pair"])
        expiry = result["expiry"]
        row.update(
            strike=f"{float(result['strike']):.6f}", expiry=str(expiry)[:10],
            delivery=str(context.delivery(expiry))[:10], vol=f"{result['vol_pct']:.4f}",
            premium_pct=f"{result['mtm_pct_base']:.4f}", premium_term=f"{result['mtm_term'] / notional:.6f}",
            spot_delta_pct=f"{result['spot_delta'] / notional * 100:.3f}",
            fwd_delta_pct=f"{result['fwd_delta'] / notional * 100:.3f}",
            forward=f"{result['_forward_rate']:.6f}", spot=f"{context.spot:.6f}",
            smile_status=result.get("smile_status", ""))
        priced.append(row)
    return priced


def read(path=FILE):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write(rows, path=FILE):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in COLUMNS})


def refresh(path=FILE):
    """Price the trades again; Bloomberg's answers already filled in are kept."""
    kept = {row["case_id"]: row for row in read(path)}
    rows = price(generate())
    for row in rows:
        for column in BBG:
            row[column] = kept.get(row["case_id"], {}).get(column, "")
    write(rows, path)
    return rows


def compare(path=FILE, out=print):
    """Our vol, premium and delta against Bloomberg's, where filled in."""
    rows = [row for row in read(path) if row.get("bbg_vol")]
    if not rows:
        out(f"no Bloomberg answers in {path} yet")
        return []
    diffs = []
    for row in rows:
        # Ours are signed by buy_sell; Bloomberg's are for one bought option.
        side = -1.0 if row["buy_sell"].upper() == "SELL" else 1.0
        vol = float(row["vol"]) - float(row["bbg_vol"])
        premium = (side * float(row["premium_pct"]) - float(row["bbg_premium_pct"])
                   if row.get("bbg_premium_pct") else None)
        delta = (side * float(row["spot_delta_pct"]) - float(row["bbg_delta_pct"])
                 if row.get("bbg_delta_pct") else None)
        diffs.append((row["case_id"], row["pair"], row["strike_input"], row["expiry_input"], vol, premium, delta))
        out(f"{row['case_id']} {row['pair']} {row['strike_input']:>9} {row['expiry_input']:>10}: "
            f"vol {vol:+.3f} pts" + (f", premium {premium:+.4f}% of base" if premium is not None else "")
            + (f", delta {delta:+.2f}%" if delta is not None else ""))
    vols = [abs(d[4]) for d in diffs]
    out(f"{len(diffs)} compared: vol |diff| mean {sum(vols) / len(vols):.3f} pts, max {max(vols):.3f} pts")
    return diffs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--compare", action="store_true", help="compare with the bbg_ columns")
    args = parser.parse_args(argv)
    if args.compare:
        compare()
        return 0
    rows = refresh()
    failed = [row["case_id"] for row in rows if row["smile_status"].startswith("failed")]
    print(f"priced {len(rows) - len(failed)} of {len(rows)} trades on market {MARKET_ID} into {FILE}"
          + (f"; failed: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

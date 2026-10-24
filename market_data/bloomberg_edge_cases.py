"""Confirm the tenor edge cases (tests/edge_cases/option_dates.csv) on Bloomberg.

For each ``to_confirm`` row without a market answer, Bloomberg says, as of the
trade date:

* the expiry: the vol surface's expiry for the tenor (``VOL_SURF_EXPIRY_OVR``,
  ``market_data.bloomberg_historical_expiry_dates``);
* the delivery: the pair's spot date as of that expiry (``<PAIR> Curncy``,
  ``SETTLE_DT`` with ``REFERENCE_DATE``). This matched all 77 deliveries read
  on the pricing page, holiday expiries included (checked 2026-10-22).

Every ``to_confirm`` row with a market answer then becomes ``confirmed`` when
the answer equals the solver's, else ``disputed`` with the difference in
``note``. Rows already ``confirmed`` or ``disputed`` are left alone. Answers
typed in by hand (``market_expiry`` filled, no ``source``) are pricing-page
reads.

Caveat: the vol surface's expiry does not move off holidays for day and week
tenors, where the pricing page sometimes does (O0008, O0012). A disputed
surface row is therefore a case to read on the pricing page, not a verdict.

Run on a machine with a Bloomberg terminal (``xbbg``):

    python -m market_data.bloomberg_edge_cases            # fetch, then write the file
    python -m market_data.bloomberg_edge_cases --dry-run  # report only
"""
import argparse
import sys
from collections import defaultdict
from datetime import date

from apeiron.gaia.market_dates import edge_cases

SURFACE_SOURCE = "Bloomberg vol surface expiry; spot date as of expiry"
PAGE_SOURCE = "Bloomberg option pricing page"


def _iso(value):
    text = "" if value is None else str(value)[:10]
    return "" if text in ("", "NaT", "nan", "None") else date.fromisoformat(text).isoformat()


def surface_expiries(wanted):
    """``{(pair, trade_date, tenor): expiry}``; one request per trade date and tenor."""
    from market_data.bloomberg_historical_expiry_dates import fetch_historical_expiry_dates
    groups = defaultdict(set)
    for pair, trade, tenor in wanted:
        groups[(trade, tenor)].add(pair)
    found = {}
    for (trade, tenor), pairs in sorted(groups.items()):
        try:
            frames = [fetch_historical_expiry_dates(trade, sorted(pairs), tenor)]
        except ValueError:      # one pair without an answer: ask pair by pair
            frames = []
            for pair in sorted(pairs):
                try:
                    frames.append(fetch_historical_expiry_dates(trade, pair, tenor))
                except ValueError:
                    pass
        for frame in frames:
            for row in frame.itertuples():
                found[(row.currency_pair, trade, tenor)] = row.expiry_date.isoformat()
    return found


def spot_dates(wanted):
    """``{(pair, day): spot date as of day}``; one request per day."""
    from market_data.xbbg_compat import patch_xbbg
    patch_xbbg()
    from xbbg.blp import bdp
    groups = defaultdict(set)
    for pair, day in wanted:
        groups[day].add(pair)
    found = {}
    for day, pairs in sorted(groups.items()):
        tickers = {f"{pair} Curncy": pair for pair in sorted(pairs)}
        frame = bdp(tickers=list(tickers), flds="SETTLE_DT", REFERENCE_DATE=day.replace("-", ""))
        for ticker, pair in tickers.items():
            if ticker in frame.index and "settle_dt" in frame.columns:
                spot = _iso(frame.at[ticker, "settle_dt"])
                if spot:
                    found[(pair, day)] = spot
    return found


def validate(rows, expiries=surface_expiries, spots=spot_dates, today=None):
    """Fill and judge ``rows`` in place; ``{"fetched", "confirmed", "disputed", "unanswered"}`` case ids."""
    today = (today or date.today()).isoformat()
    report = {"fetched": [], "confirmed": [], "disputed": [], "unanswered": []}
    open_rows = [row for row in rows if row["status"] == edge_cases.TO_CONFIRM]
    for row in open_rows:
        if row["market_expiry"] and not row["source"]:
            row["source"], row["checked_on"] = PAGE_SOURCE, row["checked_on"] or today
    blank = [row for row in open_rows if not row["market_expiry"]]
    found = expiries({(row["pair"], row["trade_date"], row["tenor"]) for row in blank}) if blank else {}
    wanted = {(row["pair"], found[key]) for row in blank
              for key in [(row["pair"], row["trade_date"], row["tenor"])] if key in found}
    delivered = spots(wanted) if wanted else {}
    for row in blank:
        expiry = found.get((row["pair"], row["trade_date"], row["tenor"]))
        delivery = delivered.get((row["pair"], expiry))
        if expiry and delivery:
            row.update(market_expiry=expiry, market_delivery=delivery,
                       source=SURFACE_SOURCE, checked_on=today)
            report["fetched"].append(row["case_id"])
    for row in open_rows:
        if not (row["market_expiry"] and row["market_delivery"]):
            report["unanswered"].append(row["case_id"])
            continue
        market = (row["market_expiry"], row["market_delivery"])
        if market == (row["expiry"], row["delivery"]):
            row["status"] = edge_cases.CONFIRMED
            report["confirmed"].append(row["case_id"])
        else:
            row["status"] = edge_cases.DISPUTED
            row["note"] = "; ".join(filter(None, [row["note"], (
                f"market {market[0]} -> {market[1]}, solver {row['expiry']} -> {row['delivery']}"
                + ("; read it on the pricing page" if row["source"] == SURFACE_SOURCE else ""))]))
            report["disputed"].append(row["case_id"])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="fetch and report, write nothing")
    args = parser.parse_args(argv)
    path = edge_cases.DIRECTORY / edge_cases.OPTION_FILE
    stamp = path.stat().st_mtime
    rows = edge_cases.read(edge_cases.OPTION_FILE)
    report = validate(rows)
    for name, ids in report.items():
        print(f"{name}: {len(ids)}" + (f"  {' '.join(ids)}" if name in ("disputed", "unanswered") else ""))
    if args.dry_run:
        return 0
    if path.stat().st_mtime != stamp:
        print(f"{path} changed while fetching; nothing written. Run again.")
        return 1
    edge_cases.write(edge_cases.OPTION_FILE, rows, edge_cases.OPTION_COLUMNS)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

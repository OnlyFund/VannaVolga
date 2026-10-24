"""Bloomberg's historical FX dates: the reference for the pillar date solver.

For each reference date (a past trade date) and pair, Bloomberg says, as of
that day:

* ``spot``: the spot date (``<PAIR> Curncy``, ``SETTLE_DT``);
* ``forward``: each swap tenor's settlement date and days to maturity (the
  forward point tickers of ``market_data.fwd_tickers``: TN, SN, 1W .. 1Y,
  ``SETTLE_DT`` and ``DAYS_TO_MTY``);
* ``vol``: each option tenor's delivery (the ATM vol tickers of
  ``market_data.vol_tickers``: ON .. 1Y, ``SETTLE_DT`` and ``DAYS_TO_MTY``).

All with the ``REFERENCE_DATE`` override; one request per reference date and
kind covers every pair.

Option EXPIRIES (``OPT_EXPIRE_DT``) are not fetched: Bloomberg returns only
today's, ignoring ``REFERENCE_DATE``, ``END_DATE_OVERRIDE`` and the other date
overrides (checked 2026-10-22), so they are no historical reference.

Bloomberg ignores an override on some fields and quietly answers for today.
At the end the script flags every ticker whose dates are the same on all
reference dates: an override that was not applied.

The reference dates are chosen for the edge cases the date rules must get
right (labels in ``REFERENCE_DATES``): month ends and the end-of-month rule,
holidays in one currency only, US holidays on crosses' intermediate day,
Christmas and New Year, Easter, 29 February, T+1 pairs, Fridays and
Wednesdays.

Run on a machine with a Bloomberg terminal (``xbbg``):

    python -m market_data.swap_dates                     # every pair and date
    python -m market_data.swap_dates --pairs EURUSD USDCAD --dates 2025-01-17
    python -m market_data.swap_dates --dry-run           # list the tickers only

It writes one CSV row per pair, reference date and ticker to
``data/local/bbg_dates/date_reference_<timestamp>.csv`` (or ``--out``),
appending after each reference date so an interrupted run keeps its rows.
"""
import argparse
import csv
import sys
from datetime import date, datetime
from pathlib import Path

from market_data.fwd_tickers import TENORS as FWD_TENORS, fwd_ticker
from market_data.vol_tickers import TENORS as VOL_TENORS, vol_ticker

PAIRS = (
    # USD pairs, T+2
    "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCHF", "USDSGD",
    # T+1
    "USDCAD", "USDTRY", "USDPHP",
    # crosses (the USD intermediate-day rule)
    "EURJPY", "EURGBP", "GBPJPY", "EURCHF", "AUDJPY", "EURPLN",
    # EM with their own holidays
    "USDMXN", "USDBRL", "USDKRW", "USDCNH", "USDZAR",
)

#: Reference (trade) date -> why it is a test case.
REFERENCE_DATES = {
    "2024-02-27": "spot Thu 29 Feb (leap day, last business day of Feb): EOM 1M -> 29 Mar = Good Friday",
    "2024-03-27": "spot would be Good Friday 29 Mar; Easter Monday 1 Apr",
    "2024-05-23": "spot would be Mon 27 May: US Memorial Day and UK bank holiday",
    "2024-07-03": "T+1 is 4 Jul (US holiday): crosses' intermediate day",
    "2024-11-26": "spot would be Thanksgiving Thu 28 Nov",
    "2024-11-27": "T+1 is Thanksgiving: crosses' intermediate day; month end",
    "2024-12-23": "spot would be 25 Dec; 26 Dec UK/EUR holiday",
    "2024-12-30": "spot would be 1 Jan; year end",
    "2025-01-17": "Fri; Mon 20 Jan MLK (US only): USD pairs' intermediate day",
    "2025-02-26": "spot Fri 28 Feb, last business day: EOM tenors",
    "2025-02-28": "Fri, month end; Brazil carnival Mon 3 / Tue 4 Mar",
    "2025-04-17": "Good Friday 18 Apr, Easter Monday 21 Apr",
    "2025-04-29": "spot would be 1 May (TARGET and many EM closed)",
    "2025-05-02": "Fri; Japan Golden Week Mon 5 / Tue 6 May",
    "2025-06-30": "Canada Day Tue 1 Jul: USDCAD (T+1) spot",
    "2025-08-22": "Fri; UK summer bank holiday Mon 25 Aug",
    "2025-09-12": "Fri; Japan Respect for the Aged Day Mon 15 Sep",
    "2025-09-30": "quarter end; China Golden Week from 1 Oct (CNH)",
    "2025-10-02": "Korea Chuseok / National Foundation Day 3 - 9 Oct (KRW)",
    "2025-10-10": "Fri; Mon 13 Oct US Columbus and Canada Thanksgiving",
    "2025-10-29": "spot Fri 31 Oct, last business day: EOM tenors",
    "2025-11-26": "Wed; T+1 Thanksgiving: crosses' intermediate day",
    "2025-12-24": "spot would be 26 Dec (UK/EUR closed)",
    "2025-12-31": "year end; 1 Jan",
    "2026-01-30": "Fri, month end: spot in February",
    "2026-02-12": "Thu; Mon 16 Feb Presidents' Day (US only)",
    "2026-04-02": "Good Friday 3 Apr, Easter Monday 6 Apr",
    "2026-07-01": "spot would be Fri 3 Jul (US Independence Day observed)",
    "2026-07-02": "T+1 is 3 Jul (US observed): intermediate day",
    "2026-08-28": "Fri, month end; UK bank holiday Mon 31 Aug",
    "2026-09-30": "Wed, quarter end",
    "2026-10-14": "plain Wednesday: spot jumps Fri -> Mon on the roll",
    "2026-10-15": "plain Thursday",
    "2026-10-21": "the reference book's market day (Wed)",
}

#: Forward tickers: the short swaps around spot, then the pillars.
FORWARD_TENORS = ("TN", "SN") + tuple(FWD_TENORS)
COLUMNS = ("reference_date", "label", "pair", "kind", "tenor", "ticker", "override",
           "settle_dt", "days_to_mty", "error")


def tickers(pair):
    """``[(kind, tenor, ticker)]`` for one pair."""
    rows = [("spot", "SPOT", f"{pair} Curncy")]
    rows += [("forward", tenor, fwd_ticker(pair, tenor)) for tenor in FORWARD_TENORS]
    rows += [("vol", tenor, vol_ticker(pair, "V", tenor)) for tenor in VOL_TENORS]
    return rows


def _text(value):
    """Bloomberg's answer as ISO text (dates) or plain text; blank if missing."""
    if value is None:
        return ""
    try:
        import pandas as pd
        if pd.isna(value):
            return ""
        if isinstance(value, (pd.Timestamp, datetime, date)):
            return pd.Timestamp(value).strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        pass
    return str(value)


def _fetch(bdp, tickers_, fields, override, reference):
    """``{ticker: {field: text}}`` from one request; ``{}`` and the error on failure."""
    try:
        frame = bdp(tickers=list(tickers_), flds=list(fields), **{override: reference})
    except Exception as error:  # a failed request is recorded, not fatal
        return {}, f"{type(error).__name__}: {error}"
    if frame is None or len(frame) == 0:
        return {}, ""
    columns = {str(column).lower(): column for column in frame.columns}
    answers = {}
    for ticker in frame.index:
        answers[str(ticker)] = {field: _text(frame.at[ticker, columns[field.lower()]])
                                for field in fields if field.lower() in columns}
    return answers, ""


def fetch_date(bdp, reference_date, pairs):
    """Every row for one reference date."""
    reference = reference_date.replace("-", "")
    label = REFERENCE_DATES.get(reference_date, "")
    plan = [(pair,) + row for pair in pairs for row in tickers(pair)]
    dated = [row for row in plan if row[1] in ("spot", "forward")]
    vols = [row for row in plan if row[1] == "vol"]
    requests = [
        (dated, ("SETTLE_DT", "DAYS_TO_MTY"), "REFERENCE_DATE"),
        (vols, ("SETTLE_DT", "DAYS_TO_MTY"), "REFERENCE_DATE"),
    ]
    rows = []
    for plan_rows, fields, override in requests:
        answers, error = _fetch(bdp, [row[3] for row in plan_rows], fields, override, reference)
        errors = {}
        if error:   # retry pair by pair, so one bad pair does not cost the others
            for pair in pairs:
                mine = [row[3] for row in plan_rows if row[0] == pair]
                more, errors[pair] = _fetch(bdp, mine, fields, override, reference)
                answers.update(more)
        for pair, kind, tenor, ticker in plan_rows:
            error = errors.get(pair, "")
            answer = answers.get(ticker, {})
            rows.append({
                "reference_date": reference_date, "label": label, "pair": pair, "kind": kind,
                "tenor": tenor, "ticker": ticker, "override": override,
                "settle_dt": answer.get("SETTLE_DT", ""),
                "days_to_mty": answer.get("DAYS_TO_MTY", ""),
                "error": error or ("" if answer else "no answer"),
            })
    return rows


def ignored_overrides(rows):
    """``[(ticker, override)]`` whose answers are the same on every reference date."""
    seen = {}
    for row in rows:
        value = row["settle_dt"]
        if value:
            seen.setdefault((row["ticker"], row["override"]), set()).add((row["reference_date"], value))
    flagged = []
    for key, answers in seen.items():
        dates = {reference for reference, _ in answers}
        if len(dates) > 1 and len({value for _, value in answers}) == 1:
            flagged.append(key)
    return sorted(flagged)


def run(pairs=PAIRS, dates=tuple(REFERENCE_DATES), out=None, bdp=None, log=print):
    """Fetch every pair on every reference date; write and return the rows."""
    if bdp is None:
        from market_data.xbbg_compat import patch_xbbg
        patch_xbbg()
        from xbbg.blp import bdp
    if out is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        out = Path(__file__).resolve().parents[1] / "data" / "local" / "bbg_dates" / f"date_reference_{stamp}.csv"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    every = []
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for number, reference_date in enumerate(dates, 1):
            rows = fetch_date(bdp, reference_date, pairs)
            writer.writerows(rows)
            handle.flush()
            every += rows
            missing = sum(1 for row in rows if row["error"])
            log(f"[{number}/{len(dates)}] {reference_date}: {len(rows)} rows, {missing} without an answer")
    flagged = ignored_overrides(every)
    log(f"wrote {len(every)} rows to {out}")
    if flagged:
        log(f"WARNING: {len(flagged)} tickers answer the same on every date (override ignored?), e.g.:")
        for ticker, override in flagged[:10]:
            log(f"   {ticker} with {override}")
    return every, out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pairs", nargs="+", default=list(PAIRS))
    parser.add_argument("--dates", nargs="+", default=list(REFERENCE_DATES),
                        help="reference dates, YYYY-MM-DD")
    parser.add_argument("--out", help="CSV path (default data/local/bbg_dates/...)")
    parser.add_argument("--dry-run", action="store_true", help="list the tickers, fetch nothing")
    args = parser.parse_args(argv)
    for reference_date in args.dates:
        datetime.strptime(reference_date, "%Y-%m-%d")      # refuse a malformed date up front
    if args.dry_run:
        for pair in args.pairs:
            print(pair, " ".join(ticker for _, _, ticker in tickers(pair)))
        print(f"{len(args.pairs)} pairs x {len(args.dates)} dates, 2 requests per date")
        return 0
    run(args.pairs, args.dates, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

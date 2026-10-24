"""Bloomberg's FX settlement holidays per currency, as a static tracked list.

The settlement calendars of the date rules (``apeiron.gaia.market_dates.calendars``)
come from ``market_data/calendars/settlement_holidays.csv``: one row per
currency, Bloomberg calendar code and non-settlement date. The list is static
and refreshed by hand when it nears its end or a government moves a holiday:

    python -m market_data.settlement_holidays            # every currency, 2005 to 5 years ahead
    python -m market_data.settlement_holidays --ccys CNH PHP --start 2005 --end 2031

For each currency the calendar code is ``CALENDAR_CODE`` on ``<CCY> Curncy``
(EUR ``TE``, USD ``FD``, JPY ``JN``, CNH ``C+`` ...) and its holidays are the
``CALENDAR_NON_SETTLEMENT_DATES`` bulk field with ``SETTLEMENT_CALENDAR_CODE``
set to that code. Weekends are not listed. A currency's coverage is the
span it was fetched for (``start``/``end`` columns); outside it the date rules
fall back to QuantLib.

Checked 2026-10-22 against Bloomberg's historical settlement dates
(``market_data.swap_dates``): 21 pairs on 34 edge-case dates, 0 differences.
"""
import argparse
import csv
import sys
from datetime import date
from pathlib import Path

PATH = Path(__file__).resolve().parent / "calendars" / "settlement_holidays.csv"
COLUMNS = ("ccy", "code", "date", "start", "end")
CURRENCIES = (
    "ARS", "AUD", "BRL", "CAD", "CHF", "CLP", "CNH", "COP", "CZK", "DKK", "EUR", "GBP",
    "HKD", "HUF", "IDR", "ILS", "INR", "JPY", "KRW", "MXN", "MYR", "NOK", "NZD", "PHP",
    "PLN", "SEK", "SGD", "THB", "TRY", "TWD", "USD", "VND", "ZAR",
)
#: Years per bulk request (long spans are split so no answer is truncated).
CHUNK_YEARS = 8
FIRST_YEAR = 2005
#: Holidays are fetched this many years past the current one: further out
#: they are projections that governments have not yet announced.
YEARS_AHEAD = 5


def default_end_year(today=None):
    return (today or date.today()).year + YEARS_AHEAD


def calendar_code(bdp, ccy):
    frame = bdp(f"{ccy} Curncy", "CALENDAR_CODE")
    if frame is None or len(frame) == 0 or not str(frame.iloc[0, 0]).strip():
        raise ValueError(f"no Bloomberg calendar code for {ccy}")
    return str(frame.iloc[0, 0]).strip()


def holidays(bds, ccy, code, start_year, end_year):
    """Sorted ISO non-settlement dates of ``code`` from 1 Jan start to 31 Dec end."""
    import pandas as pd
    found = set()
    for first in range(start_year, end_year + 1, CHUNK_YEARS):
        last = min(first + CHUNK_YEARS - 1, end_year)
        frame = bds(f"{ccy} Curncy", "CALENDAR_NON_SETTLEMENT_DATES",
                    SETTLEMENT_CALENDAR_CODE=code,
                    CALENDAR_START_DATE=f"{first}0101", CALENDAR_END_DATE=f"{last}1231")
        if frame is not None and len(frame):
            found |= {pd.Timestamp(value).date().isoformat() for value in frame.iloc[:, 0]}
    return sorted(found)


def fetch(ccys=CURRENCIES, start_year=FIRST_YEAR, end_year=None, bdp=None, bds=None, log=print):
    """``[row]`` for every currency; a currency that fails is reported and left out."""
    end_year = end_year or default_end_year()
    if bdp is None or bds is None:
        from market_data.xbbg_compat import patch_xbbg
        patch_xbbg()
        from xbbg import blp
        bdp, bds = bdp or blp.bdp, bds or blp.bds
    rows, failed = [], []
    for ccy in ccys:
        try:
            code = calendar_code(bdp, ccy)
            dates = holidays(bds, ccy, code, start_year, end_year)
        except Exception as error:   # one currency must not cost the others
            failed.append(ccy)
            log(f"{ccy}: FAILED {type(error).__name__}: {error}")
            continue
        if not dates:
            failed.append(ccy)
            log(f"{ccy} ({code}): no dates returned, left out")
            continue
        # Coverage ends with the last year Bloomberg actually lists.
        end = min(end_year, int(dates[-1][:4]))
        rows += [{"ccy": ccy, "code": code, "date": day, "start": f"{start_year}-01-01",
                  "end": f"{end}-12-31"} for day in dates if day <= f"{end}-12-31"]
        log(f"{ccy} ({code}): {len(dates)} dates, {start_year} - {end}")
    return rows, failed


def merge(existing, fresh):
    """``existing`` rows with each freshly fetched currency replaced whole."""
    replaced = {row["ccy"] for row in fresh}
    return sorted([row for row in existing if row["ccy"] not in replaced] + list(fresh),
                  key=lambda row: (row["ccy"], row["date"]))


def read(path=PATH):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write(rows, path=PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)          # all or nothing


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ccys", nargs="+", default=list(CURRENCIES))
    parser.add_argument("--start", type=int, default=FIRST_YEAR)
    parser.add_argument("--end", type=int, default=None, help="default: 5 years ahead")
    parser.add_argument("--out", default=str(PATH))
    args = parser.parse_args(argv)
    rows, failed = fetch(args.ccys, args.start, args.end)
    existing = read(args.out) if Path(args.out).exists() else []
    write(merge(existing, rows), args.out)
    print(f"wrote {args.out} ({date.today()}); failed: {failed or 'none'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

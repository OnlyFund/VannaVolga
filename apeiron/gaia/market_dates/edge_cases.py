"""Option date edge cases: proposed by the solver, confirmed against the market.

The practice (``tests/edge_cases/README.md``): every date rule is checked on
cases where conventions bite, and each case is confirmed against the market
(Bloomberg's option pricing page) before it is trusted. The cases live in CSV
files under ``tests/edge_cases/``; ``tests/test_date_edge_cases.py`` holds the
solver to every confirmed market answer, and to its own stored answer on every
row, so a rule change that moves a case fails until the case is re-checked.

This module proposes the cases:

* **tenor cases** (``option_dates.csv``): trade dates the pricing page can
  price (``TRADE_START`` to today), each tenor of ``TENORS``, tagged by why
  the date is an edge case (``CATEGORIES``), a few per pair and category;
* **typed expiries** (``typed_expiries.csv``): weekdays in the next year on
  which a currency of the pair is closed, and weekends next to them.

Rerun it to add cases; rows already in the files keep their market answers:

    python -m apeiron.gaia.market_dates.edge_cases
"""
import csv
import sys
from datetime import date, timedelta
from pathlib import Path

import QuantLib as ql

from apeiron.gaia.market_dates import legacy_dates, pillars
from apeiron.gaia.market_dates.calendars import CALENDARS, as_date

DIRECTORY = Path(__file__).resolve().parents[3] / "tests" / "edge_cases"
OPTION_FILE, TYPED_FILE = "option_dates.csv", "typed_expiries.csv"

PAIRS = ("EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCHF", "USDSGD",
         "USDCAD", "USDTRY", "USDPHP", "EURJPY", "EURGBP", "GBPJPY", "EURCHF",
         "AUDJPY", "EURPLN", "USDMXN", "USDBRL", "USDKRW", "USDCNH", "USDZAR",
         "USDILS", "USDINR")
TENORS = ("ON", "1W", "2W", "1M", "2M", "3M", "6M", "1Y")
TRADE_START = date(2024, 1, 1)
#: Why a case is an edge case, most telling first (a case's primary category).
CATEGORIES = (
    "closed_day_roll",          # a day/week expiry lands on a day the pair is closed: moved
    "expiry_on_holiday",        # the expiry is a holiday of one currency (still an expiry day)
    "usd_intermediate",         # a US-only holiday between expiry and delivery
    "holiday_in_spot_window",   # another holiday between expiry and delivery
    "year_end",                 # delivery 20 Dec - 10 Jan
    "month_end",                # a month tenor from a month-end spot (end-of-month rule)
    "rule_changed",             # the old rules (legacy_dates) gave other dates
)
PER_PAIR_AND_CATEGORY = 2
#: Pair groups for the first batch to confirm (``priority`` 1: one case per
#: group and category).
GROUPS = {
    "usd_majors": ("EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCHF"),
    "jpy": ("USDJPY", "EURJPY", "GBPJPY", "AUDJPY"),
    "crosses": ("EURGBP", "EURCHF", "EURPLN"),
    "t_plus_1": ("USDCAD", "USDTRY", "USDPHP"),
    "latam": ("USDMXN", "USDBRL"),
    "asia_em": ("USDCNH", "USDKRW", "USDSGD", "USDILS", "USDINR", "USDZAR"),
}
OPTION_COLUMNS = ("case_id", "priority", "category", "pair", "trade_date", "tenor", "expiry", "delivery",
                  "market_expiry", "market_delivery", "status", "source", "checked_on", "note")
TYPED_COLUMNS = ("case_id", "priority", "category", "pair", "typed_expiry", "expiry", "delivery",
                 "market_delivery", "status", "source", "checked_on", "note")
CONFIRMED, TO_CONFIRM, DISPUTED = "confirmed", "to_confirm", "disputed"


def _settlement(pair):
    from helper_functions.fx_conventions import settlement   # same level: pair conventions
    return settlement(pair)


def _open(ccy, day):
    return CALENDARS[ccy].isBusinessDay(ql.Date(day.day, day.month, day.year))


def _between(start, end):
    day = start + timedelta(days=1)
    while day < end:
        yield day
        day += timedelta(days=1)


def categories(pair, trade, tenor):
    """``(dates, [category ...])`` of one tenor case; no categories: not an edge case."""
    rules = _settlement(pair)
    trade = as_date(trade)
    dates = pillars.tenor_dates(rules, trade, tenor)
    currencies = {rules.base, rules.quote, "USD"}
    found = []
    count, unit = (1, "D") if tenor in pillars.OVERNIGHT else pillars._parse(tenor)
    if unit in ("D", "W"):
        raw = trade + timedelta(days=count * (7 if unit == "W" else 1))
        if raw.weekday() < 5 and not pillars.is_expiry_day(rules, raw):
            found.append("closed_day_roll")
    if any(not _open(ccy, dates.expiry) for ccy in currencies):
        found.append("expiry_on_holiday")
    gap = [day for day in _between(dates.expiry, dates.delivery) if day.weekday() < 5]
    if any(not _open("USD", day) and all(_open(c, day) for c in currencies - {"USD"}) for day in gap):
        found.append("usd_intermediate")
    if any(not _open(ccy, day) for day in gap for ccy in currencies):
        found.append("holiday_in_spot_window")
    if (dates.delivery.month, dates.delivery.day) >= (12, 20) or (dates.delivery.month, dates.delivery.day) <= (1, 10):
        found.append("year_end")
    if unit in ("M", "Y"):
        next_good = pillars.next_business_day(rules, dates.spot)
        if next_good.month != dates.spot.month:
            found.append("month_end")
    old = legacy_dates.tenor_dates(rules, trade, tenor)
    if (old.expiry, old.delivery) != (dates.expiry, dates.delivery):
        found.append("rule_changed")
    return dates, sorted(found, key=CATEGORIES.index)


def tenor_cases(pairs=PAIRS, start=TRADE_START, end=None, tenors=TENORS, per=PER_PAIR_AND_CATEGORY):
    """Proposed tenor cases: the first ``per`` of each pair and primary category."""
    end = as_date(end or date.today())
    taken, rows = {}, []
    for pair in pairs:
        rules = _settlement(pair)
        day = as_date(start)
        while day <= end:
            if pillars.is_business_day(rules, day):
                for tenor in tenors:
                    dates, found = categories(pair, day, tenor)
                    if not found or taken.get((pair, found[0]), 0) >= per:
                        continue
                    taken[(pair, found[0])] = taken.get((pair, found[0]), 0) + 1
                    rows.append({"category": "+".join(found), "pair": pair, "trade_date": str(day),
                                 "tenor": tenor, "expiry": str(dates.expiry),
                                 "delivery": str(dates.delivery), "status": TO_CONFIRM})
            day += timedelta(days=1)
    return rows


def typed_cases(pairs=PAIRS, start=None, days=365, per=6):
    """Proposed typed expiries: closed weekdays of a pair's currency, and weekends next to them."""
    start = as_date(start or date.today())
    rows = []
    for pair in pairs:
        rules = _settlement(pair)
        currencies = (rules.base, rules.quote) + (() if "USD" in (rules.base, rules.quote) else ("USD",))
        picked = []
        for offset in range(days):
            day = start + timedelta(days=offset)
            if day.weekday() >= 5:
                continue
            closed = [ccy for ccy in currencies if not _open(ccy, day)]
            if closed:
                kind = "both_closed" if not pillars.is_expiry_day(rules, day) else "one_closed"
                picked.append((day, f"{kind}:{'/'.join(closed)}"))
                weekend = day + timedelta(days=(5 - day.weekday()) % 7 or 7)
                if (weekend - day).days <= 3:
                    picked.append((weekend, "weekend_after_holiday"))
        for day, category in picked[:per]:
            rows.append({"category": category, "pair": pair, "typed_expiry": str(day),
                         "expiry": str(pillars.typed_expiry(day)),
                         "delivery": str(pillars.delivery_for_expiry(rules, day)), "status": TO_CONFIRM})
    return rows


def read(name, directory=DIRECTORY):
    path = Path(directory) / name
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def merge(existing, proposed, key_fields, prefix):
    """``existing`` rows kept whole (with their market answers), new proposals appended."""
    keys = {tuple(row[k] for k in key_fields) for row in existing}
    rows = list(existing)
    number = max([int(row["case_id"][len(prefix):]) for row in existing if row.get("case_id", "").startswith(prefix)] or [0])
    for row in proposed:
        if tuple(row[k] for k in key_fields) in keys:
            continue
        number += 1
        rows.append(dict(row, case_id=f"{prefix}{number:04d}"))
        keys.add(tuple(row[k] for k in key_fields))
    return rows


def _group(pair):
    return next((name for name, pairs in GROUPS.items() if pair in pairs), "other")


def prioritise(rows, primary=lambda row: row["category"].split("+")[0].split(":")[0]):
    """``priority`` 1 for the first case to confirm per pair group and category, else 2.

    A row that already has a priority keeps it.
    """
    first = {(_group(row["pair"]), primary(row)) for row in rows if row.get("priority") == "1"}
    for row in rows:
        if row.get("priority"):
            continue
        key = (_group(row["pair"]), primary(row))
        if row["status"] == TO_CONFIRM and key not in first:
            row["priority"] = "1"
            first.add(key)
        else:
            row["priority"] = "2"
    return rows


def write(name, rows, columns, directory=DIRECTORY):
    path = Path(directory) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    tmp.replace(path)


def main(argv=None):
    options = prioritise(merge(read(OPTION_FILE), tenor_cases(), ("pair", "trade_date", "tenor"), "O"))
    typed = prioritise(merge(read(TYPED_FILE), typed_cases(), ("pair", "typed_expiry"), "T"))
    write(OPTION_FILE, options, OPTION_COLUMNS)
    write(TYPED_FILE, typed, TYPED_COLUMNS)
    for name, rows in ((OPTION_FILE, options), (TYPED_FILE, typed)):
        status = {}
        for row in rows:
            status[row["status"]] = status.get(row["status"], 0) + 1
        first = sum(1 for row in rows if row["status"] == TO_CONFIRM and row["priority"] == "1")
        print(f"{name}: {len(rows)} cases {status}; first batch to confirm: {first}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

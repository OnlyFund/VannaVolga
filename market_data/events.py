"""Event calendars: import a CSV as a new event set, list sets, re-pin a market.

The CSV has a header row and one event per row::

    pair,date,event,variance_days
    EURUSD,2026-11-06,NFP,1.5
    EURUSD,2026-11-06,ECB speech,0.25
    USDJPY,2026-10-30,BOJ,1.0

Several events on one date are separate rows; their variance days add up.
The whole file is validated first: any bad row rejects the import, with
every problem listed, and nothing is stored.

    python -m market_data.events import calendar.csv [--note "..."]
    python -m market_data.events list
    python -m market_data.events pin MARKET_ID EVENT_SET_ID --reason "..."

New markets pin the latest event set when they are fetched; ``pin`` moves
an existing market to another set and records why.
"""
import argparse
import csv
from pathlib import Path

import pandas as pd

from market_data.store import open_market_data_store

COLUMNS = ("pair", "date", "event", "variance_days")


def read_events_csv(path):
    """``[(pair, date, event, variance_days)]`` from the CSV at ``path``.

    Raises ``ValueError`` when the header lacks a column; row checks are
    ``store.validate_events``'s (rows are numbered after the header).
    """
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header = [name.strip().lower() for name in (reader.fieldnames or [])]
        missing = [column for column in COLUMNS if column not in header]
        if missing:
            raise ValueError(f"{path}: missing column(s) {', '.join(missing)}; "
                             f"expected {', '.join(COLUMNS)}")
        rows = []
        for record in reader:
            record = {str(key).strip().lower(): value for key, value in record.items()}
            rows.append(tuple(record.get(column) for column in COLUMNS))
    return rows


def import_csv(path, store=None, note=""):
    """Store the CSV as a new event set and return its id."""
    store = store or open_market_data_store()
    return store.save_event_set(read_events_csv(path), "csv", note=note or Path(path).name)


def event_table(event_set, pair):
    """The pair's events as the app's event table (``event_date``, ``event``,
    ``variance_days``): one named row per event, by date. Same-day events
    stay separate rows; the variance days add them up."""
    rows = (event_set.events.get(pair, []) if event_set is not None else [])
    return event_rows_table(rows)


def event_rows_table(rows):
    """``[(date, event, variance_days)]`` as the app's event table, by date then name."""
    frame = pd.DataFrame(list(rows), columns=["event_date", "event", "variance_days"])
    if frame.empty:
        return pd.DataFrame({"event_date": pd.Series(dtype="datetime64[ns]"),
                             "event": pd.Series(dtype=object),
                             "variance_days": pd.Series(dtype=float)})
    frame["event_date"] = pd.to_datetime(frame["event_date"])
    frame["event"] = frame["event"].fillna("").astype(str)
    frame["variance_days"] = pd.to_numeric(frame["variance_days"])
    return frame.sort_values(["event_date", "event"]).reset_index(drop=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    load = commands.add_parser("import", help="store a CSV as a new event set")
    load.add_argument("csv")
    load.add_argument("--note", default="")
    commands.add_parser("list", help="list the event sets")
    pin = commands.add_parser("pin", help="re-pin a market to an event set")
    pin.add_argument("market_id", type=int)
    pin.add_argument("event_set_id", type=int)
    pin.add_argument("--reason", required=True)
    args = parser.parse_args(argv)
    try:
        _run(args, open_market_data_store())
    except (OSError, ValueError) as exc:          # a bad file or id: say so, change nothing
        raise SystemExit(f"Nothing changed: {exc}")


def _run(args, store):
    if args.command == "import":
        event_set_id = import_csv(args.csv, store, args.note)
        event_set = store.load_event_set(event_set_id)
        print(f"Stored event set {event_set_id}: "
              f"{sum(len(rows) for rows in event_set.events.values())} events for "
              f"{len(event_set.events)} pairs. New fetches pin it; existing markets keep "
              f"theirs (re-pin with: python -m market_data.events pin MARKET_ID {event_set_id} "
              f"--reason ...)")
    elif args.command == "list":
        sets = store.event_sets()
        print(sets.to_string(index=False) if not sets.empty else
              "No event sets yet: python -m market_data.events import calendar.csv")
    else:
        store.pin_events(args.market_id, args.event_set_id, args.reason)
        print(f"Market {args.market_id} now uses event set {args.event_set_id}.")


if __name__ == "__main__":
    main()

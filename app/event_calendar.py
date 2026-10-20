"""A pair's event calendar between the Markets tab and the market data store.

* ``table_rows`` turns the Markets tab's event table into ``(date, event,
  variance_days)`` rows (blank lines dropped, nothing else changed);
* ``save`` stores them as the pair's next calendar version
  (``MarketDataStore.save_pair_events``: validated whole, append-only);
* ``csv_text`` / ``rows_from_csv`` move a calendar to and from a CSV file in
  the event-set format (``pair,date,event,variance_days``).

Failsafes: a calendar is validated as a whole and any bad row saves or
loads nothing, with every problem listed; a CSV's rows for other pairs are
left out and counted, never mixed in; a version only reaches valuation
dates from its save date on (``applies_from``).
"""
import csv
import io

import pandas as pd

from market_data.events import COLUMNS as CSV_COLUMNS, event_rows_table


def _blank(value):
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def table_rows(event_rows):
    """``[(date, event, variance_days)]`` from the event table; fully blank
    lines (the editor's empty row) are dropped, partly filled ones are kept
    for validation to name."""
    if event_rows is None or len(event_rows) == 0:
        return []
    frame = event_rows.reindex(columns=["event_date", "event", "variance_days"])
    rows = []
    for _, row in frame.iterrows():
        values = (row["event_date"], row["event"], row["variance_days"])
        if all(_blank(value) for value in values):
            continue
        date = None if _blank(row["event_date"]) else row["event_date"]
        name = "" if _blank(row["event"]) else str(row["event"]).strip()
        days = None if _blank(row["variance_days"]) else row["variance_days"]
        rows.append((date, name, days))
    return rows


def save(store, pair, event_rows, user="", note=""):
    """Save the table as the pair's next version; returns ``(version, events)``."""
    rows = table_rows(event_rows)
    version = store.save_pair_events(pair, rows, user=user, note=note)
    return version, len(rows)


def applies_from(saved_at):
    """The first valuation date a version saved at ``saved_at`` reaches."""
    return pd.Timestamp(saved_at).normalize()


def csv_text(pair, event_rows):
    """The table as CSV text in the event-set format."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for date, name, days in table_rows(event_rows):
        day = "" if date is None else pd.Timestamp(date).strftime("%Y-%m-%d")
        writer.writerow([pair, day, name, "" if days is None else days])
    return buffer.getvalue()


def rows_from_csv(text, pair):
    """``(table, notes)``: this pair's events from CSV text, as the event table.

    A blank pair cell means this pair; rows for other pairs are left out and
    counted in ``notes``. Raises ``ValueError`` (every problem listed) on a
    missing column or any bad row: nothing is loaded.
    """
    from market_data.store import validate_events
    if isinstance(text, bytes):
        text = text.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    header = [str(name).strip().lower() for name in (reader.fieldnames or [])]
    needed = [column for column in ("date", "event", "variance_days") if column not in header]
    if needed:
        raise ValueError(f"missing column(s) {', '.join(needed)}; expected "
                         f"{', '.join(CSV_COLUMNS)}")
    mine, others = [], 0
    for record in reader:
        record = {str(key).strip().lower(): value for key, value in record.items()}
        row_pair = str(record.get("pair") or "").replace("/", "").replace(" ", "").upper()
        if row_pair and row_pair != pair:
            others += 1
            continue
        mine.append((pair, record.get("date"), record.get("event"), record.get("variance_days")))
    clean = validate_events(mine, "CSV not loaded") if mine else []
    notes = [f"{others} row(s) for other pairs left out"] if others else []
    return event_rows_table([(day, name, days) for _, day, name, days in clean]), notes

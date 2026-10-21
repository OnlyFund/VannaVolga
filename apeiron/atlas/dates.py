"""Date parsing for the core (no calendars, no QuantLib import).

``to_date(value)`` coerces what the pricers are given (``date``, ``datetime``,
pandas ``Timestamp``, a QuantLib ``Date``, or text) to ``datetime.date``. It
accepts exactly what ``helper_functions.fx_discounting._to_date`` accepts, so
the core parses dates without reaching into the calendar layer.
"""
from datetime import date, datetime

#: Text formats tried in order (ISO first).
DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%b-%Y", "%d%b%Y", "%d%b%y")


def _is_ql_date(value):
    # Duck-typed so Atlas need not import QuantLib.
    return all(callable(getattr(value, name, None)) for name in ("year", "month", "dayOfMonth"))


def to_date(value) -> date:
    """Coerce a date-like value to ``datetime.date``."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if _is_ql_date(value):
        return date(value.year(), value.month(), value.dayOfMonth())
    text = str(value).strip()
    if not text:
        raise ValueError("A date is required")
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValueError("Unrecognised date {!r}".format(value))

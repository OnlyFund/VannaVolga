"""Historical FX option expiry dates from a local Bloomberg Desktop session."""
import argparse
from datetime import date, datetime
import re

import pandas as pd


DEFAULT_TENORS = ("ON", "1W", "1M", "2M", "3M", "6M", "9M", "1Y")
EXPIRY_FIELD = "VOL_SURF_EXPIRY_OVR"
# The vol surface reads "ON" (and "O/N") as the reference date itself; "1D"
# gives the overnight expiry the pricing page shows (checked 2026-10-22).
SURFACE_TENORS = {"ON": "1D"}


def _items(values):
    if isinstance(values, (str, date, datetime, pd.Timestamp)):
        return [values]
    return list(values)


def _date(value):
    if not isinstance(value, (str, date, datetime, pd.Timestamp)):
        raise ValueError("Dates must be date objects or strings such as 20240112 or 2024-01-12")
    result = pd.Timestamp(value)
    if pd.isna(result):
        raise ValueError("A valid date is required")
    return result.date()


def fetch_historical_expiry_dates(reference_dates, pairs, tenors=DEFAULT_TENORS):
    """Return reference_date, currency_pair, tenor and expiry_date columns.

    Accept one date/pair/tenor or an iterable of each. Pairs are six-letter FX
    codes (EURUSD or EUR/USD), not individual currencies. Dates may be Python
    dates, YYYYMMDD or ISO strings. Duplicate inputs are removed in input order.
    Each reference-date/tenor combination uses one Bloomberg request for all
    pairs, because overrides apply to the entire request.

    Requires a running, logged-in Bloomberg Desktop session and xbbg. Raises
    on missing/invalid results instead of inventing expiry dates.
    """
    dates = list(dict.fromkeys(_date(value) for value in _items(reference_dates)))
    pairs = list(dict.fromkeys(
        str(value).replace("/", "").replace(" ", "").upper() for value in _items(pairs)
    ))
    tenors = list(dict.fromkeys(str(value).strip().upper() for value in _items(tenors)))
    if not dates or not pairs or not tenors:
        raise ValueError("At least one reference date, currency pair and tenor are required")
    for pair in pairs:
        if not re.fullmatch(r"[A-Z]{6}", pair) or pair[:3] == pair[3:]:
            raise ValueError("Invalid FX pair: {!r}; use a pair such as EURUSD".format(pair))
    for tenor in tenors:
        if tenor != "ON" and not re.fullmatch(r"[1-9][0-9]*[DWMY]", tenor):
            raise ValueError("Invalid option tenor: {!r}".format(tenor))

    # Keep Bloomberg dependencies out of imports used by the Cloud app.
    from market_data.xbbg_compat import patch_xbbg
    patch_xbbg()
    from xbbg.blp import bdp

    tickers = ["{} BGN Curncy".format(pair) for pair in pairs]
    rows = []
    for reference in dates:
        for tenor in tenors:
            result = bdp(
                tickers=tickers,
                flds=EXPIRY_FIELD,
                VOL_SURF_MTY_OVR=SURFACE_TENORS.get(tenor, tenor),
                REFERENCE_DATE=reference.strftime("%Y%m%d"),
            )
            for pair, ticker in zip(pairs, tickers):
                context = "{} {} as of {}".format(pair, tenor, reference.isoformat())
                column = EXPIRY_FIELD.lower()
                if (ticker not in result.index or column not in result.columns
                        or pd.isna(result.at[ticker, column])):
                    raise ValueError("Bloomberg returned no expiry for " + context)
                expiry = _date(result.at[ticker, column])
                if expiry < reference:
                    raise ValueError("Bloomberg returned an expiry before the reference date for " + context)
                rows.append(dict(reference_date=reference, currency_pair=pair,
                                 tenor=tenor, expiry_date=expiry))
    return pd.DataFrame(rows, columns=["reference_date", "currency_pair", "tenor", "expiry_date"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dates", nargs="+", required=True, help="Historical dates: YYYYMMDD or YYYY-MM-DD")
    parser.add_argument("--pairs", nargs="+", required=True, help="FX pairs, e.g. EURUSD USDIDR")
    parser.add_argument("--tenors", nargs="+", default=DEFAULT_TENORS)
    args = parser.parse_args()
    print(fetch_historical_expiry_dates(args.dates, args.pairs, args.tenors).to_string(index=False))


if __name__ == "__main__":
    main()

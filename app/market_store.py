"""Saved market files: single-pair snapshots and multi-pair bundles.

A single-pair snapshot is one market dict (``currency_pair``, ``spot``,
``vol_data``, ``forward_points``, ``event_calendar`` ...), the format the app
has always saved. A bundle maps pairs to such dicts::

    {"format": "market_bundle", "version": 1, "valuation_date": "2026-10-02",
     "markets": {"EURUSD": {...}, "USDJPY": {...}}}

:func:`read_market_file` accepts either, plus two looser shapes for
hand-made files: a bare ``{"EURUSD": {...}, ...}`` mapping and a ``markets``
list. Each pair is validated on its own, so one bad pair is reported and
skipped without losing the others; only a file with no usable pair fails.
"""
import json
import re

import numpy as np
import pandas as pd

BUNDLE_FORMAT = "market_bundle"
BUNDLE_VERSION = 1
SINGLE_FORMAT = "market"

VOL_COLUMNS = ("atm", "rr_25", "bf_25", "rr_10", "bf_10")
FORWARD_COLUMNS = ("fwd_pts", "term_discount")
#: Simple term rate used for files saved before ``term_discount`` existed.
DEFAULT_TERM_DISCOUNT = 0.05

_REQUIRED = ("currency_pair", "valuation_date", "spot",
             "vol_data", "forward_points", "event_calendar")


def normalize_pair(value):
    """Return ``value`` as an upper-case pair code such as ``EURUSD``."""
    return str(value or "").replace("/", "").replace(" ", "").strip().upper()


def is_pair(value):
    return re.fullmatch(r"[A-Z]{6}", normalize_pair(value)) is not None


def _records(frame):
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def market_payload(currency_pair, valuation_date, spot, forward_divisor,
                   broker_strangle, vol_rows, forward_rows, event_rows):
    """Return the JSON-ready single-pair market dict."""
    return {
        "format": SINGLE_FORMAT,
        "currency_pair": normalize_pair(currency_pair),
        "valuation_date": pd.Timestamp(valuation_date).strftime("%Y-%m-%d"),
        "spot": float(spot),
        "forward_divisor": float(forward_divisor),
        "broker_strangle": bool(broker_strangle),
        "rate_compounding": "simple",
        "vol_data": _records(vol_rows),
        "forward_points": _records(forward_rows),
        "event_calendar": _records(event_rows),
    }


def bundle_payload(markets, valuation_date=None):
    """Return the JSON-ready bundle for ``{pair: market dict}``."""
    bundle = {"format": BUNDLE_FORMAT, "version": BUNDLE_VERSION}
    if valuation_date is not None:
        bundle["valuation_date"] = pd.Timestamp(valuation_date).strftime("%Y-%m-%d")
    bundle["markets"] = {
        normalize_pair(pair): dict(market, currency_pair=normalize_pair(pair))
        for pair, market in markets.items()
    }
    return bundle


def validate_market(payload):
    """Validate one market dict and return its typed session-state values.

    Raises ``ValueError`` describing the first problem found. The result maps
    ``currency_pair``, ``valuation_date``, ``spot``, ``forward_divisor``,
    ``broker_strangle`` and the ``vol_rows``/``forward_rows``/``event_rows``
    frames.
    """
    if not isinstance(payload, dict):
        raise ValueError("Market entry must be a JSON object")
    missing = set(_REQUIRED).difference(payload)
    if missing:
        raise ValueError("Market file is missing fields: " + ", ".join(sorted(missing)))
    pair = normalize_pair(payload["currency_pair"])
    if not is_pair(pair):
        raise ValueError(f"Invalid currency pair {payload['currency_pair']!r}")
    frames = {
        "vol_rows": pd.DataFrame(payload["vol_data"]),
        "forward_rows": pd.DataFrame(payload["forward_points"]),
        "event_rows": pd.DataFrame(
            payload["event_calendar"], columns=["event_date", "variance_days"]
        ),
    }
    if "term_discount" not in frames["forward_rows"]:
        # Keep saved market files from earlier app versions loadable.
        frames["forward_rows"]["term_discount"] = DEFAULT_TERM_DISCOUNT
    expected = {
        "vol_rows": {"tenor", "expiry", *VOL_COLUMNS},
        "forward_rows": {"tenor", "delivery_date", *FORWARD_COLUMNS},
        "event_rows": {"event_date", "variance_days"},
    }
    for name, frame in frames.items():
        absent = expected[name].difference(frame.columns)
        if absent:
            raise ValueError(f"{name} is missing columns: " + ", ".join(sorted(absent)))
    if frames["vol_rows"].empty or frames["forward_rows"].empty:
        raise ValueError("Market file must include volatility and forward-point rows")
    # Saved ISO dates carry a "Z"; the editors hold naive dates.
    for name, column in (("vol_rows", "expiry"), ("forward_rows", "delivery_date"),
                         ("event_rows", "event_date")):
        frames[name][column] = pd.to_datetime(
            frames[name][column], errors="raise", utc=True
        ).dt.tz_localize(None)
    for name, columns in (("vol_rows", VOL_COLUMNS),
                          ("forward_rows", FORWARD_COLUMNS),
                          ("event_rows", ("variance_days",))):
        for column in columns:
            frames[name][column] = pd.to_numeric(frames[name][column], errors="raise")
            if not np.isfinite(frames[name][column].to_numpy(dtype=float)).all():
                raise ValueError(f"{name}.{column} values must be finite")
    if (frames["vol_rows"]["atm"] <= 0).any():
        raise ValueError("ATM volatilities must be positive")
    if frames["vol_rows"]["expiry"].duplicated().any():
        raise ValueError("Volatility expiry dates must be unique")
    if frames["vol_rows"]["expiry"].isna().any() or frames["forward_rows"]["delivery_date"].isna().any():
        raise ValueError("Market expiry and delivery dates must be valid")
    if frames["forward_rows"]["delivery_date"].duplicated().any():
        raise ValueError("Forward delivery dates must be unique")
    spot_value = float(payload["spot"])
    divisor_value = float(payload.get("forward_divisor", 10000.0))
    if not np.isfinite(spot_value) or spot_value <= 0:
        raise ValueError("Spot must be finite and positive")
    if not np.isfinite(divisor_value) or divisor_value <= 0:
        raise ValueError("Forward divisor must be finite and positive")
    if not isinstance(payload.get("broker_strangle", True), bool):
        raise ValueError("broker_strangle must be a boolean")
    return {
        "currency_pair": pair,
        "broker_strangle": bool(payload.get("broker_strangle", True)),
        "valuation_date": pd.Timestamp(payload["valuation_date"]).date(),
        "spot": spot_value,
        "forward_divisor": divisor_value,
        **frames,
    }


def _entries(data):
    """Yield ``(key, market dict)`` from any supported file shape.

    ``key`` is the pair the container files the market under, or ``None``
    when the market only names itself.
    """
    if not isinstance(data, dict):
        if isinstance(data, list):
            return [(None, entry) for entry in data], {}
        raise ValueError("Market file must contain a JSON object")
    if "vol_data" in data:                                  # single pair
        return [(None, data)], {}
    markets = data.get("markets")
    defaults = {"valuation_date": data["valuation_date"]} if "valuation_date" in data else {}
    if isinstance(markets, dict):
        return list(markets.items()), defaults
    if isinstance(markets, list):
        return [(None, entry) for entry in markets], defaults
    if markets is None:
        # Bare {"EURUSD": {...}, ...}; ignore metadata keys like "format".
        keyed = [(key, value) for key, value in data.items()
                 if is_pair(key) and isinstance(value, dict)]
        if keyed:
            return keyed, defaults
    raise ValueError(
        "Not a market file: expected a single market (with vol_data) "
        "or a bundle with a 'markets' mapping"
    )


def read_market_file(raw):
    """Parse a saved market file into ``({pair: market dict}, {label: error})``.

    ``raw`` is bytes, text or an already decoded object. Valid markets are
    returned as their (normalized) JSON dicts in file order, ready to store
    and to pass to :func:`validate_market`. A market filed under a pair key
    takes that pair when it does not name one, and the bundle's valuation
    date when it has none; a market whose own pair contradicts its key, or a
    pair that appears twice, is rejected. Raises ``ValueError`` only when the
    file is unreadable or holds no valid market at all.
    """
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8-sig")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Market file is not valid JSON: {exc}") from None
    entries, defaults = _entries(raw)
    if bundle_version(raw) > BUNDLE_VERSION:
        raise ValueError(
            f"Market bundle version {raw['version']} is newer than supported "
            f"version {BUNDLE_VERSION}"
        )
    markets, errors = {}, {}
    for index, (key, entry) in enumerate(entries):
        label = normalize_pair(key) if key is not None else f"entry {index + 1}"
        try:
            if not isinstance(entry, dict):
                raise ValueError("Market entry must be a JSON object")
            entry = {**defaults, **entry}
            if key is not None:
                if not is_pair(key):
                    raise ValueError(f"Invalid currency pair {key!r}")
                own = entry.get("currency_pair")
                if own not in (None, "") and normalize_pair(own) != normalize_pair(key):
                    raise ValueError(
                        f"Filed under {normalize_pair(key)} but quotes {normalize_pair(own)}"
                    )
                entry["currency_pair"] = normalize_pair(key)
            validated = validate_market(entry)
            pair = validated["currency_pair"]
            label = pair
            if pair in markets:
                raise ValueError("Duplicate market for this pair; kept the first")
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            errors[label if label not in errors else f"{label} ({index + 1})"] = str(exc)
            continue
        markets[pair] = dict(entry, currency_pair=pair)
    if not markets:
        detail = "; ".join(f"{label}: {error}" for label, error in errors.items())
        raise ValueError("No valid market in file" + (f" ({detail})" if detail else ""))
    return markets, errors


def bundle_version(data):
    try:
        return int(data.get("version", BUNDLE_VERSION)) if isinstance(data, dict) else BUNDLE_VERSION
    except (TypeError, ValueError):
        return BUNDLE_VERSION


def choose_active_pair(pairs, current_pair):
    """Keep the current pair active when available, else the file's first."""
    current_pair = normalize_pair(current_pair)
    pairs = list(pairs)
    return current_pair if current_pair in pairs else pairs[0]

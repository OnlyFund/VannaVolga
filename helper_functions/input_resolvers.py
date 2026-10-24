"""Reusable portfolio input resolvers for the application."""

import re
from math import isfinite
from datetime import date, datetime

import pandas as pd

from helper_functions.VannaVolga_function import (
    get_atm_strike,
    get_strike_from_delta,
    normalize_atm_type,
)


_TENOR_RE = re.compile(r"(\d+(?:\.\d+)?)([wmy])", re.IGNORECASE)
_BROKEN_DATE_RE = re.compile(
    r"(?P<day>\d{1,2})(?P<month>jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
    r"(?P<year>\d{2})",
    re.IGNORECASE,
)
_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
# Ordered so that ambiguous numeric dates resolve month-first, matching the
# mm/dd/yyyy convention the editor advertises. Day-first inputs such as
# 24/03/2027 are unambiguous once month-first parsing fails.
_DATE_FORMATS = (
    "%m/%d/%Y", "%m/%d/%y",
    "%m-%d-%Y", "%m-%d-%y",
    "%Y-%m-%d", "%Y/%m/%d",
    "%d%b%Y", "%d%b%y",
    "%d-%b-%Y", "%d-%b-%y",
    "%d %b %Y", "%d %b %y",
    "%d.%m.%Y", "%d.%m.%y",
    "%b %d %Y", "%b %d, %Y", "%b %d %y",
    "%d/%m/%Y", "%d/%m/%y",
)


def parse_notional(value):
    """Parse plain notionals and k/m shorthand, case-insensitively."""
    text = str(value).strip().lower().replace(",", "").replace(" ", "")
    match = re.fullmatch(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))(k|m)?", text)
    if not match:
        raise ValueError(
            "Notional must be numeric or use k/m shorthand, e.g. 100k or 1m"
        )
    amount = float(match.group(1))
    multiplier = {"k": 1_000.0, "m": 1_000_000.0}.get(match.group(2), 1.0)
    return amount * multiplier


def _date_text(value):
    """Normalize any supported date input to a plain ``YYYY-MM-DD`` string.

    Downstream year-fraction helpers parse with ``%Y-%m-%d`` exactly, so a
    midnight timestamp must not keep its time component: ``pd.Timestamp`` and
    ``datetime`` both isoformat to ``YYYY-MM-DDTHH:MM:SS``, which fails to
    parse and leaves strike aliases unresolved.
    """
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    try:
        return pd.Timestamp(text).date().isoformat()
    except (ValueError, TypeError):
        return text


def parse_expiry(value, valuation_date, pillar_tenors=None):
    """Resolve a tenor or broken date to a concrete date."""
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, date):
        return value

    text = str(value).strip()
    valuation = pd.Timestamp(valuation_date).date()
    tenor_match = _TENOR_RE.fullmatch(text)
    if tenor_match:
        tenor = tenor_match.group(1) + tenor_match.group(2).upper()
        if pillar_tenors is not None and tenor not in {
            str(item).strip().upper() for item in pillar_tenors
        }:
            raise ValueError(
                f"Expiry tenor must be one of: {', '.join(map(str, pillar_tenors))}"
            )
        units = {"W": 7, "M": 30, "Y": 365}
        return valuation + pd.Timedelta(
            days=float(tenor_match.group(1)) * units[tenor_match.group(2).upper()]
        ).to_pytimedelta()

    broken_match = _BROKEN_DATE_RE.fullmatch(text)
    if broken_match:
        try:
            return date(
                2000 + int(broken_match.group("year")),
                _MONTHS[broken_match.group("month").lower()],
                int(broken_match.group("day")),
            )
        except ValueError as exc:
            raise ValueError(f"Invalid expiry date: {value}") from exc

    try:
        parsed = pd.to_datetime(text, format="%m/%d/%Y", errors="raise")
        if not pd.isna(parsed):              # pandas 2 gives NaT for "", not an error
            return parsed.date()
    except (TypeError, ValueError):
        pass

    # Accept the common written date forms rather than only mm/dd/yyyy, since
    # the editor is a free-text cell and month names, ISO order, hyphens and
    # two-digit years are all natural things to type.
    collapsed = re.sub(r"\s+", " ", text)
    for date_format in _DATE_FORMATS:
        try:
            return datetime.strptime(collapsed, date_format).date()
        except ValueError:
            continue

    raise ValueError(
        "Expiry must be a pillar tenor (e.g. 1w, 2w, 1m, 1y) or a date such as "
        "03/24/2027, 24mar27, 24-Mar-2027 or 2027-03-24"
    )


def resolve_expiry(value, valuation_date, pillar_tenors):
    """Resolve an app expiry against the configured market pillars."""
    return parse_expiry(value, valuation_date, pillar_tenors=pillar_tenors)


def _delta_strike_on_smile(market, delta, call_put, expiry_date, valuation_date,
                           atm_type, delta_base, tol=1e-10, max_iter=50):
    """Resolve a delta-quoted strike consistently with the smile.

    A delta-quoted strike is defined by the volatility *at that strike*, not by
    the ATM volatility: the 10-delta call strike is the strike whose delta is
    10% when evaluated with the 10-delta call volatility. Solving with the ATM
    volatility instead places the wings too close to the money, because it
    ignores the butterfly and risk reversal entirely.

    Volatility and strike are mutually dependent, so iterate to a fixed point.
    At a quoted pillar this converges onto the quote, reproducing the market
    pillar strikes exactly; between pillars it follows the interpolated smile.
    """
    from VannaVolga_ImpliedVol import VannaVolga

    def strike_for_vol(vol):
        return get_strike_from_delta(
            S=market.s,
            fwd=market.fwd_pts,
            v=vol,
            expiry_date=expiry_date,
            valuation_date=valuation_date,
            CallPut=call_put,
            delta=delta,
            atm_type=atm_type,
            delta_base=delta_base,
            fwd_divisor=market.forward_divisor,
        )

    def smile_vol(strike):
        return float(
            VannaVolga(
                market=market,
                K=strike,
                expiry_date=expiry_date,
                valuation_date=valuation_date,
                CallPut=call_put,
                deltaBase=delta_base,
                atm_conv=atm_type,
            ).get_vol_for_strike()
        )

    vol = float(market.v_atm)
    strike = strike_for_vol(vol)
    for iteration in range(max_iter):
        try:
            updated = smile_vol(strike)
        except Exception as exc:
            if iteration == 0:
                # The very first evaluation failing means no smile information
                # was ever used, so this is the ATM-vol answer the fixed point
                # exists to replace. Silently returning it looks like a working
                # resolution while placing the wings too close to the money.
                raise ValueError(
                    "Could not evaluate the smile for a {:.0f}-delta {} strike: "
                    "{}".format(delta * 100.0, str(call_put).upper(), exc)
                )
            # A later failure still leaves a smile-informed strike in hand.
            return strike
        if not isfinite(updated) or updated <= 0.0:
            if iteration == 0:
                raise ValueError(
                    "Smile returned a non-positive volatility ({}) for a "
                    "{:.0f}-delta {} strike".format(
                        updated, delta * 100.0, str(call_put).upper()
                    )
                )
            return strike
        if abs(updated - vol) < tol:
            break
        vol = updated
        strike = strike_for_vol(vol)
    return strike


def resolve_strike(market, row, atm_type, delta_base):
    """Resolve numeric, delta, or ATM alias strike input."""
    value = str(row["strike"]).strip().lower().replace(" ", "")
    if (getattr(market, "wrapped_smile", False) and
            not market.force_legacy_smile and
            (value in {"atm", "atmf", "atms"} or re.fullmatch(r"(\d+(?:\.\d+)?)d", value))):
        result = _wrapped_strike(market, row, value)
        if result is not None:
            return result
    atm_aliases = {"atm": atm_type, "atmf": "atmf", "atms": "spot"}
    if value in atm_aliases:
        resolved_atm_type = normalize_atm_type(atm_aliases[value])
        return get_atm_strike(
            atm_type=resolved_atm_type,
            S=market.s,
            fwd=market.fwd_pts,
            v=market.v_atm,
            expiry_date=_date_text(row["expiry"]),
            valuation_date=_date_text(row["valuation"]),
            fwd_divisor=market.forward_divisor,
            delta_base=delta_base,
        )
    delta_match = re.fullmatch(r"(\d+(?:\.\d+)?)d", value)
    if delta_match:
        delta_pct = float(delta_match.group(1))
        if not 0.0 < delta_pct < 50.0:
            raise ValueError("Delta strike must be between 0d and 50d")
        return _delta_strike_on_smile(
            market=market,
            delta=delta_pct / 100.0,
            call_put=row["call_put"],
            expiry_date=_date_text(row["expiry"]),
            valuation_date=_date_text(row["valuation"]),
            atm_type=atm_type,
            delta_base=delta_base,
        )
    try:
        strike = float(value.replace(",", ""))
    except ValueError as exc:
        raise ValueError(
            "Strike must be numeric, an ATM alias (atm/atmf/atms), "
            "or a delta such as 25d"
        ) from exc
    if not isfinite(strike) or strike <= 0:
        raise ValueError("Strike must be a finite number greater than zero")
    return strike


def _wrapped_strike(market, row, value):
    from math import exp, log, sqrt
    import numpy as np
    from scipy.optimize import brentq
    from scipy.special import ndtr
    from app.smile_service import prepared_smile, record
    from wrapped_vol_resolver import WrappedVolResolver

    expiry, valuation = _date_text(row["expiry"]), _date_text(row["valuation"])
    if expiry == valuation:
        return None
    prepared = prepared_smile(market, expiry, valuation, market.smile_model)
    def legacy_alias():
        legacy = prepared.raw.to_market().copy_with(wrapped_smile=False)
        return resolve_strike(legacy, row, "dn", True)
    if prepared.diagnostics.degraded:
        return legacy_alias()
    ctx = prepared.conventions
    if value == "atm":
        return prepared.pillars.strikes[2]
    if value == "atmf":
        return ctx.forward
    if value == "atms":
        return market.s
    target = float(value[:-1]) / 100.0
    if not 0 < target < .5:
        raise ValueError("Delta strike must be between 0d and 50d")
    cp = str(row["call_put"]).upper()
    if cp not in ("CALL", "PUT"):
        raise ValueError("Call/Put must be CALL or PUT")
    phi = 1 if cp == "CALL" else -1
    resolver = WrappedVolResolver(prepared=prepared, CallPut=cp)

    def error(log_strike):
        strike = ctx.forward * exp(log_strike)
        vol = resolver.get_vol_for_strike(strike) / 100.0
        if resolver.diagnostics.degraded:
            record(market, market.smile_model, expiry, resolver.diagnostics.reason)
            raise _AliasDegraded()
        std = vol * sqrt(ctx.time)
        d1 = -log_strike / std + std / 2
        delta = (strike / ctx.forward * ndtr(phi * (d1 - std))
                 if ctx.premium_adjusted else ndtr(phi * d1))
        return ctx.delta_factor * delta - target

    try:
        # Start near the delta maximum and move outward; do not probe
        # extreme wings unnecessarily before a quoted-delta root is bracketed.
        std = market.v_atm / 100.0 * sqrt(ctx.time)
        grid = np.linspace(-phi * .5 * std, phi * 6 * std, 180)
        previous, previous_error = grid[0], error(grid[0])
        for point in grid[1:]:
            current_error = error(point)
            if current_error * previous_error <= 0:
                root = brentq(error, min(previous, point), max(previous, point),
                              xtol=1e-13)
                if abs(error(root)) > 1e-8:
                    raise ValueError("Smile delta strike failed numerical verification")
                return ctx.forward * exp(root)
            previous, previous_error = point, current_error
    except _AliasDegraded:
        return legacy_alias()
    raise ValueError("No verified {}-delta {} strike on selected smile".format(target * 100, cp))


class _AliasDegraded(Exception):
    pass

"""Today's pillar dates from Bloomberg (``bdp``), checked, with the solver as fallback.

Once a day, before the 17:00 New York cut, Bloomberg's live dates for every
pair and tenor:

* **ON, day and week tenors**: the expiry is ``OPT_EXPIRE_DT`` on the ATM vol
  ticker (``<PAIR>V<TENOR> Curncy``); the delivery is the expiry's spot date
  (expiry plus the spot lag);
* **month and year tenors**: the delivery is ``SETTLE_DT`` on the outright
  forward ticker, the date the forward reference rate maps to; the expiry is
  derived from it (``pillars.expiry_for_delivery``: the latest expiry day whose
  spot date is that delivery). ``OPT_EXPIRE_DT`` is then a cross-check only:
  kept as ``quoted`` and, where it differs, listed in ``mismatches`` (decided
  2026-10-23: NZDUSD 1M's ``OPT_EXPIRE_DT`` 25 Nov is not the pricing page's
  26 Nov, the date the 30 Nov forward implies). A month tenor without a
  forward date falls back to the solver;
* **spot**: ``SETTLE_DT`` on ``<PAIR> Curncy``, a check only.

Bloomberg's dates win over the solver's (decided 2026-10-23); a difference is a
*dispute*, kept for the edge cases. Bloomberg only answers for today, so the
dates are good for this FX date only.

Fail-safes: anything doubtful falls back to the solver, as narrowly as is safe,
and says why (``LiveDates.rejected``):

* **the fetch** is refused after the cut or for any FX date but today's
  (``LiveDates.refused``); a failed request is retried pair by pair; a pair or
  ticker without an answer falls back alone;
* **a tenor** falls back when its date is missing or unreadable, no expiry
  day delivers on a month tenor's forward date, its expiry is not a weekday
  after the FX date, its delivery is not after its expiry, or its expiry or
  delivery is further from the solver's than ``TOLERANCE_DAYS`` (a stale or
  mis-mapped ticker, not a holiday rule);
* **a pair** falls back whole when its expiries or deliveries do not rise with
  the tenor, or Bloomberg's spot date is not after the FX date (stale data);
* **the whole fetch** is refused when more than ``MAX_DISPUTED`` of the accepted
  tenors disagree with the solver: something is systematically wrong.

Stored (``save``, ``--store``) in the market store's pillar date tables
(schema 7), with the solver's answer and ``OPT_EXPIRE_DT`` beside each date
and every fallback with its reason. A fetch refused before asking Bloomberg (after the cut, at a
weekend) stores nothing; one refused after (the cut passed, too many
disputes) is stored as refused, for the record, and never used.

Run on a machine with a Bloomberg terminal (``xbbg``):

    python -m market_data.live_dates            # fetch and report
    python -m market_data.live_dates --store    # ... and store them
"""
import argparse
import sys
from dataclasses import dataclass, field
from datetime import date, datetime

from apeiron.gaia.market_dates import pillars
from apeiron.gaia.market_dates.calendars import OVERNIGHT, PillarDates, as_date
from apeiron.gaia.market_dates.rollover import fx_date

#: The tenors fetched, shortest first (the ATM vol tickers' tenors).
TENORS = ("ON", "1W", "2W", "1M", "2M", "3M", "6M", "9M", "1Y")
#: How far (calendar days) Bloomberg's expiry may be from the solver's.
TOLERANCE_DAYS = {"short": 4, "long": 7}
#: The largest share of accepted tenors that may disagree with the solver.
MAX_DISPUTED = 0.3

EXPIRY_FIELD, SETTLE_FIELD = "OPT_EXPIRE_DT", "SETTLE_DT"


@dataclass
class LiveDates:
    """One day's fetch: accepted dates, what fell back and why, and the disputes."""
    fx_date: date
    fetched_at: datetime
    timezone: str = None
    dates: dict = field(default_factory=dict)       # (pair, tenor) -> PillarDates
    sources: dict = field(default_factory=dict)     # (pair, tenor) -> "bloomberg"
    quoted: dict = field(default_factory=dict)      # (pair, tenor) -> OPT_EXPIRE_DT (or None)
    mismatches: list = field(default_factory=list)  # (pair, tenor, OPT_EXPIRE_DT, expiry used)
    rejected: list = field(default_factory=list)    # (pair, tenor or "*", reason)
    disputes: list = field(default_factory=list)    # (pair, tenor, bloomberg PillarDates, solver PillarDates)
    refused: str = ""                               # why nothing is used
    fetched: bool = False                           # Bloomberg was asked


def _short(tenor):
    return tenor in OVERNIGHT or tenor[-1] in "DW"


def vol_ticker(pair, tenor):
    return f"{pair}V{tenor} Curncy"


def _text(value):
    text = "" if value is None else str(value)[:10]
    return "" if text in ("", "NaT", "nan", "None") else text


def _ask(bdp, tickers, fld):
    """``{ticker: text}``; a failed request is retried ticker group by group."""
    def once(group):
        frame = bdp(tickers=list(group), flds=fld)
        column = fld.lower()
        if frame is None or column not in getattr(frame, "columns", ()):
            return {}
        return {str(t): _text(frame.at[t, column]) for t in frame.index}
    try:
        return once(tickers)
    except Exception:            # one bad ticker can fail the batch: ask one by one
        answers = {}
        for ticker in tickers:
            try:
                answers.update(once([ticker]))
            except Exception:
                pass
        return answers


def fetch_raw(bdp, pairs, tenors=None, forward_ticker=None):
    """``{pair: {"spot": text, tenor: {"expiry": text, "forward": text}}}`` from Bloomberg."""
    tenors = tenors or TENORS
    if forward_ticker is None:
        from market_data.fwd_tickers import TENORS as FWD, fwd_ticker
        forward_ticker = lambda pair, tenor: fwd_ticker(pair, tenor) if tenor in FWD else None
    vols = {(p, t): vol_ticker(p, t) for p in pairs for t in tenors}
    fwds = {(p, t): forward_ticker(p, t) for p in pairs for t in tenors if not _short(t)}
    fwds = {k: v for k, v in fwds.items() if v}
    expiries = _ask(bdp, list(vols.values()), EXPIRY_FIELD)
    settles = _ask(bdp, list(fwds.values()) + [f"{p} Curncy" for p in pairs], SETTLE_FIELD)
    raw = {}
    for pair in pairs:
        raw[pair] = {"spot": settles.get(f"{pair} Curncy", "")}
        for tenor in tenors:
            raw[pair][tenor] = {"expiry": expiries.get(vols[(pair, tenor)], ""),
                                "forward": settles.get(fwds.get((pair, tenor)), "")}
    return raw


def _readable(text):
    try:
        return as_date(text) if text else None
    except (TypeError, ValueError):
        return None


def _tenor(rules, fx_day, tenor, got):
    """``(PillarDates, OPT_EXPIRE_DT or None, solver PillarDates)``, or why it falls back."""
    quoted = _readable(got.get("expiry"))
    if got.get("expiry") and quoted is None:
        return f"unreadable expiry {got['expiry']!r}"
    solver = pillars.tenor_dates(rules, fx_day, tenor)
    limit = TOLERANCE_DAYS["short" if _short(tenor) else "long"]
    if _short(tenor):
        if quoted is None:
            return "no expiry from Bloomberg"
        expiry = quoted
        if expiry <= fx_day or expiry.weekday() >= 5:
            return f"expiry {expiry} is no weekday after {fx_day}"
        delivery = pillars.delivery_for_expiry(rules, expiry)
    else:
        if not got.get("forward"):
            return "no forward settlement date from Bloomberg"
        delivery = _readable(got["forward"])
        if delivery is None:
            return f"unreadable forward settlement date {got['forward']!r}"
        if abs((delivery - solver.delivery).days) > limit:
            return f"forward date {delivery} is over {limit} days from the solver's {solver.delivery}"
        try:
            expiry = pillars.expiry_for_delivery(rules, delivery)
        except ValueError as error:
            return f"no expiry for the forward date {delivery}: {error}"
        if expiry <= fx_day:
            return f"expiry {expiry} is not after {fx_day}"
    if abs((expiry - solver.expiry).days) > limit:
        return f"expiry {expiry} is over {limit} days from the solver's {solver.expiry}"
    if delivery <= expiry:
        return f"delivery {delivery} is not after expiry {expiry}"
    return PillarDates(tenor, fx_day, solver.spot, expiry, delivery), quoted, solver


def check(raw, fx_day, settlement_of, fetched_at=None, timezone=None, tenors=None):
    """Apply the fail-safes to ``raw`` (``fetch_raw``'s answer) for ``fx_day``."""
    tenors = tenors or TENORS
    fx_day = as_date(fx_day)
    result = LiveDates(fx_day, fetched_at or datetime.now(), timezone)
    for pair, answers in raw.items():
        try:
            rules = settlement_of(pair)
        except Exception as error:
            result.rejected.append((pair, "*", f"no conventions: {error}"))
            continue
        spot = _readable(answers.get("spot"))
        if spot is not None and spot <= fx_day:
            result.rejected.append((pair, "*", f"stale: Bloomberg spot {spot} is not after {fx_day}"))
            continue
        mine = {}
        for tenor in tenors:
            got = answers.get(tenor) or {}
            dated = _tenor(rules, fx_day, tenor, got)
            if isinstance(dated, str):
                result.rejected.append((pair, tenor, dated))
                continue
            dates, quoted, solver = dated
            result.quoted[(pair, tenor)] = quoted
            if quoted is not None and quoted != dates.expiry:
                result.mismatches.append((pair, tenor, quoted, dates.expiry))
            mine[tenor] = (dates, "bloomberg", solver)
        ordered = [mine[t] for t in tenors if t in mine]
        rising = all(a[0].expiry < b[0].expiry and a[0].delivery < b[0].delivery
                     for a, b in zip(ordered, ordered[1:]))
        if not rising:
            result.rejected.append((pair, "*", "expiries or deliveries do not rise with the tenor"))
            continue
        for tenor, (dates, source, solver) in mine.items():
            result.dates[(pair, tenor)] = dates
            result.sources[(pair, tenor)] = source
            if (dates.expiry, dates.delivery) != (solver.expiry, solver.delivery):
                result.disputes.append((pair, tenor, dates, solver))
    if result.dates and len(result.disputes) > MAX_DISPUTED * len(result.dates):
        result.refused = (f"{len(result.disputes)} of {len(result.dates)} tenors disagree with the "
                          f"solver (over {MAX_DISPUTED:.0%}): not used")
    return result


def in_session(moment, timezone=None):
    """``moment``'s FX date, if Bloomberg's live dates are for it, else ``None``.

    They are while New York's calendar date is the FX date: a weekday before
    17:00 New York. After the cut or at a weekend the FX date is a later day
    than Bloomberg's today.
    """
    from apeiron.gaia.market_dates.rollover import ROLLOVER_TIMEZONE, _moment
    day = fx_date(moment, timezone)
    new_york = _moment(moment, timezone).tz_convert(ROLLOVER_TIMEZONE).date()
    return day if new_york == day else None


def live_dates(pairs=None, timezone=None, bdp=None, settlement_of=None, clock=datetime.now):
    """Fetch and check today's dates; refused (``LiveDates.refused``) outside the session."""
    from apeiron.gaia.market_dates.rollover import MACHINE_TIMEZONE
    timezone = timezone or MACHINE_TIMEZONE
    now = clock()
    today = in_session(now, timezone)
    if today is None:
        return LiveDates(fx_date(now, timezone), now, timezone,
                         refused=f"{now} ({timezone}) is after the 17:00 New York cut or at a weekend")
    if settlement_of is None:
        from helper_functions.fx_conventions import settlement as settlement_of
    if pairs is None:
        from market_data.universe import FX_UNIVERSE as pairs
    if bdp is None:
        from market_data.xbbg_compat import patch_xbbg
        patch_xbbg()
        from xbbg.blp import bdp
    result = check(fetch_raw(bdp, list(pairs)), today, settlement_of, now, timezone)
    result.fetched = True
    if in_session(clock(), timezone) != today:
        result.refused = "the 17:00 New York cut passed during the fetch"
    return result


SOURCE = "bloomberg"


def save(result, store):
    """Store ``result`` (``LiveDates``); returns the ``fetch_id``, or ``None`` when
    Bloomberg was never asked."""
    if not result.fetched:
        return None
    solver = {(pair, tenor): ours for pair, tenor, _, ours in result.disputes}
    rows = []
    for (pair, tenor), dates in result.dates.items():
        ours = solver.get((pair, tenor), dates)           # no dispute: the solver agreed
        rows.append((pair, tenor, dates.expiry, dates.delivery, result.sources[(pair, tenor)],
                     ours.expiry, ours.delivery, result.quoted.get((pair, tenor)) or ""))
    return store.save_pillar_dates(result.fx_date, result.fetched_at, result.timezone, SOURCE,
                                   rows, result.rejected, result.refused)


def report(result, out=print):
    out(f"FX date {result.fx_date}, fetched {result.fetched_at} ({result.timezone})")
    if result.refused:
        out(f"REFUSED: {result.refused}")
    out(f"accepted {len(result.dates)}, fell back {len(result.rejected)}, "
        f"disputes {len(result.disputes)}, OPT_EXPIRE_DT differs on {len(result.mismatches)}")
    for pair, tenor, reason in result.rejected:
        out(f"  fallback {pair} {tenor}: {reason}")
    for pair, tenor, bbg, ours in result.disputes:
        out(f"  dispute {pair} {tenor}: bloomberg {bbg.expiry} -> {bbg.delivery}, solver {ours.expiry} -> {ours.delivery}")
    for pair, tenor, quoted, used in result.mismatches:
        out(f"  cross-check {pair} {tenor}: OPT_EXPIRE_DT {quoted}, expiry used {used}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pairs", nargs="+")
    parser.add_argument("--store", action="store_true", help="store the dates in the market store")
    args = parser.parse_args(argv)
    result = live_dates(pairs=args.pairs)
    report(result)
    if args.store:
        from market_data.store import open_market_data_store
        store = open_market_data_store()
        fetch_id = save(result, store)
        print(f"stored as fetch {fetch_id} in {store.path}" if fetch_id else "nothing stored")
    return 1 if result.refused else 0


if __name__ == "__main__":
    sys.exit(main())

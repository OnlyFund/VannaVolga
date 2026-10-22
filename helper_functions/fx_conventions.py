"""FX market conventions for option pricing and curve construction.

Verify delta/ATM conventions against your market data provider before production use.

This version folds in the full pair coverage from `fx_conventions_full`
(all G10 crosses incl. Scandi/DKK, plus the USD- and EUR-based EM set).
Two conventions from that source were reconciled against the richer
PairConv model already in this file:

  * `delta_base` / `premium_ccy` in the source dict map directly onto
    `premium_adjusted` / `premium_ccy` here (premium quoted in the base
    currency <=> premium-adjusted delta).
  * Every pair, G10 and EM, reads its smile quotes in spot delta up to 1Y
    and forward delta beyond (the default `fwd_delta_beyond="1Y"`), as
    Bloomberg does (checked on USDZAR, 2026-10-19). The source's EM
    `delta_type: "forward"` at every tenor was modelled with
    `fwd_delta_beyond="0D"` until then. Hedges are booked as forwards
    regardless: the app sizes them as forward amounts.

Currencies/pairs added here that have no dedicated QuantLib calendar
(Colombia, Malaysia, Philippines, Vietnam) fall back to `ql.WeekendsOnly()`
-- supplement with an explicit holiday list before using these in
production. `deliverable` flags for the EM names are best-effort market
convention (most offshore EM pairs trade NDF) and should be confirmed
against your prime broker / venue documentation.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional
import QuantLib as ql

from apeiron.gaia.time import pillars
from apeiron.gaia.time.pillars import CALENDARS, Settlement


# ---------------------------------------------------------------- currencies

@dataclass(frozen=True)
class CcyConv:
    code: str
    day_count: ql.DayCounter
    calendar: ql.Calendar
    # rank orders the base currency in a cross: lower rank wins the base slot
    rank: int


def _cal(c):
    return c


CCY: Dict[str, CcyConv] = {
    # --- G10 -----------------------------------------------------------
    "EUR": CcyConv("EUR", ql.Actual360(),      CALENDARS["EUR"], rank=10),
    "GBP": CcyConv("GBP", ql.Actual365Fixed(), CALENDARS["GBP"], rank=20),
    "AUD": CcyConv("AUD", ql.Actual365Fixed(), CALENDARS["AUD"], rank=30),
    "NZD": CcyConv("NZD", ql.Actual365Fixed(), CALENDARS["NZD"], rank=40),
    "USD": CcyConv("USD", ql.Actual360(),      CALENDARS["USD"], rank=50),
    "CAD": CcyConv("CAD", ql.Actual365Fixed(), CALENDARS["CAD"], rank=60),
    "CHF": CcyConv("CHF", ql.Actual360(),      CALENDARS["CHF"], rank=70),
    "NOK": CcyConv("NOK", ql.Actual360(),      CALENDARS["NOK"], rank=80),
    "SEK": CcyConv("SEK", ql.Actual360(),      CALENDARS["SEK"], rank=90),
    "DKK": CcyConv("DKK", ql.Actual360(),      CALENDARS["DKK"], rank=95),
    "JPY": CcyConv("JPY", ql.Actual360(),      CALENDARS["JPY"], rank=100),

    # --- Asia G10-adjacent ----------------------------------------------
    "SGD": CcyConv("SGD", ql.Actual365Fixed(), CALENDARS["SGD"], rank=110),
    "HKD": CcyConv("HKD", ql.Actual365Fixed(), CALENDARS["HKD"], rank=120),

    # --- LatAm -----------------------------------------------------------
    "MXN": CcyConv("MXN", ql.Actual360(),      CALENDARS["MXN"], rank=130),
    "BRL": CcyConv("BRL", ql.Actual360(),      CALENDARS["BRL"], rank=190),
    "CLP": CcyConv("CLP", ql.Actual360(),      CALENDARS["CLP"], rank=200),
    "COP": CcyConv("COP", ql.Actual360(),      CALENDARS["COP"],  # no QuantLib Colombia calendar
                    rank=210),
    "ARS": CcyConv("ARS", ql.Actual360(),      CALENDARS["ARS"], rank=220),

    # --- EMEA --------------------------------------------------------
    "ZAR": CcyConv("ZAR", ql.Actual365Fixed(), CALENDARS["ZAR"], rank=140),
    "PLN": CcyConv("PLN", ql.Actual365Fixed(), CALENDARS["PLN"], rank=150),
    "CZK": CcyConv("CZK", ql.Actual360(),      CALENDARS["CZK"], rank=160),
    "HUF": CcyConv("HUF", ql.Actual360(),      CALENDARS["HUF"], rank=170),
    "TRY": CcyConv("TRY", ql.Actual360(),      CALENDARS["TRY"], rank=180),
    "ILS": CcyConv("ILS", ql.Actual365Fixed(), CALENDARS["ILS"], rank=230),

    # --- Asia EM -----------------------------------------------------
    "CNH": CcyConv("CNH", ql.Actual360(),      CALENDARS["CNH"],  # approximation; offshore
                    rank=240),                                          # CNH really trades HK+CN
    "KRW": CcyConv("KRW", ql.Actual365Fixed(), CALENDARS["KRW"], rank=250),
    "INR": CcyConv("INR", ql.Actual365Fixed(), CALENDARS["INR"], rank=260),
    "TWD": CcyConv("TWD", ql.Actual365Fixed(), CALENDARS["TWD"], rank=270),
    "MYR": CcyConv("MYR", ql.Actual365Fixed(), CALENDARS["MYR"],  # no QuantLib Malaysia calendar
                    rank=280),
    "THB": CcyConv("THB", ql.Actual365Fixed(), CALENDARS["THB"], rank=290),
    "IDR": CcyConv("IDR", ql.Actual365Fixed(), CALENDARS["IDR"], rank=300),
    "PHP": CcyConv("PHP", ql.Actual365Fixed(), CALENDARS["PHP"],  # no QuantLib Philippines calendar
                    rank=310),
    "VND": CcyConv("VND", ql.Actual365Fixed(), CALENDARS["VND"],  # no QuantLib Vietnam calendar
                    rank=320),
}


# ---------------------------------------------------------------- pairs

# quote currencies conventionally shown to 2dp -> forward points divide by 100;
# everything else is conventionally shown to 4dp -> forward points divide by 10,000
# Forward points are quoted to 2dp (divisor 100) against these; TRY points are
# quoted to 4dp like most pairs (Bloomberg USDTRY 3M ~39,000 on spot ~41).
_2DP_QUOTES = {"JPY", "KRW"}


@dataclass(frozen=True)
class PairConv:
    base: str
    quote: str
    spot_lag: int            # business days, trade -> spot
    pip: float               # F = S + points * pip
    premium_adjusted: bool   # delta is premium-adjusted
    premium_ccy: str         # currency premium is normally quoted in
    atm: str                 # "dn" (delta-neutral straddle) or "ATMF"
    fwd_delta_beyond: str = "1Y"   # spot delta up to here, forward delta after
    deliverable: bool = True
    fwd_divisor: Optional[int] = None  # forward pts -> price: F = S + points / fwd_divisor
                                        # auto-derived from quote ccy if not given explicitly

    def __post_init__(self):
        if self.fwd_divisor is None:
            object.__setattr__(
                self, "fwd_divisor",
                100 if self.quote in _2DP_QUOTES else 10_000,
            )

    @property
    def name(self) -> str:
        return self.base + self.quote


_P = PairConv

PAIRS: Dict[str, PairConv] = {p.name: p for p in [
    # ================= G10 (spot delta) =================================

    # --- USD quote (G10) : premium in USD, not premium-adjusted ---------
    _P("EUR", "USD", 2, 1e-4, False, "USD", "dn"),
    _P("GBP", "USD", 2, 1e-4, False, "USD", "dn"),
    _P("AUD", "USD", 2, 1e-4, False, "USD", "dn"),
    _P("NZD", "USD", 2, 1e-4, False, "USD", "dn"),

    # --- USD base (G10) : premium in USD -> premium-adjusted ------------
    _P("USD", "JPY", 2, 1e-2, True,  "USD", "dn"),
    _P("USD", "CHF", 2, 1e-4, True,  "USD", "dn"),
    _P("USD", "CAD", 1, 1e-4, True,  "USD", "dn"),   # T+1
    _P("USD", "NOK", 2, 1e-4, True,  "USD", "dn"),
    _P("USD", "SEK", 2, 1e-4, True,  "USD", "dn"),
    _P("USD", "DKK", 2, 1e-4, True,  "USD", "dn"),

    # --- EUR crosses ------------------------------------------------------
    _P("EUR", "GBP", 2, 1e-4, True,  "EUR", "dn"),
    _P("EUR", "AUD", 2, 1e-4, True,  "EUR", "dn"),
    _P("EUR", "NZD", 2, 1e-4, True,  "EUR", "dn"),
    _P("EUR", "CAD", 2, 1e-4, True,  "EUR", "dn"),
    _P("EUR", "CHF", 2, 1e-4, True,  "EUR", "dn"),
    _P("EUR", "NOK", 2, 1e-4, True,  "EUR", "dn"),
    _P("EUR", "SEK", 2, 1e-4, True,  "EUR", "dn"),
    _P("EUR", "DKK", 2, 1e-4, True,  "EUR", "dn"),
    _P("EUR", "JPY", 2, 1e-2, True,  "EUR", "dn"),
    _P("EUR", "PLN", 2, 1e-4, True,  "EUR", "dn"),
    _P("EUR", "CZK", 2, 1e-4, True,  "EUR", "dn", fwd_divisor=1_000),   # points to 3dp
    _P("EUR", "HUF", 2, 1e-2, True,  "EUR", "dn", fwd_divisor=100),     # points to 2dp

    # --- GBP crosses --------------------------------------------------
    _P("GBP", "AUD", 2, 1e-4, True,  "GBP", "dn"),
    _P("GBP", "NZD", 2, 1e-4, True,  "GBP", "dn"),
    _P("GBP", "CAD", 2, 1e-4, True,  "GBP", "dn"),
    _P("GBP", "CHF", 2, 1e-4, True,  "GBP", "dn"),
    _P("GBP", "NOK", 2, 1e-4, True,  "GBP", "dn"),
    _P("GBP", "SEK", 2, 1e-4, True,  "GBP", "dn"),
    _P("GBP", "DKK", 2, 1e-4, True,  "GBP", "dn"),
    _P("GBP", "JPY", 2, 1e-2, True,  "GBP", "dn"),

    # --- AUD crosses --------------------------------------------------
    _P("AUD", "NZD", 2, 1e-4, True,  "AUD", "dn"),
    _P("AUD", "CAD", 2, 1e-4, True,  "AUD", "dn"),
    _P("AUD", "CHF", 2, 1e-4, True,  "AUD", "dn"),
    _P("AUD", "NOK", 2, 1e-4, True,  "AUD", "dn"),
    _P("AUD", "SEK", 2, 1e-4, True,  "AUD", "dn"),
    _P("AUD", "DKK", 2, 1e-4, True,  "AUD", "dn"),
    _P("AUD", "JPY", 2, 1e-2, True,  "AUD", "dn"),

    # --- NZD crosses --------------------------------------------------
    _P("NZD", "CAD", 2, 1e-4, True,  "NZD", "dn"),
    _P("NZD", "CHF", 2, 1e-4, True,  "NZD", "dn"),
    _P("NZD", "NOK", 2, 1e-4, True,  "NZD", "dn"),
    _P("NZD", "SEK", 2, 1e-4, True,  "NZD", "dn"),
    _P("NZD", "DKK", 2, 1e-4, True,  "NZD", "dn"),
    _P("NZD", "JPY", 2, 1e-2, True,  "NZD", "dn"),

    # --- CAD / CHF / Scandi crosses ------------------------------------
    _P("CAD", "CHF", 2, 1e-4, True,  "CAD", "dn"),
    _P("CAD", "NOK", 2, 1e-4, True,  "CAD", "dn"),
    _P("CAD", "SEK", 2, 1e-4, True,  "CAD", "dn"),
    _P("CAD", "DKK", 2, 1e-4, True,  "CAD", "dn"),
    _P("CAD", "JPY", 2, 1e-2, True,  "CAD", "dn"),

    _P("CHF", "NOK", 2, 1e-4, True,  "CHF", "dn"),
    _P("CHF", "SEK", 2, 1e-4, True,  "CHF", "dn"),
    _P("CHF", "DKK", 2, 1e-4, True,  "CHF", "dn"),
    _P("CHF", "JPY", 2, 1e-2, True,  "CHF", "dn"),

    _P("NOK", "SEK", 2, 1e-4, True,  "NOK", "dn"),
    _P("NOK", "DKK", 2, 1e-4, True,  "NOK", "dn"),
    _P("NOK", "JPY", 2, 1e-2, True,  "NOK", "dn"),

    _P("SEK", "DKK", 2, 1e-4, True,  "SEK", "dn"),
    _P("SEK", "JPY", 2, 1e-2, True,  "SEK", "dn"),

    _P("DKK", "JPY", 2, 1e-2, True,  "DKK", "dn"),

    # ================= EM (spot delta up to 1Y, forward beyond) ==========

    # --- USD base -------------------------------------------------------
    _P("USD", "SGD", 2, 1e-4, True,  "USD", "dn"),
    _P("USD", "HKD", 2, 1e-4, True,  "USD", "dn"),
    _P("USD", "MXN", 2, 1e-4, True,  "USD", "dn"),
    _P("USD", "BRL", 2, 1e-4, True,  "USD", "dn", deliverable=False),
    _P("USD", "CLP", 2, 1,    True,  "USD", "dn", deliverable=False),
    _P("USD", "COP", 2, 1,    True,  "USD", "dn", deliverable=False),
    _P("USD", "ARS", 2, 1e-2, True,  "USD", "dn", deliverable=False),
    _P("USD", "TRY", 1, 1e-4, True,  "USD", "dn"),   # T+1
    _P("USD", "ZAR", 2, 1e-4, True,  "USD", "dn"),
    _P("USD", "PLN", 2, 1e-4, True,  "USD", "dn"),
    _P("USD", "HUF", 2, 1e-2, True,  "USD", "dn", fwd_divisor=100),
    _P("USD", "CZK", 2, 1e-4, True,  "USD", "dn", fwd_divisor=1_000),
    _P("USD", "ILS", 2, 1e-4, True,  "USD", "dn"),
    _P("USD", "CNH", 2, 1e-4, True,  "USD", "dn"),
    _P("USD", "KRW", 2, 1,    True,  "USD", "dn", deliverable=False),
    _P("USD", "INR", 2, 1e-2, True,  "USD", "dn", deliverable=False),
    _P("USD", "TWD", 2, 1e-3, True,  "USD", "dn", deliverable=False),
    _P("USD", "MYR", 2, 1e-4, True,  "USD", "dn", deliverable=False),
    _P("USD", "THB", 2, 1e-2, True,  "USD", "dn"),
    _P("USD", "IDR", 2, 1,    True,  "USD", "dn", deliverable=False),
    _P("USD", "PHP", 1, 1e-4, True,  "USD", "dn", deliverable=False),
    _P("USD", "VND", 2, 1,    True,  "USD", "dn", deliverable=False),

    # --- EUR base ---------------------------------------------------------
    _P("EUR", "BRL", 2, 1e-4, True,  "EUR", "dn", deliverable=False),
    _P("EUR", "MXN", 2, 1e-4, True,  "EUR", "dn"),
    _P("EUR", "CLP", 2, 1,    True,  "EUR", "dn", deliverable=False),
    _P("EUR", "COP", 2, 1,    True,  "EUR", "dn", deliverable=False),
    _P("EUR", "ARS", 2, 1e-2, True,  "EUR", "dn", deliverable=False),
    _P("EUR", "TRY", 1, 1e-4, True,  "EUR", "dn"),   # T+1
    _P("EUR", "ZAR", 2, 1e-4, True,  "EUR", "dn"),
    _P("EUR", "ILS", 2, 1e-4, True,  "EUR", "dn"),
    _P("EUR", "SGD", 2, 1e-4, True,  "EUR", "dn"),
    _P("EUR", "HKD", 2, 1e-4, True,  "EUR", "dn"),
    _P("EUR", "CNH", 2, 1e-4, True,  "EUR", "dn"),
    _P("EUR", "KRW", 2, 1,    True,  "EUR", "dn", deliverable=False),
    _P("EUR", "INR", 2, 1e-2, True,  "EUR", "dn", deliverable=False),
    _P("EUR", "TWD", 2, 1e-3, True,  "EUR", "dn", deliverable=False),
    _P("EUR", "MYR", 2, 1e-4, True,  "EUR", "dn", deliverable=False),
    _P("EUR", "THB", 2, 1e-2, True,  "EUR", "dn"),
    _P("EUR", "IDR", 2, 1,    True,  "EUR", "dn", deliverable=False),
    _P("EUR", "PHP", 1, 1e-4, True,  "EUR", "dn", deliverable=False),
    _P("EUR", "VND", 2, 1,    True,  "EUR", "dn", deliverable=False),
]}


# ---------------------------------------------------------------- lookup

def normalise(pair: str) -> str:
    return pair.replace("/", "").replace(" ", "").upper()


def get_pair(pair: str) -> PairConv:
    """Look up a pair, deriving an inverted-quote entry if needed."""
    key = normalise(pair)
    if key in PAIRS:
        return PAIRS[key]
    inv = key[3:] + key[:3]
    if inv in PAIRS:
        raise KeyError(
            f"{key} is quoted as {inv} in the market; "
            f"invert spot/strike rather than flipping the convention."
        )
    raise KeyError(f"unknown pair {key}")


def market_pair(ccy_a: str, ccy_b: str) -> str:
    """Return the market-conventional quote order for two currencies."""
    a, b = CCY[ccy_a], CCY[ccy_b]
    return (a.code + b.code) if a.rank < b.rank else (b.code + a.code)


def settlement(pair: str) -> Settlement:
    """What the date rules (``apeiron.gaia.time.pillars``) need about a pair."""
    p = get_pair(pair)
    return Settlement(p.base, p.quote, p.spot_lag)


def joint_calendar(pair: str) -> ql.Calendar:
    """Both currencies' calendar; USD holidays still block settlement on crosses."""
    return pillars.calendar(settlement(pair))


# ---------------------------------------------------------------- dates

@dataclass
class OptionDates:
    trade: ql.Date
    spot: ql.Date
    expiry: ql.Date
    delivery: ql.Date
    premium: ql.Date


def _ql_date(day) -> ql.Date:
    return ql.Date(day.day, day.month, day.year)


def spot_date(pair: str, trade: ql.Date) -> ql.Date:
    """Trade date -> spot date, honouring the pair's settlement lag.

    The rule lives in ``apeiron.gaia.time.pillars.spot_date``. Note: for USD
    crosses the strict market rule advances over non-USD holidays first and
    then checks USD as a final good-business-day test. This approximation uses
    the joint calendar throughout, which agrees in the overwhelming majority of
    cases but can differ around US-only holidays.
    """
    return _ql_date(pillars.spot_date(settlement(pair), trade))


def pillar_dates(pair: str, trade, tenor: str) -> "pillars.PillarDates":
    """Spot, expiry and delivery (``datetime.date``) of ``tenor``, O/N included."""
    return pillars.tenor_dates(settlement(pair), trade, tenor)


def option_dates(pair: str,
                 trade: ql.Date,
                 tenor: Optional[str] = None,
                 expiry: Optional[ql.Date] = None,
                 premium_at_spot: bool = True) -> OptionDates:
    """Build the full date set for an FX option.

    Tenors are spot-relative and end-of-month aware: expiry is derived by
    advancing spot by the tenor, then unwinding the settlement lag. The rules
    live in ``apeiron.gaia.time.pillars``.
    """
    rules = settlement(pair)
    spot = spot_date(pair, trade)

    if expiry is None:
        if tenor is None:
            raise ValueError("supply either tenor or expiry")
        if str(tenor).strip().upper() in pillars.OVERNIGHT:
            raise ValueError("an overnight tenor has no spot-relative option dates")
        dates = pillars.tenor_dates(rules, trade, tenor)
        expiry, delivery = _ql_date(dates.expiry), _ql_date(dates.delivery)
    else:
        expiry = _ql_date(pillars.roll_back(rules, expiry))
        delivery = _ql_date(pillars.delivery_for_expiry(rules, expiry))

    premium = spot if premium_at_spot else trade
    return OptionDates(trade, spot, expiry, delivery, premium)


# ---------------------------------------------------------------- deltas

def delta_type(pair: str, expiry: ql.Date, trade: ql.Date) -> int:
    """QuantLib BlackDeltaCalculator delta type for this pair and tenor."""
    p = get_pair(pair)
    cal = joint_calendar(pair)
    cutoff = cal.advance(spot_date(pair, trade), ql.Period(p.fwd_delta_beyond))
    if expiry <= cutoff:
        return (ql.DeltaVolQuote.PaSpot if p.premium_adjusted
                else ql.DeltaVolQuote.Spot)
    return (ql.DeltaVolQuote.PaFwd if p.premium_adjusted
            else ql.DeltaVolQuote.Fwd)


def atm_type(pair: str) -> int:
    return (ql.DeltaVolQuote.AtmDeltaNeutral
            if get_pair(pair).atm == "dn"
            else ql.DeltaVolQuote.AtmFwd)


# ---------------------------------------------------------------- example

if __name__ == "__main__":
    trade = ql.Date(25, 9, 2026)
    ql.Settings.instance().evaluationDate = trade

    for pair in ("EURUSD", "USDJPY", "USDCAD", "EURJPY",
                 "EURDKK", "GBPNOK", "USDMXN", "EURINR"):
        d = option_dates(pair, trade, "3M")
        p = get_pair(pair)
        print(f"{pair}  T+{p.spot_lag}  pip={p.pip:g}  pa={p.premium_adjusted}  "
              f"fwd_delta_beyond={p.fwd_delta_beyond}  fwd_divisor={p.fwd_divisor}")
        print(f"   spot={d.spot}  expiry={d.expiry}  deliv={d.delivery}")


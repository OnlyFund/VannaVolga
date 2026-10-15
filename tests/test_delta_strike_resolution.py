"""Regression tests for delta-quoted strike resolution.

A delta-quoted strike is defined by the volatility *at that strike*. The
resolver previously solved every delta alias with the ATM volatility, ignoring
the butterfly and risk reversal, which pulled the wings towards the money. For
a 9m EURUSD 10-delta call that placed the strike at 1.235 instead of 1.2431.
"""

import datetime

import pandas as pd
import pytest

from helper_functions.input_resolvers import resolve_strike
from helper_functions.vanna_volga_helpers import GetMainPillars
from market_data import MarketData


VALUATION = "2026-09-25"
EXPIRY = "2027-06-25"

QUOTES = dict(
    s=1.1386,
    fwd_pts=123.2,
    v_atm=6.34,
    rr_25=-0.512,
    rr_10=-0.968,
    bf_25=0.27,
    bf_10=1.023,
)


def _market():
    return MarketData(**QUOTES)


def _row(strike, call_put):
    return pd.Series(
        {
            "strike": strike,
            "call_put": call_put,
            "expiry": EXPIRY,
            "valuation": VALUATION,
        }
    )


def _resolve(strike, call_put):
    return resolve_strike(_market(), _row(strike, call_put), "dn", True)


def _pillars():
    p = GetMainPillars(
        k=1.24,
        expiry_date=EXPIRY,
        valuation_date=VALUATION,
        atm_conv="dn",
        deltaBase=True,
        forward_divisor=10000,
        term_discount=0.04257,
        smooth_pillar_interpolation=True,
        **QUOTES
    )
    strikes, vols = p.get_five_pillar_arrays()
    return list(strikes), list(vols)


@pytest.mark.parametrize(
    "alias,call_put,index",
    [
        ("10d", "PUT", 0),
        ("25d", "PUT", 1),
        ("25d", "CALL", 3),
        ("10d", "CALL", 4),
    ],
)
def test_delta_aliases_match_the_engine_pillar_strikes(alias, call_put, index):
    """The resolver and the smile engine must agree on the same strike."""
    strikes, _ = _pillars()
    assert _resolve(alias, call_put) == pytest.approx(strikes[index], abs=1e-6)


def test_ten_delta_call_is_not_solved_with_the_atm_vol():
    """The ATM-vol solve produced 1.235; the smile-consistent solve gives 1.2431."""
    resolved = _resolve("10d", "CALL")

    assert resolved == pytest.approx(1.243114, abs=1e-4)
    assert resolved > 1.2400, (
        "10-delta call strike collapsed towards the money: %s" % resolved
    )


def test_ten_delta_put_is_not_solved_with_the_atm_vol():
    resolved = _resolve("10d", "PUT")

    assert resolved == pytest.approx(1.056170, abs=1e-4)
    assert resolved < 1.0600, (
        "10-delta put strike collapsed towards the money: %s" % resolved
    )


def test_wing_strikes_straddle_the_atm_strike():
    atm = _resolve("atm", "CALL")
    ordered = [
        _resolve("10d", "PUT"),
        _resolve("25d", "PUT"),
        atm,
        _resolve("25d", "CALL"),
        _resolve("10d", "CALL"),
    ]
    assert ordered == sorted(ordered)


def test_numeric_and_atm_strike_inputs_still_resolve():
    assert _resolve("1.2431", "CALL") == pytest.approx(1.2431)
    assert _resolve("atmf", "CALL") == pytest.approx(1.1386 + 123.2 / 10000.0)


def test_invalid_delta_alias_is_rejected():
    with pytest.raises(ValueError):
        _resolve("75d", "CALL")


@pytest.mark.parametrize(
    "expiry,valuation",
    [
        (pd.Timestamp(EXPIRY), pd.Timestamp(VALUATION)),
        (datetime.datetime(2027, 6, 25), datetime.datetime(2026, 9, 25)),
        (datetime.date(2027, 6, 25), datetime.date(2026, 9, 25)),
        (EXPIRY, VALUATION),
        (
            pd.Timestamp(EXPIRY).to_datetime64(),
            pd.Timestamp(VALUATION).to_datetime64(),
        ),
    ],
)
@pytest.mark.parametrize("alias", ["10d", "atm"])
def test_date_inputs_normalize_before_resolution(expiry, valuation, alias):
    """Midnight timestamps must not keep a time component.

    ``pd.Timestamp`` and ``datetime`` isoformat to ``YYYY-MM-DDTHH:MM:SS``,
    which the ``%Y-%m-%d`` year-fraction parser rejects. The app swallows the
    resulting error and silently leaves the alias unresolved.
    """
    row = pd.Series(
        {
            "strike": alias,
            "call_put": "CALL",
            "expiry": expiry,
            "valuation": valuation,
        }
    )
    resolved = resolve_strike(_market(), row, "dn", True)
    expected = 1.243114 if alias == "10d" else 1.149191
    assert resolved == pytest.approx(expected, abs=1e-5)


def test_unusable_smile_raises_instead_of_returning_the_atm_strike(monkeypatch):
    """A first-iteration failure must not masquerade as a resolved strike.

    Returning the ATM-vol solve here looks like a working resolution while
    silently reintroducing the exact bug the fixed point exists to fix.
    """
    import helper_functions.input_resolvers as resolvers
    import VannaVolga_ImpliedVol

    class Broken(VannaVolga_ImpliedVol.VannaVolga):
        def get_vol_for_strike(self, *args, **kwargs):
            raise RuntimeError("smile unavailable")

    monkeypatch.setattr(VannaVolga_ImpliedVol, "VannaVolga", Broken)

    with pytest.raises(ValueError) as excinfo:
        _resolve("10d", "CALL")

    assert "smile" in str(excinfo.value).lower()


@pytest.mark.parametrize(
    "text",
    [
        "03/24/2027", "3/24/2027", "03-24-2027", "03/24/27",
        "2027-03-24", "2027/03/24",
        "24mar27", "24MAR27", "24mar2027",
        "24-mar-27", "24-Mar-2027",
        "24 mar 27", "24 Mar 2027",
        "mar 24 2027", "Mar 24, 2027",
        "24/03/2027", "24.03.2027",
    ],
)
def test_written_date_forms_all_parse_to_the_same_expiry(text):
    """The expiry cell is free text, so accept the natural written forms.

    Month names, ISO order, hyphens, dots and two-digit years are all things a
    user reasonably types; previously only mm/dd/yyyy and ddmonyy worked and
    everything else fell back to the tenor-only error.
    """
    from helper_functions.input_resolvers import parse_expiry

    assert parse_expiry(text, datetime.date(2026, 9, 25)) == datetime.date(2027, 3, 24)


def test_ambiguous_numeric_dates_resolve_month_first():
    """03/04/2027 must be 4 March, matching the advertised mm/dd/yyyy order."""
    from helper_functions.input_resolvers import parse_expiry

    assert parse_expiry("03/04/2027", datetime.date(2026, 9, 25)) == datetime.date(
        2027, 3, 4
    )


@pytest.mark.parametrize("text", ["garbage", "13/45/2027", ""])
def test_unparseable_expiries_still_raise(text):
    from helper_functions.input_resolvers import parse_expiry

    with pytest.raises(ValueError):
        parse_expiry(text, datetime.date(2026, 9, 25))


def test_tenors_still_resolve_against_pillars():
    from helper_functions.input_resolvers import parse_expiry

    valuation = datetime.date(2026, 9, 25)
    assert parse_expiry("6m", valuation) == datetime.date(2027, 3, 24)
    assert parse_expiry("2w", valuation) == datetime.date(2026, 10, 9)
    with pytest.raises(ValueError):
        parse_expiry("7m", valuation, pillar_tenors=["1M", "6M", "1Y"])

# --- ATM strike: premium adjustment / convention consistency -----------------

def test_dn_strike_premium_adjustment_matches_closed_form():
    """Unadjusted DN solves d1=0 (K=F*exp(+s^2T/2)); premium-adjusted solves d2=0."""
    from math import exp
    from helper_functions.VannaVolga_function import get_atm_strike, get_dn_strike
    from helper_functions.helper_functions import year_fraction

    s, fwd, v = 1.1386, 84.0, 5.92
    t = year_fraction(EXPIRY, VALUATION)
    f = s + fwd / 10000.0
    sig = v / 100.0

    unadj = get_atm_strike(atm_type="dn", S=s, fwd=fwd, v=v,
                           expiry_date=EXPIRY, valuation_date=VALUATION)
    pa = get_dn_strike(S=s, fwd=fwd, v=v, expiry_date=EXPIRY,
                       valuation_date=VALUATION, deltaBase=True)

    assert unadj == pytest.approx(f * exp(0.5 * sig ** 2 * t), rel=1e-12)
    assert pa == pytest.approx(f * exp(-0.5 * sig ** 2 * t), rel=1e-12)
    # The premium-adjusted strike is lower, by F * sigma^2 * T to first order.
    assert unadj - pa == pytest.approx(f * sig ** 2 * t, rel=1e-3)


@pytest.mark.parametrize("delta_base", [True, False])
@pytest.mark.parametrize("atm_conv", ["dn", "DN", "atmf", "spot"])
def test_atm_pillar_matches_resolved_atm_strike(delta_base, atm_conv):
    """The ATM smile pillar must be the same strike the input resolver returns."""
    from helper_functions.vanna_volga_helpers import calculate_strikes
    from helper_functions.VannaVolga_function import get_atm_strike, get_dn_strike
    from helper_functions.helper_functions import year_fraction

    s, fwd, v = 1.1386, 84.0, 5.92
    t = year_fraction(EXPIRY, VALUATION)
    f = s + fwd / 10000.0

    k2 = calculate_strikes(s, f, fwd, EXPIRY, VALUATION, t, 7.0, v, 6.2,
                           {"RR": -0.518, "BF": 0.235, "delta": 0.10},
                           delta_base, atm_conv, 10000.0)["K2"]

    if delta_base and atm_conv.lower() == "dn":
        expected = get_dn_strike(S=s, fwd=fwd, v=v, expiry_date=EXPIRY,
                                 valuation_date=VALUATION, deltaBase=True)
    else:
        expected = get_atm_strike(atm_type=atm_conv, S=s, fwd=fwd, v=v,
                                  expiry_date=EXPIRY, valuation_date=VALUATION)

    assert k2 == pytest.approx(expected, rel=1e-12)


def test_atm_pillar_honours_atm_conv_when_unadjusted():
    """Regression: the unadjusted branch used to return the DN strike for every
    ATM convention, silently ignoring 'atmf' and 'spot'."""
    from helper_functions.vanna_volga_helpers import calculate_strikes
    from helper_functions.helper_functions import year_fraction

    s, fwd, v = 1.1386, 84.0, 5.92
    t = year_fraction(EXPIRY, VALUATION)
    f = s + fwd / 10000.0
    rr_bf = {"RR": -0.518, "BF": 0.235, "delta": 0.10}

    def k2(conv):
        return calculate_strikes(s, f, fwd, EXPIRY, VALUATION, t, 7.0, v, 6.2,
                                 rr_bf, False, conv, 10000.0)["K2"]

    assert k2("atmf") == pytest.approx(f, rel=1e-12)
    assert k2("spot") == pytest.approx(s, rel=1e-12)
    assert k2("dn") > f

@pytest.mark.parametrize("atm_conv", ["atmf", "spot"])
def test_get_atm_strike_delta_base_only_affects_dn(atm_conv):
    from helper_functions.VannaVolga_function import get_atm_strike

    kwargs = dict(S=1.1386, fwd=84.0, v=5.92,
                  expiry_date=EXPIRY, valuation_date=VALUATION)
    assert (get_atm_strike(atm_type=atm_conv, delta_base=True, **kwargs)
            == get_atm_strike(atm_type=atm_conv, delta_base=False, **kwargs))


def test_get_atm_strike_delta_base_flag_matches_get_dn_strike():
    """get_atm_strike exposes the delta convention directly; get_dn_strike is
    a thin wrapper over it so the two can never drift apart."""
    from helper_functions.VannaVolga_function import get_atm_strike, get_dn_strike

    for delta_base in (True, False):
        assert get_atm_strike(atm_type="dn", S=1.1386, fwd=84.0, v=5.92,
                              expiry_date=EXPIRY, valuation_date=VALUATION,
                              delta_base=delta_base) == pytest.approx(
            get_dn_strike(S=1.1386, fwd=84.0, v=5.92, expiry_date=EXPIRY,
                          valuation_date=VALUATION, deltaBase=delta_base),
            rel=1e-15)


def test_get_atm_strike_defaults_to_unadjusted():
    """The default must stay unadjusted so positional callers are unaffected."""
    from helper_functions.VannaVolga_function import get_atm_strike

    kwargs = dict(atm_type="dn", S=1.1386, fwd=84.0, v=5.92,
                  expiry_date=EXPIRY, valuation_date=VALUATION)
    assert get_atm_strike(**kwargs) == get_atm_strike(delta_base=False, **kwargs)

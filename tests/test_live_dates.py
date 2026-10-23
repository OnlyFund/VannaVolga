"""Today's Bloomberg dates: the fail-safes, without Bloomberg."""
from datetime import date, datetime

import pandas as pd
import pytest

from helper_functions.fx_conventions import settlement
from market_data import live_dates as live

DAY = "2026-10-23"           # a Friday; EURUSD spot Tue 27 Oct
TENORS = ("ON", "1W", "1M", "2M")


def _raw(**changes):
    """EURUSD answers equal to the solver's, with ``changes`` per tenor."""
    raw = {"spot": "2026-10-27",
           "ON": {"expiry": "2026-10-26", "forward": ""},
           "1W": {"expiry": "2026-10-30", "forward": ""},
           "1M": {"expiry": "2026-11-25", "forward": "2026-11-27"},
           "2M": {"expiry": "2026-12-23", "forward": "2026-12-28"}}
    for tenor, value in changes.items():
        if tenor == "spot":
            raw["spot"] = value
        else:
            raw[tenor] = dict(raw[tenor], **value)
    return {"EURUSD": raw}


def _check(raw):
    return live.check(raw, DAY, settlement, tenors=TENORS)


def test_agreeing_answers_are_accepted_without_disputes():
    result = _check(_raw())
    assert len(result.dates) == 4 and not result.rejected and not result.disputes
    assert result.dates[("EURUSD", "1M")].delivery == date(2026, 11, 27)
    assert set(result.sources.values()) == {"bloomberg"}


def test_bloomberg_wins_and_the_difference_is_a_dispute():
    result = _check(_raw(**{"1M": {"expiry": "2026-11-24", "forward": "2026-11-27"}}))
    dates = result.dates[("EURUSD", "1M")]
    assert (dates.expiry, dates.delivery) == (date(2026, 11, 24), date(2026, 11, 27))
    assert [(p, t) for p, t, _, _ in result.disputes] == [("EURUSD", "1M")]


def test_week_delivery_is_the_expirys_spot_date_and_a_missing_forward_is_partial():
    result = _check(_raw(**{"1W": {"expiry": "2026-11-02"}, "2M": {"forward": ""}}))
    assert result.dates[("EURUSD", "1W")].delivery == date(2026, 11, 4)
    assert result.sources[("EURUSD", "2M")] == "partial"
    assert result.dates[("EURUSD", "2M")].delivery == date(2026, 12, 28)


@pytest.mark.parametrize("change,reason", [
    ({"1M": {"expiry": ""}}, "no expiry"),
    ({"1M": {"expiry": "soon"}}, "unreadable"),
    ({"1M": {"expiry": "2026-11-28"}}, "no weekday"),              # a Saturday
    ({"ON": {"expiry": "2026-10-23"}}, "no weekday after"),        # not after the FX date
    ({"1M": {"expiry": "2026-12-09"}}, "days from the solver"),
    ({"1M": {"forward": "2026-11-20"}}, "not after expiry"),
])
def test_a_doubtful_tenor_falls_back_alone(change, reason):
    result = _check(_raw(**change))
    (pair, tenor, why), = result.rejected
    assert reason in why and (pair, tenor) == ("EURUSD", list(change)[0])
    assert len(result.dates) == 3 and ("EURUSD", tenor) not in result.dates


@pytest.mark.parametrize("change,reason", [
    ({"spot": "2026-10-23"}, "stale"),
    ({"2M": {"expiry": "2026-11-24", "forward": "2026-11-26"}, "1M": {"expiry": "2026-11-25", "forward": "2026-11-27"}},
     "do not rise"),
])
def test_a_doubtful_pair_falls_back_whole(change, reason, monkeypatch):
    monkeypatch.setitem(live.TOLERANCE_DAYS, "long", 40)          # let the 2M through to the pair check
    result = _check(_raw(**change))
    assert not result.dates and result.rejected[-1][1] == "*" and reason in result.rejected[-1][2]


def test_too_many_disputes_refuse_the_fetch(monkeypatch):
    monkeypatch.setattr(live, "MAX_DISPUTED", 0.2)
    result = _check(_raw(**{"1M": {"expiry": "2026-11-24"}}))      # 1 of 4 disagree
    assert result.refused and "disagree" in result.refused


def test_unknown_pairs_fall_back():
    result = live.check({"ABCXYZ": {}}, DAY, settlement, tenors=TENORS)
    assert result.rejected[0][:2] == ("ABCXYZ", "*")


@pytest.mark.parametrize("moment,expected", [
    ("2026-10-23 07:44", date(2026, 10, 23)),      # London morning
    ("2026-10-23 21:59", date(2026, 10, 23)),      # 16:59 New York
    ("2026-10-23 22:00", None),                    # 17:00 New York: Monday's FX date
    ("2026-10-24 10:00", None),                    # Saturday
])
def test_the_session(moment, expected):
    assert live.in_session(pd.Timestamp(moment).to_pydatetime(), "Europe/London") == expected


def _fake_bdp(answers):
    def bdp(tickers, flds):
        column = flds.lower()
        rows = {t: answers[t] for t in tickers if t in answers}
        return pd.DataFrame({column: list(rows.values())}, index=list(rows))
    return bdp


def test_fetch_and_check_with_a_fake_bloomberg(monkeypatch):
    monkeypatch.setattr(live, "TENORS", TENORS)
    answers = {"EURUSDVON Curncy": "2026-10-26", "EURUSDV1W Curncy": "2026-10-30",
               "EURUSDV1M Curncy": "2026-11-25", "EURUSDV2M Curncy": "2026-12-23",
               "EURUSD Curncy": "2026-10-27",
               "EUR1M Curncy": "2026-11-27", "EUR2M Curncy": "2026-12-28"}
    raw = live.fetch_raw(_fake_bdp(answers), ["EURUSD"], TENORS,
                         forward_ticker=lambda pair, tenor: f"EUR{tenor} Curncy")
    assert raw["EURUSD"]["1M"] == {"expiry": "2026-11-25", "forward": "2026-11-27"}
    clock = iter([datetime(2026, 10, 23, 8), datetime(2026, 10, 23, 8, 5)])
    result = live.live_dates(["EURUSD"], "Europe/London", _fake_bdp(answers), settlement,
                             clock=lambda: next(clock))
    assert not result.refused and len(result.dates) == 4 and not result.disputes and not result.rejected


def test_the_cut_passing_mid_fetch_refuses(monkeypatch):
    monkeypatch.setattr(live, "TENORS", TENORS)
    clock = iter([datetime(2026, 10, 23, 21, 55), datetime(2026, 10, 23, 22, 1)])
    result = live.live_dates(["EURUSD"], "Europe/London", _fake_bdp({}), settlement, clock=lambda: next(clock))
    assert "during the fetch" in result.refused


def test_a_failing_request_is_retried_ticker_by_ticker():
    def bdp(tickers, flds):
        if len(tickers) > 1:
            raise RuntimeError("bad ticker in batch")
        if tickers[0] == "BAD Curncy":
            raise RuntimeError("bad ticker")
        return pd.DataFrame({flds.lower(): ["2026-10-26"]}, index=tickers)
    assert live._ask(bdp, ["A Curncy", "BAD Curncy"], "OPT_EXPIRE_DT") == {"A Curncy": "2026-10-26"}

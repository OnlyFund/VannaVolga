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
    result = _check(_raw(**{"1W": {"expiry": "2026-11-02"}, "2M": {"forward": "2026-12-29"}}))
    week, two = result.dates[("EURUSD", "1W")], result.dates[("EURUSD", "2M")]
    assert (week.expiry, week.delivery) == (date(2026, 11, 2), date(2026, 11, 4))   # spot of expiry
    assert (two.expiry, two.delivery) == (date(2026, 12, 24), date(2026, 12, 29))   # from the forward
    assert [(p, t) for p, t, _, _ in result.disputes] == [("EURUSD", "1W"), ("EURUSD", "2M")]


def test_a_month_expiry_comes_from_the_forward_date_and_opt_expire_dt_is_a_cross_check():
    """NZDUSD 1M on 23 Oct 2026: OPT_EXPIRE_DT 25 Nov, forward 30 Nov -> 26 Nov (the pricing page)."""
    result = _check(_raw(**{"1M": {"expiry": "2026-11-24"}}))
    dates = result.dates[("EURUSD", "1M")]
    assert (dates.expiry, dates.delivery) == (date(2026, 11, 25), date(2026, 11, 27))
    assert not result.disputes
    assert result.mismatches == [("EURUSD", "1M", date(2026, 11, 24), date(2026, 11, 25))]
    assert result.quoted[("EURUSD", "1M")] == date(2026, 11, 24)
    nzd = live.check({"NZDUSD": {"spot": "2026-10-28",
                                 "1M": {"expiry": "2026-11-25", "forward": "2026-11-30"}}},
                     DAY, settlement, tenors=("1M",))
    assert nzd.dates[("NZDUSD", "1M")].expiry == date(2026, 11, 26)


def test_a_month_tenor_needs_no_opt_expire_dt():
    result = _check(_raw(**{"2M": {"expiry": ""}}))
    assert result.dates[("EURUSD", "2M")].expiry == date(2026, 12, 23)
    assert result.quoted[("EURUSD", "2M")] is None and not result.mismatches


@pytest.mark.parametrize("change,reason", [
    ({"1W": {"expiry": ""}}, "no expiry"),
    ({"1W": {"expiry": "soon"}}, "unreadable"),
    ({"1W": {"expiry": "2026-10-31"}}, "no weekday"),              # a Saturday
    ({"ON": {"expiry": "2026-10-23"}}, "no weekday after"),        # not after the FX date
    ({"1W": {"expiry": "2026-11-06"}}, "days from the solver"),
    ({"1M": {"forward": ""}}, "no forward settlement date"),
    ({"1M": {"forward": "soon"}}, "unreadable forward"),
    ({"1M": {"forward": "2026-12-10"}}, "days from the solver"),
    ({"1M": {"expiry": "Nov 25"}}, "unreadable expiry"),
])
def test_a_doubtful_tenor_falls_back_alone(change, reason):
    result = _check(_raw(**change))
    (pair, tenor, why), = result.rejected
    assert reason in why and (pair, tenor) == ("EURUSD", list(change)[0])
    assert len(result.dates) == 3 and ("EURUSD", tenor) not in result.dates


@pytest.mark.parametrize("change,reason", [
    ({"spot": "2026-10-23"}, "stale"),
    ({"2M": {"forward": "2026-11-26"}}, "do not rise"),             # 2M before 1M
])
def test_a_doubtful_pair_falls_back_whole(change, reason, monkeypatch):
    monkeypatch.setitem(live.TOLERANCE_DAYS, "long", 40)          # let the 2M through to the pair check
    result = _check(_raw(**change))
    assert not result.dates and result.rejected[-1][1] == "*" and reason in result.rejected[-1][2]


def test_too_many_disputes_refuse_the_fetch(monkeypatch):
    monkeypatch.setattr(live, "MAX_DISPUTED", 0.2)
    result = _check(_raw(**{"1W": {"expiry": "2026-11-02"}}))      # 1 of 4 disagree
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


def test_save_stores_dates_solver_answers_and_fallbacks(tmp_path, monkeypatch):
    from market_data.store import MarketDataStore
    monkeypatch.setattr(live, "MAX_DISPUTED", 0.5)            # 1 of 3 disagree
    result = _check(_raw(**{"1W": {"expiry": "2026-11-02"}, "1M": {"expiry": "2026-11-24"},
                            "2M": {"forward": ""}}))
    result.fetched = True
    store = MarketDataStore(tmp_path / "market_data.sqlite")
    fetch_id = live.save(result, store)
    day = store.load_pillar_dates(DAY)
    assert day.fetch_id == fetch_id and len(day.dates) == 3
    assert day.dates[("EURUSD", "1W")] == ("2026-11-02", "2026-11-04", "bloomberg")
    assert day.fallbacks == [("EURUSD", "2M", "no forward settlement date from Bloomberg")]
    disputes = store.pillar_date_disputes(DAY)
    assert list(zip(disputes.tenor, disputes.solver_expiry)) == [("1W", "2026-10-30")]
    mismatches = store.pillar_expiry_mismatches(DAY)
    assert list(zip(mismatches.tenor, mismatches.quoted_expiry, mismatches.expiry)) == [
        ("1M", "2026-11-24", "2026-11-25")]


def test_nothing_is_stored_when_bloomberg_was_never_asked(tmp_path):
    from market_data.store import MarketDataStore
    store = MarketDataStore(tmp_path / "market_data.sqlite")
    result = live.live_dates(["EURUSD"], "Europe/London", _fake_bdp({}), settlement,
                             clock=lambda: datetime(2026, 10, 24, 10))     # Saturday
    assert result.refused and live.save(result, store) is None


def test_a_fetch_refused_after_asking_is_recorded_but_never_used(tmp_path, monkeypatch):
    from market_data.store import MarketDataStore
    monkeypatch.setattr(live, "MAX_DISPUTED", 0.2)
    result = _check(_raw(**{"1W": {"expiry": "2026-11-02"}}))
    result.fetched = True
    store = MarketDataStore(tmp_path / "market_data.sqlite")
    fetch_id = live.save(result, store)
    assert store.load_pillar_dates(DAY) is None
    assert "disagree" in store.load_pillar_dates(fetch_id=fetch_id).refused

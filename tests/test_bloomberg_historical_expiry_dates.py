"""Historical expiry requests without requiring Bloomberg connectivity."""
from datetime import date
import sys
from types import ModuleType

import pandas as pd
import pytest

from market_data.bloomberg_historical_expiry_dates import fetch_historical_expiry_dates


@pytest.fixture
def requests(monkeypatch):
    calls = []

    def bdp(**kwargs):
        calls.append(kwargs)
        reference = pd.Timestamp(kwargs["REFERENCE_DATE"])
        days = 7 if kwargs["VOL_SURF_MTY_OVR"] == "1W" else 365
        return pd.DataFrame(
            {"vol_surf_expiry_ovr": [reference + pd.Timedelta(days=days)] * len(kwargs["tickers"])},
            index=kwargs["tickers"],
        )

    package = ModuleType("xbbg")
    module = ModuleType("xbbg.blp")
    module.bdp = bdp
    package.blp = module
    monkeypatch.setitem(sys.modules, "xbbg", package)
    monkeypatch.setitem(sys.modules, "xbbg.blp", module)
    return calls, module


def test_batches_by_date_and_tenor_and_normalizes_pairs(requests):
    calls, _ = requests
    result = fetch_historical_expiry_dates(
        ["20240112", date(2026, 1, 14), "2024-01-12"],
        ["eur/usd", "USDIDR", "EURUSD"], ["1w", "1Y", "1W"])
    assert len(calls) == 4
    assert len(result) == 8
    assert list(result.columns) == ["reference_date", "currency_pair", "tenor", "expiry_date"]
    assert calls[0] == dict(tickers=["EURUSD BGN Curncy", "USDIDR BGN Curncy"],
                           flds="VOL_SURF_EXPIRY_OVR", VOL_SURF_MTY_OVR="1W",
                           REFERENCE_DATE="20240112")
    assert calls[-1]["REFERENCE_DATE"] == "20260114"
    assert result.iloc[0].expiry_date == date(2024, 1, 19)
    assert result.iloc[0].reference_date == date(2024, 1, 12)


def test_overnight_is_requested_as_one_day(requests):
    calls, _ = requests
    result = fetch_historical_expiry_dates("20250117", "EURUSD", ["ON", "on"])
    assert [call["VOL_SURF_MTY_OVR"] for call in calls] == ["1D"]
    assert list(result.tenor) == ["ON"]


def test_scalar_arguments(requests):
    result = fetch_historical_expiry_dates("20240112", "EURUSD", "1W")
    assert len(result) == 1


@pytest.mark.parametrize("dates,pairs,tenors", [
    ([], ["EURUSD"], ["1W"]), (["20240112"], [], ["1W"]),
    (["20240112"], ["EURUSD"], []), (["NaT"], ["EURUSD"], ["1W"]),
    ([20240112], ["EURUSD"], ["1W"]), (["20240112"], ["USD"], ["1W"]),
    (["20240112"], ["USDUSD"], ["1W"]), (["20240112"], ["EURUSD"], ["bad"]),
])
def test_invalid_inputs_do_not_request_bloomberg(requests, dates, pairs, tenors):
    calls, _ = requests
    with pytest.raises(ValueError):
        fetch_historical_expiry_dates(dates, pairs, tenors)
    assert calls == []


@pytest.mark.parametrize("response", [
    pd.DataFrame(),
    pd.DataFrame({"vol_surf_expiry_ovr": [None]}, index=["EURUSD BGN Curncy"]),
    pd.DataFrame({"vol_surf_expiry_ovr": ["2023-01-01"]}, index=["EURUSD BGN Curncy"]),
])
def test_missing_or_invalid_expiry_is_an_error(requests, response):
    _, module = requests
    module.bdp = lambda **kwargs: response
    with pytest.raises(ValueError, match="EURUSD 1W as of 2024-01-12"):
        fetch_historical_expiry_dates("20240112", "EURUSD", "1W")

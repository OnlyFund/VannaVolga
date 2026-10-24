"""The Greek memo: extra Greeks computed when their column is added, and kept.

A line priced again with another set of extras reuses its unit price and
base Greeks and computes only the extras not computed before; the numbers
are exactly those of a fresh pricing.
"""
import pandas as pd
import pytest

from app import pricer
from app.market_context import MarketContext
from golden.app_harness import load_markets
from market_data import Modifications
from pricing import greeks as greek_module

pytestmark = pytest.mark.filterwarnings("ignore")

MODS = Modifications(beta_atm=0.7, beta_rr=1.2, beta_bf=0.0)
LINES = [dict(currency_pair="EURUSD", strike=strike, expiry=expiry, call_put=call_put,
              buy_sell=side, notional="1m", premium=0.0, premium_ccy="USD")
         for strike, expiry, call_put, side in [("atm", "1M", "CALL", "BUY"),
                                                ("25d", "3M", "PUT", "SELL")]]
DECAYS = ("delta_decay", "gamma_decay")


@pytest.fixture
def context():
    return MarketContext.from_payload(load_markets("MARKETS_20261015.json")["EURUSD"])


@pytest.fixture(autouse=True)
def empty_memo():
    pricer._GREEK_MEMO.clear()
    yield
    pricer._GREEK_MEMO.clear()


def price(context, extras=()):
    results, errors = pricer.price_lines(context, pd.DataFrame(LINES), MODS, extras=extras)
    assert not errors
    return results


def fresh(context, extras=()):
    with pricer.bypass_greek_memo():
        return price(context, extras)


def watch(monkeypatch):
    """The extras each ``calculate_extra_greeks`` call is asked for, and the
    line's own base Greek computations (not those a decay makes on the next
    day's market inside an extras call)."""
    asked, base, inside = [], [], []
    extra, greeks = greek_module.calculate_extra_greeks, greek_module.calculate_greeks

    def counted_extra(market, row, names, *args, **kwargs):
        asked.append(tuple(names))
        inside.append(1)
        try:
            return extra(market, row, names, *args, **kwargs)
        finally:
            inside.pop()

    def counted_base(*args, **kwargs):
        if not inside:
            base.append(1)
        return greeks(*args, **kwargs)
    monkeypatch.setattr(greek_module, "calculate_extra_greeks", counted_extra)
    monkeypatch.setattr(greek_module, "calculate_greeks", counted_base)
    return asked, base


def test_adding_a_column_computes_only_its_greeks(context, monkeypatch):
    price(context, ("rho",))
    asked, base = watch(monkeypatch)
    with_decays = price(context, ("rho", *DECAYS))
    assert base == [], "base Greeks recomputed"
    assert asked and all(set(call) == set(DECAYS) for call in asked)
    assert with_decays == fresh(context, ("rho", *DECAYS))


def test_removing_and_readding_a_column_computes_nothing(context, monkeypatch):
    price(context, ("rho", *DECAYS))
    asked, base = watch(monkeypatch)
    assert price(context, ("rho",)) == fresh(context, ("rho",))
    asked.clear(), base.clear()
    again = price(context, ("rho", *DECAYS))
    assert asked == [] and base == []
    assert again == fresh(context, ("rho", *DECAYS))


def test_a_changed_market_never_reads_an_old_value(context, monkeypatch):
    price(context, ("rho",))
    moved = context.with_spot(context.spot * 1.01)
    asked, base = watch(monkeypatch)
    result = price(moved, ("rho",))
    assert base, "the moved market's Greeks were read from the memo"
    assert result == fresh(moved, ("rho",))


def test_changed_betas_never_read_an_old_value(context, monkeypatch):
    price(context, ("rho",))
    asked, base = watch(monkeypatch)
    pricer.price_lines(context, pd.DataFrame(LINES), Modifications(beta_atm=0.5, beta_rr=1.2,
                                                                   beta_bf=0.0), extras=("rho",))
    assert base


def test_the_off_switch_keeps_nothing(context, monkeypatch):
    monkeypatch.setenv("VV_MARKET_CACHE", "0")
    price(context, ("rho",))
    assert pricer.greek_memo() is None and len(pricer._GREEK_MEMO._entries) == 0


def test_a_check_against_fresh_values_bypasses_the_memo(context, monkeypatch):
    price(context, ("rho",))
    asked, base = watch(monkeypatch)
    fresh(context, ("rho",))
    assert base and asked


def test_the_memo_is_bounded():
    memo = pricer.GreekMemo(maxsize=2)
    for key in range(5):
        memo.put(key, key)
    assert len(memo._entries) == 2 and memo.get(4) == 4 and memo.get(0) is None


def test_extras_that_do_not_split_by_name_are_not_kept():
    from pricing.pricing import _memo_extras
    memo = pricer.GreekMemo()

    def compute(names):
        return {"rho": 1.0, "fixed_rho": 2.0, "shared": 3.0}, {}
    values, _ = _memo_extras(memo, "base", ("rho",), compute)
    assert values == {"rho": 1.0, "fixed_rho": 2.0, "shared": 3.0}
    assert memo.get(("base", "extra", "rho")) is None


def test_an_extra_that_fails_keeps_its_reason_and_the_others_stand():
    from pricing.pricing import _memo_extras
    memo = pricer.GreekMemo()

    def compute(names):
        return ({"rho": 1.0, "fixed_rho": 2.0, "phi": None, "fixed_phi": None},
                {"phi": "no foreign curve"})
    first = _memo_extras(memo, "base", ("rho", "phi"), compute)
    again = _memo_extras(memo, "base", ("rho", "phi"), lambda names: pytest.fail("recomputed"))
    assert first == again
    assert again[1] == {"phi": "no foreign curve"} and again[0]["rho"] == 1.0

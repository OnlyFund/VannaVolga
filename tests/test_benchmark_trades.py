"""The Bloomberg benchmark's trades are fixed by their seed (tests/benchmark)."""
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "benchmark" / "bloomberg_vols.py"


def _module():
    spec = importlib.util.spec_from_file_location("bloomberg_vols", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_trades_are_the_same_every_run_and_match_the_file():
    module = _module()
    trades = module.generate()
    assert trades == module.generate() and len(trades) == module.COUNT
    stored = module.read()
    assert [row["case_id"] for row in stored] == [trade["case_id"] for trade in trades]
    for row, trade in zip(stored, trades):
        assert {key: row[key] for key in trade} == {key: str(value) for key, value in trade.items()}


def test_dates_excel_rewrote_are_read_back(tmp_path):
    module = _module()
    path = tmp_path / "trades.csv"
    row = {column: "" for column in module.COLUMNS}
    row.update(case_id="B99", expiry_input="7/15/2027", expiry="2027-07-15")
    module.write([row], path)
    assert module.read(path)[0]["expiry_input"] == "2027-07-15"


def test_the_compare_takes_signs_as_entered_and_flags_the_other_way(tmp_path):
    module = _module()
    path = tmp_path / "trades.csv"
    base = {column: "" for column in module.COLUMNS}
    sold = dict(base, case_id="B98", pair="EURUSD", buy_sell="SELL", strike_input="atm", expiry_input="1M",
                vol="7.0", premium_pct="-1.20", bbg_vol="7.1", bbg_premium_pct="-1.25", bbg_delta_pct="-50.5")
    flipped = dict(sold, case_id="B99", bbg_premium_pct="1.25", bbg_delta_pct="50.5")
    module.write([sold, flipped], path)
    lines = []
    diffs = module.compare(path, out=lines.append, deltas={"B98": (-50.0, -49.0), "B99": (-50.0, -49.0)})
    for case, *_, vol, premium, delta in diffs:
        assert round(vol, 6) == -0.1 and round(premium, 6) == 0.05 and round(delta, 6) == 0.5, case
    assert "other way" in lines[1] and "other way" not in lines[0]

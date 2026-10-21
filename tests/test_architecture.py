"""Apeiron's level rules hold: every library module has a level, and no
import crosses the rules except the known crossings, which may only shrink.

``apeiron/known_crossings.json`` lists today's crossings (the code that still
has to be untangled). A NEW crossing fails: place the code in the right level
instead. A crossing that is GONE also fails until it is removed from the list,
so the list always shows exactly what is left to untangle.
"""
import json
from pathlib import Path

from apeiron.layers import ALLOWED, BASE, CONSUMERS, level_of, may_import
from apeiron.scan import scan

ROOT = Path(__file__).resolve().parents[1]
KNOWN = ROOT / "apeiron" / "known_crossings.json"


def test_every_library_module_has_a_level():
    _, _, unplaced = scan(ROOT)
    assert unplaced == [], f"place these in apeiron/layers.py: {unplaced}"


def test_no_new_crossings_and_the_known_ones_only_shrink():
    _, crossings, _ = scan(ROOT)
    known = json.loads(KNOWN.read_text(encoding="utf-8"))
    new = sorted(set(crossings) - set(known))
    gone = sorted(set(known) - set(crossings))
    assert new == [], f"imports breaking the level rules (fix, don't list): {new}"
    assert gone == [], f"untangled - remove from apeiron/known_crossings.json: {gone}"


def test_the_rules_are_the_agreed_ones():
    # Atlas and each part of Gaia import nothing else in the suite.
    for level in BASE:
        assert ALLOWED[level] == set()
    assert ALLOWED["hyperion"] == set(BASE)
    # Consumers may use the levels below, never each other.
    for consumer in CONSUMERS:
        assert not any(may_import(consumer, other) for other in CONSUMERS if other != consumer)
        assert all(may_import(consumer, level) for level in (*BASE, "hyperion"))
    assert not may_import("gaia.time", "gaia.market")
    assert not may_import("hyperion", "olympus")


def test_new_apeiron_folders_take_their_level_from_their_path():
    assert level_of("apeiron.hyperion.cache.store") == "hyperion"
    assert level_of("apeiron.gaia.time.pillars") == "gaia.time"
    assert level_of("apeiron.atlas.black_scholes") == "atlas"
    assert level_of("research.anything") == "delphi"


def test_the_scanner_catches_a_crossing(tmp_path):
    # An Atlas module importing the UI (even inside a function) is caught.
    for package in ("apeiron", "apeiron/atlas", "apeiron/olympus"):
        (tmp_path / package).mkdir(parents=True, exist_ok=True)
        (tmp_path / package / "__init__.py").write_text("")
    (tmp_path / "apeiron/olympus/page.py").write_text("X = 1\n")
    (tmp_path / "apeiron/atlas/engine.py").write_text(
        "def f():\n    from apeiron.olympus.page import X\n    return X\n")
    levels, crossings, unplaced = scan(tmp_path)
    assert crossings == ["apeiron.atlas.engine -> apeiron.olympus.page"]
    assert levels["apeiron.atlas.engine"] == "atlas" and unplaced == []


def test_no_new_import_cycles_and_the_known_ones_only_shrink():
    # apeiron/known_cycles.json: load-time cycles and cycles through deferred
    # (in-function) imports that exist today; they work, but each is a knot
    # to untangle. A new one fails; a gone one must be removed from the list.
    from apeiron.scan import cycles
    known = json.loads((ROOT / "apeiron" / "known_cycles.json").read_text(encoding="utf-8"))
    for key, module_level_only in (("module_level", True), ("any_depth", False)):
        found = cycles(ROOT, module_level_only)
        new = [group for group in found if group not in known[key]]
        gone = [group for group in known[key] if group not in found]
        assert new == [], f"new import cycle(s) ({key}): {new}"
        assert gone == [], f"untangled - remove from apeiron/known_cycles.json ({key}): {gone}"

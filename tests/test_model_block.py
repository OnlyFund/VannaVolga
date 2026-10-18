"""The Model tab's settings: same keys as before, defaults for anything invalid."""
import pytest

from app.ui import model_block
from app.ui.model_block import DEFAULTS, model_settings, render_model_tab


def test_defaults_are_the_former_sidebar_defaults():
    assert DEFAULTS == {"pricing_model": "vv", "beta_atm": 0.7, "beta_rr": 1.2, "beta_bf": 0.0}
    assert model_settings({}) == DEFAULTS


def test_stored_values_are_used():
    state = {"pricing_model": "vv-sabr", "beta_atm": 0.8, "beta_rr": 1.0, "beta_bf": 0.25}
    assert model_settings(state) == state


@pytest.mark.parametrize("key, bad", [
    ("pricing_model", "black"), ("pricing_model", None), ("beta_atm", float("nan")),
    ("beta_atm", 7.0), ("beta_rr", -0.1), ("beta_bf", "0.1"), ("beta_bf", True),
])
def test_invalid_values_fall_back_to_their_default(key, bad):
    state = {"pricing_model": "vv-sabr", "beta_atm": 0.8, "beta_rr": 1.0, "beta_bf": 0.25}
    state[key] = bad
    settings = model_settings(state)
    assert settings[key] == DEFAULTS[key]
    assert all(settings[k] == v for k, v in state.items() if k != key)


def test_a_failing_tab_still_returns_settings(monkeypatch):
    class Broken:
        def __enter__(self):
            raise RuntimeError("no tab")

        def __exit__(self, *args):
            return False
    state = {"pricing_model": "vv-sabr", "beta_atm": 0.8}
    warnings = []
    monkeypatch.setattr(model_block.st, "session_state", state)
    monkeypatch.setattr(model_block.st, "warning", warnings.append)
    settings = render_model_tab(Broken())
    assert settings == dict(DEFAULTS, pricing_model="vv-sabr", beta_atm=0.8)
    assert warnings and "no tab" in warnings[0]


def test_an_invalid_stored_value_is_reset_before_its_widget_draws(monkeypatch):
    state = {"pricing_model": "black", "beta_atm": 9.0, "beta_rr": 1.0}
    monkeypatch.setattr(model_block.st, "session_state", state)
    model_block._reset_invalid()
    assert state == {"pricing_model": "vv", "beta_atm": 0.7, "beta_rr": 1.0}

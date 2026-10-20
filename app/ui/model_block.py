"""The Model tab: pricing model and smile dynamics, shared by every tab.

These are global settings (they apply to every pair and every report), so
they live in their own tab rather than the sidebar. The widgets keep the
session keys they always had (``pricing_model``, ``beta_atm``, ``beta_rr``,
``beta_bf``), so moving them changes no value. A stored value a widget
cannot take is reset to its default first; should the tab fail to draw,
pricing goes on with the stored values (or the defaults) and says so.
"""
from math import isfinite

import streamlit as st

PRICING_MODELS = ("vv", "vv-sabr")
#: Slider range and step per beta: (minimum, maximum, step).
BETA_RANGES = {"beta_atm": (0.0, 5.0, 0.05), "beta_rr": (0.0, 2.0, 0.05),
               "beta_bf": (0.0, 2.0, 0.05)}
DEFAULTS = {"pricing_model": "vv", "beta_atm": 0.7, "beta_rr": 0.5, "beta_bf": 0.0}
#: Settings the app fixes (shown, not editable).
FIXED_SETTINGS = (
    ("ATM convention", "Delta-neutral (dn)"),
    ("Delta base", "Premium adjustment per pair convention"),
    ("Pillar interpolation", "Smooth 5-pillar"),
    ("Delta reporting", "Spot-equivalent totals; each line in its pair's convention"),
)


def _valid(key, value):
    if key == "pricing_model":
        return value in PRICING_MODELS
    low, high, _ = BETA_RANGES[key]
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and isfinite(value) and low <= value <= high)


def model_settings(state=None):
    """The model settings from ``state`` (the session by default); a missing
    or invalid value gives its default."""
    state = st.session_state if state is None else state
    settings = {}
    for key, default in DEFAULTS.items():
        try:
            value = state.get(key, default)
        except Exception:
            value = default
        settings[key] = value if _valid(key, value) else default
    settings["beta_atm"], settings["beta_rr"], settings["beta_bf"] = (
        float(settings[k]) for k in ("beta_atm", "beta_rr", "beta_bf"))
    return settings


def _reset_invalid():
    for key, default in DEFAULTS.items():
        if key in st.session_state and not _valid(key, st.session_state[key]):
            st.session_state[key] = default


def render_model_tab(tab):
    """Draw the Model tab; returns the model settings (see ``model_settings``)."""
    try:
        _reset_invalid()
        with tab:
            st.subheader("Model")
            st.caption("Global settings: they apply to every pair, the Pricer, the "
                       "Portfolio and the Risk reports.")
            st.markdown("**Pricing**")
            st.selectbox("Pricing model", list(PRICING_MODELS), key="pricing_model",
                         help="vv: Vanna-Volga. vv-sabr: Vanna-Volga with a SABR smile.")
            st.markdown("**Smile dynamics**")
            st.caption("How the smile moves with spot for the smile Greeks and the spot "
                       "ladder. Betas 0/0/0 are sticky delta (the smile Greeks equal the "
                       "sticky-delta ones); 1/1/1 is close to sticky strike (the vol at "
                       "a fixed strike barely moves, as in the Black-Scholes Greeks). "
                       "Beta ATM 1 alone is not sticky strike: the ATM follows the skew "
                       "but the wings keep their spread to it.")
            for key, (low, high, step) in BETA_RANGES.items():
                st.slider(key, low, high, DEFAULTS[key], step, key=key)
            st.markdown("**Fixed by the app**")
            for name, value in FIXED_SETTINGS:
                st.caption(f"{name}: {value}")
    except Exception as exc:
        try:
            st.warning(f"Model settings could not be shown ({exc}); pricing uses the "
                       "last settings or the defaults.")
        except Exception:
            pass
    return model_settings()

"""The Greek picker shared by the Pricer, Portfolio and Risk tabs.

Offers every Greek in ``pricing.greek_catalog`` grouped as first order,
decay, cross and weighted. The choice is kept per tab for the session;
names no longer in the catalogue are dropped rather than failing. An empty
choice (Streamlit's multiselect clears on Escape) shows the default Greeks
rather than none.
"""
import streamlit as st

from pricing import greek_catalog as catalog


def _option_label(name):
    greek = catalog.CATALOG[name]
    return greek.label if greek.group == "first order" else f"{greek.label} · {greek.group}"


def chosen_greeks(key, default=catalog.BASE_GREEKS):
    """The Greeks chosen in the picker ``key`` (catalogue order); ``default``
    when nothing is chosen."""
    return catalog.known(st.session_state.get(key) or default)


def render_greek_picker(key, default=catalog.BASE_GREEKS, label="Greeks"):
    """A multiselect of every Greek; returns the chosen names in catalogue order."""
    if key in st.session_state:
        # A stale saved choice (a Greek renamed or removed) must not break the widget.
        st.session_state[key] = catalog.known(st.session_state[key])
    else:
        st.session_state[key] = catalog.known(default)
    chosen = st.multiselect(
        label, list(catalog.CATALOG), key=key, format_func=_option_label,
        help="Each Greek shows as a sticky-delta column (smile quotes unchanged as spot "
             "moves) and a smile column (smile moved by the betas). Decay is the "
             "change to the next business day; gamma per vol is gamma's change per vol "
             "point; weighted Greeks are scaled by 0.25/T^0.4. Extras are computed "
             "when you price.")
    if not chosen:
        st.caption("No Greeks chosen: showing " + ", ".join(
            catalog.CATALOG[name].label for name in catalog.known(default)) + ".")
    return catalog.known(chosen or default)


def missing_extras(chosen, results):
    """Chosen extra Greeks that ``results`` (priced lines) do not carry yet."""
    if not results:
        return []
    return [name for name in catalog.extras(chosen) if name not in results[0]]

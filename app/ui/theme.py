"""The app's look: the Bloomberg_Data app's font (Inter), a title that sits
near the top of the page and larger tab labels.

The web font is loaded from Google Fonts; without network access the
browser falls back to the next font in ``FONT_STACK`` (Arial ships with
Windows). The data grids are drawn on a canvas and keep Streamlit's own
sans-serif font.
"""
import streamlit as st

FONT_URL = ("https://fonts.googleapis.com/css2?"
            "family=Inter:wght@300;400;500;600;700&display=swap")
FONT_STACK = '"Inter", "Helvetica Neue", Arial, sans-serif'

CSS = """
<style>
    @import url('%(font_url)s');

    html, body, [class*="css"],
    [data-testid="stAppViewContainer"] *,
    section[data-testid="stSidebar"] * {
        font-family: %(font)s !important;
    }
    /* Streamlit's icon font must keep its own family. */
    [data-testid="stAppViewContainer"] .material-icons,
    [data-testid="stAppViewContainer"] [class*="material-symbols"] {
        font-family: "Material Icons", "Material Symbols Rounded" !important;
    }

    /* Title near the top: Streamlit's default leaves about 2rem above it. */
    /* .block-container: Streamlit 1.23; the test ids: newer releases. */
    [data-testid="stAppViewContainer"] .block-container,
    [data-testid="stAppViewBlockContainer"],
    [data-testid="stMainBlockContainer"] {
        padding-top: 1.5rem !important;
    }
    /* Newer releases (the online app) fix a 2rem header over the page:
       start the title below it. 1.23's container has no test id. */
    [data-testid="stAppViewContainer"] [data-testid="stAppViewBlockContainer"],
    [data-testid="stAppViewContainer"] [data-testid="stMainBlockContainer"] {
        padding-top: 3rem !important;
    }
    [data-testid="stAppViewContainer"] h1 {
        padding-top: 0 !important;
        padding-bottom: 0.25rem !important;
    }

    [data-testid="stAppViewContainer"] p,
    [data-testid="stAppViewContainer"] label,
    [data-testid="stAppViewContainer"] input,
    [data-testid="stAppViewContainer"] button {
        font-size: 13px !important;
    }

    /* Tab labels larger than body text. */
    [data-testid="stAppViewContainer"] button[data-baseweb="tab"],
    [data-testid="stAppViewContainer"] button[data-baseweb="tab"] p {
        font-size: 17px !important;
        font-weight: 600 !important;
    }
    [data-testid="stAppViewContainer"] button[data-baseweb="tab"] {
        padding: 0.35rem 1rem !important;
    }

    [data-testid="stDataEditor"] div[role="gridcell"],
    [data-testid="stDataEditor"] div[role="columnheader"],
    [data-testid="stDataFrame"] div[role="gridcell"],
    [data-testid="stDataFrame"] div[role="columnheader"] {
        font-size: 12px !important;
    }
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div {
        width: 260px !important;
        min-width: 260px !important;
        max-width: 260px !important;
    }
    section[data-testid="stSidebar"] * {
        font-size: 12px !important;
    }
    /* Sidebar content starts under its close button, not 6rem down. */
    section[data-testid="stSidebar"] div:has(> .block-container),
    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
        padding-top: 2.5rem !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 0 !important;
    }
</style>
""" % {"font_url": FONT_URL, "font": FONT_STACK}


def apply_theme():
    """Inject the app's CSS; call once per run, right after set_page_config."""
    st.markdown(CSS, unsafe_allow_html=True)

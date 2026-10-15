"""Streamlit application entry point.

Run from the project root with:

    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app.ui import render_app

st.set_page_config(page_title="FXO Risk Analyzer", layout="wide")
st.markdown(
    """
    <style>
        [data-testid="stAppViewContainer"] p,
        [data-testid="stAppViewContainer"] label,
        [data-testid="stAppViewContainer"] input,
        [data-testid="stAppViewContainer"] button {
            font-size: 13px !important;
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
            font-size: 13px !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
render_app()

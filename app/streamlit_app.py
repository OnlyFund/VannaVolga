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
from app.ui.theme import apply_theme

st.set_page_config(page_title="FXO Risk Analyzer", layout="wide")
apply_theme()
render_app()

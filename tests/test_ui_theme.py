"""The app's CSS: well formed, with an offline font fallback."""
from pathlib import Path

from app.ui import theme


def test_css_braces_balance():
    assert theme.CSS.count("{") == theme.CSS.count("}")


def test_font_stack_falls_back_to_installed_font():
    fonts = [name.strip().strip('"') for name in theme.FONT_STACK.split(",")]
    assert fonts[0] == "Inter"
    assert "Arial" in fonts and fonts[-1] == "sans-serif"


def test_css_covers_title_tabs_and_font():
    assert "padding-top: 1.5rem" in theme.CSS
    assert 'button[data-baseweb="tab"]' in theme.CSS
    assert theme.FONT_STACK in theme.CSS and theme.FONT_URL in theme.CSS


def test_entry_point_applies_theme():
    source = (Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py").read_text()
    assert source.index("set_page_config") < source.index("apply_theme()") < source.index(
        "render_app()")

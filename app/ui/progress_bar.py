"""A Streamlit progress bar to pass as a calculation's ``progress`` callback.

Use it as a context manager inside the tab (or container) it belongs to::

    with ProgressBar("Pricing selection") as bar:
        service.price_trades(..., progress=bar)

It shows "Pricing selection: 12 / 40 · TR-000012" as the calculation calls
it, and disappears when the block ends, also on an error. Drawing never
breaks the calculation (``app.progress.safe``).
"""
import streamlit as st


class ProgressBar:
    def __init__(self, title):
        self.title = title
        self.slot = None

    def __enter__(self):
        self.slot = st.empty()
        try:
            self.slot.progress(0.0, text=f"{self.title}...")
        except Exception:
            pass
        return self

    def __call__(self, done, total, label=""):
        if self.slot is None:
            return
        try:
            fraction = min(max(float(done) / float(total), 0.0), 1.0) if total else 1.0
            text = f"{self.title}: {done} / {total}" + (f" · {label}" if label else "")
            self.slot.progress(fraction, text=text)
        except Exception:
            pass

    def __exit__(self, *exc):
        try:
            self.slot.empty()
        except Exception:
            pass
        return False

"""Progress reporting for long calculations, free of any UI.

A calculation takes an optional ``progress(done, total, label)`` callback
and calls it as it goes; the app passes one that draws a progress bar, a
service or a test may pass anything else, or nothing. ``safe`` wraps a
callback so a failing one (a closed browser tab, a bad display) never stops
the calculation it reports on.

``reporting`` / ``current`` carry a callback through code that does not
pass it explicitly (the spot ladder's scenario runner). The value lives in a
context variable: each Streamlit session runs in its own thread, so one
session's bar never shows another's progress. Worker threads do not call it;
results are gathered, and progress reported, on the calling thread.
"""
from contextlib import contextmanager
from contextvars import ContextVar

_CURRENT = ContextVar("vv_progress", default=None)


def safe(progress):
    """``progress`` that swallows its own errors; a no-op for ``None``."""
    if progress is None:
        return lambda done, total, label="": None

    def report(done, total, label=""):
        try:
            progress(done, total, label)
        except Exception:
            pass                       # reporting never stops the work
    return report


@contextmanager
def reporting(progress):
    """Make ``progress`` the current callback inside the block."""
    token = _CURRENT.set(progress)
    try:
        yield
    finally:
        _CURRENT.reset(token)


def current():
    """The current callback (``reporting``), made safe."""
    return safe(_CURRENT.get())

"""Shared test set-up."""
import pytest


@pytest.fixture(autouse=True)
def _fresh_greek_memo():
    """Every test starts with an empty Greek memo (``app.pricer.GreekMemo``):
    a test that patches a Greek calculator must see it run, not a value kept
    by an earlier test."""
    try:
        from app import pricer
    except Exception:                     # a test that cannot import the app
        yield
        return
    pricer._GREEK_MEMO.clear()
    yield
    pricer._GREEK_MEMO.clear()

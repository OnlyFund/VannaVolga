"""xbbg 0.7 keeps one Bloomberg session and its opened services on blpapi 3.2x."""
import pytest

from market_data.xbbg_compat import patch_xbbg


class _Service:
    def __init__(self, name):
        self.name = name

    def _handle(self):
        return self.name


class _Session:
    """New-style blpapi: the handle is behind ``_handle()``; services must be opened."""

    def __init__(self):
        self.opened, self.handle = set(), object()

    def _handle(self):
        return self.handle

    def openService(self, name):
        self.opened.add(name)

    def getService(self, name):
        return _Service(name) if name in self.opened else _Service(None)


@pytest.fixture
def conn(monkeypatch):
    pytest.importorskip("xbbg")
    from xbbg.core import conn
    for name in [n for n in vars(conn) if n.startswith(conn._CON_SYM_)]:
        monkeypatch.delitem(vars(conn), name)
    for name in ("bbg_session", "bbg_service", "_vv_session_reuse"):
        if hasattr(conn, name):
            monkeypatch.setattr(conn, name, getattr(conn, name))
    started = []
    monkeypatch.setattr(conn, "connect_bbg", lambda **kw: started.append(_Session()) or started[-1])
    monkeypatch.setattr(conn, "_vv_session_reuse", False, raising=False)
    patch_xbbg()
    return conn, started


def test_one_session_serves_every_call_and_the_service_it_opened(conn):
    conn, started = conn
    assert conn.bbg_session() is conn.bbg_session()
    service = conn.bbg_service("//blp/refdata")
    assert service._handle() == "//blp/refdata"
    assert conn.bbg_service("//blp/refdata") is service and len(started) == 1


def test_a_dead_session_is_replaced(conn):
    conn, started = conn
    first = conn.bbg_session()
    first.handle = None
    assert conn.bbg_session() is not first and len(started) == 2


def test_patching_twice_keeps_the_first_patch(conn):
    conn, _ = conn
    session_fn = conn.bbg_session
    patch_xbbg()
    assert conn.bbg_session is session_fn

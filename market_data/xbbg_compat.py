"""Keep xbbg 0.7's Bloomberg session and services cached on newer blpapi.

xbbg 0.7.7 checks that its cached session and services are still alive by
reading blpapi's private ``_Session__handle`` / ``_Service__handle``. blpapi
3.2x keeps the handle behind a ``_handle()`` method instead, so the check
always fails: every call starts a new session, and a service opened on one
session is fetched from another, where it was never opened
(``InvalidArgumentException: Null service handle``).

``patch_xbbg()`` swaps xbbg's ``bbg_session`` / ``bbg_service`` for versions
that ask either form. The cache stays in xbbg's own module globals, so its
``send_request`` can still drop a broken session. Call it before the first
request; calling it again does nothing, and without xbbg it does nothing.
"""


def _alive(obj, legacy_attr):
    handle = getattr(obj, "_handle", None)
    if callable(handle):
        return handle() is not None
    return getattr(obj, legacy_attr, None) is not None


def patch_xbbg():
    try:
        from xbbg.core import conn
    except ImportError:
        return
    if getattr(conn, "_vv_session_reuse", False):
        return
    cache = vars(conn)

    def bbg_session(**kwargs):
        key = f"{conn._CON_SYM_}//{kwargs.get('port', conn._PORT_)}"
        if key in cache and not _alive(cache[key], "_Session__handle"):
            del cache[key]
        if key not in cache:
            cache[key] = conn.connect_bbg(**kwargs)
        return cache[key]

    def bbg_service(service, **kwargs):
        key = f"{conn._CON_SYM_}/{kwargs.get('port', conn._PORT_)}{service}"
        if key in cache and not _alive(cache[key], "_Service__handle"):
            del cache[key]
        if key not in cache:
            session = bbg_session(**kwargs)
            session.openService(service)
            cache[key] = session.getService(service)
        return cache[key]

    conn.bbg_session = bbg_session
    conn.bbg_service = bbg_service
    conn._vv_session_reuse = True

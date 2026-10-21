"""Apeiron - FX option and volatility analytics: research, pricing, risk and strategy.

Six levels (``apeiron.layers`` holds the map and the rules; ``tests/test_architecture.py``
enforces them):

* ``atlas``    - core numerical engine, imports nothing else in the suite;
* ``gaia``     - data foundation: ``time``, ``market``, ``book`` (each independent);
* ``hyperion`` - analytics, risk, services and the result cache;
* ``athena``   - strategy, optimizer, backtest;
* ``delphi``   - research;
* ``olympus``  - the Streamlit user interface.

The library is moving here one level at a time; until then the modules live
in their old places, each already assigned its level in ``apeiron.layers``.
"""

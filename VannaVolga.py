"""Compatibility layer for the legacy Vanna-Volga import path.

This module keeps older imports such as:
    from VannaVolga import GetStrikeFromDelta, GetStrikeFromDeltaPA
working while the implementation now lives under helper_functions/.
"""

from helper_functions.VannaVolga_function import (
    get_strike_from_delta as GetStrikeFromDelta,
    get_strike_from_delta_pa as GetStrikeFromDeltaPA,
)

__all__ = ["GetStrikeFromDelta", "GetStrikeFromDeltaPA"]



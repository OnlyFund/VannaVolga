from datetime import datetime
from functools import lru_cache


# Helper functions
@lru_cache(maxsize=4096)
def year_fraction(expiry_date : str, value_date : str,annual_factor=365.0) -> float:
    """Return year fraction between two YYYY-MM-DD dates."""
    d2 = datetime.strptime(expiry_date, '%Y-%m-%d')
    d1 = datetime.strptime(value_date, '%Y-%m-%d')
    delta = (d2 - d1).days
    if delta < 0:
        raise ValueError("Expiry date must be after value date")
    return delta / annual_factor

def cp_flag(CallPut):
    """Map CALL/PUT to ±1."""
    return {'CALL': 1, 'PUT': -1}[CallPut.upper()]

def pos_flag(BuySell):
    """Map BUY/SELL to ±1."""
    return {'BUY': 1, 'SELL': -1}[BuySell.upper()]


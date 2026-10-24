from xbbg import blp

from market_data.xbbg_compat import patch_xbbg

patch_xbbg()

holidays = blp.bds(
    "USD Curncy",
    "CALENDAR_NON_SETTLEMENT_DATES",
    SETTLEMENT_CALENDAR_CODE="FD",
    CALENDAR_START_DATE="20250101",
    CALENDAR_END_DATE="20261231",
)

print(holidays)
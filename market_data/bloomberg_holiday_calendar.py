from xbbg import blp

holidays = blp.bds(
    "USD Curncy",
    "CALENDAR_NON_SETTLEMENT_DATES",
    SETTLEMENT_CALENDAR_CODE="FD",
    CALENDAR_START_DATE="20250101",
    CALENDAR_END_DATE="20261231",
)

print(holidays)
from datetime import date, datetime
from zoneinfo import ZoneInfo

# NBA schedule dates are US Eastern; servers run in UTC.
ET = ZoneInfo("America/New_York")


def today_et() -> date:
    return datetime.now(ET).date()

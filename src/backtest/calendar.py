"""NYSE trading calendar for 2024–2027.

Provides trading-day-aware date arithmetic for backtesting.
Weekdays that fall on NYSE holidays are treated as non-trading days.
"""
from datetime import datetime, date, timedelta

# NYSE holidays 2024–2027 (observed dates — if holiday falls on
# Saturday it is observed Friday; if Sunday, observed Monday).
_NYSE_HOLIDAYS = {
    # 2024
    "2024-01-01",  # New Year's Day
    "2024-01-15",  # Martin Luther King Jr. Day
    "2024-02-19",  # Presidents' Day
    "2024-03-29",  # Good Friday
    "2024-05-27",  # Memorial Day
    "2024-06-19",  # Juneteenth
    "2024-07-04",  # Independence Day
    "2024-09-02",  # Labor Day
    "2024-11-28",  # Thanksgiving
    "2024-12-25",  # Christmas
    # 2025
    "2025-01-01",  # New Year's Day
    "2025-01-20",  # Martin Luther King Jr. Day
    "2025-02-17",  # Presidents' Day
    "2025-04-18",  # Good Friday
    "2025-05-26",  # Memorial Day
    "2025-06-19",  # Juneteenth
    "2025-07-04",  # Independence Day
    "2025-09-01",  # Labor Day
    "2025-11-27",  # Thanksgiving
    "2025-12-25",  # Christmas
    # 2026
    "2026-01-01",  # New Year's Day
    "2026-01-19",  # Martin Luther King Jr. Day
    "2026-02-16",  # Presidents' Day
    "2026-04-03",  # Good Friday
    "2026-05-25",  # Memorial Day
    "2026-06-19",  # Juneteenth
    "2026-07-03",  # Independence Day (observed — Jul 4 is Saturday)
    "2026-09-07",  # Labor Day
    "2026-11-26",  # Thanksgiving
    "2026-12-25",  # Christmas
    # 2027
    "2027-01-01",  # New Year's Day
    "2027-01-18",  # Martin Luther King Jr. Day
    "2027-02-15",  # Presidents' Day
    "2027-03-26",  # Good Friday
    "2027-05-31",  # Memorial Day
    "2027-06-18",  # Juneteenth (observed — Jun 19 is Saturday)
    "2027-07-05",  # Independence Day (observed — Jul 4 is Sunday)
    "2027-09-06",  # Labor Day
    "2027-11-25",  # Thanksgiving
    "2027-12-24",  # Christmas (observed — Dec 25 is Saturday)
}


def _to_date_str(d: str | datetime | date) -> str:
    """Normalize input to YYYY-MM-DD string."""
    if isinstance(d, datetime):
        return d.strftime("%Y-%m-%d")
    if isinstance(d, date):
        return d.isoformat()
    return d


def is_trading_day(d: str | datetime | date) -> bool:
    """Return True if ``d`` is a NYSE trading day (Mon–Fri, not a holiday)."""
    ds = _to_date_str(d)
    dt = datetime.strptime(ds, "%Y-%m-%d")
    if dt.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False
    if ds in _NYSE_HOLIDAYS:
        return False
    return True


def next_trading_day(d: str | datetime | date) -> str:
    """First trading day strictly after ``d``."""
    ds = _to_date_str(d)
    dt = datetime.strptime(ds, "%Y-%m-%d")
    while True:
        dt += timedelta(days=1)
        candidate = dt.strftime("%Y-%m-%d")
        if is_trading_day(candidate):
            return candidate


def trading_days_between(start: str, end: str) -> list[str]:
    """All trading days in [start, end] inclusive."""
    result = []
    dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    while dt <= end_dt:
        ds = dt.strftime("%Y-%m-%d")
        if is_trading_day(ds):
            result.append(ds)
        dt += timedelta(days=1)
    return result


def n_trading_days_later(d: str | datetime | date, n: int) -> str:
    """Date ``n`` trading days after ``d`` (0 = same day if trading day)."""
    ds = _to_date_str(d)
    dt = datetime.strptime(ds, "%Y-%m-%d")
    count = 0
    while count < n:
        dt += timedelta(days=1)
        if is_trading_day(dt.strftime("%Y-%m-%d")):
            count += 1
    return dt.strftime("%Y-%m-%d")

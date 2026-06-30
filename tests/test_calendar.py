"""Tests for NYSE trading calendar."""
import pytest
from datetime import datetime, date
from src.backtest.calendar import (
    is_trading_day,
    next_trading_day,
    trading_days_between,
    n_trading_days_later,
)


class TestIsTradingDay:
    def test_weekend_returns_false(self):
        """Saturday and Sunday are not trading days."""
        # 2024-06-15 is Saturday, 2024-06-16 is Sunday
        assert is_trading_day("2024-06-15") is False
        assert is_trading_day("2024-06-16") is False

    def test_weekday_returns_true(self):
        """Normal weekday is a trading day."""
        # 2024-06-12 is Wednesday
        assert is_trading_day("2024-06-12") is True

    def test_new_years_day(self):
        """Jan 1 is a NYSE holiday."""
        assert is_trading_day("2024-01-01") is False

    def test_christmas_day(self):
        """Dec 25 is a NYSE holiday."""
        assert is_trading_day("2024-12-25") is False

    def test_accepts_datetime(self):
        """is_trading_day accepts datetime objects."""
        dt = datetime(2024, 6, 12)
        assert is_trading_day(dt) is True

    def test_accepts_date(self):
        """is_trading_day accepts date objects."""
        d = date(2024, 6, 15)  # Saturday
        assert is_trading_day(d) is False


class TestNextTradingDay:
    def test_friday_goes_to_monday(self):
        """Friday's next trading day is Monday."""
        # 2024-06-14 is Friday
        assert next_trading_day("2024-06-14") == "2024-06-17"

    def test_thursday_goes_to_friday(self):
        """Normal weekday advances one day."""
        # 2024-06-13 is Thursday
        assert next_trading_day("2024-06-13") == "2024-06-14"

    def test_skips_holiday(self):
        """Next trading day skips holidays."""
        # Dec 24 2024 is Tuesday, Dec 25 is Christmas (holiday)
        assert next_trading_day("2024-12-24") == "2024-12-26"


class TestTradingDaysBetween:
    def test_returns_list_of_strings(self):
        """Returns list of date strings."""
        result = trading_days_between("2024-06-10", "2024-06-14")
        assert isinstance(result, list)
        assert all(isinstance(d, str) for d in result)

    def test_excludes_weekends(self):
        """Weekend dates are not included."""
        # June 10 (Mon) through June 16 (Sun) = 5 trading days
        result = trading_days_between("2024-06-10", "2024-06-16")
        assert "2024-06-15" not in result  # Saturday
        assert "2024-06-16" not in result  # Sunday
        assert len(result) == 5

    def test_includes_both_endpoints(self):
        """Start and end dates are inclusive."""
        result = trading_days_between("2024-06-10", "2024-06-10")
        assert result == ["2024-06-10"]


class TestNTradingDaysLater:
    def test_one_day(self):
        """1 trading day later = next trading day."""
        assert n_trading_days_later("2024-06-14", 1) == "2024-06-17"

    def test_five_days(self):
        """5 trading days from Monday = next Monday."""
        assert n_trading_days_later("2024-06-10", 5) == "2024-06-17"

    def test_zero_days(self):
        """0 trading days later = same day if it is a trading day."""
        assert n_trading_days_later("2024-06-12", 0) == "2024-06-12"

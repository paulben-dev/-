"""Tests for signal transformation and features."""
import pytest
from datetime import datetime
from src.model.signal import compute_return_ratio, compute_expert_availability
from src.model.features import compute_momentum
from src.data.models import ExpertRecord


class TestReturnRatio:
    def test_positive_return(self):
        assert compute_return_ratio(110.0, 100.0) == 0.1

    def test_negative_return(self):
        assert compute_return_ratio(95.0, 100.0) == -0.05

    def test_zero_prev(self):
        assert compute_return_ratio(110.0, 0.0) == 0.0


class TestExpertAvailability:
    def test_some_stocks_have_experts(self):
        records = [
            ExpertRecord("u1", "AAPL", datetime(2024, 6, 15), 0.85, 0.70, "expert", "Bullish"),
            ExpertRecord("u2", "MSFT", datetime(2024, 6, 15), 0.15, 0.30, "inverse_expert", "Bearish"),
        ]
        avail = compute_expert_availability(records, ["AAPL", "MSFT", "GOOGL", "AMZN"])
        assert avail["AAPL"] == 1
        assert avail["MSFT"] == 1
        assert avail["GOOGL"] == 0
        assert avail["AMZN"] == 0

    def test_no_experts(self):
        avail = compute_expert_availability([], ["AAPL", "MSFT"])
        assert avail == {"AAPL": 0, "MSFT": 0}

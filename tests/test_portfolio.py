"""Tests for portfolio construction and backtest."""
import pytest
import pandas as pd
from src.backtest.portfolio import construct_long_short, _max_drawdown


class TestConstructLongShort:
    def test_basic_construction(self):
        df = pd.DataFrame({
            "stock": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
            "predicted_return": [0.5, 0.4, 0.3, 0.2, 0.1, -0.1, -0.2, -0.3, -0.4, -0.5],
            "date": "2024-06-15",
        })
        result = construct_long_short(df, quantile=0.1)
        assert result["long"] == ["A"]
        assert result["short"] == ["J"]
        assert result["long_weight"] == 1.0
        assert result["short_weight"] == 1.0

    def test_quantile_20(self):
        df = pd.DataFrame({
            "stock": ["A", "B", "C", "D", "E"],
            "predicted_return": [0.5, 0.3, 0.0, -0.3, -0.5],
            "date": "2024-06-15",
        })
        result = construct_long_short(df, quantile=0.2)
        assert len(result["long"]) == 1  # 20% of 5 = 1
        assert len(result["short"]) == 1


class TestMaxDrawdown:
    def test_no_drawdown(self):
        cum = pd.Series([1.0, 1.05, 1.10, 1.15])
        assert _max_drawdown(cum) == 0.0

    def test_with_drawdown(self):
        cum = pd.Series([1.0, 1.10, 1.05, 0.95, 1.02])
        dd = _max_drawdown(cum)
        assert dd < 0
        assert dd == pytest.approx(-0.136, abs=0.01)  # (0.95 - 1.10) / 1.10

"""Tests for portfolio construction and backtest."""
import pytest
import numpy as np
import pandas as pd
from src.backtest.portfolio import run_backtest, construct_long_short, _max_drawdown
from src.backtest.slippage import SlippageConfig
from src.backtest.position import PositionConfig


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


class TestBacktestCalendar:
    """Tests that enhanced backtest uses trading calendar and slippage."""

    def test_backtest_uses_trading_calendar(self, prepopulated_db):
        """When use_calendar=True, backtest skips non-trading days."""
        stocks = ["AAPL", "MSFT"]
        dates = pd.date_range("2024-06-10", "2024-06-16", freq="D")
        preds = []
        for d in dates:
            ds = d.strftime("%Y-%m-%d")
            preds.append({
                "stock": stocks[0], "date": ds,
                "predicted_return": 0.01, "signal_source": "test",
            })
            preds.append({
                "stock": stocks[1], "date": ds,
                "predicted_return": -0.01, "signal_source": "test",
            })
        pred_df = pd.DataFrame(preds)

        result = run_backtest(
            pred_df, stocks, "2024-06-10", "2024-06-16",
            use_calendar=True,
        )
        # June 15-16 is weekend, should have at most 5 trading days
        assert result["n_trading_days"] <= 5

    def test_backtest_no_calendar_preserves_old_behavior(self, prepopulated_db):
        """Without use_calendar, backtest matches v0.4 behavior."""
        stocks = ["AAPL", "MSFT"]
        dates = pd.date_range("2024-06-10", "2024-06-14", freq="D")
        preds = []
        for d in dates:
            ds = d.strftime("%Y-%m-%d")
            preds.append({
                "stock": stocks[0], "date": ds,
                "predicted_return": 0.01, "signal_source": "test",
            })
            preds.append({
                "stock": stocks[1], "date": ds,
                "predicted_return": -0.01, "signal_source": "test",
            })
        pred_df = pd.DataFrame(preds)

        result_old = run_backtest(pred_df, stocks, "2024-06-10", "2024-06-14")
        # Without calendar, uses old +1 day logic
        assert result_old["n_trading_days"] == 5

    def test_slippage_reduces_returns(self, prepopulated_db):
        """Enabling slippage reduces annualized return vs disabled."""
        stocks = ["AAPL", "MSFT"]
        preds = [
            {"stock": stocks[0], "date": "2024-06-12",
             "predicted_return": 0.05, "signal_source": "test"},
            {"stock": stocks[1], "date": "2024-06-12",
             "predicted_return": -0.05, "signal_source": "test"},
        ]
        pred_df = pd.DataFrame(preds)

        no_slip = run_backtest(pred_df, stocks, "2024-06-12", "2024-06-13")
        with_slip = run_backtest(
            pred_df, stocks, "2024-06-12", "2024-06-13",
            slippage_config=SlippageConfig(fixed_cost=0.01, impact_factor=0.0),
        )
        # Slippage should not increase returns
        assert with_slip["annualized_return"] <= no_slip["annualized_return"]

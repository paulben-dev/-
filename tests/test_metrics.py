"""Tests for backtest metrics."""
import pytest
import numpy as np
import pandas as pd
from src.backtest.metrics import (
    compute_accuracy, compute_ic, compute_ric, compute_icir,
    compute_annualized_return, compute_sharpe,
)


class TestAccuracy:
    def test_perfect_accuracy(self):
        pred = pd.Series([0.1, -0.1, 0.05], index=["A", "B", "C"])
        actual = pd.Series([0.05, -0.02, 0.03], index=["A", "B", "C"])
        assert compute_accuracy(pred, actual) == 1.0

    def test_bad_accuracy(self):
        pred = pd.Series([0.1, -0.1], index=["A", "B"])
        actual = pd.Series([-0.05, 0.02], index=["A", "B"])
        assert compute_accuracy(pred, actual) == 0.0

    def test_partial_match(self):
        pred = pd.Series([0.1, -0.1], index=["A", "B"])
        actual = pd.Series([0.05, 0.02], index=["A", "B"])
        assert compute_accuracy(pred, actual) == 0.5

    def test_misaligned_indices(self):
        pred = pd.Series([0.1, -0.1, 0.05], index=["A", "B", "C"])
        actual = pd.Series([0.05, -0.02], index=["A", "B"])
        acc = compute_accuracy(pred, actual)
        assert 0.0 <= acc <= 1.0

    def test_empty_series_returns_neutral(self):
        pred = pd.Series([], dtype=float)
        actual = pd.Series([], dtype=float)
        assert compute_accuracy(pred, actual) == 0.5


class TestIC:
    def test_perfect_positive_correlation(self):
        pred = pd.Series([1, 2, 3], index=["A", "B", "C"])
        actual = pd.Series([0.5, 1.0, 1.5], index=["A", "B", "C"])
        assert compute_ic(pred, actual) == pytest.approx(1.0)

    def test_perfect_negative_correlation(self):
        pred = pd.Series([3, 2, 1], index=["A", "B", "C"])
        actual = pd.Series([0.5, 1.0, 1.5], index=["A", "B", "C"])
        assert compute_ic(pred, actual) == pytest.approx(-1.0)

    def test_no_correlation(self):
        np.random.seed(42)
        pred = pd.Series(np.random.randn(30))
        actual = pd.Series(np.random.randn(30))
        ic = compute_ic(pred, actual)
        assert -0.5 < ic < 0.5

    def test_too_few_points_returns_zero(self):
        pred = pd.Series([1, 2], index=["A", "B"])
        actual = pd.Series([0.5, 1.0], index=["A", "B"])
        assert compute_ic(pred, actual) == 0.0


class TestRIC:
    def test_perfect_rank_correlation(self):
        pred = pd.Series([1, 2, 3, 4, 5], index=["A", "B", "C", "D", "E"])
        actual = pd.Series([10, 20, 30, 40, 50], index=["A", "B", "C", "D", "E"])
        assert compute_ric(pred, actual) == pytest.approx(1.0)

    def test_perfect_negative_rank_correlation(self):
        pred = pd.Series([5, 4, 3, 2, 1], index=["A", "B", "C", "D", "E"])
        actual = pd.Series([10, 20, 30, 40, 50], index=["A", "B", "C", "D", "E"])
        assert compute_ric(pred, actual) == pytest.approx(-1.0)


class TestICIR:
    def test_positive_icir(self):
        ic = pd.Series([0.05, 0.06, 0.04, 0.07, 0.05])
        assert compute_icir(ic) > 0

    def test_short_series(self):
        assert compute_icir(pd.Series([0.05])) == 0.0

    def test_all_same_values(self):
        ic = pd.Series([0.05, 0.05, 0.05])
        assert compute_icir(ic) == 0.0


class TestReturns:
    def test_annualized_return_positive(self):
        returns = pd.Series([0.001] * 100)  # 0.1% daily
        ar = compute_annualized_return(returns)
        assert ar > 0

    def test_annualized_return_zero(self):
        returns = pd.Series([0.0] * 50)
        ar = compute_annualized_return(returns)
        assert ar <= 0  # negative after transaction cost

    def test_annualized_return_empty(self):
        returns = pd.Series([], dtype=float)
        assert compute_annualized_return(returns) == 0.0

    def test_sharpe_positive(self):
        returns = pd.Series(np.random.normal(0.001, 0.01, 252))
        sr = compute_sharpe(returns)
        assert isinstance(sr, float)

    def test_sharpe_too_few_days(self):
        returns = pd.Series([0.001, 0.002, 0.001])
        assert compute_sharpe(returns) == 0.0

    def test_sharpe_flat_returns(self):
        returns = pd.Series([0.0] * 10)
        sr = compute_sharpe(returns)
        assert isinstance(sr, float)
        assert sr <= 0  # negative due to transaction costs minus risk-free

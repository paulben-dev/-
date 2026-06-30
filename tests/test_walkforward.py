"""Tests for walk-forward validation engine."""
import pytest
import numpy as np
import pandas as pd
from src.backtest.walkforward import (
    WalkForwardConfig,
    WalkForwardResult,
    run_walk_forward,
)


class TestWalkForwardConfig:
    def test_defaults(self):
        cfg = WalkForwardConfig()
        assert cfg.train_days == 252
        assert cfg.validate_days == 63
        assert cfg.step_days == 21
        assert cfg.mode == "full"


class TestRunWalkForward:
    def test_params_mode_produces_windows(self, prepopulated_db):
        """params mode generates windows covering the date range."""
        cfg = WalkForwardConfig(
            train_days=30,
            validate_days=10,
            step_days=5,
            mode="params",
            min_train_days=5,
        )
        stocks = ["AAPL", "MSFT"]
        result = run_walk_forward(stocks, "2024-05-01", "2024-07-15", cfg)
        assert isinstance(result, WalkForwardResult)
        assert len(result.windows) > 0
        # Each window should have required keys
        for w in result.windows:
            assert "window_id" in w
            assert "train_start" in w
            assert "train_end" in w
            assert "val_start" in w
            assert "val_end" in w
            assert "sharpe_ratio" in w

    def test_no_lookahead_bias(self, prepopulated_db):
        """Validation dates never overlap with training dates for the same window."""
        cfg = WalkForwardConfig(
            train_days=30,
            validate_days=10,
            step_days=5,
            mode="params",
            min_train_days=5,
        )
        stocks = ["AAPL", "MSFT"]
        result = run_walk_forward(stocks, "2024-05-01", "2024-07-15", cfg)
        for w in result.windows:
            # Validation end > train end (no overlap)
            assert w["val_start"] > w["train_end"], \
                f"Window {w['window_id']}: val_start={w['val_start']} <= train_end={w['train_end']}"

    def test_summary_has_mean_std(self, prepopulated_db):
        """Summary contains mean and std for key metrics."""
        cfg = WalkForwardConfig(
            train_days=30,
            validate_days=10,
            step_days=5,
            mode="params",
            min_train_days=5,
        )
        stocks = ["AAPL", "MSFT"]
        result = run_walk_forward(stocks, "2024-05-01", "2024-07-15", cfg)
        assert "sharpe_mean" in result.summary
        assert "sharpe_std" in result.summary
        assert "mean_ic_mean" in result.summary
        assert "mean_ic_std" in result.summary

    @pytest.mark.slow
    def test_full_mode_produces_windows(self, prepopulated_db):
        """full mode runs without crashing and produces windows.

        Uses a tiny window so retraining is fast; model fits may fail
        on insufficient data but the framework must not crash.
        """
        cfg = WalkForwardConfig(
            train_days=5,
            validate_days=5,
            step_days=3,
            mode="full",
            min_train_days=5,
        )
        stocks = ["AAPL", "MSFT"]
        result = run_walk_forward(stocks, "2024-06-01", "2024-07-15", cfg)
        assert isinstance(result, WalkForwardResult)
        assert len(result.windows) > 0
        # lookahead must hold even in full mode
        for w in result.windows:
            assert w["val_start"] > w["train_end"], \
                f"Window {w['window_id']}: val_start={w['val_start']} <= train_end={w['train_end']}"

    def test_insufficient_data_returns_empty(self, prepopulated_db):
        """Date range too short for one window returns empty result."""
        cfg = WalkForwardConfig(
            train_days=252,
            validate_days=63,
            step_days=21,
            mode="params",
            min_train_days=252,
        )
        stocks = ["AAPL", "MSFT"]
        result = run_walk_forward(stocks, "2024-05-01", "2024-05-20", cfg)
        assert len(result.windows) == 0

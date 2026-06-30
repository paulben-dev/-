"""Tests for parameter scanner."""
import pytest
import numpy as np
import pandas as pd
from src.backtest.scanner import (
    ParamSpec,
    build_param_grid,
    random_search,
    run_scan,
)


class TestBuildParamGrid:
    def test_grid_covers_all_combinations(self):
        """Cartesian product of two parameters."""
        specs = [
            ParamSpec("quantile", [0.05, 0.10]),
            ParamSpec("lookback", [10, 20, 30]),
        ]
        grid = build_param_grid(specs)
        assert len(grid) == 6  # 2 * 3
        combos = {(g["quantile"], g["lookback"]) for g in grid}
        assert (0.05, 10) in combos
        assert (0.10, 30) in combos

    def test_single_param(self):
        """Single parameter returns one entry per value."""
        specs = [ParamSpec("x", [1, 2, 3])]
        grid = build_param_grid(specs)
        assert len(grid) == 3

    def test_empty_specs(self):
        """Empty specs returns single empty dict."""
        grid = build_param_grid([])
        assert grid == [{}]


class TestRandomSearch:
    def test_respects_n_iter(self):
        """Random search returns exactly n_iter combinations."""
        specs = [
            ParamSpec("a", list(range(100))),
            ParamSpec("b", list(range(100))),
        ]
        grid = build_param_grid(specs)
        sampled = random_search(grid, n_iter=10)
        assert len(sampled) == 10

    def test_n_iter_exceeds_grid_size(self):
        """When n_iter > |grid|, returns all (deduplicated)."""
        specs = [ParamSpec("x", [1, 2])]
        grid = build_param_grid(specs)
        sampled = random_search(grid, n_iter=100)
        assert len(sampled) == 2


class TestRunScan:
    def test_run_scan_returns_dataframe(self, prepopulated_db):
        """run_scan returns a DataFrame with expected columns."""
        specs = [
            ParamSpec("quantile", [0.10, 0.20]),
        ]
        grid = build_param_grid(specs)
        stocks = ["AAPL", "MSFT"]
        result = run_scan(stocks, "2024-05-01", "2024-06-15", grid,
                          wf_config=None, metric="sharpe_ratio")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        for col in ["sharpe_ratio", "mean_ic", "params"]:
            assert col in result.columns

    def test_returns_best_first(self, prepopulated_db):
        """Results are sorted by metric descending."""
        specs = [
            ParamSpec("quantile", [0.05, 0.10, 0.15]),
        ]
        grid = build_param_grid(specs)
        stocks = ["AAPL", "MSFT"]
        result = run_scan(stocks, "2024-05-01", "2024-06-15", grid,
                          wf_config=None, metric="sharpe_ratio")
        sharpe_vals = result["sharpe_ratio"].tolist()
        assert sharpe_vals == sorted(sharpe_vals, reverse=True)

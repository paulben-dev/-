"""Parameter scanner for backtest hyperparameter optimization.

Supports grid search (Cartesian product) and random search with
optional walk-forward cross-validation.
"""
from dataclasses import dataclass, field
import logging
import random
import numpy as np
import pandas as pd

from config import SCAN_DEFAULT_METRIC, SCAN_RANDOM_N_ITER

logger = logging.getLogger(__name__)


@dataclass
class ParamSpec:
    """A single parameter to scan over."""
    name: str
    values: list


def build_param_grid(specs: list[ParamSpec]) -> list[dict]:
    """Build Cartesian product of all parameter value lists.

    Args:
        specs: List of parameter specifications.

    Returns:
        List of {param_name: value} dicts, one per combination.
    """
    if not specs:
        return [{}]

    grid: list[dict] = [{}]
    for spec in specs:
        new_grid = []
        for combo in grid:
            for val in spec.values:
                new_combo = dict(combo)
                new_combo[spec.name] = val
                new_grid.append(new_combo)
        grid = new_grid
    return grid


def random_search(grid: list[dict], n_iter: int = SCAN_RANDOM_N_ITER) -> list[dict]:
    """Randomly sample ``n_iter`` combinations from the grid.

    If ``n_iter`` >= len(grid), returns all combinations in shuffled order.

    Args:
        grid: Full parameter grid from build_param_grid.
        n_iter: Number of combinations to sample.

    Returns:
        Sampled subset of the grid.
    """
    if n_iter >= len(grid):
        result = list(grid)
        random.shuffle(result)
        return result
    return random.sample(grid, n_iter)


def run_scan(
    stocks: list[str],
    start_date: str,
    end_date: str,
    param_grid: list[dict],
    wf_config=None,     # WalkForwardConfig | None
    metric: str = SCAN_DEFAULT_METRIC,
) -> pd.DataFrame:
    """Evaluate each parameter combination and return results sorted by metric.

    Args:
        stocks: Ticker symbols.
        start_date / end_date: Backtest date range.
        param_grid: List of {param: value} dicts to evaluate.
        wf_config: If provided, each combination is evaluated via
                   walk-forward (OOS metric). If None, simple hold-out backtest.
        metric: Column name to sort by descending.

    Returns:
        DataFrame with columns: params, sharpe_ratio, mean_ic, max_drawdown,
        annualized_return, icir — sorted by ``metric`` descending.
    """
    from src.backtest.portfolio import run_backtest
    from src.model.baseline import BaselinePredictor
    from src.backtest.calendar import trading_days_between

    results = []

    for combo in param_grid:
        try:
            if wf_config is not None:
                from src.backtest.walkforward import run_walk_forward
                wf_result = run_walk_forward(
                    stocks, start_date, end_date, wf_config,
                    param_grid=combo,
                )
                # Use mean Sharpe across windows as the evaluation metric
                sharpe = wf_result.summary.get("sharpe_mean", 0.0)
                mean_ic = wf_result.summary.get("mean_ic_mean", 0.0)
                results.append({
                    "params": combo,
                    "sharpe_ratio": sharpe,
                    "mean_ic": mean_ic,
                    "max_drawdown": min((w.get("max_drawdown", 0.0) for w in wf_result.windows), default=0.0),
                    "annualized_return": sum(w.get("annualized_return", 0.0) for w in wf_result.windows) / max(len(wf_result.windows), 1),
                    "icir": sum(w.get("icir", 0.0) for w in wf_result.windows) / max(len(wf_result.windows), 1),
                })
            else:
                # Simple hold-out backtest
                dates = trading_days_between(start_date, end_date)
                predictor = BaselinePredictor()
                all_preds = []
                for d in dates:
                    pred_df = predictor.predict(stocks, d, [])
                    all_preds.append(pred_df)

                if all_preds:
                    combined = pd.concat(all_preds, ignore_index=True)
                    bt = run_backtest(combined, stocks, start_date, end_date,
                                      use_calendar=True, **combo)
                    results.append({
                        "params": combo,
                        "sharpe_ratio": bt.get("sharpe_ratio", 0.0),
                        "mean_ic": bt.get("mean_ic", 0.0),
                        "max_drawdown": bt.get("max_drawdown", 0.0),
                        "annualized_return": bt.get("annualized_return", 0.0),
                        "icir": bt.get("icir", 0.0),
                    })
        except Exception as e:
            logger.warning(f"Scan combo {combo} failed: {e}")

    if not results:
        return pd.DataFrame(columns=[
            "params", "sharpe_ratio", "mean_ic",
            "max_drawdown", "annualized_return", "icir",
        ])

    df = pd.DataFrame(results)
    if metric in df.columns:
        df = df.sort_values(metric, ascending=False).reset_index(drop=True)
    return df

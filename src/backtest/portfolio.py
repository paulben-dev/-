"""Long-short portfolio construction and backtest simulation."""
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.db import schema as db
from src.backtest.metrics import (
    compute_ic,
    compute_icir,
    compute_annualized_return,
    compute_sharpe,
    compute_daily_ic_series,
)
from config import PORTFOLIO_QUANTILE, TRANSACTION_COST

logger = logging.getLogger(__name__)


def construct_long_short(pred_df: pd.DataFrame, quantile: float = PORTFOLIO_QUANTILE) -> dict:
    """Construct long-short portfolio for a single day.

    Args:
        pred_df: Predictions for one day with columns [stock, predicted_return].
        quantile: Fraction of stocks to long/short (default 0.10 = top/bottom 10%).

    Returns:
        Dict with keys: long (list of stocks), short (list of stocks),
                        long_weight, short_weight.
    """
    df = pred_df.sort_values("predicted_return", ascending=False)
    n_stocks = len(df)
    n_positions = max(1, int(n_stocks * quantile))

    long_stocks = df.head(n_positions)["stock"].tolist()
    short_stocks = df.tail(n_positions)["stock"].tolist()

    weight = 1.0 / n_positions if n_positions > 0 else 0.0

    return {
        "long": long_stocks,
        "short": short_stocks,
        "long_weight": weight,
        "short_weight": weight,
        "date": df["date"].iloc[0] if "date" in df.columns else None,
    }


def run_backtest(
    pred_df: pd.DataFrame,
    stocks: list[str],
    start_date: str,
    end_date: str,
    quantile: float = PORTFOLIO_QUANTILE,
    transaction_cost: float = TRANSACTION_COST,
) -> dict:
    """Run a full backtest simulation.

    Strategy: Long top quantile, short bottom quantile, daily rebalancing.
    Returns daily P&L and summary metrics.
    """
    dates = sorted(pred_df["date"].unique())
    daily_returns = []
    daily_long_short = []

    for date_str in dates:
        day_pred = pred_df[pred_df["date"] == date_str]
        if len(day_pred) == 0:
            continue

        portfolio = construct_long_short(day_pred, quantile)

        # Get next-day returns for selected stocks
        next_date = _get_next_trading_day(date_str)
        if next_date is None:
            continue

        long_ret = _get_portfolio_return(portfolio["long"], date_str, next_date)
        short_ret = _get_portfolio_return(portfolio["short"], date_str, next_date)

        daily_ret = (long_ret - short_ret) / 2
        daily_returns.append(daily_ret)
        daily_long_short.append({
            "date": date_str,
            "long_return": long_ret,
            "short_return": short_ret,
            "long_short_return": daily_ret,
            "long_stocks": portfolio["long"],
            "short_stocks": portfolio["short"],
        })

    dr_series = pd.Series(daily_returns)
    cumulative = (1 + dr_series).cumprod()

    # Compute metrics
    daily_ic = compute_daily_ic_series(pred_df, stocks, start_date, end_date)

    return {
        "daily_returns": dr_series,
        "cumulative_returns": cumulative,
        "daily_long_short": pd.DataFrame(daily_long_short),
        "annualized_return": compute_annualized_return(dr_series, transaction_cost),
        "sharpe_ratio": compute_sharpe(dr_series, transaction_cost=transaction_cost),
        "max_drawdown": float(_max_drawdown(cumulative)),
        "mean_ic": daily_ic.mean() if len(daily_ic) > 0 else 0.0,
        "icir": compute_icir(daily_ic),
        "n_trading_days": len(dr_series),
    }


def _get_portfolio_return(stocks: list[str], date_str: str, next_date: str) -> float:
    """Get equal-weighted portfolio return from date to next_date."""
    if not stocks:
        return 0.0
    prices = db.get_prices(stocks, date_str, next_date)
    returns = []
    for stock in stocks:
        sp = prices.get(stock, [])
        if len(sp) >= 2:
            sp.sort(key=lambda x: x["date"])
            ret = (sp[-1]["close"] - sp[0]["close"]) / sp[0]["close"] if sp[0]["close"] else 0.0
            returns.append(ret)
    return np.mean(returns) if returns else 0.0


def _get_next_trading_day(date_str: str) -> str | None:
    """Get the next calendar day (simplified — ignores weekends/holidays)."""
    dt = datetime.fromisoformat(date_str) + timedelta(days=1)
    return dt.strftime("%Y-%m-%d")


def _max_drawdown(cumulative: pd.Series) -> float:
    """Compute maximum drawdown from cumulative returns."""
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return float(drawdown.min())

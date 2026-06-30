"""Long-short portfolio construction and backtest simulation."""
from __future__ import annotations
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
    use_calendar: bool = False,
    slippage_config: SlippageConfig | None = None,
    position_config: PositionConfig | None = None,
) -> dict:
    """Run a full backtest simulation.

    Strategy: Long top quantile, short bottom quantile, daily rebalancing.
    Returns daily P&L and summary metrics.

    New optional kwargs (all default False/None = backward compatible):
        use_calendar: Use NYSE calendar for next-trading-day logic and skip
                      non-trading prediction dates.
        slippage_config: Apply per-trade slippage costs.
        position_config: Use risk-aware position sizing instead of simple
                         top/bottom quantile.
    """
    all_dates = sorted(pred_df["date"].unique())
    daily_returns = []
    daily_long_short = []

    prev_weights: dict[str, float] = {}

    for date_str in all_dates:
        # Optionally skip non-trading days (weekends, holidays)
        if use_calendar:
            from src.backtest.calendar import is_trading_day as _is_td
            if not _is_td(date_str):
                continue

        day_pred = pred_df[pred_df["date"] == date_str]
        if len(day_pred) == 0:
            continue

        # Determine next trading day
        if use_calendar:
            from src.backtest.calendar import next_trading_day as _next_td
            next_date = _next_td(date_str)
        else:
            next_date = _get_next_trading_day(date_str)

        # Optionally use position sizing
        if position_config is not None:
            from src.backtest.position import size_positions
            preds_arr = day_pred.set_index("stock")["predicted_return"].reindex(stocks).fillna(0.0).values
            prices_for_vol = _get_recent_prices(stocks, date_str, lookback=20)
            pos_weights = size_positions(
                stocks, preds_arr, prev_weights, prices_for_vol, position_config, quantile,
            )
            long_stocks = [s for s, w in pos_weights.items() if w > 0]
            short_stocks = [s for s, w in pos_weights.items() if w < 0]
            long_ret = _get_weighted_return(pos_weights, date_str, next_date, side="long")
            short_ret = _get_weighted_return(pos_weights, date_str, next_date, side="short")
            prev_weights = pos_weights
        else:
            portfolio = construct_long_short(day_pred, quantile)
            long_ret = _get_portfolio_return(portfolio["long"], date_str, next_date)
            short_ret = _get_portfolio_return(portfolio["short"], date_str, next_date)
            long_stocks = portfolio["long"]
            short_stocks = portfolio["short"]

        # Apply slippage costs
        slippage = 0.0
        if slippage_config is not None:
            from src.backtest.slippage import estimate_slippage
            # Estimate slippage for each traded stock
            for stock in list(set(long_stocks + short_stocks)):
                vol = _get_stock_volume(stock, date_str)
                px = _get_stock_close(stock, date_str)
                if position_config is not None:
                    w = pos_weights.get(stock, 0.0)
                else:
                    w = (1.0 / max(len(long_stocks), 1) if stock in long_stocks
                         else -1.0 / max(len(short_stocks), 1) if stock in short_stocks
                         else 0.0)
                slippage += abs(w) * estimate_slippage(w, vol, px, slippage_config)

        daily_ret = (long_ret - short_ret) / 2 - slippage
        daily_returns.append(daily_ret)
        daily_long_short.append({
            "date": date_str,
            "long_return": long_ret,
            "short_return": short_ret,
            "long_short_return": daily_ret,
            "long_stocks": long_stocks,
            "short_stocks": short_stocks,
            "slippage": slippage,
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


def _get_recent_prices(stocks: list[str], date_str: str, lookback: int = 20) -> dict[str, list[float]]:
    """Get recent closing prices for volatility estimation."""
    start = (datetime.fromisoformat(date_str) - timedelta(days=lookback + 5)).strftime("%Y-%m-%d")
    raw = db.get_prices(stocks, start, date_str)
    result = {}
    for stock in stocks:
        sp = raw.get(stock, [])
        result[stock] = [p["close"] for p in sorted(sp, key=lambda x: x["date"])]
    return result


def _get_weighted_return(weights: dict[str, float], date_str: str,
                         next_date: str, side: str) -> float:
    """Get weighted return for long or short side."""
    stocks = [s for s, w in weights.items() if (w > 0 and side == "long") or (w < 0 and side == "short")]
    if not stocks:
        return 0.0
    prices = db.get_prices(stocks, date_str, next_date)
    total_return = 0.0
    for stock in stocks:
        sp = prices.get(stock, [])
        if len(sp) >= 2:
            sp.sort(key=lambda x: x["date"])
            ret = (sp[-1]["close"] - sp[0]["close"]) / sp[0]["close"] if sp[0]["close"] else 0.0
            total_return += abs(weights.get(stock, 0.0)) * ret
    return total_return


def _get_stock_volume(stock: str, date_str: str) -> int:
    """Get reported volume for a stock on a date."""
    prices = db.get_prices([stock], date_str, date_str)
    sp = prices.get(stock, [])
    return sp[0].get("volume", 0) if sp else 0


def _get_stock_close(stock: str, date_str: str) -> float:
    """Get closing price for a stock on a date."""
    prices = db.get_prices([stock], date_str, date_str)
    sp = prices.get(stock, [])
    return sp[0].get("close", 0.0) if sp else 0.0

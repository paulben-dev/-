"""Quantitative evaluation metrics for stock prediction."""
import logging
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta
from src.db import schema as db
from config import TRANSACTION_COST, PORTFOLIO_QUANTILE

logger = logging.getLogger(__name__)


def compute_accuracy(predictions: pd.Series, actual_returns: pd.Series) -> float:
    """Directional accuracy: did we predict the right sign?"""
    aligned = pd.concat([predictions, actual_returns], axis=1, join="inner").dropna()
    aligned.columns = ["pred", "actual"]
    if len(aligned) == 0:
        return 0.5
    correct = ((aligned["pred"] > 0) & (aligned["actual"] > 0)) | \
              ((aligned["pred"] < 0) & (aligned["actual"] < 0))
    return float(correct.mean())


def compute_ic(predictions: pd.Series, actual_returns: pd.Series) -> float:
    """Information Coefficient: Pearson correlation between predicted and actual."""
    aligned = predictions.align(actual_returns, join="inner")
    combined = pd.concat([aligned[0], aligned[1]], axis=1).dropna()
    if len(combined) < 3:
        return 0.0
    return float(combined.iloc[:, 0].corr(combined.iloc[:, 1]))


def compute_ric(predictions: pd.Series, actual_returns: pd.Series) -> float:
    """Rank IC: Spearman rank correlation."""
    aligned = predictions.align(actual_returns, join="inner")
    combined = pd.concat([aligned[0], aligned[1]], axis=1).dropna()
    if len(combined) < 3:
        return 0.0
    r, _ = stats.spearmanr(combined.iloc[:, 0], combined.iloc[:, 1])
    return float(r) if not np.isnan(r) else 0.0


def compute_icir(ic_series: pd.Series) -> float:
    """IC Information Ratio: mean(IC) / std(IC)."""
    if len(ic_series) < 2 or ic_series.std() < 1e-12:
        return 0.0
    return float(ic_series.mean() / ic_series.std())


def compute_annualized_return(daily_returns: pd.Series,
                              transaction_cost: float = None) -> float:
    """Annualized return from daily returns, accounting for transaction costs."""
    if transaction_cost is None:
        transaction_cost = TRANSACTION_COST
    net_returns = daily_returns - transaction_cost
    cumulative = (1 + net_returns).prod()
    n_days = len(daily_returns)
    if n_days == 0:
        return 0.0
    annualized = cumulative ** (252 / n_days) - 1
    return float(annualized)


def compute_sharpe(daily_returns: pd.Series, risk_free_rate: float = 0.04,
                   transaction_cost: float = None) -> float:
    """Sharpe Ratio: (annualized_return - risk_free) / annualized_volatility."""
    if transaction_cost is None:
        transaction_cost = TRANSACTION_COST
    if len(daily_returns) < 5:
        return 0.0
    net_returns = daily_returns - transaction_cost
    excess = net_returns.mean() * 252 - risk_free_rate
    vol = net_returns.std() * np.sqrt(252)
    return float(excess / vol) if vol > 0 else 0.0


def compute_daily_ic_series(
    pred_df: pd.DataFrame, stocks: list[str],
    start_date: str, end_date: str,
) -> pd.Series:
    """Compute IC for each day in the date range.

    For each date in pred_df, computes the actual next-day return
    by querying price data spanning the prediction date through
    the following calendar day, then calculates the Pearson IC
    between predicted returns and actual returns for that day.
    """
    dates = sorted(pred_df["date"].unique())
    daily_ic = {}

    for date_str in dates:
        day_preds = pred_df[pred_df["date"] == date_str].set_index("stock")

        # Look up prices for this date through the next day to compute
        # the next-close return as the actual outcome.
        next_date = (datetime.strptime(date_str, "%Y-%m-%d")
                     + timedelta(days=1)).strftime("%Y-%m-%d")
        day_prices = db.get_prices(stocks, date_str, next_date)

        # Compute next-day actual return for each stock
        actuals = {}
        for stock in stocks:
            sp = day_prices.get(stock, [])
            if len(sp) >= 2:
                prev_close = sp[0]["close"]
                next_close = sp[-1]["close"]
                if prev_close and prev_close != 0:
                    actuals[stock] = (next_close - prev_close) / prev_close

        if not actuals:
            continue

        actual_series = pd.Series(actuals, name="actual")
        pred_series = day_preds["predicted_return"]

        if pred_series.index.intersection(actual_series.index).size < 3:
            continue

        ic = compute_ic(pred_series, actual_series)
        daily_ic[date_str] = ic

    return pd.Series(daily_ic, name="daily_ic")


def compute_all_metrics(
    pred_df: pd.DataFrame, stocks: list[str],
    start_date: str, end_date: str,
) -> dict:
    """Compute all quantitative metrics for predictions.

    Returns:
        dict with keys: mean_ic, icir, ic_std, n_days, portfolio_quantile,
        transaction_cost
    """
    daily_ic = compute_daily_ic_series(pred_df, stocks, start_date, end_date)

    return {
        "mean_ic": float(daily_ic.mean()) if len(daily_ic) > 0 else 0.0,
        "icir": compute_icir(daily_ic),
        "ic_std": float(daily_ic.std()) if len(daily_ic) > 0 else 0.0,
        "n_days": len(daily_ic),
        "portfolio_quantile": PORTFOLIO_QUANTILE,
        "transaction_cost": TRANSACTION_COST,
    }

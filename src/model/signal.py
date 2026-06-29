"""Expert signal transformation: binary predictions to continuous return signals."""
import logging
from datetime import datetime, timedelta
from collections import defaultdict
from src.data.models import ExpertRecord
from src.db import schema as db
from config import SIGNAL_LOOKBACK_DAYS

logger = logging.getLogger(__name__)


def compute_return_ratio(close_today: float, close_prev: float) -> float:
    """Compute return ratio: (close_today - close_prev) / close_prev."""
    if close_prev == 0:
        return 0.0
    return (close_today - close_prev) / close_prev


def transform_expert_signal(records: list[ExpertRecord], date_str: str) -> dict[str, float]:
    """Transform binary expert predictions into continuous return signals.

    For Bullish predictions: use average return of past 30 up-days.
    For Bearish predictions: use average return of past 30 down-days.
    For multiple experts on same stock -> randomly sample one.

    Returns:
        Dict mapping stock ticker to expert signal value (0 if no expert signal).
    """
    if not records:
        return {}

    target_date = datetime.fromisoformat(date_str)
    lookback_start = (target_date - timedelta(days=SIGNAL_LOOKBACK_DAYS + 5)).strftime("%Y-%m-%d")

    # Group records by stock, pick one expert per stock randomly
    stock_experts = defaultdict(list)
    for r in records:
        stock_experts[r.stock].append(r)

    import random
    signals = {}
    for stock, experts in stock_experts.items():
        expert = random.choice(experts)
        avg_return = _compute_directional_average(stock, target_date, expert.predicted_direction)
        signals[stock] = avg_return

    return signals


def _compute_directional_average(stock: str, target_date: datetime, direction: str) -> float:
    """Compute average return ratio for a given direction over past 30 days.

    Bullish -> average of positive-return days.
    Bearish -> average of negative-return days (returned as negative value).
    """
    start_date = (target_date - timedelta(days=SIGNAL_LOOKBACK_DAYS + 5)).strftime("%Y-%m-%d")
    end_date = target_date.strftime("%Y-%m-%d")

    prices = db.get_prices([stock], start_date, end_date)
    stock_prices = prices.get(stock, [])
    if len(stock_prices) < 5:
        return 0.0

    returns = []
    for i in range(1, len(stock_prices)):
        ret = compute_return_ratio(
            stock_prices[i]["close"],
            stock_prices[i - 1]["close"],
        )
        returns.append(ret)

    if direction == "Bullish":
        positive = [r for r in returns if r > 0]
        return sum(positive) / len(positive) if positive else 0.01  # Default small positive
    else:
        negative = [r for r in returns if r < 0]
        return sum(negative) / len(negative) if negative else -0.01  # Default small negative


def compute_expert_availability(records: list[ExpertRecord], stocks: list[str]) -> dict[str, int]:
    """Binary indicator: does this stock have an expert signal today?"""
    stocks_with_experts = set(r.stock for r in records)
    return {s: (1 if s in stocks_with_experts else 0) for s in stocks}

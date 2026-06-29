"""Feature engineering for stock prediction."""
import logging
from datetime import datetime, timedelta
from src.db import schema as db
from config import MOMENTUM_LOOKBACK_DAYS

logger = logging.getLogger(__name__)


def compute_momentum(
    stocks: list[str],
    date_str: str,
    lookback: int = MOMENTUM_LOOKBACK_DAYS,
) -> dict[str, float]:
    """Compute momentum factor: return over the past N days.

    Positive momentum -> expect continuation (short-term).
    Used as baseline when no expert signal is available.
    """
    target_date = datetime.fromisoformat(date_str)
    start_date = (target_date - timedelta(days=lookback + 5)).strftime("%Y-%m-%d")
    end_date = date_str

    prices = db.get_prices(stocks, start_date, end_date)
    momentum = {}

    for stock in stocks:
        stock_prices = prices.get(stock, [])
        if len(stock_prices) < 2:
            momentum[stock] = 0.0
            continue
        # Find price closest to target_date
        recent = [p for p in stock_prices if p["date"] <= date_str]
        if len(recent) < 2:
            momentum[stock] = 0.0
            continue
        recent.sort(key=lambda x: x["date"])
        # Return over lookback window
        first_close = recent[0]["close"]
        last_close = recent[-1]["close"]
        if first_close == 0:
            momentum[stock] = 0.0
        else:
            momentum[stock] = (last_close - first_close) / first_close
    return momentum


def build_feature_vector(
    stocks: list[str],
    date_str: str,
    expert_signals: dict[str, float],
    expert_availability: dict[str, int],
) -> dict[str, dict]:
    """Build the complete feature vector for each stock on a given day.

    Returns:
        Dict mapping stock to {momentum, expert_signal, expert_available, ...}
    """
    momentum = compute_momentum(stocks, date_str)
    features = {}
    for stock in stocks:
        features[stock] = {
            "momentum": momentum.get(stock, 0.0),
            "expert_signal": expert_signals.get(stock, 0.0),
            "expert_available": expert_availability.get(stock, 0),
        }
    return features

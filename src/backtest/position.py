"""Risk-aware position sizing for long-short portfolios.

Extends basic top/bottom-quantile construction with:
  - volatility targeting
  - single-stock position caps
  - sector neutrality (GICS level-1)
  - turnover constraints
"""
from dataclasses import dataclass
import numpy as np
from config import (
    PORTFOLIO_QUANTILE,
    POSITION_TARGET_VOL,
    POSITION_MAX_SINGLE,
    POSITION_SECTOR_NEUTRAL,
    POSITION_MAX_TURNOVER,
)

# Hardcoded GICS sector mapping for the 20-stock universe.
GICS_SECTORS: dict[str, str] = {
    "AAPL":  "Information Technology",
    "MSFT":  "Information Technology",
    "GOOGL": "Communication Services",
    "AMZN":  "Consumer Discretionary",
    "NVDA":  "Information Technology",
    "META":  "Communication Services",
    "TSLA":  "Consumer Discretionary",
    "JPM":   "Financials",
    "JNJ":   "Health Care",
    "V":     "Financials",
    "PG":    "Consumer Staples",
    "XOM":   "Energy",
    "WMT":   "Consumer Staples",
    "MA":    "Financials",
    "UNH":   "Health Care",
    "HD":    "Consumer Discretionary",
    "BAC":   "Financials",
    "DIS":   "Communication Services",
    "NFLX":  "Communication Services",
    "ADBE":  "Information Technology",
}


@dataclass
class PositionConfig:
    """Position sizing constraints. Zero/False = disabled."""
    target_vol: float = POSITION_TARGET_VOL
    max_single_weight: float = POSITION_MAX_SINGLE
    sector_neutral: bool = POSITION_SECTOR_NEUTRAL
    max_turnover: float = POSITION_MAX_TURNOVER


def size_positions(
    stocks: list[str],
    preds: np.ndarray,
    prev_weights: dict[str, float],
    prices: dict[str, list[float]],
    config: PositionConfig,
    quantile: float = PORTFOLIO_QUANTILE,
) -> dict[str, float]:
    """Compute risk-constrained long-short portfolio weights.

    Algorithm:
      1. Rank stocks by preds -> select top/bottom quantile
      2. Vol scaling: scale weights so portfolio ex-ante vol == target_vol
      3. Single-stock cap: clip any weight exceeding max_single_weight
      4. Sector neutral: equalize long/short exposure per GICS sector
      5. Turnover cap: if distance from prev_weights > max_turnover,
         scale toward prev_weights

    Args:
        stocks: Ordered list of tickers (aligned with preds).
        preds: Predicted returns, one per stock.
        prev_weights: Previous day's weights (empty dict on first day).
        prices: {stock: [recent close prices]} for volatility estimation.
        config: Active constraints.
        quantile: Fraction of stocks to select on each side.

    Returns:
        {stock: weight} dict. Positive = long, negative = short.
    """
    if len(stocks) == 0 or len(preds) == 0:
        return {}

    n_stocks = len(stocks)
    n_positions = max(1, int(n_stocks * quantile))

    # Sort by prediction descending
    order = np.argsort(preds)[::-1]
    sorted_stocks = [stocks[i] for i in order]

    long_stocks = sorted_stocks[:n_positions]
    short_stocks = sorted_stocks[-n_positions:]

    raw_weight = 1.0 / n_positions if n_positions > 0 else 0.0
    weights: dict[str, float] = {}
    for s in long_stocks:
        weights[s] = raw_weight
    for s in short_stocks:
        weights[s] = -raw_weight

    # --- 1. Volatility scaling ---
    if config.target_vol > 0:
        weights = _apply_vol_scaling(weights, prices, config.target_vol)

    # --- 2. Single-stock cap ---
    if config.max_single_weight > 0:
        weights = _apply_single_cap(weights, config.max_single_weight)

    # --- 3. Sector neutrality ---
    if config.sector_neutral:
        weights = _apply_sector_neutral(weights)

    # --- 4. Turnover constraint ---
    if config.max_turnover > 0 and prev_weights:
        weights = _apply_turnover_cap(weights, prev_weights, config.max_turnover)

    return weights


def _estimate_portfolio_vol(
    weights: dict[str, float],
    prices: dict[str, list[float]],
) -> float:
    """Estimate ex-ante annualized portfolio volatility from price history.

    Computes aligned daily portfolio returns, then takes the standard
    deviation of that series.  Annualizes with sqrt(252).
    """
    # Build aligned daily return series for all stocks
    stock_returns: dict[str, list[float]] = {}
    for stock in weights:
        ps = prices.get(stock, [])
        if len(ps) < 2:
            continue
        rets = [(ps[i] - ps[i - 1]) / ps[i - 1] for i in range(1, len(ps))
                if ps[i - 1] > 0]
        if rets:
            stock_returns[stock] = rets

    if len(stock_returns) < 2:
        # Single stock: use its own vol
        if stock_returns:
            all_rets = list(stock_returns.values())[0]
            daily_vol = float(np.std(all_rets)) if len(all_rets) > 1 else 0.01
            return daily_vol * np.sqrt(252)
        return 0.15  # default assumption

    # Align lengths to the minimum across stocks.  Using the shortest history
    # is conservative: it avoids overfitting to stocks with longer lookbacks
    # and ensures every return series covers a comparable recent window.  The
    # trade-off is discarding older data for stocks with longer histories, but
    # for portfolio vol estimation the most recent data matter most.
    min_len = min(len(r) for r in stock_returns.values())
    aligned = {s: np.array(r[-min_len:]) for s, r in stock_returns.items()}

    # Portfolio daily returns: sum of weight * daily_return for each day
    portfolio_daily = np.zeros(min_len)
    for s, w in weights.items():
        if s in aligned:
            portfolio_daily += w * aligned[s]

    daily_vol = float(np.std(portfolio_daily)) if min_len > 1 else 0.01
    return daily_vol * np.sqrt(252)


def _apply_vol_scaling(
    weights: dict[str, float],
    prices: dict[str, list[float]],
    target_vol: float,
) -> dict[str, float]:
    """Scale weights so ex-ante annualized vol matches target."""
    current_vol = _estimate_portfolio_vol(weights, prices)
    if current_vol <= 0:
        return weights
    scale = target_vol / current_vol
    return {s: w * scale for s, w in weights.items()}


def _apply_single_cap(
    weights: dict[str, float],
    max_weight: float,
) -> dict[str, float]:
    """Clip any absolute weight exceeding max_weight."""
    return {s: max(-max_weight, min(max_weight, w)) for s, w in weights.items()}


def _apply_sector_neutral(
    weights: dict[str, float],
) -> dict[str, float]:
    """Equalize long and short exposure within each GICS sector.

    For each sector with both long and short positions, scale exposures
    so the net is zero.
    """
    sectors: dict[str, dict[str, list[tuple[str, float]]]] = {
        "long": {}, "short": {},
    }

    for stock, w in weights.items():
        sector = GICS_SECTORS.get(stock, "Unknown")
        side = "long" if w > 0 else "short"
        sectors[side].setdefault(sector, []).append((stock, w))

    result = dict(weights)
    for sector in set(sectors["long"]) & set(sectors["short"]):
        long_positions = sectors["long"][sector]
        short_positions = sectors["short"][sector]
        long_total = sum(abs(w) for _, w in long_positions)
        short_total = sum(abs(w) for _, w in short_positions)
        if long_total > 0 and short_total > 0:
            # Scale the larger side down to match the smaller
            if long_total > short_total:
                scale = short_total / long_total
                for s, w in long_positions:
                    result[s] = w * scale
            else:
                scale = long_total / short_total
                for s, w in short_positions:
                    result[s] = w * scale

    return result


def _apply_turnover_cap(
    weights: dict[str, float],
    prev_weights: dict[str, float],
    max_turnover: float,
) -> dict[str, float]:
    """Limit turnover between consecutive days.

    Turnover = 0.5 * sum(|w_new - w_old|) across all stocks.
    """
    all_stocks = set(weights) | set(prev_weights)
    raw_turnover = sum(
        abs(weights.get(s, 0.0) - prev_weights.get(s, 0.0))
        for s in all_stocks
    ) / 2.0

    if raw_turnover <= max_turnover:
        return weights

    # Blend toward previous weights
    alpha = max_turnover / raw_turnover
    result = {}
    for s in all_stocks:
        w_new = weights.get(s, 0.0)
        w_old = prev_weights.get(s, 0.0)
        result[s] = w_old + alpha * (w_new - w_old)
    return result

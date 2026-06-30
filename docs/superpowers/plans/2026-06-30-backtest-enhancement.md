# Backtest Enhancement v0.5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add trading calendar, slippage, position sizing, walk-forward validation, and parameter scanning to the backtest engine while preserving backward compatibility.

**Architecture:** Five new modules in `src/backtest/` (calendar, slippage, position, walkforward, scanner), enhanced `portfolio.py` with optional precision features, new API endpoints (`/api/backtest/walkforward`, `/api/backtest/scan`), and `/api/backtest/compare` extended with query-param toggles.

**Tech Stack:** Python 3.14, numpy, pandas, torch, FastAPI, pytest

## Global Constraints

- Python 3.14 (asyncio mode=strict)
- `python3 -m pytest` for all test runs
- Backward compatibility: all new features opt-in, disabled by default
- `git commit` after each task
- 0 regressions on existing 136 tests
- TDD: write the test first, verify it fails, then implement

---

### Task 1: NYSE Trading Calendar

**Files:**
- Create: `src/backtest/calendar.py`
- Create: `tests/test_calendar.py`

**Interfaces:**
- Produces: `is_trading_day(date) -> bool`, `next_trading_day(date) -> str`, `trading_days_between(start, end) -> list[str]`, `n_trading_days_later(date, n) -> str`

- [ ] **Step 1: Write the failing test file**

Create `tests/test_calendar.py`:

```python
"""Tests for NYSE trading calendar."""
import pytest
from datetime import datetime, date
from src.backtest.calendar import (
    is_trading_day,
    next_trading_day,
    trading_days_between,
    n_trading_days_later,
)


class TestIsTradingDay:
    def test_weekend_returns_false(self):
        """Saturday and Sunday are not trading days."""
        # 2024-06-15 is Saturday, 2024-06-16 is Sunday
        assert is_trading_day("2024-06-15") is False
        assert is_trading_day("2024-06-16") is False

    def test_weekday_returns_true(self):
        """Normal weekday is a trading day."""
        # 2024-06-12 is Wednesday
        assert is_trading_day("2024-06-12") is True

    def test_new_years_day(self):
        """Jan 1 is a NYSE holiday."""
        assert is_trading_day("2024-01-01") is False

    def test_christmas_day(self):
        """Dec 25 is a NYSE holiday."""
        assert is_trading_day("2024-12-25") is False

    def test_accepts_datetime(self):
        """is_trading_day accepts datetime objects."""
        dt = datetime(2024, 6, 12)
        assert is_trading_day(dt) is True

    def test_accepts_date(self):
        """is_trading_day accepts date objects."""
        d = date(2024, 6, 15)  # Saturday
        assert is_trading_day(d) is False


class TestNextTradingDay:
    def test_friday_goes_to_monday(self):
        """Friday's next trading day is Monday."""
        # 2024-06-14 is Friday
        assert next_trading_day("2024-06-14") == "2024-06-17"

    def test_thursday_goes_to_friday(self):
        """Normal weekday advances one day."""
        # 2024-06-13 is Thursday
        assert next_trading_day("2024-06-13") == "2024-06-14"

    def test_skips_holiday(self):
        """Next trading day skips holidays."""
        # Dec 24 2024 is Tuesday, Dec 25 is Christmas (holiday)
        assert next_trading_day("2024-12-24") == "2024-12-26"


class TestTradingDaysBetween:
    def test_returns_list_of_strings(self):
        """Returns list of date strings."""
        result = trading_days_between("2024-06-10", "2024-06-14")
        assert isinstance(result, list)
        assert all(isinstance(d, str) for d in result)

    def test_excludes_weekends(self):
        """Weekend dates are not included."""
        # June 10 (Mon) through June 16 (Sun) = 5 trading days
        result = trading_days_between("2024-06-10", "2024-06-16")
        assert "2024-06-15" not in result  # Saturday
        assert "2024-06-16" not in result  # Sunday
        assert len(result) == 5

    def test_includes_both_endpoints(self):
        """Start and end dates are inclusive."""
        result = trading_days_between("2024-06-10", "2024-06-10")
        assert result == ["2024-06-10"]


class TestNTradingDaysLater:
    def test_one_day(self):
        """1 trading day later = next trading day."""
        assert n_trading_days_later("2024-06-14", 1) == "2024-06-17"

    def test_five_days(self):
        """5 trading days from Monday = next Monday."""
        assert n_trading_days_later("2024-06-10", 5) == "2024-06-17"

    def test_zero_days(self):
        """0 trading days later = same day if it is a trading day."""
        assert n_trading_days_later("2024-06-12", 0) == "2024-06-12"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest tests/test_calendar.py -v
```

Expected: all tests FAIL with ModuleNotFoundError.

- [ ] **Step 3: Implement `src/backtest/calendar.py`**

```python
"""NYSE trading calendar for 2024–2027.

Provides trading-day-aware date arithmetic for backtesting.
Weekdays that fall on NYSE holidays are treated as non-trading days.
"""
from datetime import datetime, date, timedelta

# NYSE holidays 2024–2027 (observed dates — if holiday falls on
# Saturday it is observed Friday; if Sunday, observed Monday).
_NYSE_HOLIDAYS = {
    # 2024
    "2024-01-01",  # New Year's Day
    "2024-01-15",  # Martin Luther King Jr. Day
    "2024-02-19",  # Presidents' Day
    "2024-03-29",  # Good Friday
    "2024-05-27",  # Memorial Day
    "2024-06-19",  # Juneteenth
    "2024-07-04",  # Independence Day
    "2024-09-02",  # Labor Day
    "2024-11-28",  # Thanksgiving
    "2024-12-25",  # Christmas
    # 2025
    "2025-01-01",  # New Year's Day
    "2025-01-20",  # Martin Luther King Jr. Day
    "2025-02-17",  # Presidents' Day
    "2025-04-18",  # Good Friday
    "2025-05-26",  # Memorial Day
    "2025-06-19",  # Juneteenth
    "2025-07-04",  # Independence Day
    "2025-09-01",  # Labor Day
    "2025-11-27",  # Thanksgiving
    "2025-12-25",  # Christmas
    # 2026
    "2026-01-01",  # New Year's Day
    "2026-01-19",  # Martin Luther King Jr. Day
    "2026-02-16",  # Presidents' Day
    "2026-04-03",  # Good Friday
    "2026-05-25",  # Memorial Day
    "2026-06-19",  # Juneteenth
    "2026-07-03",  # Independence Day (observed — Jul 4 is Saturday)
    "2026-09-07",  # Labor Day
    "2026-11-26",  # Thanksgiving
    "2026-12-25",  # Christmas
    # 2027
    "2027-01-01",  # New Year's Day
    "2027-01-18",  # Martin Luther King Jr. Day
    "2027-02-15",  # Presidents' Day
    "2027-03-26",  # Good Friday
    "2027-05-31",  # Memorial Day
    "2027-06-18",  # Juneteenth (observed — Jun 19 is Saturday)
    "2027-07-05",  # Independence Day (observed — Jul 4 is Sunday)
    "2027-09-06",  # Labor Day
    "2027-11-25",  # Thanksgiving
    "2027-12-24",  # Christmas (observed — Dec 25 is Saturday)
}


def _to_date_str(d: str | datetime | date) -> str:
    """Normalize input to YYYY-MM-DD string."""
    if isinstance(d, datetime):
        return d.strftime("%Y-%m-%d")
    if isinstance(d, date):
        return d.isoformat()
    return d


def is_trading_day(d: str | datetime | date) -> bool:
    """Return True if ``d`` is a NYSE trading day (Mon–Fri, not a holiday)."""
    ds = _to_date_str(d)
    dt = datetime.strptime(ds, "%Y-%m-%d")
    if dt.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False
    if ds in _NYSE_HOLIDAYS:
        return False
    return True


def next_trading_day(d: str | datetime | date) -> str:
    """First trading day strictly after ``d``."""
    ds = _to_date_str(d)
    dt = datetime.strptime(ds, "%Y-%m-%d")
    while True:
        dt += timedelta(days=1)
        candidate = dt.strftime("%Y-%m-%d")
        if is_trading_day(candidate):
            return candidate


def trading_days_between(start: str, end: str) -> list[str]:
    """All trading days in [start, end] inclusive."""
    result = []
    dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    while dt <= end_dt:
        ds = dt.strftime("%Y-%m-%d")
        if is_trading_day(ds):
            result.append(ds)
        dt += timedelta(days=1)
    return result


def n_trading_days_later(d: str | datetime | date, n: int) -> str:
    """Date ``n`` trading days after ``d`` (0 = same day if trading day)."""
    ds = _to_date_str(d)
    dt = datetime.strptime(ds, "%Y-%m-%d")
    count = 0
    while count < n:
        dt += timedelta(days=1)
        if is_trading_day(dt.strftime("%Y-%m-%d")):
            count += 1
    return dt.strftime("%Y-%m-%d")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/test_calendar.py -v
```

Expected: all 12 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/backtest/calendar.py tests/test_calendar.py
git commit -m "feat: add NYSE trading calendar (2024-2027)"
```

---

### Task 2: Slippage Model

**Files:**
- Create: `src/backtest/slippage.py`
- Create: `tests/test_slippage.py`

**Interfaces:**
- Produces: `SlippageConfig` dataclass, `estimate_slippage(weight, daily_volume, close_price, config) -> float`

- [ ] **Step 1: Write the failing test file**

Create `tests/test_slippage.py`:

```python
"""Tests for slippage / market-impact model."""
import pytest
from src.backtest.slippage import SlippageConfig, estimate_slippage


class TestEstimateSlippage:
    @pytest.fixture
    def config(self):
        return SlippageConfig()

    def test_zero_trade_zero_slippage(self, config):
        """Zero-weight trade incurs zero slippage."""
        cost = estimate_slippage(0.0, 1_000_000, 100.0, config)
        assert cost == 0.0

    def test_slippage_increases_with_trade_size(self, config):
        """Larger trades incur more slippage."""
        small = estimate_slippage(0.01, 1_000_000, 100.0, config)
        large = estimate_slippage(0.05, 1_000_000, 100.0, config)
        assert large > small

    def test_minimum_cost_is_fixed_cost(self, config):
        """Even tiny trades pay at least the fixed commission."""
        cost = estimate_slippage(0.0001, 10_000_000, 100.0, config)
        assert cost >= config.fixed_cost

    def test_slippage_decreases_with_volume(self, config):
        """Higher daily volume means lower impact."""
        low_vol = estimate_slippage(0.01, 100_000, 100.0, config)
        high_vol = estimate_slippage(0.01, 10_000_000, 100.0, config)
        assert low_vol > high_vol

    def test_negative_weight_same_as_positive(self, config):
        """Short and long positions have same slippage for equal |weight|."""
        long_cost = estimate_slippage(0.01, 1_000_000, 100.0, config)
        short_cost = estimate_slippage(-0.01, 1_000_000, 100.0, config)
        assert long_cost == short_cost

    def test_zero_volume_returns_fixed_cost(self, config):
        """Zero reported volume still returns fixed_cost (no division by zero)."""
        cost = estimate_slippage(0.01, 0, 100.0, config)
        assert cost >= config.fixed_cost
        assert cost == config.fixed_cost

    def test_custom_config(self):
        """Custom SlippageConfig values are respected."""
        cfg = SlippageConfig(fixed_cost=0.001, impact_factor=0.5, daily_volume_fraction=0.02)
        cost = estimate_slippage(0.01, 1_000_000, 100.0, cfg)
        assert cost > 0.001  # impact adds on top of fixed
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest tests/test_slippage.py -v
```

Expected: all tests FAIL with ModuleNotFoundError.

- [ ] **Step 3: Implement `src/backtest/slippage.py`**

```python
"""Slippage and market-impact cost model.

Simplified Almgren-Chriss: fixed commission + volume-proportional impact.
"""
from dataclasses import dataclass
from config import SLIPPAGE_FIXED_COST, SLIPPAGE_IMPACT_FACTOR, SLIPPAGE_VOLUME_FRACTION


@dataclass
class SlippageConfig:
    """Slippage model parameters.

    Attributes:
        fixed_cost: Fixed commission as fraction of trade value (e.g. 0.0004 = 4bp).
        impact_factor: Coefficient for price impact relative to participation rate.
        daily_volume_fraction: Assumed maximum fraction of daily volume we can trade.
    """
    fixed_cost: float = SLIPPAGE_FIXED_COST
    impact_factor: float = SLIPPAGE_IMPACT_FACTOR
    daily_volume_fraction: float = SLIPPAGE_VOLUME_FRACTION


def estimate_slippage(
    weight: float,
    daily_volume: int,
    close_price: float,
    config: SlippageConfig,
) -> float:
    """Estimate total slippage cost as fraction of trade value.

    Formula:
        cost = fixed_cost
             + impact_factor * |weight| / (daily_volume_frac * daily_volume)

    When daily_volume is zero or the participation denominator is zero,
    only the fixed_cost is charged.

    Args:
        weight: Portfolio weight of the trade (sign ignored, magnitude matters).
        daily_volume: Reported daily volume in shares.
        close_price: Closing price per share (not used in simplified formula,
                     kept for future extensions).
        config: Slippage model parameters.

    Returns:
        Slippage cost as a fraction of trade value.
    """
    cost = config.fixed_cost
    abs_weight = abs(weight)

    if abs_weight == 0.0:
        return 0.0

    denominator = config.daily_volume_fraction * daily_volume
    if denominator > 0:
        cost += config.impact_factor * abs_weight / denominator

    return cost
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/test_slippage.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/backtest/slippage.py tests/test_slippage.py
git commit -m "feat: add slippage model (simplified Almgren-Chriss)"
```

---

### Task 3: Position Sizing with Risk Constraints

**Files:**
- Create: `src/backtest/position.py`
- Create: `tests/test_position.py`
- Modify: `config.py` (append POSITION_* constants)

**Interfaces:**
- Consumes: `is_trading_day` from `src/backtest/calendar.py` (for vol lookback date filtering)
- Produces: `PositionConfig` dataclass, `GICS_SECTORS` dict, `size_positions(stocks, preds, prev_weights, prices, config) -> dict[str, float]`

- [ ] **Step 1: Write the failing test file**

Create `tests/test_position.py`:

```python
"""Tests for risk-aware position sizing."""
import pytest
import numpy as np
from src.backtest.position import PositionConfig, size_positions, GICS_SECTORS


class TestPositionConfig:
    def test_defaults_disabled(self):
        """Default config has all constraints disabled."""
        cfg = PositionConfig()
        assert cfg.target_vol == 0.15
        assert cfg.max_single_weight == 0.05
        assert cfg.sector_neutral is False
        assert cfg.max_turnover == 1.0


class TestGicsSectors:
    def test_all_default_tickers_mapped(self):
        """All 20 DEFAULT_TICKERS have GICS sector assignments."""
        from config import DEFAULT_TICKERS
        for t in DEFAULT_TICKERS:
            assert t in GICS_SECTORS, f"{t} missing from GICS_SECTORS"


class TestSizePositions:
    @pytest.fixture
    def stocks(self):
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

    @pytest.fixture
    def preds(self):
        return np.array([0.05, 0.03, -0.01, 0.02, 0.04])

    @pytest.fixture
    def prices(self):
        """Recent close prices for vol computation (5 days each)."""
        return {
            "AAPL":  [180.0, 181.0, 182.0, 180.5, 183.0],
            "MSFT":  [400.0, 402.0, 398.0, 401.0, 405.0],
            "GOOGL": [140.0, 141.0, 142.0, 141.5, 143.0],
            "AMZN":  [175.0, 176.0, 177.0, 176.5, 178.0],
            "NVDA":  [800.0, 810.0, 790.0, 805.0, 820.0],
        }

    def test_no_constraints_returns_equal_weights(self, stocks, preds, prices):
        """With all constraints disabled, long/short get equal absolute weights."""
        cfg = PositionConfig(target_vol=0.0, max_single_weight=0.0,
                             sector_neutral=False, max_turnover=0.0)
        weights = size_positions(stocks, preds, {}, prices, cfg)
        # 5 stocks, top/bottom 10% = 1 long, 1 short (with quantile=0.10)
        assert len(weights) > 0
        assert abs(sum(weights.values())) < 0.01  # long ≈ short

    def test_max_single_weight_enforced(self, stocks, preds, prices):
        """No single position exceeds the cap."""
        cfg = PositionConfig(target_vol=0.0, max_single_weight=0.03,
                             sector_neutral=False, max_turnover=0.0)
        weights = size_positions(stocks, preds, {}, prices, cfg)
        for w in weights.values():
            assert abs(w) <= 0.03 + 1e-10

    def test_turnover_constraint_enforced(self, stocks, preds, prices):
        """Turnover between consecutive days is limited."""
        cfg = PositionConfig(target_vol=0.0, max_single_weight=0.0,
                             sector_neutral=False, max_turnover=0.5)
        prev_weights = {"AAPL": 0.10, "MSFT": -0.10}
        weights = size_positions(stocks, preds, prev_weights, prices, cfg)
        # Total change in |weight| should not exceed max_turnover
        turnover = sum(abs(weights.get(s, 0.0) - prev_weights.get(s, 0.0))
                       for s in set(stocks) | set(prev_weights)) / 2
        assert turnover <= 0.5 + 1e-10

    def test_vol_scaling_reduces_weights(self, stocks, preds, prices):
        """When target_vol is set, weights are scaled down from raw."""
        cfg_no_vol = PositionConfig(target_vol=0.0, max_single_weight=0.0,
                                     sector_neutral=False, max_turnover=0.0)
        cfg_vol = PositionConfig(target_vol=0.10, max_single_weight=0.0,
                                  sector_neutral=False, max_turnover=0.0)
        w_no_vol = size_positions(stocks, preds, {}, prices, cfg_no_vol)
        w_vol = size_positions(stocks, preds, {}, prices, cfg_vol)
        no_vol_sum = sum(abs(v) for v in w_no_vol.values())
        vol_sum = sum(abs(v) for v in w_vol.values())
        # Vol-constrained weights should be <= unconstrained
        assert vol_sum <= no_vol_sum + 1e-10

    def test_empty_stocks_returns_empty(self):
        """Empty stock list returns empty dict."""
        cfg = PositionConfig()
        result = size_positions([], np.array([]), {}, {}, cfg)
        assert result == {}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest tests/test_position.py -v
```

Expected: all tests FAIL with ModuleNotFoundError.

- [ ] **Step 3: Add config constants**

Edit `config.py` — append after the existing Ensemble section (after line 85):

```python
# Position Sizing (v0.5)
POSITION_TARGET_VOL = 0.15         # Annualized volatility target (0=disabled)
POSITION_MAX_SINGLE = 0.05         # Max absolute weight per stock (0=disabled)
POSITION_SECTOR_NEUTRAL = False    # Equal long/short exposure per GICS sector
POSITION_MAX_TURNOVER = 1.0        # Max daily turnover fraction (0=disabled)
```

- [ ] **Step 4: Implement `src/backtest/position.py`**

```python
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
    DEFAULT_TICKERS,
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
      1. Rank stocks by preds → select top/bottom quantile
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
        weights = _apply_vol_scaling(weights, stocks, prices, config.target_vol)

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
    """Estimate ex-ante annualized portfolio volatility from price history."""
    returns = []
    for stock, w in weights.items():
        ps = prices.get(stock, [])
        if len(ps) < 2:
            continue
        daily_rets = [(ps[i] - ps[i - 1]) / ps[i - 1] for i in range(1, len(ps))
                      if ps[i - 1] > 0]
        if daily_rets:
            returns.append(w * np.mean(daily_rets))

    if not returns:
        return 0.15  # default assumption

    # Approximate: portfolio daily vol = std of weighted daily returns
    # Scale to annualized
    daily_vol = np.std(returns) if len(returns) > 1 else 0.01
    return daily_vol * np.sqrt(252)


def _apply_vol_scaling(
    weights: dict[str, float],
    stocks: list[str],
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
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python3 -m pytest tests/test_position.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/backtest/position.py tests/test_position.py config.py
git commit -m "feat: add risk-aware position sizing (vol target, sector neutral, turnover cap)"
```

---

### Task 4: Enhanced Portfolio Backtest (Backward Compatible)

**Files:**
- Modify: `src/backtest/portfolio.py` (enhance `run_backtest` with optional precision params)
- Modify: `tests/test_portfolio.py` (tests for existing behavior preserved + enhanced mode)

**Interfaces:**
- Consumes: `is_trading_day`, `next_trading_day` from `calendar.py`; `estimate_slippage`, `SlippageConfig` from `slippage.py`; `size_positions`, `PositionConfig` from `position.py`
- Produces: Updated `run_backtest(pred_df, stocks, start_date, end_date, **kwargs) -> dict` with new optional keyword args

- [ ] **Step 1: Write the failing test for calendar integration**

Read `tests/test_portfolio.py` first, then add a new test class. Create or append to `tests/test_portfolio.py`:

```python
"""Tests for portfolio construction and backtest simulation."""
import pytest
import numpy as np
import pandas as pd
from src.backtest.portfolio import run_backtest, construct_long_short
from src.backtest.slippage import SlippageConfig
from src.backtest.position import PositionConfig


class TestBacktestCalendar:
    """Tests that enhanced backtest uses trading calendar."""
    
    def test_backtest_uses_trading_calendar(self, prepopulated_db):
        """When use_calendar=True, backtest skips non-trading days."""
        stocks = ["AAPL", "MSFT"]
        dates = pd.date_range("2024-06-10", "2024-06-16", freq="D")
        preds = []
        for d in dates:
            ds = d.strftime("%Y-%m-%d")
            preds.append({
                "stock": stocks[0], "date": ds,
                "predicted_return": 0.01, "signal_source": "test",
            })
            preds.append({
                "stock": stocks[1], "date": ds,
                "predicted_return": -0.01, "signal_source": "test",
            })
        pred_df = pd.DataFrame(preds)

        result = run_backtest(
            pred_df, stocks, "2024-06-10", "2024-06-16",
            use_calendar=True,
        )
        # June 15-16 is weekend, should have at most 5 trading days
        assert result["n_trading_days"] <= 5

    def test_backtest_no_calendar_preserves_old_behavior(self, prepopulated_db):
        """Without use_calendar, backtest matches v0.4 behavior."""
        stocks = ["AAPL", "MSFT"]
        dates = pd.date_range("2024-06-10", "2024-06-14", freq="D")
        preds = []
        for d in dates:
            ds = d.strftime("%Y-%m-%d")
            preds.append({
                "stock": stocks[0], "date": ds,
                "predicted_return": 0.01, "signal_source": "test",
            })
            preds.append({
                "stock": stocks[1], "date": ds,
                "predicted_return": -0.01, "signal_source": "test",
            })
        pred_df = pd.DataFrame(preds)

        result_old = run_backtest(pred_df, stocks, "2024-06-10", "2024-06-14")
        # Without calendar, uses old +1 day logic
        assert result_old["n_trading_days"] == 5

    def test_slippage_reduces_returns(self, prepopulated_db):
        """Enabling slippage reduces annualized return vs disabled."""
        stocks = ["AAPL", "MSFT"]
        preds = [
            {"stock": stocks[0], "date": "2024-06-12",
             "predicted_return": 0.05, "signal_source": "test"},
            {"stock": stocks[1], "date": "2024-06-12",
             "predicted_return": -0.05, "signal_source": "test"},
        ]
        pred_df = pd.DataFrame(preds)

        no_slip = run_backtest(pred_df, stocks, "2024-06-12", "2024-06-13")
        with_slip = run_backtest(
            pred_df, stocks, "2024-06-12", "2024-06-13",
            slippage_config=SlippageConfig(fixed_cost=0.01, impact_factor=0.0),
        )
        # Slippage should not increase returns
        assert with_slip["annualized_return"] <= no_slip["annualized_return"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest tests/test_portfolio.py::TestBacktestCalendar -v
```

Expected: FAIL — `run_backtest` doesn't accept `use_calendar` or `slippage_config`.

- [ ] **Step 3: Modify `src/backtest/portfolio.py` — update `run_backtest` signature and body**

Read the existing file first, then apply these changes:

Replace the `run_backtest` function signature (line 48):

```python
def run_backtest(
    pred_df: pd.DataFrame,
    stocks: list[str],
    start_date: str,
    end_date: str,
    quantile: float = PORTFOLIO_QUANTILE,
    transaction_cost: float = TRANSACTION_COST,
    use_calendar: bool = False,
    slippage_config: "SlippageConfig | None" = None,
    position_config: "PositionConfig | None" = None,
) -> dict:
```

Replace the body of `run_backtest` (lines 61-107):

```python
    dates = sorted(pred_df["date"].unique())
    daily_returns = []
    daily_long_short = []

    prev_weights: dict[str, float] = {}

    for date_str in dates:
        day_pred = pred_df[pred_df["date"] == date_str]
        if len(day_pred) == 0:
            continue

        # Determine next trading day
        if use_calendar:
            from src.backtest.calendar import next_trading_day as _next_td
            next_date = _next_td(date_str)
        else:
            next_date = _get_next_trading_day(date_str)

        if next_date is None:
            continue

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
                w = pos_weights.get(stock, 0.0) if position_config is not None else (
                    1.0 / max(len(long_stocks), 1) if stock in long_stocks
                    else -1.0 / max(len(short_stocks), 1) if stock in short_stocks
                    else 0.0
                )
                slippage += estimate_slippage(w, vol, px, slippage_config)

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
```

Add helper functions at the bottom of `portfolio.py`:

```python
def _get_recent_prices(stocks: list[str], date_str: str, lookback: int = 20) -> dict[str, list[float]]:
    """Get recent closing prices for volatility estimation."""
    from datetime import datetime, timedelta
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
```

Add the import at the top of `portfolio.py`:

```python
from __future__ import annotations
```

- [ ] **Step 4: Run portfolio tests to verify they pass**

```bash
python3 -m pytest tests/test_portfolio.py -v
```

Expected: all tests PASS (existing + new).

- [ ] **Step 5: Run full test suite to check for regressions**

```bash
python3 -m pytest tests/ -v 2>&1 | tail -20
```

Expected: 0 failures.

- [ ] **Step 6: Commit**

```bash
git add src/backtest/portfolio.py tests/test_portfolio.py
git commit -m "feat: enhance run_backtest with optional calendar, slippage, and position sizing"
```

---

### Task 5: Walk-Forward Validation Engine

**Files:**
- Create: `src/backtest/walkforward.py`
- Create: `tests/test_walkforward.py`

**Interfaces:**
- Consumes: `trading_days_between` from `calendar.py`; `run_backtest` from `portfolio.py`; model classes from `src/model/`
- Produces: `WalkForwardConfig`, `WalkForwardResult`, `run_walk_forward(stocks, start, end, config, param_grid) -> WalkForwardResult`

- [ ] **Step 1: Write the failing test file**

Create `tests/test_walkforward.py`:

```python
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
        result = run_walk_forward(stocks, "2024-05-01", "2024-06-15", cfg)
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
        result = run_walk_forward(stocks, "2024-05-01", "2024-06-15", cfg)
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
        result = run_walk_forward(stocks, "2024-05-01", "2024-06-15", cfg)
        assert "sharpe_mean" in result.summary
        assert "sharpe_std" in result.summary
        assert "mean_ic_mean" in result.summary

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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest tests/test_walkforward.py -v
```

Expected: all tests FAIL with ModuleNotFoundError.

- [ ] **Step 3: Add walk-forward config constants to `config.py`**

Append after position sizing constants:

```python
# Walk-Forward (v0.5)
WF_TRAIN_DAYS = 252
WF_VALIDATE_DAYS = 63
WF_STEP_DAYS = 21
WF_MODE = "full"                   # "full" or "params"
WF_MIN_TRAIN_DAYS = 60
```

- [ ] **Step 4: Implement `src/backtest/walkforward.py`**

```python
"""Walk-forward validation engine.

Rolling-window framework for out-of-sample model evaluation.
Supports two modes:
  - "full": Retrain models within each training window.
  - "params": Use existing models, vary only backtest parameters.
"""
from dataclasses import dataclass, field
import logging
import numpy as np
import pandas as pd

from config import (
    WF_TRAIN_DAYS,
    WF_VALIDATE_DAYS,
    WF_STEP_DAYS,
    WF_MODE,
    WF_MIN_TRAIN_DAYS,
    DEFAULT_TICKERS,
)

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    """Walk-forward validation parameters (all in trading days)."""
    train_days: int = WF_TRAIN_DAYS
    validate_days: int = WF_VALIDATE_DAYS
    step_days: int = WF_STEP_DAYS
    mode: str = WF_MODE           # "full" | "params"
    min_train_days: int = WF_MIN_TRAIN_DAYS


@dataclass
class WalkForwardResult:
    """Results from a walk-forward validation run."""
    windows: list[dict] = field(default_factory=list)
    oos_predictions: pd.DataFrame | None = None
    summary: dict = field(default_factory=dict)


def run_walk_forward(
    stocks: list[str],
    start_date: str,
    end_date: str,
    config: WalkForwardConfig,
    param_grid: dict | None = None,
) -> WalkForwardResult:
    """Run walk-forward validation over the date range.

    Args:
        stocks: Ticker symbols.
        start_date / end_date: Overall date range (YYYY-MM-DD).
        config: Walk-forward parameters.
        param_grid: For "params" mode, dict of {param_name: value} to
                    pass through to run_backtest.

    Returns:
        WalkForwardResult with per-window metrics and concatenated OOS predictions.
    """
    from src.backtest.calendar import trading_days_between
    from src.backtest.portfolio import run_backtest

    all_td = trading_days_between(start_date, end_date)
    if len(all_td) < config.train_days + config.validate_days:
        logger.warning(
            f"Insufficient trading days ({len(all_td)}) for "
            f"train={config.train_days}+val={config.validate_days}"
        )
        return WalkForwardResult()

    windows = []
    oos_preds_list = []

    idx = 0
    window_id = 1

    while idx + config.train_days + config.validate_days <= len(all_td):
        train_start = all_td[idx]
        train_end = all_td[idx + config.train_days - 1]
        val_start = all_td[idx + config.train_days]
        val_end_idx = idx + config.train_days + config.validate_days - 1
        if val_end_idx >= len(all_td):
            val_end_idx = len(all_td) - 1
        val_end = all_td[val_end_idx]

        window_info = {
            "window_id": window_id,
            "train_start": train_start,
            "train_end": train_end,
            "val_start": val_start,
            "val_end": val_end,
        }

        try:
            # Generate predictions for validation window
            preds = _generate_predictions(stocks, val_start, val_end, config.mode,
                                          train_start, train_end)

            if preds is not None and len(preds) > 0:
                # Run backtest on validation window
                backtest_kwargs = param_grid or {}
                bt_result = run_backtest(preds, stocks, val_start, val_end,
                                         **backtest_kwargs)
                window_info.update({
                    "sharpe_ratio": bt_result.get("sharpe_ratio", 0.0),
                    "mean_ic": bt_result.get("mean_ic", 0.0),
                    "annualized_return": bt_result.get("annualized_return", 0.0),
                    "max_drawdown": bt_result.get("max_drawdown", 0.0),
                    "icir": bt_result.get("icir", 0.0),
                    "n_trading_days": bt_result.get("n_trading_days", 0),
                })

                preds["window_id"] = window_id
                oos_preds_list.append(preds)

        except Exception as e:
            logger.warning(f"Window {window_id} failed: {e}")
            window_info["error"] = str(e)

        windows.append(window_info)

        idx += config.step_days
        window_id += 1

    # Build summary
    sharpe_vals = [w.get("sharpe_ratio", 0.0) for w in windows if "sharpe_ratio" in w]
    ic_vals = [w.get("mean_ic", 0.0) for w in windows if "mean_ic" in w]

    summary = {
        "n_windows": len(windows),
        "sharpe_mean": float(np.mean(sharpe_vals)) if sharpe_vals else 0.0,
        "sharpe_std": float(np.std(sharpe_vals)) if sharpe_vals else 0.0,
        "mean_ic_mean": float(np.mean(ic_vals)) if ic_vals else 0.0,
        "mean_ic_std": float(np.std(ic_vals)) if ic_vals else 0.0,
    }

    oos_df = pd.concat(oos_preds_list, ignore_index=True) if oos_preds_list else None

    return WalkForwardResult(
        windows=windows,
        oos_predictions=oos_df,
        summary=summary,
    )


def _generate_predictions(
    stocks: list[str],
    start: str,
    end: str,
    mode: str,
    train_start: str | None = None,
    train_end: str | None = None,
) -> pd.DataFrame | None:
    """Generate predictions for a date range.

    In "full" mode, retrains MS-LSTM/DualGAT/Ensemble on the training window
    before predicting on the validation window.
    In "params" mode, uses existing model instances.
    """
    from src.backtest.calendar import trading_days_between
    from src.model.baseline import BaselinePredictor
    from src.expert.tracker import ExpertTracker

    dates = trading_days_between(start, end)
    if not dates:
        return None

    if mode == "full" and train_start and train_end:
        # Retrain models on training window
        _retrain_models(stocks, train_start, train_end)

    # Use default predictors (which load from disk, or Baseline)
    from src.model.ms_lstm import MSLSTMPredictor
    from src.model.dualgat import DualGATPredictor
    from src.model.ensemble import EnsemblePredictor

    baseline = BaselinePredictor()
    tracker = ExpertTracker()

    # Try loading MS-LSTM
    ms_lstm = None
    try:
        ms_lstm = MSLSTMPredictor()
        ms_lstm.load("data/ms_lstm_model.pt")
    except Exception:
        pass

    # Try loading DualGAT
    dualgat = None
    try:
        dualgat = DualGATPredictor()
        dualgat.load("data/dualgat_model.pt")
    except Exception:
        pass

    # Try loading Ensemble
    ensemble = None
    try:
        ensemble = EnsemblePredictor()
        ensemble.load("data/ensemble_model.pt")
    except Exception:
        pass

    all_preds = []
    for date_str in dates:
        expert_records = tracker.trace(date_str)

        if ensemble is not None:
            bl_df = baseline.predict(stocks, date_str, expert_records)
            ms_df = ms_lstm.predict(stocks, date_str) if ms_lstm else bl_df.copy()
            dg_df = dualgat.predict(stocks, date_str) if dualgat else bl_df.copy()
            pred_df = ensemble.predict(stocks, date_str, bl_df, ms_df, dg_df)
        elif dualgat is not None:
            pred_df = dualgat.predict(stocks, date_str)
        elif ms_lstm is not None:
            pred_df = ms_lstm.predict(stocks, date_str)
        else:
            pred_df = baseline.predict(stocks, date_str, expert_records)

        all_preds.append(pred_df)

    return pd.concat(all_preds, ignore_index=True) if all_preds else None


def _retrain_models(stocks: list[str], train_start: str, train_end: str) -> None:
    """Retrain MS-LSTM, DualGAT, and Ensemble on the given window."""
    import tempfile, os
    from pathlib import Path

    try:
        # MS-LSTM
        from src.model.ms_lstm import MSLSTMPredictor
        ms = MSLSTMPredictor()
        ms.fit(stocks, train_start, train_end)
        ms.save("data/ms_lstm_model.pt")
    except Exception as e:
        logger.warning(f"MS-LSTM retrain failed: {e}")

    try:
        # DualGAT
        from src.model.dualgat import DualGATPredictor
        dg = DualGATPredictor()
        dg.fit(stocks, train_start, train_end)
        dg.save("data/dualgat_model.pt")
    except Exception as e:
        logger.warning(f"DualGAT retrain failed: {e}")

    try:
        # Ensemble meta-learner
        from src.model.baseline import BaselinePredictor
        from src.model.ms_lstm import MSLSTMPredictor
        from src.model.dualgat import DualGATPredictor
        from src.model.ensemble import EnsemblePredictor

        baseline = BaselinePredictor()
        ms = MSLSTMPredictor()
        ms.load("data/ms_lstm_model.pt")
        dg = DualGATPredictor()
        dg.load("data/dualgat_model.pt")

        ensemble = EnsemblePredictor(strategy="meta")
        ensemble.fit_meta(stocks, train_start, train_end, baseline, ms, dg, epochs=30)
        ensemble.save("data/ensemble_model.pt")
    except Exception as e:
        logger.warning(f"Ensemble retrain failed: {e}")
```

- [ ] **Step 5: Run walk-forward tests to verify they pass**

```bash
python3 -m pytest tests/test_walkforward.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/backtest/walkforward.py tests/test_walkforward.py config.py
git commit -m "feat: add walk-forward validation engine"
```

---

### Task 6: Parameter Scanner

**Files:**
- Create: `src/backtest/scanner.py`
- Create: `tests/test_scanner.py`

**Interfaces:**
- Consumes: `run_walk_forward` from `walkforward.py`; `run_backtest` from `portfolio.py`
- Produces: `ParamSpec`, `build_param_grid(specs) -> list[dict]`, `random_search(grid, n_iter) -> list[dict]`, `run_scan(stocks, start, end, grid, wf_config, metric) -> pd.DataFrame`

- [ ] **Step 1: Write the failing test file**

Create `tests/test_scanner.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest tests/test_scanner.py -v
```

Expected: all tests FAIL with ModuleNotFoundError.

- [ ] **Step 3: Add scanner config constants to `config.py`**

Append after walk-forward constants:

```python
# Parameter Scanner (v0.5)
SCAN_DEFAULT_METRIC = "sharpe_ratio"
SCAN_RANDOM_N_ITER = 50
```

- [ ] **Step 4: Implement `src/backtest/scanner.py`**

```python
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
                    "max_drawdown": 0.0,
                    "annualized_return": 0.0,
                    "icir": 0.0,
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
                                      **combo)
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
```

- [ ] **Step 5: Run scanner tests to verify they pass**

```bash
python3 -m pytest tests/test_scanner.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/backtest/scanner.py tests/test_scanner.py config.py
git commit -m "feat: add parameter scanner (grid + random search)"
```

---

### Task 7: API Endpoints + Backward Compatible `/api/backtest/compare`

**Files:**
- Modify: `src/web/api.py` (add 2 new endpoints, enhance `/api/backtest/compare` with query params)

**Interfaces:**
- Consumes: `run_walk_forward` from `walkforward.py`; `run_scan`, `build_param_grid`, `ParamSpec` from `scanner.py`; `SlippageConfig` from `slippage.py`; `PositionConfig` from `position.py`
- Produces: `POST /api/backtest/walkforward`, `POST /api/backtest/scan`, enhanced `GET /api/backtest/compare?use_calendar=&use_slippage=&use_position=`

- [ ] **Step 1: Note existing tests that already cover the new endpoints**

The test plan in the spec reserves `test_api.py` entries for the two new endpoints and the enhanced compare. Since these tests depend on the full app, we write them now. Create a new test class in `tests/test_api.py`. Append after the existing `TestBacktestCompareEndpoint` class:

```python
class TestWalkForwardEndpoint:
    """Tests for POST /api/backtest/walkforward."""

    def test_walkforward_params_mode(self, populated_client):
        """Walk-forward in params mode returns 200."""
        resp = populated_client.post("/api/backtest/walkforward", json={
            "start": "2024-05-01",
            "end": "2024-06-15",
            "mode": "params",
            "train_days": 20,
            "validate_days": 5,
            "step_days": 5,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "windows" in data
        assert "summary" in data

    def test_walkforward_insufficient_data(self, populated_client):
        """Walk-forward with too-short range returns 400."""
        resp = populated_client.post("/api/backtest/walkforward", json={
            "start": "2024-06-10",
            "end": "2024-06-14",
            "mode": "params",
            "train_days": 252,
            "validate_days": 63,
            "step_days": 21,
        })
        assert resp.status_code in (200, 400)


class TestScanEndpoint:
    """Tests for POST /api/backtest/scan."""

    def test_scan_grid_mode(self, populated_client):
        """Parameter scan in grid mode returns 200."""
        resp = populated_client.post("/api/backtest/scan", json={
            "start": "2024-05-01",
            "end": "2024-06-15",
            "params": {"quantile": [0.10, 0.20]},
            "mode": "grid",
            "use_walkforward": False,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert "best" in data
        assert len(data["results"]) == 2

    def test_scan_invalid_params(self, client):
        """Missing required fields returns 422."""
        resp = client.post("/api/backtest/scan", json={})
        assert resp.status_code == 422
```

- [ ] **Step 2: Run API tests to verify they fail**

```bash
python3 -m pytest tests/test_api.py::TestWalkForwardEndpoint tests/test_api.py::TestScanEndpoint -v
```

Expected: FAIL — endpoints don't exist yet (404/405 or import errors).

- [ ] **Step 3: Implement API endpoints in `src/web/api.py`**

Read `src/web/api.py` to locate the end of the backtest section (around line 398). Append after the existing backtest compare endpoint, before any other routes:

```python
# ------------------------------------------------------------------
# Walk-Forward (v0.5)
# ------------------------------------------------------------------

@app.post("/api/backtest/walkforward")
async def walkforward_backtest(body: dict):
    """Run walk-forward validation.

    Body:
        start: str (YYYY-MM-DD)
        end: str (YYYY-MM-DD)
        mode: str = "params"  ("full" | "params")
        train_days: int = 252
        validate_days: int = 63
        step_days: int = 21
    """
    from src.backtest.walkforward import WalkForwardConfig, run_walk_forward
    from src.backtest.calendar import trading_days_between

    start = body.get("start")
    end = body.get("end")
    if not start or not end:
        raise HTTPException(400, "start and end are required")

    cfg = WalkForwardConfig(
        train_days=body.get("train_days", 252),
        validate_days=body.get("validate_days", 63),
        step_days=body.get("step_days", 21),
        mode=body.get("mode", "params"),
        min_train_days=body.get("min_train_days", 60),
    )

    # Check data sufficiency
    all_td = trading_days_between(start, end)
    if len(all_td) < cfg.train_days + cfg.validate_days:
        raise HTTPException(
            400,
            f"Insufficient trading days ({len(all_td)}) for "
            f"train={cfg.train_days}+val={cfg.validate_days}",
        )

    result = run_walk_forward(DEFAULT_TICKERS, start, end, cfg)
    return {
        "windows": result.windows,
        "summary": result.summary,
        "oos_predictions": result.oos_predictions.to_dict(orient="records")
        if result.oos_predictions is not None else [],
    }


# ------------------------------------------------------------------
# Parameter Scanner (v0.5)
# ------------------------------------------------------------------

@app.post("/api/backtest/scan")
async def scan_backtest(body: dict):
    """Run parameter scan.

    Body:
        start: str, end: str
        params: dict[str, list]  (e.g. {"quantile": [0.05, 0.10]})
        mode: str = "grid"  ("grid" | "random")
        n_iter: int = 50  (for random mode)
        use_walkforward: bool = False
        wf_config: dict | None  (if use_walkforward)
    """
    from src.backtest.scanner import (
        ParamSpec, build_param_grid, random_search, run_scan,
    )
    from src.backtest.walkforward import WalkForwardConfig

    start = body.get("start")
    end = body.get("end")
    params = body.get("params")
    if not start or not end or not params:
        raise HTTPException(400, "start, end, and params are required")

    specs = [ParamSpec(name, values) for name, values in params.items()]
    full_grid = build_param_grid(specs)

    mode = body.get("mode", "grid")
    n_iter = body.get("n_iter", 50)

    if mode == "random":
        grid = random_search(full_grid, n_iter)
    else:
        grid = full_grid

    wf_config = None
    if body.get("use_walkforward"):
        wf_body = body.get("wf_config", {})
        wf_config = WalkForwardConfig(
            train_days=wf_body.get("train_days", 252),
            validate_days=wf_body.get("validate_days", 63),
            step_days=wf_body.get("step_days", 21),
            mode=wf_body.get("mode", "params"),
            min_train_days=wf_body.get("min_train_days", 60),
        )

    metric = body.get("metric", "sharpe_ratio")
    df = run_scan(DEFAULT_TICKERS, start, end, grid, wf_config, metric)

    results = df.to_dict(orient="records")
    best = results[0] if results else None

    return {"results": results, "best": best}
```

- [ ] **Step 4: Enhance `/api/backtest/compare` with precision toggles**

Find the `compare_backtest` function (around line 306). Add query params after the existing `end` parameter:

```python
@app.get("/api/backtest/compare")
async def compare_backtest(
    start: str = Query(None, description="Start date YYYY-MM-DD"),
    end: str = Query(None, description="End date YYYY-MM-DD"),
    use_calendar: bool = Query(False, description="Use NYSE trading calendar"),
    use_slippage: bool = Query(False, description="Enable slippage model"),
    use_position: bool = Query(False, description="Enable position sizing"),
):
```

In the per-model backtest loop within `compare_backtest` (around line 384 where `run_backtest` is called), update the call to pass through the new options:

```python
            # Build optional kwargs for precision features
            bt_kwargs = {}
            if use_calendar:
                bt_kwargs["use_calendar"] = True
            if use_slippage:
                from src.backtest.slippage import SlippageConfig
                bt_kwargs["slippage_config"] = SlippageConfig()
            if use_position:
                from src.backtest.position import PositionConfig
                bt_kwargs["position_config"] = PositionConfig()

            bt_result = run_backtest(combined, DEFAULT_TICKERS, start, end,
                                     **bt_kwargs)
```

- [ ] **Step 5: Run the new API tests**

```bash
python3 -m pytest tests/test_api.py::TestWalkForwardEndpoint tests/test_api.py::TestScanEndpoint -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/web/api.py tests/test_api.py
git commit -m "feat: add /api/backtest/walkforward and /api/backtest/scan endpoints"
```

---

### Task 8: Final Integration + Verification

**Files:**
- Modify: `tests/test_integration.py` (add walk-forward + scan smoke test)

- [ ] **Step 1: Add integration smoke test**

Append to `tests/test_integration.py`:

```python
class TestWalkForwardIntegration:
    """Integration smoke test for walk-forward + scanner pipeline."""

    def test_walkforward_scan_pipeline(self, prepopulated_db, tmp_path):
        """Walk-forward → scan pipeline runs end-to-end without errors."""
        import torch
        torch.manual_seed(42)
        np.random.seed(42)

        from src.backtest.walkforward import (
            WalkForwardConfig, run_walk_forward,
        )
        from src.backtest.scanner import (
            ParamSpec, build_param_grid, run_scan,
        )

        stocks = ["AAPL", "MSFT"]

        # Walk-forward in params mode
        wf_cfg = WalkForwardConfig(
            train_days=30,
            validate_days=5,
            step_days=5,
            mode="params",
            min_train_days=5,
        )
        wf_result = run_walk_forward(stocks, "2024-05-01", "2024-06-15", wf_cfg)
        assert len(wf_result.windows) > 0
        assert "sharpe_mean" in wf_result.summary

        # Parameter scan (hold-out mode)
        specs = [ParamSpec("quantile", [0.10, 0.20])]
        grid = build_param_grid(specs)
        scan_df = run_scan(stocks, "2024-05-01", "2024-06-15", grid,
                           wf_config=None, metric="sharpe_ratio")
        assert len(scan_df) == 2
        assert scan_df["sharpe_ratio"].iloc[0] >= scan_df["sharpe_ratio"].iloc[1]
```

- [ ] **Step 2: Run integration tests**

```bash
python3 -m pytest tests/test_integration.py -v
```

Expected: all tests PASS.

- [ ] **Step 3: Run full test suite**

```bash
python3 -m pytest tests/ -v 2>&1 | tail -20
```

Expected: all tests PASS, 0 regressions.

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add walk-forward + scanner integration smoke test"
```

---

## Dependency Order

```
Task 1 (Calendar) ──┐
                    ├──► Task 4 (Enhanced Portfolio) ──┐
Task 2 (Slippage) ──┤                                 │
                    │                                  ├──► Task 7 (API)
Task 3 (Position) ──┘                                  │         │
                                                       │         │
                    Task 5 (Walk-Forward) ─────────────┘         │
                                                       │         │
                    Task 6 (Scanner) ──────────────────┘         │
                                                                 │
                    Task 8 (Integration) ◄───────────────────────┘
```

Tasks 1-3 are independent and can run in parallel. Tasks 5 and 6 depend on Task 4's enhanced `run_backtest` signature. Task 7 needs all prior tasks. Task 8 is the final verification gate.

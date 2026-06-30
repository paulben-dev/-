# Backtest Enhancement v0.5 — Design Spec

**Date:** 2026-06-30
**Prerequisite:** v0.4 Ensemble + Model-Switching UI

---

## Overview

Enhance the backtest engine with three capability tiers:

1. **Precision** — real NYSE trading calendar, slippage/market-impact model, risk-aware position sizing
2. **Walk-forward validation** — rolling window framework supporting both full model retraining and parameter-only optimization
3. **Parameter scanning** — systematic grid and random search over backtest hyperparameters

**Success criteria:**
- Backtest returns differ < 5% from current when all precision features are disabled (backward compatibility)
- Walk-forward produces OOS predictions with zero look-ahead bias (verifiable by date overlap check)
- Parameter scan identifies a quantile/lookback combination with >= 0.1 higher Sharpe than defaults on the same date range
- All 20 new tests pass, 0 regressions in existing 136 tests

---

## Architecture

```
src/backtest/
├── __init__.py
├── calendar.py       # NEW: NYSE trading calendar
├── slippage.py       # NEW: Market impact model
├── position.py       # NEW: Risk-aware position sizing
├── walkforward.py    # NEW: Walk-forward validation engine
├── scanner.py        # NEW: Parameter grid/random search
├── portfolio.py      # MODIFIED: Enhanced run_backtest
└── metrics.py        # MODIFIED: New metrics (turnover, beta exposure)

tests/
├── test_calendar.py      # NEW
├── test_slippage.py      # NEW
├── test_position.py      # NEW
├── test_walkforward.py   # NEW
├── test_scanner.py       # NEW
├── test_portfolio.py     # MODIFIED
└── test_integration.py   # MODIFIED
```

**Data flow:**

```
Config (params grid)
    │
    ▼
scanner.py ──► walkforward.py ──► portfolio.py (enhanced)
                    │                    │
                    ▼                    ▼
              model retrain        calendar.py (trading days)
              (per window)         slippage.py (impact cost)
                                   position.py (risk weights)
                                        │
                                        ▼
                                   metrics.py (enhanced)
                                        │
                                        ▼
                                   API endpoints
```

---

## Component 1: Trading Calendar (`calendar.py`)

Hardcoded NYSE holiday list for 2024–2027. Weekdays that are holidays count as non-trading days.

### Key interfaces

```python
def is_trading_day(date: str | datetime) -> bool
    """True for Mon-Fri excluding NYSE holidays."""

def next_trading_day(date: str) -> str
    """First trading day strictly after date."""

def trading_days_between(start: str, end: str) -> list[str]
    """All trading days in [start, end] inclusive."""

def n_trading_days_later(date: str, n: int) -> str
    """N trading days after date (for position holding periods)."""
```

Replaces the naive `_get_next_trading_day()` (+1 calendar day) in `portfolio.py`.

---

## Component 2: Slippage Model (`slippage.py`)

Two-layer cost: fixed commission + volume-based market impact (simplified Almgren-Chriss).

### Key interfaces

```python
@dataclass
class SlippageConfig:
    fixed_cost: float = 0.0004       # 4bp commission
    impact_factor: float = 0.1       # price impact coefficient
    daily_volume_fraction: float = 0.01  # assumed max participation rate

def estimate_slippage(
    weight: float,
    daily_volume: int,
    close_price: float,
    config: SlippageConfig,
) -> float:
    """Return total slippage cost as fraction of trade value."""
```

Formula: `fixed_cost + impact_factor * (|weight| * close_price) / (daily_volume * close_price * volume_fraction)`

Applied per-stock on each rebalance. Slippage costs are subtracted from daily returns.

---

## Component 3: Position Sizing (`position.py`)

Extends the basic long-short construction with optional risk constraints.

### Key interfaces

```python
@dataclass
class PositionConfig:
    target_vol: float = 0.15         # annualized vol target (0 = disabled)
    max_single_weight: float = 0.05  # max weight per stock (0 = disabled)
    sector_neutral: bool = False     # equal long/short per GICS sector
    max_turnover: float = 1.0        # max daily turnover fraction (0 = disabled)

def size_positions(
    stocks: list[str],
    preds: np.ndarray,
    prev_weights: dict[str, float],
    prices: dict,                    # {stock: recent close prices for vol calc}
    config: PositionConfig,
) -> dict[str, float]:
    """Return {stock: weight} satisfying all enabled constraints."""
```

Constraints are applied in order: pred-based ranking → vol scaling → single-stock cap → sector neutral → turnover limit. If `target_vol` is set, positions are scaled so the portfolio's ex-ante annualized volatility equals the target.

GICS sector mapping is a hardcoded dict of `{ticker: sector}` for the 20 DEFAULT_TICKERS.

---

## Component 4: Walk-Forward (`walkforward.py`)

Rolling window framework for out-of-sample validation.

```
Time axis ─────────────────────────────────────────────►

Window 1:  [── train ──][─ validate ─]
Window 2:        [── train ──][─ validate ─]
Window 3:               [── train ──][─ validate ─]
                         ...
```

### Key interfaces

```python
@dataclass
class WalkForwardConfig:
    train_days: int = 252           # training window in trading days
    validate_days: int = 63         # validation window in trading days
    step_days: int = 21             # forward step in trading days
    mode: str = "full"              # "full" = retrain models | "params" = scan params only
    min_train_days: int = 60        # minimum training data required

@dataclass
class WalkForwardResult:
    windows: list[dict]             # per-window metrics
    oos_predictions: pd.DataFrame   # concatenated validation predictions
    summary: dict                   # mean/std of key metrics across windows

def run_walk_forward(
    stocks: list[str],
    start_date: str,
    end_date: str,
    config: WalkForwardConfig,
    param_grid: dict | None = None,
) -> WalkForwardResult:
    """Run walk-forward validation."""
```

**`full` mode:** each train window retrains MS-LSTM, DualGAT, and Ensemble (meta-learner). Baseline needs no training. Predictions are generated on the validation window using the freshly-trained models.

**`params` mode:** uses existing model files, only varies portfolio construction parameters (quantile, lookback) per window. Fast — no model training.

**OOS guarantee:** validation windows are strictly non-overlapping with their corresponding training windows. The `oos_predictions` DataFrame carries a `window_id` column for per-window analysis.

---

## Component 5: Parameter Scanner (`scanner.py`)

Systematic hyperparameter search over the parameter grid.

### Key interfaces

```python
@dataclass
class ParamSpec:
    name: str
    values: list                  # discrete values

def build_param_grid(specs: list[ParamSpec]) -> list[dict]:
    """Cartesian product of all parameter values."""

def random_search(grid: list[dict], n_iter: int) -> list[dict]:
    """Random sample n_iter combinations."""

def run_scan(
    stocks: list[str],
    start_date: str,
    end_date: str,
    param_grid: list[dict],
    wf_config: WalkForwardConfig | None,
    metric: str = "sharpe_ratio",
) -> pd.DataFrame:
    """Return scan results sorted by metric descending."""
```

When `wf_config` is provided, each parameter combination is evaluated via walk-forward (OOS metric). When `None`, a simple hold-out backtest is used.

Result DataFrame columns: `params` (dict), `sharpe_ratio`, `mean_ic`, `max_drawdown`, `annualized_return`, `icir`.

---

## API Changes

### New endpoints

```
POST /api/backtest/walkforward
Body: {
    start: "2024-01-01", end: "2025-01-01",
    mode: "full" | "params",
    train_days: 252, validate_days: 63, step_days: 21,
}
Response: {
    windows: [{window_id, train_start, train_end, val_start, val_end, sharpe, mean_ic, ...}],
    summary: {sharpe_mean, sharpe_std, mean_ic_mean, ...},
    oos_predictions: [...]  # optional, large
}

POST /api/backtest/scan
Body: {
    start: "2024-01-01", end: "2025-01-01",
    params: {quantile: [0.05, 0.10, 0.15], lookback: [10, 20, 30]},
    mode: "grid" | "random",
    n_iter: 50,
    use_walkforward: true,
    wf_config: {...}
}
Response: {
    results: [{params: {...}, sharpe_ratio, mean_ic, ...}],
    best: {params: {...}, sharpe_ratio, ...}
}
```

### Modified endpoint

`GET /api/backtest/compare` — adds optional query params `?slippage=true&position_risk=true&calendar=true` to toggle precision features.

---

## Configuration Additions

```python
# Trading Calendar (v0.5)
NYSE_HOLIDAYS_2024_2027 = [...]   # hardcoded NYSE holiday dates
TRADING_CALENDAR_START = "2024-01-01"
TRADING_CALENDAR_END = "2027-12-31"

# Slippage (v0.5)
SLIPPAGE_FIXED_COST = 0.0004
SLIPPAGE_IMPACT_FACTOR = 0.1
SLIPPAGE_VOLUME_FRACTION = 0.01

# Position Sizing (v0.5)
POSITION_TARGET_VOL = 0.15
POSITION_MAX_SINGLE = 0.05
POSITION_SECTOR_NEUTRAL = False
POSITION_MAX_TURNOVER = 1.0

# Walk-Forward (v0.5)
WF_TRAIN_DAYS = 252
WF_VALIDATE_DAYS = 63
WF_STEP_DAYS = 21
WF_MODE = "full"
WF_MIN_TRAIN_DAYS = 60

# Parameter Scanner (v0.5)
SCAN_DEFAULT_METRIC = "sharpe_ratio"
SCAN_RANDOM_N_ITER = 50
```

---

## Backward Compatibility

All new position/slippage/calendar features are **opt-in**. When config values are left at their defaults (0 = disabled), the backtest produces identical results to v0.4. The existing `run_backtest()` signature gains optional keyword arguments with defaults that preserve current behavior.

---

## Testing Strategy

| # | Test | File |
|---|------|------|
| 1 | Weekend returns False | `test_calendar.py` |
| 2 | Known holiday returns False | `test_calendar.py` |
| 3 | Friday → next Monday | `test_calendar.py` |
| 4 | trading_days_between correct count | `test_calendar.py` |
| 5 | Zero trade = zero slippage | `test_slippage.py` |
| 6 | Slippage increases with trade size | `test_slippage.py` |
| 7 | Minimum cost = fixed_cost | `test_slippage.py` |
| 8 | High-vol stock gets lower weight | `test_position.py` |
| 9 | Single-stock cap enforced | `test_position.py` |
| 10 | Turnover constraint enforced | `test_position.py` |
| 11 | Window count correct | `test_walkforward.py` |
| 12 | No look-ahead bias | `test_walkforward.py` |
| 13 | params mode skips retraining | `test_walkforward.py` |
| 14 | Grid covers all combinations | `test_scanner.py` |
| 15 | Returns best params | `test_scanner.py` |
| 16 | Random respects n_iter | `test_scanner.py` |
| 17 | Enhanced backtest uses calendar | `test_portfolio.py` |
| 18 | `/api/backtest/walkforward` 200 | `test_api.py` |
| 19 | `/api/backtest/scan` 200 | `test_api.py` |
| 20 | Full pipeline smoke test | `test_integration.py` |

---

## Non-Goals

- Real-time market data streaming
- Multi-asset / cross-market backtesting
- Auto-download GICS sector mappings (hardcoded for the 20-stock universe)
- Frontend UI for walk-forward / parameter scan results (API only)
- Intraday backtesting or tick-level simulation
- Short-selling borrow cost / availability modeling

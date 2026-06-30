# Final Code Review Fix Report — v0.5

**Date:** 2026-07-01
**Branch:** master
**Review:** v0.5 final review (5 CONFIRMED findings)

---

## Fix 1 (CRITICAL): Transaction cost double-counted

- **File:** `src/backtest/portfolio.py`
- **Lines changed:** 144-155
- **Bug:** When `slippage_config` is active, `estimate_slippage` includes `fixed_cost` (4 bp) in daily P&L. But `compute_annualized_return` and `compute_sharpe` also subtracted `transaction_cost` (4 bp) from the return series -- effective drag of 8 bp instead of 4 bp.
- **Fix:** Compute `net_transaction_cost = max(0.0, transaction_cost - slippage_config.fixed_cost)` when slippage is active, and pass that to both metric functions. This ensures the total commission drag never exceeds the explicit `transaction_cost`.

## Fix 2 (IMPORTANT): Walk-forward inner backtest without calendar

- **File:** `src/backtest/walkforward.py`
- **Line changed:** 104
- **Bug:** The inner `run_backtest` call inside `run_walk_forward` did not pass `use_calendar=True`, so weekend/holiday dates in predictions were processed as if they were trading days, inflating the number of trading days and distorting metrics.
- **Fix:** Added `use_calendar=True` before `**backtest_kwargs` so the walk-forward engine always uses the NYSE calendar for its inner backtests.

## Fix 3 (IMPORTANT): Scanner hold-out path without calendar

- **File:** `src/backtest/scanner.py`
- **Line changed:** 125
- **Bug:** The simple hold-out backtest path inside `run_scan` (when `wf_config is None`) did not pass `use_calendar=True`, causing the same weekend/holiday distortion as Fix 2.
- **Fix:** Added `use_calendar=True` before `**combo`.

## Fix 4 (IMPORTANT): API sufficiency check inconsistent

- **File:** `src/web/api.py`
- **Line changed:** 453
- **Bug:** The `/api/backtest/walkforward` endpoint checked `cfg.train_days + cfg.validate_days` but the walk-forward engine itself (in `walkforward.py:68`) checks `config.min_train_days + config.validate_days`. This meant the API could reject requests that the engine would accept (when `min_train_days < train_days`), or accept requests the engine would reject (when `min_train_days > train_days`).
- **Fix:** Changed `cfg.train_days` to `cfg.min_train_days` to match the engine's guard exactly.

## Fix 5 (MEDIUM): Scanner zeros in walk-forward mode

- **File:** `src/backtest/scanner.py`
- **Lines changed:** 110-112
- **Bug:** When scanning with walk-forward mode, `max_drawdown`, `annualized_return`, and `icir` were hardcoded to 0.0, so the scan results table showed no drawdown/return/ICIR information for walk-forward parameter combos.
- **Fix:** Aggregate these metrics from `WalkForwardResult.windows`:
  - `max_drawdown` = minimum across all windows
  - `annualized_return` = mean across all windows
  - `icir` = mean across all windows

---

## Test results

```
$ python3 -m pytest tests/test_portfolio.py tests/test_walkforward.py tests/test_scanner.py tests/test_api.py -v
44 passed in 54.59s

$ python3 -m pytest tests/ -q
190 passed in 70.38s
```

One test (`test_walkforward_params_mode`) was updated to pass `min_train_days=20` explicitly since the API now uses `min_train_days` (not `train_days`) for the sufficiency guard, matching the engine behavior.

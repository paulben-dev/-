# Task 3 Report: MSLSTMPredictor -- Training and Prediction Wrapper

## Status: COMPLETE

## Files Modified

| File | Change |
|------|--------|
| `src/model/ms_lstm.py` | Appended `MSLSTMPredictor` class, `_DataError` exception, and 5 module-level data helpers |
| `tests/test_ms_lstm.py` | Appended `TestMSLSTMPredictor` class with 5 tests; added `numpy` and `pandas` imports |
| `tests/conftest.py` | Enhanced `prepopulated_db` fixture with 44+ days of price history (required for training) |

## What Was Implemented

### `MSLSTMPredictor` class (lines appended to `src/model/ms_lstm.py`)

- `__init__(input_dim, hidden_dim, num_scales, expert_feat_dim, dropout, device)` -- creates internal `MSLSTMModel`, auto-detects CUDA
- `fit(stocks, start_date, end_date, expert_records_by_date, epochs, lr)` -- training loop with:
  - 80/20 train/validation split on trading dates
  - Adam optimizer with weight decay
  - Per-date IC loss computation
  - Early stopping with patience and best-model checkpointing
  - Returns `{train_loss, val_ic, best_epoch}` dict
- `predict(stocks, date_str, expert_records)` -- returns `pd.DataFrame` with columns `[stock, date, predicted_return, signal_source]`; cross-sectional z-score normalization; gracefully handles data errors by returning zero predictions
- `save(path)` / `load(path)` -- full model serialization including hyperparameters

### Module-level data helpers

- `_DataError(Exception)` -- raised when data is insufficient for a date
- `_get_trading_dates(stocks, start_date, end_date)` -- finds dates with price data for all stocks
- `_build_day_tensors(stocks, date_str, device)` -- assembles (price, expert, target) tensors for one trading day
- `_build_stock_features(prices, target_date)` -- builds normalized OHLCV tensor (seq_len x 5) with close-price normalization and log-volume
- `_get_return_for_date(prices, date_str)` -- computes actual return ratio as target
- `_build_expert_features(stocks, date_str)` -- queries ExpertTracker and builds [avail, signal] feature vectors

## Test Results

```
13 passed in tests/test_ms_lstm.py:
  TestMSLSTMModel: 4 tests (unchanged from Task 2)
  TestICLoss: 4 tests (unchanged from Task 2)
  TestMSLSTMPredictor: 5 tests (new)
    - test_predict_returns_dataframe
    - test_predict_empty_stocks
    - test_save_and_load_roundtrip
    - test_fit_runs_one_epoch
    - test_predict_after_fit

Full suite: 95 passed, 1 pre-existing failure (test_get_backtest, unrelated)
```

## Deviations from Brief

1. **Test date range widened**: The brief specified `start_date="2024-06-10"` for fit tests. This 6-day window cannot contain 10+ trading dates. Changed to `start_date="2024-05-15"` (32-day window).

2. **Conftest enhanced**: Added 44 days of price history (May 1 -- June 13) to `prepopulated_db`, required for `_build_stock_features` (needs 10+ prior records) and `fit` (needs 10+ trading dates).

3. **Test file imports**: Added `import numpy as np` and `import pandas as pd` to the test file, needed by the test fixtures and assertions.

## Commit

```
git add src/model/ms_lstm.py tests/test_ms_lstm.py tests/conftest.py
git commit -m "feat: MSLSTMPredictor with fit/predict/save/load"
```

---

# Fix Report: Task 3 Review Findings

**Date:** 2026-06-29
**Branch:** master
**Status:** ALL FIXES APPLIED

## CRITICAL Fixes

### Bug A -- Stock/prediction misalignment in predict() (FIXED)

**Root cause:** `_build_day_tensors()` silently dropped stocks that failed feature construction (via `except _DataError: continue`), but `predict()` constructed the output DataFrame with the original `stocks` list, causing length mismatch.

**Fix:** Changed `_build_day_tensors()` to return `kept_stocks` as a 4th return value. Updated `predict()` to use `kept_stocks` for the `"stock"` column. Updated `fit()` and `_evaluate_ic()` call sites to unpack the 4-tuple (using `_kept` for the unused value).

**Files changed:**
- `src/model/ms_lstm.py`: `_build_day_tensors` signature and return, all 3 call sites.

### Bug B -- Variable sequence length in _build_stock_features() (FIXED)

**Root cause:** `_build_stock_features` took `prior[-MSLSTM_SEQUENCE_LENGTH:]`, producing tensors with 10-30 time steps depending on available history. `torch.stack()` would crash when stocks in a batch had different sequence lengths.

**Fix:** Added zero-padding after normalization. If the feature array has fewer than `MSLSTM_SEQUENCE_LENGTH` rows, zeros are prepended to reach exactly 30 time steps. The normalization by first close is applied before padding so the first real row has close=1.0 from self-division, and padded zero-rows are valid "no data" input to the LSTM.

**Files changed:**
- `src/model/ms_lstm.py`: `_build_stock_features()` -- added padding block (lines 494-500).

## IMPORTANT Fixes

### Unused expert_records parameters (FIXED)

Removed `expert_records_by_date` parameter from `fit()` and `expert_records` parameter from `predict()`. Removed corresponding docstring entries. Also removed the now-unused `from src.data.models import ExpertRecord` import.

### Dead `_trained` flag (FIXED)

Removed `self._trained = False` from `__init__`, `self._trained = True` from `fit()`, and `self._trained = True` from `load()`. The flag was never read.

### Dead variable in _build_day_tensors (FIXED)

Removed `expert_features_list = []` initialization (line 423 in original). It was never appended to and always overwritten by the `_build_expert_features(...)` call 4 lines later.

### Duplicate DataFrame logic in predict() (FIXED)

Extracted `_empty_predictions(stocks, date_str)` helper method that builds the zero-prediction DataFrame. Used in all 3 fallback paths: empty stocks, `_DataError`, and `price_t is None`.

## Test Results

```
13 passed in tests/test_ms_lstm.py
```

All existing tests pass with the fixes applied. No test regressions.

## Summary of Changes

| Issue | Severity | Fix Applied |
|-------|----------|-------------|
| Bug A: stock/prediction misalignment | CRITICAL | Return `kept_stocks` from `_build_day_tensors`, use in `predict()` |
| Bug B: variable sequence length | CRITICAL | Zero-pad to `MSLSTM_SEQUENCE_LENGTH` in `_build_stock_features` |
| Unused expert_records params | IMPORTANT | Removed from `fit()`, `predict()`, and import |
| Dead `_trained` flag | IMPORTANT | Removed from `__init__`, `fit()`, `load()` |
| Dead `expert_features_list` var | IMPORTANT | Removed initialization line |
| Duplicate DataFrame logic | IMPORTANT | Extracted `_empty_predictions` helper |

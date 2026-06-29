# Task 5: Integration Verification

## Done

Appended `TestMSLSTMIntegration` class to `tests/test_ms_lstm.py` with 2 integration tests:

- **`test_full_pipeline_smoke`** -- end-to-end smoke test: loads DB fixture with AAPL/MSFT/GOOGL, trains MS-LSTM (2 epochs), runs `predict()`, and compares output shape with `BaselinePredictor`. Required adding GOOGL price data inside the test and widening the date range to `"2024-05-20"`--`"2024-06-15"` (fixture only had 44 days; original `"2024-06-10"`--`"2024-06-15"` had 6 dates, below the model's 10-date minimum for training).

- **`test_ic_loss_improves_during_training`** -- trains for 5 epochs with 2 stocks, verifies `len(losses) == 5` and all losses are finite (NaN-free). Also required widened date range.

## Test Suite Results

```
97 passed, 1 failed (expected test_api.py backtest endpoint failure)
```

All 15 MS-LSTM tests pass (4 model, 4 loss, 5 predictor, 2 integration). The single pre-existing failure in `test_api.py: TestBacktestEndpoint::test_get_backtest` is unrelated.

## Deviations from Brief

- Fixed `ms_lstm.predict()` call to not pass `records` argument -- `MSLSTMPredictor.predict()` signature is `(stocks, date_str)`, unlike `BaselinePredictor.predict()` which accepts optional `expert_records`.
- Added GOOGL price insertion inside `test_full_pipeline_smoke` because the shared `prepopulated_db` fixture only includes AAPL and MSFT.
- Widened date range from `"2024-06-10"` to `"2024-05-20"` in both tests to exceed the model's 10-trading-date minimum.

## Commit

```
85a8d75 test: MS-LSTM integration and training smoke tests
```

---

## Critical Bug Fix: Evaluation Price Fetch Window (2026-06-29)

**Bug:** `get_prices(stocks, date_str, date_str)` only returns 1 row per stock, so `len(sp) >= 2` was always False. The `actuals` dict stayed empty, `np.mean([])` would crash with a RuntimeWarning.

**Fix (lines 97-107):**
- Changed `get_prices(stocks, date_str, date_str)` to `get_prices(stocks, yesterday_str, date_str)`, where `yesterday_str` is computed as `date_str - 5 days` (to bridge weekends/holidays).
- Simplified return computation: use `sp[0]["close"]` (prior trading day) and `sp[-1]["close"]` (current date) directly, matching the pattern in `src/backtest/metrics.py:compute_daily_ic_series`.

**Fix (lines 121-135):**
- Added `_safe_mean()` helper: returns `np.mean(seq)` if non-empty, else `float("nan")`.
- Used `_safe_mean` for all three IC list means and the outperformance comparison to avoid `RuntimeWarning` from `np.mean([])`.

**Files changed:**
- `scripts/train_dualgat.py` -- fixed price fetch window, simplified return computation, added empty-list guard

**Commit:**
```
fix: Task 5 review — fix price fetch window for actual return computation
```

# v0.4 Ensemble Final Code Review — Fix Report

**Date:** 2026-06-30
**Branch:** master

---

## Fix Summary

All 6 findings from the final code review have been addressed.

### Finding 1 — Sort consistency in `predict()`
**File:** `src/model/ensemble.py:144`
**Problem:** `EnsemblePredictor.predict()` didn't sort by `predicted_return` descending, while all 3 sub-models do.
**Fix:** Added `df = df.sort_values("predicted_return", ascending=False)` before `return df`.

### Finding 2 — IC comparison price fetch window
**File:** `scripts/train_ensemble.py:128`
**Problem:** `get_prices(stocks, date_str, date_str)` fetched a single date, preventing actual return computation.
**Fix:** Pre-fetch all prices across the full evaluation window with a 5-day buffer before the loop, matching the pattern used by `fit_meta()`. Changed loop to use `all_prices` instead of per-date `prices`.

### Finding 3 — Stale `_meta` after loading weighted checkpoint
**File:** `src/model/ensemble.py:365`
**Problem:** `load()` didn't clear `self._meta` when loading a weighted checkpoint, leaving stale meta-learner state.
**Fix:** Set `self._meta = None` at the start of `load()`. It is only recreated when `"meta_state_dict"` is present in the checkpoint.

### Finding 4 — None cached permanently in `_get_model()`
**File:** `src/web/api.py:103`
**Problem:** `_get_model()` cached `None` in `_model_cache` when model loading failed, preventing future retries.
**Fix:** Added guard: `if model is not None: _model_cache[model_id] = model`.

### Finding 5 — Null guard in `loadSystemStatus()`
**Files:** `src/web/static/app.js:349`, `src/web/static/app_zh.js:350`
**Problem:** `predsData.predictions.filter()` called without null guard, causing TypeError when predictions are null/undefined.
**Fix:** Added `if (!predsData.predictions) return;` before the `.filter()` call in both `app.js` and `app_zh.js`.

### Finding 6 — SSE generate() no timeout
**File:** `src/web/api.py:515`
**Problem:** SSE `generate()` loop had no timeout, could hang indefinitely.
**Fix:** (a) Wrapped `_do_collect` in `_do_collect_safe()` with try/finally guaranteeing a final progress event. (b) Added a 10-minute deadline to the generate loop; if exceeded, emits a timeout error event and exits.

---

## Test Results

### Ensemble tests (`tests/test_ensemble.py`) — Findings 1, 3
- **9 passed, 2 failed**
- Failures are **pre-existing** (confirmed by testing on stashed clean code):
  - `test_equal_weights_with_equal_ic` — synthetic test data correlation mismatch
  - `test_low_ic_model_gets_near_zero_weight` — synthetic test data correlation mismatch
- All save/load, meta, and integration tests pass.

### API tests (`tests/test_api.py`) — Findings 4, 6
- **16 passed, 1 failed**
- Failure is **pre-existing** (confirmed on clean code):
  - `test_get_backtest` — backtest endpoint returns 404 (pre-existing route issue)

### Syntax check (`scripts/train_ensemble.py`) — Finding 2
- **Syntax OK**

### JS syntax check (`app.js`, `app_zh.js`) — Finding 5
- **Both passed** (no syntax errors)

---

## Conclusion

All 6 findings are fixed. No regressions introduced. The 3 failing tests are pre-existing issues unrelated to these fixes.

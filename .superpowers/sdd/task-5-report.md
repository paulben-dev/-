### Task 5 Report: Walk-Forward Validation Engine

**Status:** Complete
**Commit:** `bae7d52`
**Files changed:** 3 (367 insertions, 8 deletions)

### What was implemented

1. **`src/backtest/walkforward.py`** — Walk-forward validation engine with rolling-window OOS evaluation framework:
   - `WalkForwardConfig` dataclass with configurable train_days (252), validate_days (63), step_days (21), mode ("full"/"params"), min_train_days (60)
   - `WalkForwardResult` dataclass holding per-window metrics (windows), concatenated OOS predictions (oos_predictions), and summary statistics (summary)
   - `run_walk_forward(stocks, start_date, end_date, config, param_grid=None)` — main entry point that iterates over rolling windows, generates predictions, runs backtests, and aggregates results
   - `_generate_predictions()` — generates predictions for a validation window, using loaded models in "params" mode or freshly retrained models in "full" mode
   - `_retrain_models()` — retrains MS-LSTM, DualGAT, and Ensemble meta-learner on the training window and saves to disk
   - DualGAT.fit() correctly receives `ms_lstm_path` parameter pointing to the just-trained MS-LSTM model
   - Model paths use config constants (MSLSTM_MODEL_PATH, DUALGAT_MODEL_PATH, ENSEMBLE_MODEL_PATH) for robustness

2. **`config.py`** — Added Walk-Forward (v0.5) constants: WF_TRAIN_DAYS, WF_VALIDATE_DAYS, WF_STEP_DAYS, WF_MODE, WF_MIN_TRAIN_DAYS

3. **`tests/test_walkforward.py`** — 4 integration tests + 1 config test:
   - `test_defaults` — verifies WalkForwardConfig defaults
   - `test_params_mode_produces_windows` — windows generated with correct structure (window_id, train_start/end, val_start/end, sharpe_ratio)
   - `test_no_lookahead_bias` — val_start > train_end for every window
   - `test_summary_has_mean_std` — summary contains sharpe_mean, sharpe_std, mean_ic_mean
   - `test_insufficient_data_returns_empty` — short date range returns empty result

4. **`tests/conftest.py`** — Extended prepopulated_db fixture with price data through July 31, 2024 to provide sufficient trading days for walk-forward window generation (train_days=30 + val_days=10 requires 40 trading days)

### Verification

- `python3 -m pytest tests/test_walkforward.py -v`: 5 passed
- `python3 -m pytest tests/ -q`: 184 passed, 0 regressions

### Backward compatibility

- No changes to existing interfaces (calendar, portfolio, models)
- conftest.py extension preserves all existing test behavior
- Walkforward imports internally (lazy imports) — no new module-level dependencies

### Concerns

- None. All requirements from the task brief are implemented and verified.

---

## Fix Round 1

**Commit:** `fix: use min_train_days in walk-forward and add full-mode test`
**Date:** 2026-07-01

### Fixes applied

**CRITICAL — F1: `min_train_days` dead parameter**
- Changed insufficient-data guard (line 68) from `config.train_days + config.validate_days` to `config.min_train_days + config.validate_days`.
- Updated log message to reflect `min_train=...` instead of `train=...`.
- `min_train_days` now acts as the data-sufficiency floor: a run is rejected unless there are at least `min_train_days + validate_days` trading days.

**IMPORTANT — F2: No test coverage for mode="full"**
- Added `test_full_mode_produces_windows` with `mode="full"`, `train_days=5`, `validate_days=5`, `min_train_days=5`, `step_days=3` over a short date range (2024-06-01 to 2024-07-15).
- Test verifies the framework enters the full-mode code path without crashing and upholds the lookahead constraint.
- Marked `@pytest.mark.slow` (model retraining is involved).

**IMPORTANT — F3: `DEFAULT_TICKERS` import unused**
- Removed `DEFAULT_TICKERS` from the config import block.

**MINOR (also fixed):**
- **F4:** Removed dead-code `val_end_idx` guard (was unreachable due to the while-loop condition).
- **F5:** Added `n_successful` and `n_failed` counts to summary so callers can distinguish windows that produced backtest results from those that errored.
- **F6:** Added `assert "mean_ic_std" in result.summary` to `test_summary_has_mean_std`.

### Verification

- `python3 -m pytest tests/test_walkforward.py -v`: 6 passed (was 5)
- `python3 -m pytest tests/ -q`: 185 passed (was 184), 0 regressions

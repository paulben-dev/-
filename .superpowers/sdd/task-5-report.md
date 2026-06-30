### Task 5 Report: Model-switching frontend UI

**Status:** Complete  
**Commit:** `8c320b3`  
**Files changed:** 4 (454 insertions, 98 deletions)

### What was implemented

1. **`src/web/templates/index.html`** — Added CSS for model tabs, comparison cards, and status chips. Added model tab bar (`#model-tabs`) with four tabs (Baseline, MS-LSTM, DualGAT, Ensemble) plus a "Compare All" checkbox toggle. Added `#predictions-compare` container and `#compare-table-container` to the backtest card.

2. **`src/web/templates/index_zh.html`** — Same structural changes with Chinese labels (基准模型, MS-LSTM模型, DualGAT模型, 集成模型, 对比所有).

3. **`src/web/static/app.js`** — Full rewrite with model switching:
   - `currentModel` / `compareMode` state, `MODEL_COLORS` map
   - `loadModels()` fetches `/api/models` and stores availability in `window._models`
   - `updateModelTabs()` adds `.unavailable` class to gray out missing models
   - `selectModel()` switches the active model, reloads predictions and backtest
   - `toggleCompare()` switches between single-model and side-by-side prediction views
   - `loadPredictionsCompare()` fetches all four models, renders grid of mini-cards
   - `loadBacktest()` now calls `/api/backtest/compare`, renders a multi-line Chart.js chart with one curve per model (color-coded, current model highlighted with thicker line), and a comparison table with best-value highlighting

4. **`src/web/static/app_zh.js`** — Same logic, all user-facing strings in Chinese.

### Verification

- Python TestClient: `/` returns model-tabs, all four data-model attributes, compare-mode, predictions-compare, compare-table-container. `/static/app.js` contains MODEL_COLORS, currentModel, selectModel, toggleCompare. `/zh` returns Chinese labels (基准模型, MS-LSTM模型, DualGAT模型, 集成模型, 对比所有). `/static/app_zh.js` contains Chinese metric labels.
- Node syntax check: Both app.js and app_zh.js pass `node -c`.

### Backward compatibility

- Date picker, experts panel, data collection, system status, and auto-refresh all preserved.
- Existing `/api/predictions?model=...` and `/api/backtest/compare` endpoints consumed from Tasks 1-4.

### Concerns

- None. All requirements from the task brief are implemented and verified.

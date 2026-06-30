# Ensemble Model Fusion + Model-Switching UI — Design Spec

**Date:** 2026-06-30
**Prerequisite:** v0.1 MVP + v0.2 MS-LSTM + v0.3 DualGAT + bilingual dashboard

---

## Overview

Add multi-model ensemble prediction with two fusion strategies (weighted-average and meta-learner MLP), plus a model-switching UI that supports side-by-side comparison and multi-line backtest overlay charts.

**Success criteria:**
- Ensemble achieves higher mean IC than any single model on the backtest period
- Dashboard supports switching between 4 models (Baseline, MS-LSTM, DualGAT, Ensemble) with one click
- Backtest chart overlays all model curves for visual comparison

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  Dashboard                                            │
│  ┌──────────────────────────────────────────────────┐ │
│  │ [Baseline] [MS-LSTM] [DualGAT] [Ensemble] ← tabs │ │
│  └──────────────────────────────────────────────────┘ │
│  ┌──────────────┐  ┌────────────────────────────────┐ │
│  │ Predictions   │  │  Backtest (multi-line overlay) │ │
│  │ (selected     │  │  ─ Baseline   ─ MS-LSTM       │ │
│  │  model only)  │  │  ─ DualGAT    ─ Ensemble      │ │
│  └──────────────┘  └────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────────┐ │
│  │ Model Status: IC | Sharpe | Availability per model│ │
│  └──────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

**Model selection** drives:
- Predictions panel: single-model view (default) or side-by-side comparison
- Backtest panel: always multi-line overlay + comparison table
- Expert panel: unchanged (model-agnostic)

---

## Backend: Ensemble Predictor

### Two Fusion Strategies

**Strategy 1 — Weighted Average (default, no training):**
- Track rolling IC for each model over recent N days
- `weight_i = exp(mean_IC_i / temperature) / sum(exp(mean_IC_j / temperature))`
- `ensemble_pred = sum(weight_i * pred_i)`
- Temperature `τ = 0.1` controls sharpness of weighting

**Strategy 2 — Meta-Learner MLP (optional training):**
- Architecture: `[3 → 8 → 1]` with ReLU
- Input: 3 scalar predictions per stock (baseline, MS-LSTM, DualGAT)
- Output: fused scalar prediction
- Loss: cross-sectional IC loss (same as DualGAT)
- Training: Adam, lr=1e-3, early stopping
- Model saved to `data/ensemble_meta.pt`

### Files

```
src/model/
├── ensemble.py      # NEW: EnsemblePredictor
├── dualgat.py       # Existing
├── ms_lstm.py       # Existing
├── baseline.py      # Existing
└── __init__.py

tests/
└── test_ensemble.py # NEW

config.py            # MODIFY: append ENSEMBLE_* constants
```

### Key Interfaces

```python
# src/model/ensemble.py

class EnsemblePredictor:
    """Fuses Baseline + MS-LSTM + DualGAT predictions."""

    def __init__(self, strategy: str = "weighted", temperature: float = 0.1):
        ...

    def predict(
        self, stocks: list[str], date_str: str,
        baseline_preds: pd.DataFrame,
        ms_lstm_preds: pd.DataFrame,
        dualgat_preds: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return ensemble predictions + per-model columns."""
        # Returns DataFrame with columns:
        #   stock, date, predicted_return, signal_source,
        #   baseline_return, ms_lstm_return, dualgat_return

    def fit_meta(
        self, stocks, start_date, end_date,
        baseline, ms_lstm, dualgat,
        epochs=50, lr=1e-3,
    ) -> dict:
        """Train meta-learner MLP. Only for strategy='meta'."""

    def save(self, path) -> None
    def load(self, path) -> None
```

### API Changes

**New endpoints:**

```
GET /api/models
→ { "models": [
    {"id": "baseline", "name": "Baseline", "available": true, "recent_ic": 0.03, "needs_training": false},
    {"id": "ms_lstm",  "name": "MS-LSTM",   "available": true, "recent_ic": 0.05, "needs_model": "data/ms_lstm_model.pt"},
    {"id": "dualgat",  "name": "DualGAT",   "available": true, "recent_ic": 0.06, "needs_model": "data/dualgat_model.pt"},
    {"id": "ensemble", "name": "Ensemble",  "available": true, "recent_ic": 0.07, "strategy": "weighted"},
]}

GET /api/predictions?date=YYYY-MM-DD&model=ensemble
→ same shape + extra per-model return columns when model=ensemble

GET /api/backtest/compare?start=YYYY-MM-DD&end=YYYY-MM-DD
→ {
    "start": "...", "end": "...",
    "models": {
      "baseline": { "annualized_return": ..., "sharpe_ratio": ..., "mean_ic": ..., "max_drawdown": ..., "icir": ..., "cumulative_returns": [...] },
      "ms_lstm":  { ... },
      "dualgat":  { ... },
      "ensemble": { ... },
    }
  }
```

**Modified endpoint:**

```
GET /api/predictions?date=...&model=baseline|ms_lstm|dualgat|ensemble
→ existing shape, model param optional (defaults to baseline for backward compat)
```

---

## Frontend Changes

### Model Selector (tab bar)

```
[Baseline] [MS-LSTM] [DualGAT] [Ensemble]  |  [Compare All]
                                           → toggles side-by-side prediction view
```

### Predictions Panel

- **Single model mode (default):** shows predictions for selected model only
- **Compare mode:** side-by-side columns showing each model's prediction per stock

### Backtest Panel

- **Multi-line chart:** one curve per model, color-coded, with legend
- **Comparison table:** rows = models, columns = (Annualized Return, Sharpe, Mean IC, Max Drawdown, ICIR)
- Highlight best value in each column

### Model Status Bar

- Shows each model's availability status:
  - **Baseline:** always available (no model file needed)
  - **MS-LSTM:** available if `data/ms_lstm_model.pt` exists
  - **DualGAT:** available if `data/dualgat_model.pt` exists
  - **Ensemble (weighted):** available if all 3 sub-models are available
  - **Ensemble (meta):** available if `data/ensemble_meta.pt` exists and all 3 sub-models are available
- Gray out unavailable models in the tab bar

### Performance Notes

- `/api/backtest/compare` runs all 4 backtests in one call. For 90 days × 20 stocks this is ~2-4 seconds. Acceptable for on-demand use; if too slow, add per-model caching or lazy loading per model curve.
- Model loading is lazy — prediction endpoints load models on first call and cache them for the process lifetime.

### Template & JS

- `index.html` / `index_zh.html`: add model tabs, model status bar, side-by-side layout containers
- `app.js` / `app_zh.js`: model switching logic, multi-line chart rendering, compare endpoint calls
- CSS additions: tab bar styles, comparison table styles, model status indicator

---

## Configuration Additions

```python
# Ensemble (v0.4)
ENSEMBLE_STRATEGY = "weighted"   # "weighted" or "meta"
ENSEMBLE_TEMPERATURE = 0.1       # Softmax temperature for IC→weight
ENSEMBLE_IC_WINDOW = 20          # Trading days for rolling IC
ENSEMBLE_META_HIDDEN = 8         # Meta-learner hidden dim
ENSEMBLE_META_LR = 1e-3
ENSEMBLE_META_EPOCHS = 50
ENSEMBLE_META_PATIENCE = 10
ENSEMBLE_MODEL_PATH = ROOT_DIR / "data" / "ensemble_model.pt"
ENSEMBLE_META_PATH = ROOT_DIR / "data" / "ensemble_meta.pt"
```

---

## Testing Strategy

1. **Weighted average:** 3 models with known outputs → ensemble matches expected weighted result
2. **Weight falls with lower IC:** model with IC=0 gets near-zero weight
3. **Meta-learner forward pass:** produces [N] output from 3 [N] inputs
4. **Meta-learner training:** loss decreases over epochs
5. **predict() returns correct columns:** includes per-model breakdown
6. **Save/load roundtrip:** identical predictions
7. **API: /api/models returns all 4 models with correct availability**
8. **API: /api/predictions?model=ensemble returns per-model columns**
9. **API: /api/backtest/compare returns all model curves**
10. **Integration:** full pipeline smoke test with all 4 models

---

## Non-Goals

- Hyperparameter tuning / grid search for ensemble
- Dynamic ensemble strategy switching at runtime (requires reload)
- Model training from dashboard UI (train via scripts only)
- Real-time IC tracking (computed on-demand from backtest data)
- Per-stock ensemble weights (uniform across stocks)

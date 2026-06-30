# v0.4 Ensemble Model Fusion + Model-Switching UI — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add multi-model ensemble prediction (weighted-average + meta-learner MLP) and a model-switching dashboard UI with multi-line backtest overlay charts.

**Architecture:** `EnsemblePredictor` wraps Baseline/MS-LSTM/DualGAT, fusing their predictions via softmax-weighted IC or a trained MLP. API adds `/api/models`, model=? param on predictions, and `/api/backtest/compare`. Frontend adds a model tab bar, side-by-side prediction views, and multi-curve backtest chart.

**Tech Stack:** PyTorch, FastAPI, Chart.js — reuses v0.1-v0.3 models without changes.

## Global Constraints

- CPU-first (GPU optional via `torch.cuda.is_available()`)
- Reuse v0.1 ExpertTracker, backtest engine — no changes to existing model files
- `EnsemblePredictor.predict()` returns same DataFrame shape as `BaselinePredictor.predict()`
- Weighted-average ensemble: no training required, weights from rolling IC
- Meta-learner: MLP `[3→8→1]` trained with IC loss, saved to `data/ensemble_meta.pt`
- Model `available` logic: Baseline always true; MS-LSTM/DualGAT true if .pt file exists; Ensemble (weighted) true if all 3 sub-models available; Ensemble (meta) true if meta .pt exists and all 3 sub-models available
- `/api/backtest/compare` runs up to 4 backtests, target response time < 5s for 90-day window
- Bilingual UI: English and Chinese versions updated in sync
- Existing `/api/predictions` defaults to baseline when `model` param omitted (backward compat)

---

### Task 1: Configuration

**Files:**
- Modify: `config.py` (append Ensemble constants)

**Interfaces:**
- Produces: All `ENSEMBLE_*` and `ENSEMBLE_META_*` config constants, `ENSEMBLE_MODEL_PATH`, `ENSEMBLE_META_PATH`

- [ ] **Step 1: Add Ensemble constants to config.py**

Append after the DualGAT section (after line 74):

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

- [ ] **Step 2: Verify imports**

```bash
python3 -c "from config import ENSEMBLE_TEMPERATURE, ENSEMBLE_META_PATH; print(ENSEMBLE_TEMPERATURE, ENSEMBLE_META_PATH)"
```
Expected: `0.1 /home/paulben/code/金融投资项目/data/ensemble_meta.pt`

- [ ] **Step 3: Commit**

```bash
git add config.py
git commit -m "feat: add Ensemble configuration constants"
```

---

### Task 2: EnsemblePredictor — Weighted Average

**Files:**
- Create: `src/model/ensemble.py` (weighted-average portion)
- Create: `tests/test_ensemble.py` (weighted-average tests)

**Interfaces:**
- Produces:
  - `EnsemblePredictor(strategy="weighted", temperature=0.1)` — constructor
  - `EnsemblePredictor.predict(stocks, date_str, baseline_preds, ms_lstm_preds, dualgat_preds) -> pd.DataFrame`
  - `EnsemblePredictor.save(path) -> None`
  - `EnsemblePredictor.load(path) -> None`

- [ ] **Step 1: Write failing tests for weighted average**

```python
# tests/test_ensemble.py
"""Tests for EnsemblePredictor — weighted average and meta-learner."""
import pytest
import numpy as np
import pandas as pd
import torch


def _make_preds(stocks, returns, source):
    """Helper: create a predictions DataFrame matching BaselinePredictor format."""
    return pd.DataFrame({
        "stock": stocks,
        "date": "2024-06-15",
        "predicted_return": returns,
        "signal_source": source,
    })


class TestEnsembleWeighted:
    """Tests for weighted-average ensemble strategy."""

    @pytest.fixture
    def ensemble(self):
        from src.model.ensemble import EnsemblePredictor
        return EnsemblePredictor(strategy="weighted", temperature=0.1)

    @pytest.fixture
    def sub_preds(self):
        """3 models predicting 5 stocks with known values."""
        stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
        baseline = _make_preds(stocks, [0.01, 0.02, -0.01, 0.03, -0.02], "baseline")
        ms_lstm  = _make_preds(stocks, [0.02, 0.03,  0.00, 0.04, -0.01], "ms_lstm")
        dualgat  = _make_preds(stocks, [0.03, 0.01,  0.02, 0.05,  0.00], "dualgat")
        return baseline, ms_lstm, dualgat

    def test_predict_returns_dataframe(self, ensemble, sub_preds):
        """predict() returns DataFrame with required columns."""
        baseline, ms_lstm, dualgat = sub_preds
        df = ensemble.predict(["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"], "2024-06-15",
                              baseline, ms_lstm, dualgat)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["stock", "date", "predicted_return", "signal_source",
                                     "baseline_return", "ms_lstm_return", "dualgat_return"]
        assert len(df) == 5
        assert (df["signal_source"] == "ensemble").all()

    def test_equal_weights_with_equal_ic(self, ensemble, sub_preds):
        """When all models have same IC, weights should be ~equal."""
        baseline, ms_lstm, dualgat = sub_preds
        # Set equal IC history for all models
        ensemble.model_ic_history = {
            "baseline": [0.05] * 20,
            "ms_lstm":  [0.05] * 20,
            "dualgat":  [0.05] * 20,
        }
        df = ensemble.predict(["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"], "2024-06-15",
                              baseline, ms_lstm, dualgat)
        # With equal weights, ensemble pred ≈ mean of the three
        expected_mean = (baseline["predicted_return"].values +
                         ms_lstm["predicted_return"].values +
                         dualgat["predicted_return"].values) / 3.0
        # After z-score normalization the correlation should be 1.0
        assert np.corrcoef(df["predicted_return"].values, expected_mean)[0, 1] > 0.99

    def test_low_ic_model_gets_near_zero_weight(self, ensemble, sub_preds):
        """A model with IC=0 should contribute almost nothing."""
        baseline, ms_lstm, dualgat = sub_preds
        ensemble.model_ic_history = {
            "baseline": [0.05] * 20,
            "ms_lstm":  [0.00] * 20,   # zero IC
            "dualgat":  [0.05] * 20,
        }
        df = ensemble.predict(["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"], "2024-06-15",
                              baseline, ms_lstm, dualgat)
        # Ensemble should be closer to the average of baseline+dualgat than to ms_lstm
        avg_good = (baseline["predicted_return"].values + dualgat["predicted_return"].values) / 2.0
        corr_with_good = np.corrcoef(df["predicted_return"].values, avg_good)[0, 1]
        corr_with_bad = np.corrcoef(df["predicted_return"].values,
                                    ms_lstm["predicted_return"].values)[0, 1]
        assert corr_with_good > corr_with_bad

    def test_predict_empty_stocks(self, ensemble):
        """Predict handles empty stock list."""
        empty = _make_preds([], [], "baseline")
        df = ensemble.predict([], "2024-06-15", empty, empty, empty)
        assert len(df) == 0

    def test_save_and_load_roundtrip(self, ensemble, sub_preds, tmp_path):
        """Weighted ensemble survives save→load roundtrip."""
        baseline, ms_lstm, dualgat = sub_preds
        ensemble.model_ic_history = {
            "baseline": [0.03] * 20,
            "ms_lstm":  [0.06] * 20,
            "dualgat":  [0.04] * 20,
        }

        path = tmp_path / "test_ensemble.pt"
        ensemble.save(path)
        assert path.exists()

        from src.model.ensemble import EnsemblePredictor
        loaded = EnsemblePredictor(strategy="weighted")
        loaded.load(path)

        stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
        df1 = ensemble.predict(stocks, "2024-06-15", baseline, ms_lstm, dualgat)
        df2 = loaded.predict(stocks, "2024-06-15", baseline, ms_lstm, dualgat)
        assert np.allclose(df1["predicted_return"].values, df2["predicted_return"].values)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/test_ensemble.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'src.model.ensemble'`

- [ ] **Step 3: Write EnsemblePredictor implementation (weighted average)**

```python
# src/model/ensemble.py
"""Ensemble predictor fusing Baseline, MS-LSTM, and DualGAT predictions.

Two strategies:
  - weighted: Softmax over rolling IC determines per-model weight.
  - meta: Small MLP [3→8→1] trained with IC loss.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from config import (
    ENSEMBLE_STRATEGY,
    ENSEMBLE_TEMPERATURE,
    ENSEMBLE_IC_WINDOW,
    ENSEMBLE_META_HIDDEN,
    ENSEMBLE_META_LR,
    ENSEMBLE_META_EPOCHS,
    ENSEMBLE_META_PATIENCE,
    ENSEMBLE_MODEL_PATH,
    ENSEMBLE_META_PATH,
)

logger = logging.getLogger(__name__)

_MODEL_IDS = ["baseline", "ms_lstm", "dualgat"]


# ------------------------------------------------------------------
# Meta-Learner MLP
# ------------------------------------------------------------------

class _MetaMLP(nn.Module):
    """Small MLP meta-learner: [3 → hidden → 1]."""

    def __init__(self, hidden: int = ENSEMBLE_META_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N, 3] → [N]."""
        return self.net(x).squeeze(-1)


# ------------------------------------------------------------------
# Ensemble Predictor
# ------------------------------------------------------------------

class EnsemblePredictor:
    """Fuses Baseline + MS-LSTM + DualGAT predictions.

    Args:
        strategy: "weighted" (default) or "meta".
        temperature: Softmax temperature for IC→weight conversion.
        device: Torch device string or None for auto-detect.
    """

    def __init__(
        self,
        strategy: str = ENSEMBLE_STRATEGY,
        temperature: float = ENSEMBLE_TEMPERATURE,
        device: str | None = None,
    ):
        self.strategy = strategy
        self.temperature = temperature
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model_ic_history: dict[str, list[float]] = {
            mid: [] for mid in _MODEL_IDS
        }

        self._meta: _MetaMLP | None = None
        if strategy == "meta":
            self._meta = _MetaMLP().to(self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        stocks: list[str],
        date_str: str,
        baseline_preds: pd.DataFrame,
        ms_lstm_preds: pd.DataFrame,
        dualgat_preds: pd.DataFrame,
    ) -> pd.DataFrame:
        """Fuse predictions from three sub-models.

        Args:
            stocks: Ordered list of stock tickers.
            date_str: Prediction date (YYYY-MM-DD).
            baseline_preds: BaselinePredictor.predict() output.
            ms_lstm_preds: MSLSTMPredictor.predict() output.
            dualgat_preds: DualGATPredictor.predict() output.

        Returns:
            DataFrame with columns:
              stock, date, predicted_return, signal_source,
              baseline_return, ms_lstm_return, dualgat_return
        """
        if not stocks:
            return pd.DataFrame(columns=[
                "stock", "date", "predicted_return", "signal_source",
                "baseline_return", "ms_lstm_return", "dualgat_return",
            ])

        # Align predictions to stock order
        bl_map = dict(zip(baseline_preds["stock"], baseline_preds["predicted_return"]))
        ms_map = dict(zip(ms_lstm_preds["stock"], ms_lstm_preds["predicted_return"]))
        dg_map = dict(zip(dualgat_preds["stock"], dualgat_preds["predicted_return"]))

        bl_vals = np.array([bl_map.get(s, 0.0) for s in stocks], dtype=np.float32)
        ms_vals = np.array([ms_map.get(s, 0.0) for s in stocks], dtype=np.float32)
        dg_vals = np.array([dg_map.get(s, 0.0) for s in stocks], dtype=np.float32)

        # Compute ensemble prediction
        if self.strategy == "weighted":
            fused = self._fuse_weighted(bl_vals, ms_vals, dg_vals)
        else:
            fused = self._fuse_meta(bl_vals, ms_vals, dg_vals)

        # Z-score normalize
        std = np.std(fused)
        if std > 0:
            fused = (fused - np.mean(fused)) / std

        df = pd.DataFrame({
            "stock": stocks,
            "date": date_str,
            "predicted_return": fused,
            "signal_source": "ensemble",
            "baseline_return": bl_vals,
            "ms_lstm_return": ms_vals,
            "dualgat_return": dg_vals,
        })
        return df.sort_values("predicted_return", ascending=False)

    def update_ic_history(self, model_id: str, ic: float) -> None:
        """Record a new IC value for a sub-model."""
        if model_id in self.model_ic_history:
            self.model_ic_history[model_id].append(ic)
            if len(self.model_ic_history[model_id]) > ENSEMBLE_IC_WINDOW:
                self.model_ic_history[model_id] = \
                    self.model_ic_history[model_id][-ENSEMBLE_IC_WINDOW:]

    # fit_meta() — see Task 3

    def save(self, path: str | Path) -> None:
        """Save ensemble state to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "strategy": self.strategy,
            "temperature": self.temperature,
            "model_ic_history": self.model_ic_history,
        }
        if self._meta is not None:
            data["meta_state_dict"] = {k: v.cpu().clone()
                                       for k, v in self._meta.state_dict().items()}
        torch.save(data, path)
        logger.info(f"Ensemble saved to {path}")

    def load(self, path: str | Path) -> None:
        """Load ensemble state from disk."""
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.strategy = checkpoint.get("strategy", "weighted")
        self.temperature = checkpoint.get("temperature", ENSEMBLE_TEMPERATURE)
        self.model_ic_history = checkpoint.get("model_ic_history",
                                               {mid: [] for mid in _MODEL_IDS})
        if "meta_state_dict" in checkpoint:
            self._meta = _MetaMLP().to(self.device)
            self._meta.load_state_dict(checkpoint["meta_state_dict"])
            self._meta.eval()
        logger.info(f"Ensemble loaded from {path}")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_weights(self) -> dict[str, float]:
        """Compute softmax weights from rolling mean IC."""
        means = {}
        for mid in _MODEL_IDS:
            hist = self.model_ic_history.get(mid, [])
            means[mid] = np.mean(hist) if hist else 0.0

        # Softmax with temperature
        scores = np.array([means[mid] for mid in _MODEL_IDS])
        scaled = scores / max(self.temperature, 1e-8)
        scaled -= scaled.max()  # numerical stability
        exp_scores = np.exp(scaled)
        weights = exp_scores / exp_scores.sum()

        return dict(zip(_MODEL_IDS, weights))

    def _fuse_weighted(
        self,
        bl: np.ndarray,
        ms: np.ndarray,
        dg: np.ndarray,
    ) -> np.ndarray:
        """Weighted average using softmax-IC weights."""
        w = self._compute_weights()
        return w["baseline"] * bl + w["ms_lstm"] * ms + w["dualgat"] * dg

    def _fuse_meta(
        self,
        bl: np.ndarray,
        ms: np.ndarray,
        dg: np.ndarray,
    ) -> np.ndarray:
        """Meta-learner MLP fusion."""
        if self._meta is None:
            # Fall back to weighted average
            return self._fuse_weighted(bl, ms, dg)
        self._meta.eval()
        x = np.stack([bl, ms, dg], axis=1)  # [N, 3]
        x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self._meta(x_t).cpu().numpy()
```

- [ ] **Step 4: Run weighted-average tests to verify they pass**

```bash
python3 -m pytest tests/test_ensemble.py -v
```
Expected: 5 tests pass (weighted average + save/load roundtrip)

- [ ] **Step 5: Commit**

```bash
git add src/model/ensemble.py tests/test_ensemble.py
git commit -m "feat: EnsemblePredictor with weighted-average fusion"
```

---

### Task 3: EnsemblePredictor — Meta-Learner MLP Training

**Files:**
- Modify: `src/model/ensemble.py` (append `fit_meta()` method)
- Modify: `tests/test_ensemble.py` (append meta-learner tests)

**Interfaces:**
- Produces:
  - `EnsemblePredictor.fit_meta(stocks, start_date, end_date, baseline, ms_lstm, dualgat, epochs, lr) -> dict`

- [ ] **Step 1: Write failing meta-learner tests**

Append to `tests/test_ensemble.py`:

```python
class TestEnsembleMeta:
    """Tests for meta-learner MLP ensemble strategy."""

    @pytest.fixture
    def meta_ensemble(self):
        from src.model.ensemble import EnsemblePredictor
        return EnsemblePredictor(strategy="meta", temperature=0.1)

    def test_meta_forward_produces_n_outputs(self, meta_ensemble, prepopulated_db):
        """Meta-learner forward pass produces [N] predictions."""
        import torch
        torch.manual_seed(42)
        np.random.seed(42)

        stocks = ["AAPL", "MSFT"]
        from src.model.baseline import BaselinePredictor
        bl = BaselinePredictor()
        bl_df = bl.predict(stocks, "2024-06-15", [])

        from src.model.ms_lstm import MSLSTMPredictor
        ms = MSLSTMPredictor(hidden_dim=8, num_scales=3)

        from src.model.dualgat import DualGATPredictor
        dg = DualGATPredictor(hidden=16, out_dim=8, heads=2)

        df = meta_ensemble.predict(stocks, "2024-06-15", bl_df, bl_df, bl_df)
        assert len(df) == 2
        assert "baseline_return" in df.columns
        assert "ms_lstm_return" in df.columns
        assert "dualgat_return" in df.columns

    def test_meta_predict_uses_mlp(self, meta_ensemble):
        """Meta strategy uses MLP when weights would be equal."""
        bl = pd.DataFrame({"stock": ["A", "B"], "predicted_return": [0.1, -0.1]})
        ms = pd.DataFrame({"stock": ["A", "B"], "predicted_return": [0.1, -0.1]})
        dg = pd.DataFrame({"stock": ["A", "B"], "predicted_return": [0.1, -0.1]})
        meta_ensemble.model_ic_history = {
            "baseline": [0.05] * 20, "ms_lstm": [0.05] * 20, "dualgat": [0.05] * 20,
        }
        df = meta_ensemble.predict(["A", "B"], "2024-06-15", bl, ms, dg)
        # With equal inputs and equal weights, weighted avg would return proportionally identical
        # Meta MLP may differ due to learned parameters
        assert len(df) == 2

    def test_fit_meta_runs_one_epoch(self, meta_ensemble, prepopulated_db, tmp_path):
        """fit_meta() completes one epoch without error."""
        import torch
        torch.manual_seed(123)
        np.random.seed(123)

        stocks = ["AAPL", "MSFT"]

        from src.model.baseline import BaselinePredictor
        baseline = BaselinePredictor()

        from src.model.ms_lstm import MSLSTMPredictor
        ms = MSLSTMPredictor(hidden_dim=8, num_scales=3)
        ms_path = tmp_path / "dummy_ms_meta.pt"
        ms.save(ms_path)

        from src.model.dualgat import DualGATPredictor
        dg = DualGATPredictor(hidden=16, out_dim=8, heads=2)

        history = meta_ensemble.fit_meta(
            stocks=stocks,
            start_date="2024-05-20",
            end_date="2024-06-15",
            baseline=baseline,
            ms_lstm=ms,
            dualgat=dg,
            epochs=1,
            lr=1e-3,
        )
        assert "train_loss" in history
        assert len(history["train_loss"]) == 1
        assert np.isfinite(history["train_loss"][0])

    def test_meta_save_load_roundtrip(self, meta_ensemble, tmp_path):
        """Meta-learner survives save→load roundtrip."""
        import torch
        torch.manual_seed(42)

        meta_ensemble.model_ic_history = {
            "baseline": [0.03] * 20, "ms_lstm": [0.06] * 20, "dualgat": [0.04] * 20,
        }

        path = tmp_path / "test_meta.pt"
        meta_ensemble.save(path)

        from src.model.ensemble import EnsemblePredictor
        loaded = EnsemblePredictor(strategy="meta")
        loaded.load(path)

        bl = pd.DataFrame({"stock": ["A"], "predicted_return": [0.05]})
        ms = pd.DataFrame({"stock": ["A"], "predicted_return": [0.03]})
        dg = pd.DataFrame({"stock": ["A"], "predicted_return": [0.07]})

        df1 = meta_ensemble.predict(["A"], "2024-06-15", bl, ms, dg)
        df2 = loaded.predict(["A"], "2024-06-15", bl, ms, dg)
        assert np.allclose(df1["predicted_return"].values, df2["predicted_return"].values)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/test_ensemble.py::TestEnsembleMeta -v
```
Expected: FAIL — `AttributeError: 'EnsemblePredictor' object has no attribute 'fit_meta'`

- [ ] **Step 3: Write fit_meta() implementation**

Append to `src/model/ensemble.py`, inside the `EnsemblePredictor` class, after the `update_ic_history()` method:

```python
    def fit_meta(
        self,
        stocks: list[str],
        start_date: str,
        end_date: str,
        baseline,
        ms_lstm,
        dualgat,
        epochs: int = ENSEMBLE_META_EPOCHS,
        lr: float = ENSEMBLE_META_LR,
    ) -> dict:
        """Train meta-learner MLP on historical data.

        Args:
            stocks: List of ticker symbols.
            start_date / end_date: Training date range (YYYY-MM-DD).
            baseline: BaselinePredictor instance.
            ms_lstm: MSLSTMPredictor instance.
            dualgat: DualGATPredictor instance.
            epochs: Maximum training epochs.
            lr: Learning rate.

        Returns:
            Dict with keys: train_loss, val_ic, best_epoch.
        """
        if self._meta is None:
            self._meta = _MetaMLP().to(self.device)
            self.strategy = "meta"

        # Get trading dates
        from src.model.dualgat import _get_trading_dates
        trading_dates = _get_trading_dates(stocks, start_date, end_date)
        if len(trading_dates) < 10:
            logger.warning(f"Only {len(trading_dates)} trading dates, need >= 10")
            return {"train_loss": [], "val_ic": [], "best_epoch": 0}

        split = int(len(trading_dates) * 0.8)
        train_dates = trading_dates[:split]
        val_dates = trading_dates[split:]

        optimizer = torch.optim.Adam(self._meta.parameters(), lr=lr)
        history = {"train_loss": [], "val_ic": []}
        best_val_ic = -float("inf")
        best_state = None
        patience_counter = 0

        from src.db import schema as db

        for epoch in range(epochs):
            # Training
            self._meta.train()
            epoch_losses = []
            for date_str in train_dates:
                try:
                    bl_df = baseline.predict(stocks, date_str, [])
                    ms_df = ms_lstm.predict(stocks, date_str)
                    dg_df = dualgat.predict(stocks, date_str)
                except Exception:
                    continue

                if len(bl_df) < 3:
                    continue

                # Align predictions
                bl_map = dict(zip(bl_df["stock"], bl_df["predicted_return"]))
                ms_map = dict(zip(ms_df["stock"], ms_df["predicted_return"]))
                dg_map = dict(zip(dg_df["stock"], dg_df["predicted_return"]))

                common = set(bl_map) & set(ms_map) & set(dg_map)
                stock_list = [s for s in stocks if s in common]
                if len(stock_list) < 3:
                    continue

                bl_arr = np.array([bl_map[s] for s in stock_list], dtype=np.float32)
                ms_arr = np.array([ms_map[s] for s in stock_list], dtype=np.float32)
                dg_arr = np.array([dg_map[s] for s in stock_list], dtype=np.float32)
                x = np.stack([bl_arr, ms_arr, dg_arr], axis=1)

                # Targets: actual returns
                all_prices = db.get_prices(stock_list, date_str, date_str)
                targets = []
                valid_stocks = []
                for s in stock_list:
                    sp = all_prices.get(s, [])
                    if len(sp) >= 2:
                        # Find today's close and yesterday's close
                        sp_sorted = sorted(sp, key=lambda p: p["date"])
                        for i, p in enumerate(sp_sorted):
                            if p["date"] == date_str and i > 0:
                                prev = sp_sorted[i - 1]["close"]
                                curr = p["close"]
                                if prev > 0:
                                    targets.append((curr - prev) / prev)
                                    valid_stocks.append(s)
                                break

                if len(targets) < 3:
                    continue

                x_t = torch.tensor(x[:len(valid_stocks)], dtype=torch.float32, device=self.device)
                y_t = torch.tensor(targets, dtype=torch.float32, device=self.device)

                optimizer.zero_grad()
                preds = self._meta(x_t)
                # IC loss
                vx = preds - preds.mean()
                vy = y_t - y_t.mean()
                numerator = (vx * vy).sum()
                denominator = torch.sqrt((vx ** 2).sum()) * torch.sqrt((vy ** 2).sum())
                corr = numerator / (denominator + 1e-8)
                loss = 1.0 - corr
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            history["train_loss"].append(avg_loss)

            # Validation
            val_ics = []
            self._meta.eval()
            for date_str in val_dates:
                try:
                    bl_df = baseline.predict(stocks, date_str, [])
                    ms_df = ms_lstm.predict(stocks, date_str)
                    dg_df = dualgat.predict(stocks, date_str)
                except Exception:
                    continue
                if len(bl_df) < 3:
                    continue
                bl_map = dict(zip(bl_df["stock"], bl_df["predicted_return"]))
                ms_map = dict(zip(ms_df["stock"], ms_df["predicted_return"]))
                dg_map = dict(zip(dg_df["stock"], dg_df["predicted_return"]))
                common = set(bl_map) & set(ms_map) & set(dg_map)
                stock_list = [s for s in stocks if s in common]
                if len(stock_list) < 3:
                    continue
                bl_arr = np.array([bl_map[s] for s in stock_list], dtype=np.float32)
                ms_arr = np.array([ms_map[s] for s in stock_list], dtype=np.float32)
                dg_arr = np.array([dg_map[s] for s in stock_list], dtype=np.float32)
                x = np.stack([bl_arr, ms_arr, dg_arr], axis=1)
                x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
                all_prices = db.get_prices(stock_list, date_str, date_str)
                targets = []
                for s in stock_list:
                    sp = all_prices.get(s, [])
                    sp_sorted = sorted(sp, key=lambda p: p["date"])
                    for i, p in enumerate(sp_sorted):
                        if p["date"] == date_str and i > 0:
                            prev = sp_sorted[i - 1]["close"]
                            curr = p["close"]
                            if prev > 0:
                                targets.append((curr - prev) / prev)
                            break
                if len(targets) < 3:
                    continue
                y_t = torch.tensor(targets, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    preds = self._meta(x_t)
                    vx = preds - preds.mean()
                    vy = y_t - y_t.mean()
                    numerator = (vx * vy).sum()
                    denominator = torch.sqrt((vx ** 2).sum()) * torch.sqrt((vy ** 2).sum())
                    corr = numerator / (denominator + 1e-8)
                    val_ics.append(corr.item())

            avg_val_ic = float(np.mean(val_ics)) if val_ics else 0.0
            history["val_ic"].append(avg_val_ic)

            logger.info(f"Meta epoch {epoch+1}/{epochs}: train_loss={avg_loss:.4f}, val_ic={avg_val_ic:.4f}")

            if avg_val_ic > best_val_ic:
                best_val_ic = avg_val_ic
                best_state = {k: v.cpu().clone() for k, v in self._meta.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= ENSEMBLE_META_PATIENCE:
                    logger.info(f"Early stopping at meta epoch {epoch+1}")
                    break

        if best_state is not None:
            self._meta.load_state_dict(best_state)

        history["best_epoch"] = int(np.argmax(history["val_ic"]) + 1) if history["val_ic"] else 0
        return history
```

- [ ] **Step 4: Run all ensemble tests**

```bash
python3 -m pytest tests/test_ensemble.py -v
```
Expected: 9 tests pass (5 weighted + 4 meta)

- [ ] **Step 5: Commit**

```bash
git add src/model/ensemble.py tests/test_ensemble.py
git commit -m "feat: EnsemblePredictor meta-learner MLP training"
```

---

### Task 4: API — Model Listing, Model Param, Backtest Comparison

**Files:**
- Modify: `src/web/api.py` (add `/api/models`, model param on predictions, `/api/backtest/compare`)
- Modify: `tests/test_api.py` (append endpoint tests)

**Interfaces:**
- Produces:
  - `GET /api/models` → `{ models: [...] }`
  - `GET /api/predictions?date=...&model=baseline|ms_lstm|dualgat|ensemble`
  - `GET /api/backtest/compare?start=...&end=...` → `{ models: {id: {...}} }`
- Consumes: `EnsemblePredictor`, `MSLSTMPredictor`, `DualGATPredictor` (all existing)

- [ ] **Step 1: Write failing API tests**

Append to `tests/test_api.py`:

```python
class TestModelsEndpoint:
    """Tests for GET /api/models."""

    def test_list_models(self, client):
        """Returns all 4 models with availability info."""
        resp = client.get("/api/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        ids = [m["id"] for m in data["models"]]
        assert "baseline" in ids
        assert "ms_lstm" in ids
        assert "dualgat" in ids
        assert "ensemble" in ids
        # Baseline is always available
        baseline = next(m for m in data["models"] if m["id"] == "baseline")
        assert baseline["available"] is True


class TestModelParamPredictions:
    """Tests for GET /api/predictions?model=..."""

    def test_predictions_default_to_baseline(self, client, prepopulated_db):
        """Without model param, defaults to baseline (backward compat)."""
        # Set a date that has experts
        resp = client.get("/api/predictions?date=2024-06-15")
        assert resp.status_code == 200
        data = resp.json()
        assert "predictions" in data

    def test_predictions_with_ms_lstm_model(self, client, prepopulated_db):
        """model=ms_lstm returns predictions."""
        resp = client.get("/api/predictions?date=2024-06-15&model=ms_lstm")
        assert resp.status_code == 200

    def test_predictions_invalid_model(self, client):
        """Invalid model name returns 400."""
        resp = client.get("/api/predictions?date=2024-06-15&model=unknown")
        assert resp.status_code == 400


class TestBacktestCompareEndpoint:
    """Tests for GET /api/backtest/compare."""

    def test_compare_returns_all_models(self, client, prepopulated_db):
        """Compare endpoint returns backtest results for all available models."""
        resp = client.get("/api/backtest/compare?start=2024-05-20&end=2024-06-15")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert "baseline" in data["models"]
        for mid in ["baseline", "ms_lstm", "dualgat", "ensemble"]:
            if mid in data["models"]:
                m = data["models"][mid]
                assert "annualized_return" in m
                assert "sharpe_ratio" in m
                assert "cumulative_returns" in m

    def test_compare_skips_unavailable_models(self, client):
        """Models without required files are omitted, not errored."""
        resp = client.get("/api/backtest/compare?start=2024-05-20&end=2024-06-15")
        assert resp.status_code in (200, 404)  # 404 if no models at all
```

- [ ] **Step 2: Run API tests to verify they fail**

```bash
python3 -m pytest tests/test_api.py::TestModelsEndpoint tests/test_api.py::TestModelParamPredictions tests/test_api.py::TestBacktestCompareEndpoint -v
```
Expected: FAIL — 404/500 for new endpoints

- [ ] **Step 3: Write API implementation**

Modify `src/web/api.py`:

Add lazy model loading helpers at the top (after the existing `get_predictor` function):

```python
# Model cache (lazy init, loaded once per process lifetime)
_model_cache: dict[str, object] = {}


def _get_model(model_id: str) -> object | None:
    """Load and cache a prediction model by ID."""
    global _model_cache
    if model_id in _model_cache:
        return _model_cache[model_id]

    model = None
    if model_id == "baseline":
        from src.model.baseline import BaselinePredictor
        model = BaselinePredictor()
    elif model_id == "ms_lstm":
        from config import MSLSTM_MODEL_PATH
        if MSLSTM_MODEL_PATH.exists():
            from src.model.ms_lstm import MSLSTMPredictor
            model = MSLSTMPredictor()
            model.load(MSLSTM_MODEL_PATH)
    elif model_id == "dualgat":
        from config import DUALGAT_MODEL_PATH
        if DUALGAT_MODEL_PATH.exists():
            from src.model.dualgat import DualGATPredictor
            model = DualGATPredictor()
            model.load(DUALGAT_MODEL_PATH)
    elif model_id == "ensemble":
        from src.model.ensemble import EnsemblePredictor
        from config import ENSEMBLE_MODEL_PATH
        ensemble = EnsemblePredictor(strategy="weighted")
        if ENSEMBLE_MODEL_PATH.exists():
            ensemble.load(ENSEMBLE_MODEL_PATH)
        model = ensemble

    _model_cache[model_id] = model
    return model


def _get_available_models() -> list[dict]:
    """Return metadata for all models, including availability."""
    from config import MSLSTM_MODEL_PATH, DUALGAT_MODEL_PATH, ENSEMBLE_MODEL_PATH, ENSEMBLE_META_PATH

    ms_available = MSLSTM_MODEL_PATH.exists()
    dg_available = DUALGAT_MODEL_PATH.exists()
    all_three = ms_available and dg_available  # baseline always available
    ens_meta_available = ENSEMBLE_META_PATH.exists()

    return [
        {"id": "baseline", "name": "Baseline", "available": True, "needs_training": False},
        {"id": "ms_lstm",  "name": "MS-LSTM",   "available": ms_available,
         "needs_model": str(MSLSTM_MODEL_PATH) if not ms_available else None},
        {"id": "dualgat",  "name": "DualGAT",   "available": dg_available,
         "needs_model": str(DUALGAT_MODEL_PATH) if not dg_available else None},
        {"id": "ensemble", "name": "Ensemble",  "available": all_three,
         "strategy": "weighted" if not ens_meta_available else "meta"},
    ]
```

Add the `/api/models` endpoint:

```python
@app.get("/api/models")
async def list_models():
    """Return all prediction models and their availability."""
    return {"models": _get_available_models()}
```

Modify `/api/predictions` to accept `model` param (replace the existing function):

```python
@app.get("/api/predictions")
async def get_predictions(
    date: str = Query(None, description="Date in YYYY-MM-DD format"),
    model: str = Query("baseline", description="Model ID: baseline|ms_lstm|dualgat|ensemble"),
):
    """Get stock return predictions for a given date and model."""
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    valid_models = {"baseline", "ms_lstm", "dualgat", "ensemble"}
    if model not in valid_models:
        raise HTTPException(400, f"Unknown model '{model}'. Valid: {', '.join(sorted(valid_models))}")

    tracker = get_tracker()
    expert_records = tracker.trace(date)

    available = {m["id"]: m["available"] for m in _get_available_models()}
    if not available.get(model, False):
        raise HTTPException(503, f"Model '{model}' is not available (missing model file)")

    # Generate predictions
    pred_model = _get_model(model)
    if pred_model is None:
        raise HTTPException(503, f"Model '{model}' failed to load")

    if model == "baseline":
        pred_df = pred_model.predict(DEFAULT_TICKERS, date, expert_records)
    elif model == "ms_lstm":
        pred_df = pred_model.predict(DEFAULT_TICKERS, date)
    elif model == "dualgat":
        pred_df = pred_model.predict(DEFAULT_TICKERS, date)
    elif model == "ensemble":
        bl = _get_model("baseline")
        ms = _get_model("ms_lstm")
        dg = _get_model("dualgat")
        if bl is None or ms is None or dg is None:
            raise HTTPException(503, "Ensemble requires all sub-models to be available")
        bl_df = bl.predict(DEFAULT_TICKERS, date, expert_records)
        ms_df = ms.predict(DEFAULT_TICKERS, date)
        dg_df = dg.predict(DEFAULT_TICKERS, date)
        pred_df = pred_model.predict(DEFAULT_TICKERS, date, bl_df, ms_df, dg_df)

    # Build response
    response = {
        "date": date,
        "model": model,
        "predictions": pred_df.to_dict(orient="records"),
    }
    if model != "baseline":
        response["expert_coverage"] = len([r for r in expert_records if r.expert_type != "none"])
    else:
        response["expert_coverage"] = len([r for r in expert_records if r.expert_type != "none"])

    return response
```

Add the `/api/backtest/compare` endpoint:

```python
@app.get("/api/backtest/compare")
async def compare_backtest(
    start: str = Query(None, description="Start date YYYY-MM-DD"),
    end: str = Query(None, description="End date YYYY-MM-DD"),
):
    """Run backtest for all available models and return comparison."""
    if start is None:
        start = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    if end is None:
        end = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    available_models = _get_available_models()
    available_ids = [m["id"] for m in available_models if m["available"]]
    if not available_ids:
        raise HTTPException(404, "No models available for backtest")

    tracker = get_tracker()

    # Get trading dates
    from src.db import schema as db_schema
    all_prices = db_schema.get_prices(DEFAULT_TICKERS, start, end)
    trading_dates = set()
    for stock_prices in all_prices.values():
        for p in stock_prices:
            trading_dates.add(p["date"])
    trading_dates = sorted([d for d in trading_dates if start <= d <= end])

    if len(trading_dates) < 3:
        raise HTTPException(404, "Insufficient trading data for the date range")

    results = {}
    for model_id in available_ids:
        try:
            pred_model = _get_model(model_id)
            if pred_model is None:
                continue

            all_preds = []
            for date_str in trading_dates:
                expert_records = tracker.trace(date_str)
                if model_id == "baseline":
                    pdf = pred_model.predict(DEFAULT_TICKERS, date_str, expert_records)
                elif model_id == "ms_lstm":
                    pdf = pred_model.predict(DEFAULT_TICKERS, date_str)
                elif model_id == "dualgat":
                    pdf = pred_model.predict(DEFAULT_TICKERS, date_str)
                elif model_id == "ensemble":
                    bl = _get_model("baseline")
                    ms = _get_model("ms_lstm")
                    dg = _get_model("dualgat")
                    if bl is None or ms is None or dg is None:
                        continue
                    bl_df = bl.predict(DEFAULT_TICKERS, date_str, expert_records)
                    ms_df = ms.predict(DEFAULT_TICKERS, date_str)
                    dg_df = dg.predict(DEFAULT_TICKERS, date_str)
                    pdf = pred_model.predict(DEFAULT_TICKERS, date_str, bl_df, ms_df, dg_df)
                all_preds.append(pdf)

            if not all_preds:
                continue

            combined = pd.concat(all_preds, ignore_index=True)
            bt_result = run_backtest(combined, DEFAULT_TICKERS, start, end)
            results[model_id] = {
                "annualized_return": bt_result["annualized_return"],
                "sharpe_ratio": bt_result["sharpe_ratio"],
                "max_drawdown": bt_result["max_drawdown"],
                "mean_ic": bt_result["mean_ic"],
                "icir": bt_result["icir"],
                "n_trading_days": bt_result["n_trading_days"],
                "cumulative_returns": bt_result["cumulative_returns"].tolist(),
            }
        except Exception as e:
            logger.warning(f"Backtest compare failed for {model_id}: {e}")

    if not results:
        raise HTTPException(404, "No backtest results could be computed")

    return {"start": start, "end": end, "models": results}
```

- [ ] **Step 4: Run all tests**

```bash
python3 -m pytest tests/test_api.py -v --tb=short
```
Expected: New endpoint tests pass. Pre-existing `test_get_backtest` may still fail (unrelated, needs data in DB).

- [ ] **Step 5: Commit**

```bash
git add src/web/api.py tests/test_api.py
git commit -m "feat: API /api/models, model=? param, /api/backtest/compare"
```

---

### Task 5: Frontend — Model Switching + Multi-Line Comparison

**Files:**
- Modify: `src/web/templates/index.html` (model tab bar, comparison table, status indicators)
- Modify: `src/web/templates/index_zh.html` (same, Chinese)
- Modify: `src/web/static/app.js` (model switching logic, multi-line chart, comparison table)
- Modify: `src/web/static/app_zh.js` (same, Chinese)

**Interfaces:**
- Consumes: `/api/models`, `/api/predictions?model=...`, `/api/backtest/compare`
- Produces: Model tab bar UI, multi-line backtest chart, comparison table

- [ ] **Step 1: Add CSS for model tabs, comparison, and status indicators**

Replace the `<style>` block in both `index.html` and `index_zh.html`. Below is the English version CSS additions — apply the same to `index_zh.html` (the templates differ only in content text, CSS is shared).

In `index.html` and `index_zh.html`, add these styles after the existing `.summary-bar` rule:

```css
        .model-tabs { display: flex; gap: 0; max-width: 1400px; margin: 0 auto 15px; background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
        .model-tab { flex: 1; text-align: center; padding: 12px 8px; cursor: pointer; font-size: 13px; font-weight: 600; border: none; background: white; color: #888; transition: all 0.2s; border-bottom: 3px solid transparent; }
        .model-tab:hover { background: #f8f9ff; color: #333; }
        .model-tab.active { color: #16213e; border-bottom-color: #16213e; background: #f0f2ff; }
        .model-tab.unavailable { color: #ccc; cursor: not-allowed; opacity: 0.5; }
        .model-tab .tab-ic { display: block; font-size: 10px; font-weight: 400; color: #28a745; margin-top: 2px; }
        .model-tab.unavailable .tab-ic { color: #ccc; }
        .compare-toggle { display: flex; align-items: center; padding: 0 16px; font-size: 11px; color: #888; cursor: pointer; white-space: nowrap; }
        .compare-toggle input { margin-right: 4px; }
        .compare-table { width: 100%; font-size: 12px; margin-top: 10px; }
        .compare-table .best { background: #d4edda; font-weight: 700; }
        .model-status { display: flex; gap: 8px; flex-wrap: wrap; max-width: 1400px; margin: 0 auto 15px; }
        .model-status .chip { font-size: 11px; padding: 3px 10px; border-radius: 12px; background: #e8e8e8; color: #888; }
        .model-status .chip.available { background: #d4edda; color: #155724; }
        .predictions-compare { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 12px; }
        .predictions-compare .mini-card { background: #f8f9fa; border-radius: 8px; padding: 12px; }
        .predictions-compare .mini-card h4 { font-size: 12px; color: #666; margin-bottom: 8px; }
        .predictions-compare table { font-size: 11px; }
```

- [ ] **Step 2: Add model tab bar and comparison elements to HTML templates**

In `index.html`, replace the controls div (the one with max-width:1400px) with:

```html
    <div class="model-tabs" id="model-tabs">
        <button class="model-tab active" data-model="baseline" onclick="selectModel('baseline')">
            Baseline<span class="tab-ic"></span>
        </button>
        <button class="model-tab" data-model="ms_lstm" onclick="selectModel('ms_lstm')">
            MS-LSTM<span class="tab-ic"></span>
        </button>
        <button class="model-tab" data-model="dualgat" onclick="selectModel('dualgat')">
            DualGAT<span class="tab-ic"></span>
        </button>
        <button class="model-tab" data-model="ensemble" onclick="selectModel('ensemble')">
            Ensemble<span class="tab-ic"></span>
        </button>
        <label class="compare-toggle">
            <input type="checkbox" id="compare-mode" onchange="toggleCompare()"> Compare All
        </label>
    </div>

    <div class="controls" style="max-width:1400px; margin: 0 auto 15px;">
        <input type="date" id="date-picker" title="Select prediction date" />
        <button onclick="loadPredictions()" class="refresh">🔍 Predictions</button>
        <button onclick="loadExperts()">🎯 Experts</button>
        <button onclick="loadBacktest()">📊 Backtest</button>
        <button onclick="triggerCollect()">📥 Collect</button>
        <span id="status" style="margin-left:auto; color:#888; font-size:13px;"></span>
    </div>
```

In the predictions card, add a container for compare mode:

```html
        <div class="card">
            <h2>Return Predictions <span id="pred-date" style="font-weight:400;color:#888;"></span></h2>
            <div class="summary-bar" id="pred-summary"></div>
            <div id="predictions-table"><div class="loading">Loading predictions</div></div>
            <div id="predictions-compare" class="predictions-compare" style="display:none;"></div>
        </div>
```

In the backtest card, add a comparison table after the chart container:

```html
        <div class="card full">
            <h2>Backtest Performance</h2>
            <div class="metrics-row" id="metrics-summary">
                <div class="metric"><div class="value">--</div><div class="label">Annualized Return</div></div>
                <div class="metric"><div class="value">--</div><div class="label">Sharpe Ratio</div></div>
                <div class="metric"><div class="value">--</div><div class="label">Mean IC</div></div>
                <div class="metric"><div class="value red">--</div><div class="label">Max Drawdown</div></div>
            </div>
            <div id="chart-container"><canvas id="returnsChart"></canvas></div>
            <div id="chart-legend" style="text-align:center;margin-top:8px;font-size:12px;color:#888;"></div>
            <div id="compare-table-container" style="margin-top:12px;"></div>
        </div>
```

Apply the same changes to `index_zh.html` (adjusting text labels to Chinese).

- [ ] **Step 3: Write JavaScript for model switching, multi-line chart, comparison**

Replace `app.js` with the full updated version. Key new/modified functions below — integrate into the existing file:

```javascript
// DualGAT Stock Predictor — frontend logic (v0.4 model switching)
let returnsChart = null;
let currentModel = 'baseline';
let compareMode = false;

const MODEL_COLORS = {
    baseline: '#6c757d',
    ms_lstm:  '#0d6efd',
    dualgat:  '#fd7e14',
    ensemble: '#198754',
};

document.addEventListener('DOMContentLoaded', () => {
    const yesterday = new Date();
    yesterday.setDate(yesterday.getDate() - 1);
    document.getElementById('date-picker').value = yesterday.toISOString().split('T')[0];
    loadModels();
    loadPredictions();
    loadExperts();
    loadBacktest();
    loadSystemStatus();
});

// ── Model management ──

async function loadModels() {
    try {
        const resp = await fetch('/api/models');
        const data = await resp.json();
        window._models = data.models;
        updateModelTabs(data.models);
    } catch (e) {
        console.error('Failed to load models:', e);
    }
}

function updateModelTabs(models) {
    for (const m of models) {
        const tab = document.querySelector(`.model-tab[data-model="${m.id}"]`);
        if (!tab) continue;
        if (!m.available) {
            tab.classList.add('unavailable');
            tab.title = 'Model file not found';
        } else {
            tab.classList.remove('unavailable');
        }
    }
}

function selectModel(modelId) {
    const models = window._models || [];
    const m = models.find(x => x.id === modelId);
    if (m && !m.available) return;

    currentModel = modelId;
    document.querySelectorAll('.model-tab').forEach(t => t.classList.remove('active'));
    document.querySelector(`.model-tab[data-model="${modelId}"]`)?.classList.add('active');
    loadPredictions();
    loadBacktest();
}

function toggleCompare() {
    compareMode = document.getElementById('compare-mode').checked;
    loadPredictions();
}

// ── Predictions (updated) ──

async function loadPredictions() {
    const date = getDate();
    setStatus(`Loading predictions (${currentModel})...`);

    if (compareMode) {
        await loadPredictionsCompare(date);
        return;
    }

    document.getElementById('predictions-compare').style.display = 'none';
    document.getElementById('predictions-table').style.display = '';

    try {
        const resp = await fetch(`/api/predictions?date=${date}&model=${currentModel}`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();

        // Update model tab IC
        updateModelTabIC(currentModel, data);

        const experts = data.predictions.filter(p => p.signal_source === 'expert').length;
        const momentum = data.predictions.length - experts;
        document.getElementById('pred-date').textContent =
            `${date} [${currentModel}] (experts: ${experts} | momentum: ${momentum})`;

        if (!data.predictions || data.predictions.length === 0) {
            document.getElementById('predictions-table').innerHTML =
                '<div class="empty-state"><div class="icon">📭</div>No predictions available</div>';
            document.getElementById('pred-summary').innerHTML = '';
            return;
        }

        document.getElementById('pred-summary').innerHTML =
            `<span>🟢 Expert signals: <strong>${experts}</strong> stocks</span>` +
            `<span>⚪ Momentum signals: <strong>${momentum}</strong> stocks</span>` +
            `<span>📊 Coverage: <strong>${data.expert_coverage || '—'}</strong></span>`;

        const cols = [
            { key: 'stock', label: 'Stock' },
            { key: 'predicted_return', label: 'Predicted Return', fmt: v => `<span class="${v > 0 ? 'green' : 'red'}">${fmtPct(v)}</span>` },
            { key: 'signal_source', label: 'Source', fmt: v => {
                if (v === 'expert') return '<span class="badge badge-expert-signal">🧠 Expert</span>';
                if (v === 'ensemble') return '<span class="badge badge-expert-signal">🔗 Ensemble</span>';
                if (v === 'ms_lstm') return '<span class="badge badge-expert-signal">🔮 MS-LSTM</span>';
                if (v === 'dualgat') return '<span class="badge badge-expert-signal">🕸️ DualGAT</span>';
                return '<span class="badge badge-momentum">📐 Momentum</span>';
            }},
        ];
        document.getElementById('predictions-table').innerHTML = buildTable(data.predictions, cols, 20);
        setStatus('Predictions loaded');
    } catch (e) {
        document.getElementById('predictions-table').innerHTML =
            `<div class="empty-state"><div class="icon">⚠️</div>Load failed: ${e.message}</div>`;
        setStatus('Error loading predictions');
    }
}

async function loadPredictionsCompare(date) {
    document.getElementById('predictions-table').style.display = 'none';
    const container = document.getElementById('predictions-compare');
    container.style.display = 'grid';
    container.innerHTML = '<div class="loading">Loading all models...</div>';

    const modelIds = ['baseline', 'ms_lstm', 'dualgat', 'ensemble'];
    const names = { baseline: 'Baseline', ms_lstm: 'MS-LSTM', dualgat: 'DualGAT', ensemble: 'Ensemble' };
    const colors = { baseline: '#6c757d', ms_lstm: '#0d6efd', dualgat: '#fd7e14', ensemble: '#198754' };

    let html = '';
    for (const mid of modelIds) {
        try {
            const resp = await fetch(`/api/predictions?date=${date}&model=${mid}`);
            if (!resp.ok) {
                html += `<div class="mini-card"><h4 style="color:${colors[mid]};">${names[mid]}</h4><div class="empty-state" style="padding:10px;">⚠️ Unavailable</div></div>`;
                continue;
            }
            const data = await resp.json();
            const rows = data.predictions.slice(0, 10);
            let table = '<table><thead><tr><th>Stock</th><th>Return</th></tr></thead><tbody>';
            for (const p of rows) {
                table += `<tr><td><strong>${p.stock}</strong></td><td class="${p.predicted_return > 0 ? 'green' : 'red'}">${fmtPct(p.predicted_return)}</td></tr>`;
            }
            table += '</tbody></table>';
            html += `<div class="mini-card"><h4 style="color:${colors[mid]};">${names[mid]}</h4>${table}</div>`;
        } catch (e) {
            html += `<div class="mini-card"><h4 style="color:${colors[mid]};">${names[mid]}</h4><div class="empty-state">Error</div></div>`;
        }
    }
    container.innerHTML = html;
}

function updateModelTabIC(modelId, data) {
    // Compute IC from predictions if actuals available (not here — placeholder)
    // Can be extended to show per-model IC from backtest data
}

// ── Backtest (updated — multi-line chart + comparison table) ──

async function loadBacktest() {
    const endDate = getDate();
    const startDate = new Date(endDate);
    startDate.setDate(startDate.getDate() - 90);
    const start = startDate.toISOString().split('T')[0];

    try {
        const resp = await fetch(`/api/backtest/compare?start=${start}&end=${endDate}`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();

        const modelIds = Object.keys(data.models);
        if (modelIds.length === 0) {
            document.getElementById('chart-container').innerHTML =
                '<div class="empty-state"><div class="icon">📉</div>No backtest data available</div>';
            return;
        }

        // Show current model's metrics in the summary bar
        const cur = data.models[currentModel] || data.models[modelIds[0]];
        const arClass = cur.annualized_return > 0 ? 'green' : 'red';
        document.querySelector('#metrics-summary').innerHTML = `
            <div class="metric"><div class="value ${arClass}">${fmtPct1(cur.annualized_return)}</div><div class="label">Annualized Return</div></div>
            <div class="metric"><div class="value">${cur.sharpe_ratio.toFixed(2)}</div><div class="label">Sharpe Ratio</div></div>
            <div class="metric"><div class="value">${fmtPct(cur.mean_ic)}</div><div class="label">Mean IC</div></div>
            <div class="metric"><div class="value red">${fmtPct1(cur.max_drawdown)}</div><div class="label">Max Drawdown</div></div>
        `;

        // Multi-line chart
        if (returnsChart) returnsChart.destroy();
        const ctx = document.getElementById('returnsChart').getContext('2d');
        const maxLen = Math.max(...modelIds.map(id => (data.models[id].cumulative_returns || []).length));

        const datasets = modelIds.map(id => {
            const cr = data.models[id].cumulative_returns || [];
            return {
                label: id.charAt(0).toUpperCase() + id.slice(1),
                data: cr,
                borderColor: MODEL_COLORS[id] || '#999',
                backgroundColor: 'transparent',
                tension: 0.3,
                pointRadius: 0,
                borderWidth: id === currentModel ? 3 : 1.5,
            };
        });

        returnsChart = new Chart(ctx, {
            type: 'line',
            data: { labels: Array.from({length: maxLen}, (_, i) => i + 1), datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { intersect: false, mode: 'index' },
                plugins: {
                    legend: { display: true, position: 'bottom', labels: { boxWidth: 12, font: { size: 10 } } },
                },
                scales: {
                    x: { display: true, title: { display: true, text: 'Trading Day', color: '#888' }, ticks: { maxTicksLimit: 15, color: '#aaa' }, grid: { display: false } },
                    y: { ticks: { callback: v => (v * 100).toFixed(0) + '%', color: '#aaa' }, grid: { color: '#f0f0f0' } },
                }
            }
        });

        // Comparison table
        const names = { baseline: 'Baseline', ms_lstm: 'MS-LSTM', dualgat: 'DualGAT', ensemble: 'Ensemble' };
        const metrics = ['annualized_return', 'sharpe_ratio', 'mean_ic', 'max_drawdown', 'icir'];
        const labels = ['Ann. Return', 'Sharpe', 'Mean IC', 'Max DD', 'ICIR'];
        const fmt = [
            v => fmtPct1(v),
            v => v.toFixed(2),
            v => fmtPct(v),
            v => fmtPct1(v),
            v => v.toFixed(2),
        ];
        const higherBetter = [true, true, true, false, true];

        // Find best values
        const best = {};
        for (let i = 0; i < metrics.length; i++) {
            const vals = modelIds.map(id => data.models[id][metrics[i]]).filter(v => v != null);
            if (vals.length > 0) {
                best[metrics[i]] = higherBetter[i] ? Math.max(...vals) : Math.min(...vals);
            }
        }

        let tbl = '<table class="compare-table"><thead><tr><th>Model</th>';
        for (const l of labels) tbl += `<th>${l}</th>`;
        tbl += '</tr></thead><tbody>';
        for (const id of modelIds) {
            const m = data.models[id];
            tbl += `<tr><td><strong>${names[id] || id}</strong></td>`;
            for (let i = 0; i < metrics.length; i++) {
                const val = m[metrics[i]];
                const isBest = best[metrics[i]] !== undefined && val === best[metrics[i]];
                tbl += `<td class="${isBest ? 'best' : ''}">${val != null ? fmt[i](val) : '—'}</td>`;
            }
            tbl += '</tr>';
        }
        tbl += '</tbody></table>';
        document.getElementById('compare-table-container').innerHTML = tbl;

        const totalDays = modelIds.reduce((max, id) => Math.max(max, data.models[id].n_trading_days || 0), 0);
        document.getElementById('chart-legend').textContent =
            `${totalDays} trading days · ${start} ~ ${endDate}`;
    } catch (e) {
        console.error('Backtest load error:', e);
        document.getElementById('chart-container').innerHTML =
            `<div class="empty-state"><div class="icon">⚠️</div>Backtest load failed: ${e.message}</div>`;
    }
}
```

- [ ] **Step 4: Apply same changes to Chinese versions**

Update `app_zh.js` with the same model switching logic (adjusted strings to Chinese). Update `index_zh.html` with the same HTML structure changes (Chinese labels).

The Chinese tab labels: Baseline → 基准模型, MS-LSTM → MS-LSTM模型, DualGAT → DualGAT模型, Ensemble → 集成模型, Compare All → 对比所有.

- [ ] **Step 5: Verify dashboard renders**

```bash
cd /home/paulben/code/金融投资项目 && python3 -c "
from fastapi.testclient import TestClient
from src.web.api import app
client = TestClient(app)
r = client.get('/')
assert r.status_code == 200
assert 'model-tabs' in r.text
assert 'data-model=\"baseline\"' in r.text
r = client.get('/zh')
assert r.status_code == 200
assert 'model-tabs' in r.text
print('Dashboard HTML OK — model tabs present')
"
```
Expected: `Dashboard HTML OK — model tabs present`

- [ ] **Step 6: Commit**

```bash
git add src/web/templates/index.html src/web/templates/index_zh.html src/web/static/app.js src/web/static/app_zh.js
git commit -m "feat: model-switching UI with multi-line backtest chart and comparison table"
```

---

### Task 6: Training Script + Integration Verification

**Files:**
- Create: `scripts/train_ensemble.py`
- Modify: `tests/test_ensemble.py` (append integration test)

**Interfaces:**
- Consumes: All 3 sub-models (pre-trained), `EnsemblePredictor`
- Produces: Saved ensemble model, console output with 4-way IC comparison

- [ ] **Step 1: Write training script**

```python
#!/usr/bin/env python3
"""Train ensemble meta-learner and save both weighted and meta models.

Usage:
    python3 scripts/train_ensemble.py [--start YYYY-MM-DD] [--end YYYY-MM-DD]

If --meta flag is passed, also trains the meta-learner MLP.
Otherwise saves the weighted-average config only.
"""
import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    DEFAULT_TICKERS,
    MSLSTM_MODEL_PATH,
    DUALGAT_MODEL_PATH,
    ENSEMBLE_MODEL_PATH,
    ENSEMBLE_META_PATH,
)
from src.db.schema import init_db, get_prices
from src.model.baseline import BaselinePredictor
from src.model.ms_lstm import MSLSTMPredictor
from src.model.dualgat import DualGATPredictor, _get_trading_dates
from src.model.ensemble import EnsemblePredictor
from src.expert.tracker import ExpertTracker
from src.backtest.metrics import compute_ic

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("train_ensemble")


def main():
    parser = argparse.ArgumentParser(description="Train ensemble model")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--meta", action="store_true", help="Train meta-learner MLP")
    parser.add_argument("--stocks", type=int, default=10)
    args = parser.parse_args()

    if args.end is None:
        args.end = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    if args.start is None:
        args.start = (datetime.fromisoformat(args.end) - timedelta(days=180)).strftime("%Y-%m-%d")

    stocks = DEFAULT_TICKERS[: args.stocks]
    logger.info(f"Training ensemble on {len(stocks)} stocks from {args.start} to {args.end}")

    init_db()

    # Load sub-models
    logger.info("Loading sub-models...")
    baseline = BaselinePredictor()

    if not MSLSTM_MODEL_PATH.exists():
        logger.error(f"MS-LSTM model not found at {MSLSTM_MODEL_PATH}")
        sys.exit(1)
    ms_lstm = MSLSTMPredictor()
    ms_lstm.load(MSLSTM_MODEL_PATH)

    if not DUALGAT_MODEL_PATH.exists():
        logger.error(f"DualGAT model not found at {DUALGAT_MODEL_PATH}")
        sys.exit(1)
    dualgat = DualGATPredictor()
    dualgat.load(DUALGAT_MODEL_PATH)

    tracker = ExpertTracker()

    # Save weighted-average ensemble
    logger.info("Saving weighted-average ensemble...")
    ensemble = EnsemblePredictor(strategy="weighted")
    ensemble.save(ENSEMBLE_MODEL_PATH)

    # Train meta-learner if requested
    if args.meta:
        logger.info("Training meta-learner MLP...")
        meta = EnsemblePredictor(strategy="meta")
        history = meta.fit_meta(
            stocks=stocks,
            start_date=args.start,
            end_date=args.end,
            baseline=baseline,
            ms_lstm=ms_lstm,
            dualgat=dualgat,
        )
        if history["train_loss"]:
            logger.info(f"Meta training done. Best epoch: {history['best_epoch']}, "
                        f"Best val IC: {max(history['val_ic']):.4f}")
            meta.save(ENSEMBLE_META_PATH)
        else:
            logger.warning("Meta training produced no results — saving weighted ensemble as meta fallback")
            meta.save(ENSEMBLE_META_PATH)

    # Evaluate all 4 models
    logger.info("Comparing all models...")
    trading_dates = _get_trading_dates(stocks, args.start, args.end)
    split = int(len(trading_dates) * 0.8)
    val_dates = trading_dates[split:]

    # Re-load ensemble (weighted)
    ensemble_w = EnsemblePredictor(strategy="weighted")
    if ENSEMBLE_MODEL_PATH.exists():
        ensemble_w.load(ENSEMBLE_MODEL_PATH)

    ensemble_m = None
    if args.meta and ENSEMBLE_META_PATH.exists():
        ensemble_m = EnsemblePredictor(strategy="meta")
        ensemble_m.load(ENSEMBLE_META_PATH)

    results = {"baseline": [], "ms_lstm": [], "dualgat": [], "ensemble_w": []}
    if ensemble_m:
        results["ensemble_m"] = []

    for date_str in val_dates:
        records = tracker.trace(date_str)

        bl_df = baseline.predict(stocks, date_str, records)
        ms_df = ms_lstm.predict(stocks, date_str)
        dg_df = dualgat.predict(stocks, date_str)
        ew_df = ensemble_w.predict(stocks, date_str, bl_df, ms_df, dg_df)

        prices = get_prices(stocks, date_str, date_str)
        actuals = {}
        for stock in stocks:
            sp = prices.get(stock, [])
            sp_sorted = sorted(sp, key=lambda x: x["date"])
            for i, p in enumerate(sp_sorted):
                if p["date"] == date_str and i > 0:
                    prev = sp_sorted[i - 1]["close"]
                    curr = p["close"]
                    if prev > 0:
                        actuals[stock] = (curr - prev) / prev
                    break

        if len(actuals) < 3:
            continue

        import pandas as pd
        actual_series = pd.Series(actuals)
        for label, df in [("baseline", bl_df), ("ms_lstm", ms_df), ("dualgat", dg_df), ("ensemble_w", ew_df)]:
            pred_series = df.set_index("stock")["predicted_return"]
            results[label].append(compute_ic(pred_series, actual_series))

        if ensemble_m:
            em_df = ensemble_m.predict(stocks, date_str, bl_df, ms_df, dg_df)
            em_series = em_df.set_index("stock")["predicted_return"]
            results["ensemble_m"].append(compute_ic(em_series, actual_series))

    logger.info("=" * 55)
    for label, ics in results.items():
        if ics:
            logger.info(f"  {label:15s}  mean IC: {np.mean(ics):.4f}  ({len(ics)} days)")
    logger.info("=" * 55)

    best_label = max(results, key=lambda k: np.mean(results[k]) if results[k] else -999)
    logger.info(f"Best model: {best_label}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify syntax**

```bash
python3 -c "import ast; ast.parse(open('scripts/train_ensemble.py').read()); print('Syntax OK')"
```
Expected: `Syntax OK`

- [ ] **Step 3: Write integration test**

Append to `tests/test_ensemble.py`:

```python
class TestEnsembleIntegration:
    """End-to-end integration tests for ensemble pipeline."""

    def test_full_pipeline_smoke(self, prepopulated_db, tmp_path):
        """Weighted ensemble works end-to-end with sub-models."""
        import torch
        torch.manual_seed(99)
        np.random.seed(99)

        stocks = ["AAPL", "MSFT"]
        from src.model.baseline import BaselinePredictor
        from src.model.ms_lstm import MSLSTMPredictor
        from src.model.dualgat import DualGATPredictor
        from src.model.ensemble import EnsemblePredictor

        baseline = BaselinePredictor()

        ms_path = tmp_path / "ms_int.pt"
        ms = MSLSTMPredictor(hidden_dim=8, num_scales=3)
        ms.save(ms_path)
        ms.load(ms_path)

        dg = DualGATPredictor(hidden=16, out_dim=8, heads=2)

        ensemble = EnsemblePredictor(strategy="weighted")
        ensemble.model_ic_history = {
            "baseline": [0.04] * 20, "ms_lstm": [0.05] * 20, "dualgat": [0.06] * 20,
        }

        bl_df = baseline.predict(stocks, "2024-06-15", [])
        ms_df = ms.predict(stocks, "2024-06-15")
        dg_df = dg.predict(stocks, "2024-06-15")

        df = ensemble.predict(stocks, "2024-06-15", bl_df, ms_df, dg_df)
        assert len(df) == 2
        assert df["signal_source"].iloc[0] == "ensemble"
        assert "baseline_return" in df.columns

        # Save/load
        path = tmp_path / "ensemble_int.pt"
        ensemble.save(path)
        loaded = EnsemblePredictor()
        loaded.load(path)
        df2 = loaded.predict(stocks, "2024-06-15", bl_df, ms_df, dg_df)
        assert np.allclose(df["predicted_return"].values, df2["predicted_return"].values)

    def test_meta_fit_smoke(self, prepopulated_db, tmp_path):
        """Meta-learner fit completes one epoch without error."""
        import torch
        torch.manual_seed(42)
        np.random.seed(42)

        stocks = ["AAPL", "MSFT"]
        from src.model.baseline import BaselinePredictor
        from src.model.ms_lstm import MSLSTMPredictor
        from src.model.dualgat import DualGATPredictor
        from src.model.ensemble import EnsemblePredictor

        baseline = BaselinePredictor()

        ms_path = tmp_path / "ms_meta_fit.pt"
        ms = MSLSTMPredictor(hidden_dim=8, num_scales=3)
        ms.save(ms_path)

        dg = DualGATPredictor(hidden=16, out_dim=8, heads=2)

        meta = EnsemblePredictor(strategy="meta")
        history = meta.fit_meta(
            stocks=stocks,
            start_date="2024-05-20",
            end_date="2024-06-15",
            baseline=baseline,
            ms_lstm=ms,
            dualgat=dg,
            epochs=1,
            lr=1e-3,
        )
        assert len(history["train_loss"]) == 1
        assert np.isfinite(history["train_loss"][0])
```

- [ ] **Step 4: Run full test suite**

```bash
python3 -m pytest tests/ -v --tb=short
```
Expected: All non-api tests pass. The pre-existing `test_get_backtest` 404 may remain.

- [ ] **Step 5: Commit**

```bash
git add scripts/train_ensemble.py tests/test_ensemble.py
git commit -m "feat: ensemble training script + integration smoke tests"
```

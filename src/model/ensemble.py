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
        return df

    def update_ic_history(self, model_id: str, ic: float) -> None:
        """Record a new IC value for a sub-model."""
        if model_id in self.model_ic_history:
            self.model_ic_history[model_id].append(ic)
            if len(self.model_ic_history[model_id]) > ENSEMBLE_IC_WINDOW:
                self.model_ic_history[model_id] = \
                    self.model_ic_history[model_id][-ENSEMBLE_IC_WINDOW:]

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
        history: dict = {"train_loss": [], "val_ic": []}
        best_val_ic = -float("inf")
        best_state = None
        patience_counter = 0

        from datetime import datetime, timedelta
        from src.db import schema as db

        # Pre-fetch all prices once for the full window (+5 day buffer for returns)
        fetch_start = (datetime.fromisoformat(start_date) - timedelta(days=5)).strftime("%Y-%m-%d")
        all_prices_full = db.get_prices(stocks, fetch_start, end_date)

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

                # Compute targets using pre-fetched prices
                targets = []
                valid_stocks = []
                target_dt = datetime.fromisoformat(date_str)
                for s in stock_list:
                    sp = all_prices_full.get(s, [])
                    sp_sorted = sorted(sp, key=lambda p: p["date"])
                    for i, p in enumerate(sp_sorted):
                        if p["date"] == date_str and i > 0:
                            prev = float(sp_sorted[i - 1]["close"])
                            curr = float(p["close"])
                            if prev > 0:
                                targets.append((curr - prev) / prev)
                                valid_stocks.append(s)
                            break

                if len(targets) < 3:
                    continue

                # Build features aligned to valid_stocks
                bl_arr = np.array([bl_map[s] for s in valid_stocks], dtype=np.float32)
                ms_arr = np.array([ms_map[s] for s in valid_stocks], dtype=np.float32)
                dg_arr = np.array([dg_map[s] for s in valid_stocks], dtype=np.float32)
                x = np.stack([bl_arr, ms_arr, dg_arr], axis=1)

                x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
                y_t = torch.tensor(targets, dtype=torch.float32, device=self.device)

                optimizer.zero_grad()
                preds = self._meta(x_t)
                # IC loss: maximize Pearson correlation => minimize 1 - corr
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

                targets = []
                valid_stocks = []
                target_dt = datetime.fromisoformat(date_str)
                for s in stock_list:
                    sp = all_prices_full.get(s, [])
                    sp_sorted = sorted(sp, key=lambda p: p["date"])
                    for i, p in enumerate(sp_sorted):
                        if p["date"] == date_str and i > 0:
                            prev = float(sp_sorted[i - 1]["close"])
                            curr = float(p["close"])
                            if prev > 0:
                                targets.append((curr - prev) / prev)
                                valid_stocks.append(s)
                            break

                if len(targets) < 3:
                    continue

                bl_arr = np.array([bl_map[s] for s in valid_stocks], dtype=np.float32)
                ms_arr = np.array([ms_map[s] for s in valid_stocks], dtype=np.float32)
                dg_arr = np.array([dg_map[s] for s in valid_stocks], dtype=np.float32)
                x = np.stack([bl_arr, ms_arr, dg_arr], axis=1)
                x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
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
        # weights_only=False is required for dict checkpoints; safe for local trusted files
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
            logger.warning("Meta-learner not loaded, falling back to weighted average")
            return self._fuse_weighted(bl, ms, dg)
        self._meta.eval()
        x = np.stack([bl, ms, dg], axis=1)  # [N, 3]
        x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self._meta(x_t).cpu().numpy()

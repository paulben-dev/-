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

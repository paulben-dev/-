"""Multi-Scale LSTM model for stock return prediction.

Architecture (from DualGAT paper):
  5 LSTM branches at strides [1, 2, 4, 8, 16] process 30-day OHLCV windows.
  Last hidden states are mean-pooled, concatenated with expert features,
  and passed through a 2-layer MLP to produce scalar return predictions.

Trained with cross-sectional IC loss: 1 - Pearson_correlation(pred, actual).
"""
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from config import (
    MSLSTM_DROPOUT,
    MSLSTM_EARLY_STOP_PATIENCE,
    MSLSTM_EPOCHS,
    MSLSTM_HIDDEN_DIM,
    MSLSTM_LEARNING_RATE,
    MSLSTM_NUM_SCALES,
    MSLSTM_SEQUENCE_LENGTH,
    MSLSTM_WEIGHT_DECAY,
)
from src.db import schema as db

logger = logging.getLogger(__name__)


def ic_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-sectional Information Coefficient loss.

    IC = Pearson correlation(pred, actual) computed over stocks in one day.
    Loss = 1 - IC  (minimized when predictions are perfectly correlated with actuals).
    """
    vx = predictions - predictions.mean()
    vy = targets - targets.mean()
    numerator = (vx * vy).sum()
    denominator = torch.sqrt((vx ** 2).sum()) * torch.sqrt((vy ** 2).sum())
    corr = numerator / (denominator + 1e-8)
    return 1.0 - corr


class MSLSTMModel(nn.Module):
    """Multi-Scale LSTM for stock return prediction.

    Args:
        input_dim: Number of price features per time step (default 5: OHLCV).
        hidden_dim: Hidden size of each LSTM branch.
        num_scales: Number of LSTM branches at different strides.
        expert_feat_dim: Number of expert features (expert_available + expert_signal).
        dropout: Dropout rate applied in the MLP.
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = MSLSTM_HIDDEN_DIM,
        num_scales: int = MSLSTM_NUM_SCALES,
        expert_feat_dim: int = 2,
        dropout: float = MSLSTM_DROPOUT,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_scales = num_scales
        self.input_dim = input_dim
        self.expert_feat_dim = expert_feat_dim

        # Strides: 2^0, 2^1, 2^2, 2^3, 2^4 = 1, 2, 4, 8, 16
        self.strides = [2 ** i for i in range(num_scales)]

        # One LSTM per scale
        self.lstms = nn.ModuleList([
            nn.LSTM(input_dim, hidden_dim, batch_first=True)
            for _ in range(num_scales)
        ])

        # MLP: pooled hidden (hidden_dim) + expert features -> 32 -> 1
        mlp_input_dim = hidden_dim + expert_feat_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        price_features: torch.Tensor,
        expert_features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass producing scalar return predictions per stock.

        Args:
            price_features: [N_stocks, seq_len=30, input_dim] normalized OHLCV.
            expert_features: [N_stocks, expert_feat_dim] expert_available + signal.

        Returns:
            Tensor of shape [N_stocks] with predicted return ratios.
        """
        branch_outputs = []
        for lstm, stride in zip(self.lstms, self.strides):
            # Sample the sequence at the given stride
            sampled = price_features[:, ::stride, :]  # [N, seq_len//stride, input_dim]
            _, (h_n, _) = lstm(sampled)
            # h_n: [1, N, hidden_dim] -> squeeze to [N, hidden_dim]
            branch_outputs.append(h_n.squeeze(0))

        # Mean-pool across branches -> [N, hidden_dim]
        stacked = torch.stack(branch_outputs, dim=1)  # [N, num_scales, hidden_dim]
        pooled = stacked.mean(dim=1)  # [N, hidden_dim]

        # Concatenate expert features -> [N, hidden_dim + expert_feat_dim]
        combined = torch.cat([pooled, expert_features], dim=1)

        # MLP -> [N, 1] -> squeeze to [N]
        return self.mlp(combined).squeeze(-1)


class MSLSTMPredictor:
    """Training and prediction wrapper for MSLSTMModel.

    Compatible with BaselinePredictor interface: predict() returns
    a DataFrame with columns [stock, date, predicted_return, signal_source].
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = MSLSTM_HIDDEN_DIM,
        num_scales: int = MSLSTM_NUM_SCALES,
        expert_feat_dim: int = 2,
        dropout: float = MSLSTM_DROPOUT,
        device: str | None = None,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_scales = num_scales
        self.expert_feat_dim = expert_feat_dim
        self.dropout = dropout
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = MSLSTMModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_scales=num_scales,
            expert_feat_dim=expert_feat_dim,
            dropout=dropout,
        ).to(self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        stocks: list[str],
        start_date: str,
        end_date: str,
        epochs: int = MSLSTM_EPOCHS,
        lr: float = MSLSTM_LEARNING_RATE,
    ) -> dict:
        """Train the model on historical data.

        Args:
            stocks: List of ticker symbols.
            start_date: Training start date (YYYY-MM-DD).
            end_date: Training end date (YYYY-MM-DD).
            epochs: Maximum training epochs.
            lr: Learning rate for Adam optimizer.

        Returns:
            Dict with keys: train_loss (list), val_ic (list), best_epoch.
        """
        trading_dates = _get_trading_dates(stocks, start_date, end_date)
        if len(trading_dates) < 10:
            logger.warning(f"Only {len(trading_dates)} trading dates, need >= 10")
            return {"train_loss": [], "val_ic": [], "best_epoch": 0}

        # Split: first 80% train, last 20% validation
        split = int(len(trading_dates) * 0.8)
        train_dates = trading_dates[:split]
        val_dates = trading_dates[split:]

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=MSLSTM_WEIGHT_DECAY
        )

        history = {"train_loss": [], "val_ic": []}
        best_val_ic = -float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(epochs):
            # --- Training ---
            self.model.train()
            epoch_losses = []
            for date_str in train_dates:
                try:
                    price_t, expert_t, targets_t, _kept = _build_day_tensors(
                        stocks, date_str, self.device
                    )
                except _DataError:
                    continue

                if price_t is None or len(targets_t) < 3:
                    continue

                optimizer.zero_grad()
                predictions = self.model(price_t, expert_t)
                loss = ic_loss(predictions, targets_t)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            avg_train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            history["train_loss"].append(avg_train_loss)

            # --- Validation ---
            val_ic = self._evaluate_ic(val_dates, stocks)
            history["val_ic"].append(val_ic)

            logger.info(
                f"Epoch {epoch+1}/{epochs}: train_loss={avg_train_loss:.4f}, val_ic={val_ic:.4f}"
            )

            # Early stopping
            if val_ic > best_val_ic:
                best_val_ic = val_ic
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= MSLSTM_EARLY_STOP_PATIENCE:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        history["best_epoch"] = int(
            (np.argmax(history["val_ic"]) + 1) if history["val_ic"] else 0
        )
        return history

    def predict(
        self,
        stocks: list[str],
        date_str: str,
    ) -> pd.DataFrame:
        """Generate return predictions for all stocks on a given date.

        Compatible interface with BaselinePredictor.predict().
        """
        if not stocks:
            return self._empty_predictions([], date_str)

        self.model.eval()
        try:
            price_t, expert_t, _, kept_stocks = _build_day_tensors(
                stocks, date_str, self.device
            )
        except _DataError:
            return self._empty_predictions(stocks, date_str)

        if price_t is None:
            return self._empty_predictions(stocks, date_str)

        with torch.no_grad():
            preds = self.model(price_t, expert_t).cpu().numpy()

        df = pd.DataFrame({
            "stock": kept_stocks,
            "date": date_str,
            "predicted_return": preds,
            "signal_source": "ms_lstm",
        })

        # Normalize cross-sectionally like baseline
        std = df["predicted_return"].std()
        if std > 0:
            df["predicted_return"] = (df["predicted_return"] - df["predicted_return"].mean()) / std

        return df.sort_values("predicted_return", ascending=False)

    def save(self, path: str | Path) -> None:
        """Save model state to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "num_scales": self.num_scales,
                "expert_feat_dim": self.expert_feat_dim,
                "dropout": self.dropout,
            },
            path,
        )
        logger.info(f"Model saved to {path}")

    def load(self, path: str | Path) -> None:
        """Load model state from disk."""
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.input_dim = checkpoint["input_dim"]
        self.hidden_dim = checkpoint["hidden_dim"]
        self.num_scales = checkpoint["num_scales"]
        self.expert_feat_dim = checkpoint["expert_feat_dim"]
        self.dropout = checkpoint["dropout"]

        self.model = MSLSTMModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_scales=self.num_scales,
            expert_feat_dim=self.expert_feat_dim,
            dropout=self.dropout,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        logger.info(f"Model loaded from {path}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _empty_predictions(
        self, stocks: list[str], date_str: str
    ) -> pd.DataFrame:
        """Return zero-prediction DataFrame for the given stocks."""
        if not stocks:
            return pd.DataFrame(
                columns=["stock", "date", "predicted_return", "signal_source"]
            )
        return pd.DataFrame([
            {"stock": s, "date": date_str,
             "predicted_return": 0.0, "signal_source": "ms_lstm"}
            for s in stocks
        ])

    def _evaluate_ic(self, dates: list[str], stocks: list[str]) -> float:
        """Compute mean IC over a list of validation dates."""
        self.model.eval()
        ics = []
        for date_str in dates:
            try:
                price_t, expert_t, targets_t, _kept = _build_day_tensors(
                    stocks, date_str, self.device
                )
            except _DataError:
                continue

            if price_t is None or len(targets_t) < 3:
                continue

            with torch.no_grad():
                preds = self.model(price_t, expert_t)
                ic = 1.0 - ic_loss(preds, targets_t).item()
                ics.append(ic)

        return float(np.mean(ics)) if ics else 0.0


# ------------------------------------------------------------------
# Module-level data helpers
# ------------------------------------------------------------------

class _DataError(Exception):
    """Raised when data for a date is insufficient."""


def _get_trading_dates(
    stocks: list[str], start_date: str, end_date: str
) -> list[str]:
    """Return sorted list of trading dates that have price data for all stocks."""
    all_prices = db.get_prices(stocks, start_date, end_date)
    if not all_prices:
        return []

    # Find dates where ALL requested stocks have prices
    date_sets = []
    for stock in stocks:
        dates = set(p["date"] for p in all_prices.get(stock, []))
        date_sets.append(dates)

    common = date_sets[0]
    for ds in date_sets[1:]:
        common = common & ds

    return sorted(common)


def _build_day_tensors(
    stocks: list[str], date_str: str, device: str
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, list[str]]:
    """Build feature and target tensors for a single trading day.

    Args:
        stocks: List of ticker symbols.
        date_str: Target date (YYYY-MM-DD).
        device: Torch device string.

    Returns:
        (price_features, expert_features, targets) — each is a tensor or None.

    Raises:
        _DataError: When the date has insufficient data overall.
    """
    target_date = datetime.fromisoformat(date_str)
    window_start = (target_date - timedelta(days=MSLSTM_SEQUENCE_LENGTH + 10)).strftime(
        "%Y-%m-%d"
    )

    all_prices = db.get_prices(stocks, window_start, date_str)

    price_features_list = []
    targets_list = []
    kept_stocks = []

    for stock in stocks:
        try:
            feat = _build_stock_features(all_prices.get(stock, []), target_date)
            price_features_list.append(feat)

            # Target: return ratio for target date
            sp = all_prices.get(stock, [])
            target = _get_return_for_date(sp, date_str)
            targets_list.append(target)

            kept_stocks.append(stock)
        except _DataError:
            continue

    if len(kept_stocks) < 3:
        raise _DataError(f"Insufficient stocks with data for {date_str}")

    # Expert features
    expert_features_list = _build_expert_features(kept_stocks, date_str)

    if not price_features_list:
        return None, None, None, []

    price_t = torch.stack(price_features_list).to(device)
    expert_t = torch.tensor(expert_features_list, dtype=torch.float32, device=device)
    targets_t = torch.tensor(targets_list, dtype=torch.float32, device=device)

    return price_t, expert_t, targets_t, kept_stocks


def _build_stock_features(
    prices: list[dict], target_date: datetime
) -> torch.Tensor:
    """Build normalized OHLCV feature tensor for one stock.

    Extracts up to MSLSTM_SEQUENCE_LENGTH trading days before target_date,
    normalizes price columns by the first close, and volume by log.

    Returns:
        Tensor of shape [seq_len, 5] (OHLCV).

    Raises:
        _DataError: If fewer than 10 price records available.
    """
    # Filter to dates <= target_date
    prior = [p for p in prices if p["date"] <= target_date.strftime("%Y-%m-%d")]
    prior.sort(key=lambda x: x["date"])

    if len(prior) < 10:
        raise _DataError("Not enough price history")

    # Take last N days
    prior = prior[-MSLSTM_SEQUENCE_LENGTH:]

    rows = []
    for p in prior:
        rows.append([
            p["open"], p["high"], p["low"], p["close"],
            np.log1p(max(p["volume"], 1)),
        ])

    arr = np.array(rows, dtype=np.float32)

    # Normalize OHLC by first close price
    first_close = arr[0, 3]
    if first_close > 0:
        arr[:, :4] /= first_close

    # Pad to exactly MSLSTM_SEQUENCE_LENGTH time steps with zeros at the beginning.
    # This guarantees all stocks in a batch have uniform sequence length so
    # torch.stack() does not fail.
    if arr.shape[0] < MSLSTM_SEQUENCE_LENGTH:
        pad_count = MSLSTM_SEQUENCE_LENGTH - arr.shape[0]
        pad = np.zeros((pad_count, arr.shape[1]), dtype=np.float32)
        arr = np.concatenate([pad, arr], axis=0)

    return torch.tensor(arr, dtype=torch.float32)


def _get_return_for_date(prices: list[dict], date_str: str) -> float:
    """Compute actual return ratio for a stock on a given date.

    Return = (close_on_date - close_before_date) / close_before_date.

    Raises:
        _DataError: If price data is insufficient.
    """
    sorted_prices = sorted(prices, key=lambda x: x["date"])
    for i, p in enumerate(sorted_prices):
        if p["date"] == date_str and i > 0:
            prev_close = sorted_prices[i - 1]["close"]
            curr_close = p["close"]
            if prev_close > 0:
                return (curr_close - prev_close) / prev_close
    raise _DataError(f"No return data for {date_str}")


def _build_expert_features(
    stocks: list[str], date_str: str
) -> list[list[float]]:
    """Build expert feature vectors for each stock.

    Returns:
        List of [expert_available, expert_signal] for each stock.
    """
    from src.expert.tracker import ExpertTracker
    from src.model.signal import transform_expert_signal, compute_expert_availability

    tracker = ExpertTracker()
    records = tracker.trace(date_str)

    avail = compute_expert_availability(records, stocks)
    signals = transform_expert_signal(records, date_str)

    features = []
    for stock in stocks:
        features.append([
            float(avail.get(stock, 0)),
            float(signals.get(stock, 0.0)),
        ])
    return features

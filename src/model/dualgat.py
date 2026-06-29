"""Dual Graph Attention Network for stock return prediction.

Architecture (from DualGAT paper):
  Two graph structures (industry + correlation) with 2-hop GATConv
  and learnable dual-graph attention fusion. Trained with IC loss
  on top of frozen MS-LSTM features.
"""
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from pathlib import Path

from torch_geometric.nn import GATConv

from src.db import schema as db
from config import (
    CORR_WINDOW_DAYS,
    CORR_THRESHOLD_NORMAL,
    CORR_THRESHOLD_EXPERT,
    DUALGAT_IN_DIM,
    DUALGAT_HIDDEN_DIM,
    DUALGAT_OUT_DIM,
    DUALGAT_DROPOUT,
    DUALGAT_GAT_HEADS,
    DUALGAT_LEARNING_RATE,
    DUALGAT_WEIGHT_DECAY,
    DUALGAT_EPOCHS,
    DUALGAT_EARLY_STOP_PATIENCE,
    MSLSTM_MODEL_PATH,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Graph Builders
# ------------------------------------------------------------------

class IndustryGraphBuilder:
    """Build industry graph from GICS sector data.

    Two stocks are connected if they share the same GICS sector.
    Self-loops are always included.
    """

    def build(
        self,
        stocks: list[str],
        fundamentals: pd.DataFrame,
    ) -> torch.Tensor:
        """Build edge_index for industry graph.

        Args:
            stocks: Ordered list of stock tickers (determines node indices).
            fundamentals: DataFrame with columns [stock, sector].

        Returns:
            edge_index tensor of shape [2, num_edges].
        """
        n = len(stocks)
        sector_map = {}
        if not fundamentals.empty and "sector" in fundamentals.columns:
            for _, row in fundamentals.iterrows():
                sector_map[row["stock"]] = row.get("sector", "") or ""

        sources = []
        targets = []

        # Self-loops
        for i in range(n):
            sources.append(i)
            targets.append(i)

        # Cross-edges: same sector
        for i in range(n):
            for j in range(i + 1, n):
                si = sector_map.get(stocks[i], "")
                sj = sector_map.get(stocks[j], "")
                if si and sj and si == sj:
                    sources.append(i)
                    targets.append(j)
                    sources.append(j)
                    targets.append(i)

        return torch.tensor([sources, targets], dtype=torch.long)


class CorrelationGraphBuilder:
    """Build correlation graph from 30-day price data.

    Two stocks are connected if their Pearson correlation over
    the trailing window exceeds a threshold. Lower threshold (theta2)
    is used when either stock has an expert label.
    """

    def __init__(
        self,
        window: int = CORR_WINDOW_DAYS,
        theta1: float = CORR_THRESHOLD_NORMAL,
        theta2: float = CORR_THRESHOLD_EXPERT,
    ):
        self.window = window
        self.theta1 = theta1
        self.theta2 = theta2

    def build(
        self,
        stocks: list[str],
        date_str: str,
        expert_stocks: set[str],
    ) -> torch.Tensor:
        """Build edge_index for correlation graph.

        Args:
            stocks: Ordered list of stock tickers.
            date_str: Target date (YYYY-MM-DD).
            expert_stocks: Set of stock tickers that have expert coverage.

        Returns:
            edge_index tensor of shape [2, num_edges].
        """
        n = len(stocks)
        target_date = datetime.fromisoformat(date_str)
        window_start = (target_date - timedelta(days=self.window + 10)).strftime(
            "%Y-%m-%d"
        )

        # Fetch price data
        all_prices = db.get_prices(stocks, window_start, date_str)

        # Build close price matrix [n, window]
        price_matrix = np.full((n, self.window), np.nan)
        for i, stock in enumerate(stocks):
            sp = all_prices.get(stock, [])
            closes = [p["close"] for p in sp if p["date"] <= date_str]
            closes = closes[-self.window:]
            for j, c in enumerate(closes):
                price_matrix[i, j] = c

        # Compute returns and correlation
        returns = np.diff(price_matrix, axis=1) / price_matrix[:, :-1]
        # Replace NaN/inf with 0
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

        # Correlation matrix
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.corrcoef(returns)
        corr = np.nan_to_num(corr, nan=0.0)

        # Build edges
        sources = []
        targets = []

        # Self-loops
        for i in range(n):
            sources.append(i)
            targets.append(i)

        # Cross-edges based on threshold
        for i in range(n):
            for j in range(i + 1, n):
                threshold = self.theta1
                if stocks[i] in expert_stocks or stocks[j] in expert_stocks:
                    threshold = self.theta2
                if abs(corr[i, j]) > threshold:
                    sources.append(i)
                    targets.append(j)
                    sources.append(j)
                    targets.append(i)

        return torch.tensor([sources, targets], dtype=torch.long)


# ------------------------------------------------------------------
# DualGAT Model
# ------------------------------------------------------------------


class DualGATFusion(nn.Module):
    """Learnable dual-graph attentive fusion layer.

    Computes per-node scalar scores for each graph, softmax-normalizes
    into beta weights, and returns a weighted combination of the two
    graph representations.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.q_ind = nn.Parameter(torch.randn(hidden_dim))
        self.q_cor = nn.Parameter(torch.randn(hidden_dim))

    def forward(
        self,
        h_ind: torch.Tensor,
        h_cor: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse two graph representations with learned weights.

        Args:
            h_ind: [N, d] industry graph node embeddings.
            h_cor: [N, d] correlation graph node embeddings.

        Returns:
            h_fused: [N, d] weighted combination.
        """
        # Compute per-node scores
        score_ind = h_ind @ self.q_ind  # [N]
        score_cor = h_cor @ self.q_cor  # [N]
        scores = torch.stack([score_ind, score_cor], dim=1)  # [N, 2]
        beta = torch.softmax(scores, dim=1)  # [N, 2]

        # Weighted fusion
        h_fused = beta[:, 0:1] * h_ind + beta[:, 1:2] * h_cor
        return h_fused


class DualGATModel(nn.Module):
    """2-hop Dual Graph Attention Network.

    Architecture:
      Hop 1: GATConv on each graph -> DualGATFusion
      Hop 2: GATConv on each graph -> DualGATFusion
      MLP: [out_dim -> 1] scalar prediction

    Args:
        in_dim: Input feature dimension per node (default 3).
        hidden: Hidden dimension for GAT layers.
        out_dim: Output dimension after hop 2.
        heads: Number of attention heads.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_dim: int = DUALGAT_IN_DIM,
        hidden: int = DUALGAT_HIDDEN_DIM,
        out_dim: int = DUALGAT_OUT_DIM,
        heads: int = DUALGAT_GAT_HEADS,
        dropout: float = DUALGAT_DROPOUT,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden = hidden
        self.out_dim = out_dim
        self.heads = heads

        # Hop 1: GATConv per graph
        # heads * (hidden // heads) = hidden -> per-head dim = hidden // heads
        per_head_1 = hidden // heads
        self.gat_ind_1 = GATConv(in_dim, per_head_1, heads=heads, dropout=dropout)
        self.gat_cor_1 = GATConv(in_dim, per_head_1, heads=heads, dropout=dropout)
        self.fusion_1 = DualGATFusion(hidden)

        # Hop 2: GATConv on fused features
        per_head_2 = out_dim // heads
        self.gat_ind_2 = GATConv(hidden, per_head_2, heads=heads, dropout=dropout)
        self.gat_cor_2 = GATConv(hidden, per_head_2, heads=heads, dropout=dropout)
        self.fusion_2 = DualGATFusion(out_dim)

        # Output MLP
        self.mlp = nn.Linear(out_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index_ind: torch.Tensor,
        edge_index_cor: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features [N, in_dim].
            edge_index_ind: Industry graph edges [2, E_ind].
            edge_index_cor: Correlation graph edges [2, E_cor].

        Returns:
            Predicted returns [N].
        """
        # Hop 1
        h_ind_1 = self.gat_ind_1(x, edge_index_ind)  # [N, hidden]
        h_cor_1 = self.gat_cor_1(x, edge_index_cor)  # [N, hidden]
        h_fused_1 = self.fusion_1(h_ind_1, h_cor_1)  # [N, hidden]

        # Hop 2
        h_ind_2 = self.gat_ind_2(h_fused_1, edge_index_ind)  # [N, out_dim]
        h_cor_2 = self.gat_cor_2(h_fused_1, edge_index_cor)  # [N, out_dim]
        h_fused_2 = self.fusion_2(h_ind_2, h_cor_2)  # [N, out_dim]

        # Output
        return self.mlp(h_fused_2).squeeze(-1)  # [N]


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

class _DataError(Exception):
    """Raised when data for a date is insufficient."""


def _ic_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-sectional IC loss: 1 - Pearson correlation."""
    vx = predictions - predictions.mean()
    vy = targets - targets.mean()
    numerator = (vx * vy).sum()
    denominator = torch.sqrt((vx ** 2).sum()) * torch.sqrt((vy ** 2).sum())
    corr = numerator / (denominator + 1e-8)
    return 1.0 - corr


def _get_trading_dates(stocks: list[str], start_date: str, end_date: str) -> list[str]:
    """Return sorted trading dates with price data for all stocks."""
    all_prices = db.get_prices(stocks, start_date, end_date)
    if not all_prices:
        return []
    date_sets = [set(p["date"] for p in all_prices.get(s, [])) for s in stocks]
    common = date_sets[0]
    for ds in date_sets[1:]:
        common = common & ds
    return sorted(common)


def _build_input_features(
    stocks: list[str], date_str: str, device: str,
    ms_lstm=None, expert_records=None,
) -> tuple[torch.Tensor, list[str]]:
    """Build 3-dim input features for DualGAT.

    Features: [MS-LSTM prediction, expert_available, expert_signal]
    All three default to 0 when data is missing.

    Args:
        stocks: Ordered list of stock tickers.
        date_str: Target date (YYYY-MM-DD).
        device: Torch device string.
        ms_lstm: Optional MSLSTMPredictor for populating feature 0.
        expert_records: Optional pre-fetched ExpertRecord list (avoids
            redundant ExpertTracker.trace() calls when the caller has
            already fetched them).

    Returns:
        (features_tensor [N, 3], kept_stocks)

    Raises:
        _DataError: If fewer than 3 stocks have valid data.
    """
    from src.model.signal import transform_expert_signal, compute_expert_availability

    # Expert features — use pre-fetched records if provided (H1 fix)
    if expert_records is not None:
        records = expert_records
    else:
        from src.expert.tracker import ExpertTracker
        tracker = ExpertTracker()
        records = tracker.trace(date_str)

    avail = compute_expert_availability(records, stocks)
    signals = transform_expert_signal(records, date_str)

    # MS-LSTM predictions (CRITICAL fix: use real predictions instead of 0.0)
    ms_pred_map = {}
    if ms_lstm is not None:
        try:
            ms_preds = ms_lstm.predict(stocks, date_str)
            ms_pred_map = dict(zip(ms_preds["stock"], ms_preds["predicted_return"]))
        except Exception:
            pass  # Fall back to zeros

    features = []
    kept = []

    for stock in stocks:
        ms_val = ms_pred_map.get(stock, 0.0)
        features.append([
            ms_val,
            float(avail.get(stock, 0)),
            float(signals.get(stock, 0.0)),
        ])
        kept.append(stock)

    if len(kept) < 3:
        raise _DataError(f"Insufficient stocks for {date_str}")

    return torch.tensor(features, dtype=torch.float32, device=device), kept


def _get_return_for_date(prices: list[dict], date_str: str) -> float:
    """Compute actual return ratio for a stock on a given date."""
    sorted_prices = sorted(prices, key=lambda x: x["date"])
    for i, p in enumerate(sorted_prices):
        if p["date"] == date_str and i > 0:
            prev_close = sorted_prices[i - 1]["close"]
            curr_close = p["close"]
            if prev_close > 0:
                return (curr_close - prev_close) / prev_close
    raise _DataError(f"No return data for {date_str}")


def _build_day_tensors_dualgat(
    stocks: list[str],
    date_str: str,
    ms_lstm,
    corr_builder: CorrelationGraphBuilder,
    device: str,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Build feature tensors, targets, and correlation graph for one day.

    Returns:
        (x, targets, edge_cor) — each is a tensor or None.

    Raises:
        _DataError: When data is insufficient.
    """
    from config import MSLSTM_SEQUENCE_LENGTH
    from src.expert.tracker import ExpertTracker
    from src.model.signal import transform_expert_signal, compute_expert_availability

    target_date = datetime.fromisoformat(date_str)
    window_start = (target_date - timedelta(days=MSLSTM_SEQUENCE_LENGTH + 10)).strftime("%Y-%m-%d")
    all_prices = db.get_prices(stocks, window_start, date_str)

    # Build input features and targets
    tracker = ExpertTracker()
    records = tracker.trace(date_str)
    avail = compute_expert_availability(records, stocks)
    signals = transform_expert_signal(records, date_str)

    # Get MS-LSTM predictions for this date
    ms_preds = ms_lstm.predict(stocks, date_str)
    ms_pred_map = dict(zip(ms_preds["stock"], ms_preds["predicted_return"]))

    features = []
    targets_list = []
    kept_stocks = []

    for stock in stocks:
        # MS-LSTM prediction
        ms_val = ms_pred_map.get(stock, 0.0)
        # Expert features
        exp_avail = float(avail.get(stock, 0))
        exp_sig = float(signals.get(stock, 0.0))

        features.append([ms_val, exp_avail, exp_sig])
        kept_stocks.append(stock)

        # Target: actual return for this date
        sp = all_prices.get(stock, [])
        target_val = _get_return_for_date(sp, date_str)
        targets_list.append(target_val)

    if len(kept_stocks) < 3:
        raise _DataError(f"Insufficient stocks for {date_str}")

    # Build correlation graph
    expert_stocks = set(r.stock for r in records if r.expert_type != "none")
    edge_cor = corr_builder.build(stocks, date_str, expert_stocks).to(device)

    x = torch.tensor(features, dtype=torch.float32, device=device)
    targets = torch.tensor(targets_list, dtype=torch.float32, device=device)

    return x, targets, edge_cor


def _empty_predictions(stocks: list[str], date_str: str, source: str) -> pd.DataFrame:
    """Return zero-prediction fallback DataFrame."""
    return pd.DataFrame([
        {"stock": s, "date": date_str, "predicted_return": 0.0, "signal_source": source}
        for s in stocks
    ])


# ------------------------------------------------------------------
# DualGAT Predictor
# ------------------------------------------------------------------

class DualGATPredictor:
    """Training and prediction wrapper for DualGATModel.

    Compatible with BaselinePredictor and MSLSTMPredictor interface.
    Uses frozen MS-LSTM for feature extraction.
    """

    def __init__(
        self,
        in_dim: int = DUALGAT_IN_DIM,
        hidden: int = DUALGAT_HIDDEN_DIM,
        out_dim: int = DUALGAT_OUT_DIM,
        heads: int = DUALGAT_GAT_HEADS,
        dropout: float = DUALGAT_DROPOUT,
        device: str | None = None,
    ):
        self.in_dim = in_dim
        self.hidden = hidden
        self.out_dim = out_dim
        self.heads = heads
        self.dropout = dropout
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DualGATModel(
            in_dim=in_dim,
            hidden=hidden,
            out_dim=out_dim,
            heads=heads,
            dropout=dropout,
        ).to(self.device)

        self._ind_builder = IndustryGraphBuilder()
        self._corr_builder = CorrelationGraphBuilder()
        self._edge_index_ind: torch.Tensor | None = None
        self._ms_lstm = None  # Cached MS-LSTM for inference (set by fit())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        stocks: list[str],
        start_date: str,
        end_date: str,
        ms_lstm_path: str,
        epochs: int = DUALGAT_EPOCHS,
        lr: float = DUALGAT_LEARNING_RATE,
    ) -> dict:
        """Train DualGAT on historical data.

        Args:
            stocks: List of ticker symbols.
            start_date / end_date: Training date range (YYYY-MM-DD).
            ms_lstm_path: Path to pre-trained MS-LSTM model.
            epochs: Maximum training epochs.
            lr: Learning rate.

        Returns:
            Dict with keys: train_loss, val_ic, best_epoch.
        """
        # Load frozen MS-LSTM
        from src.model.ms_lstm import MSLSTMPredictor
        ms_lstm = MSLSTMPredictor()
        ms_lstm.load(ms_lstm_path)
        ms_lstm.model.eval()
        self._ms_lstm = ms_lstm  # Cache for inference (CRITICAL fix)

        # Build static industry graph
        from src.data.yfinance import YFinanceCollector
        yf = YFinanceCollector()
        fundamentals = yf.collect_fundamentals(stocks)
        self._edge_index_ind = self._ind_builder.build(stocks, fundamentals).to(self.device)

        # Get trading dates
        trading_dates = _get_trading_dates(stocks, start_date, end_date)
        if len(trading_dates) < 10:
            logger.warning(f"Only {len(trading_dates)} trading dates, need >= 10")
            return {"train_loss": [], "val_ic": [], "best_epoch": 0}

        split = int(len(trading_dates) * 0.8)
        train_dates = trading_dates[:split]
        val_dates = trading_dates[split:]

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=DUALGAT_WEIGHT_DECAY
        )

        history = {"train_loss": [], "val_ic": []}
        best_val_ic = -float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            self.model.train()
            epoch_losses = []
            for date_str in train_dates:
                try:
                    x, targets, edge_cor = _build_day_tensors_dualgat(
                        stocks, date_str, ms_lstm, self._corr_builder, self.device
                    )
                except _DataError:
                    continue

                if x is None or len(targets) < 3:
                    continue

                optimizer.zero_grad()
                predictions = self.model(x, self._edge_index_ind, edge_cor)
                loss = _ic_loss(predictions, targets)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            history["train_loss"].append(avg_loss)

            # Validation
            val_ic = self._evaluate_ic(val_dates, stocks, ms_lstm)
            history["val_ic"].append(val_ic)

            logger.info(f"Epoch {epoch+1}/{epochs}: train_loss={avg_loss:.4f}, val_ic={val_ic:.4f}")

            if val_ic > best_val_ic:
                best_val_ic = val_ic
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= DUALGAT_EARLY_STOP_PATIENCE:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        history["best_epoch"] = int(np.argmax(history["val_ic"]) + 1) if history["val_ic"] else 0
        return history

    def predict(self, stocks: list[str], date_str: str,
                ms_lstm_path: str | None = None) -> pd.DataFrame:
        """Generate return predictions for all stocks on a given date.

        Args:
            stocks: List of ticker symbols.
            date_str: Target date (YYYY-MM-DD).
            ms_lstm_path: Optional path to a pre-trained MS-LSTM model.
                Overrides the model stored during fit() and the default
                MSLSTM_MODEL_PATH config. The MS-LSTM is used to populate
                input feature 0 (CRITICAL fix: eliminates train/inference skew).

        Returns:
            pd.DataFrame with columns [stock, date, predicted_return,
            signal_source].
        """
        if not stocks:
            return pd.DataFrame(columns=["stock", "date", "predicted_return", "signal_source"])

        self.model.eval()

        # --- Resolve MS-LSTM model (CRITICAL fix) ---
        ms_lstm = None
        if ms_lstm_path is not None:
            from src.model.ms_lstm import MSLSTMPredictor
            ms_lstm = MSLSTMPredictor()
            ms_lstm.load(ms_lstm_path)
            ms_lstm.model.eval()
        elif self._ms_lstm is not None:
            ms_lstm = self._ms_lstm
        else:
            # Fallback: try default path from config
            try:
                from src.model.ms_lstm import MSLSTMPredictor
                ms_lstm = MSLSTMPredictor()
                ms_lstm.load(MSLSTM_MODEL_PATH)
                ms_lstm.model.eval()
                logger.info("Loaded MS-LSTM from %s for inference", MSLSTM_MODEL_PATH)
            except Exception:
                logger.warning(
                    "predict() called before fit() and no MS-LSTM model available. "
                    "Feature 0 (MS-LSTM prediction) will be zero-filled. "
                    "Predictions may be unreliable. Call fit() first, or provide "
                    "ms_lstm_path to load a pre-trained MS-LSTM."
                )

        # --- Fetch expert data once (H1 fix) ---
        try:
            from src.expert.tracker import ExpertTracker
            tracker = ExpertTracker()
            records = tracker.trace(date_str)
            expert_stocks = set(r.stock for r in records if r.expert_type != "none")
        except Exception:
            records = None
            expert_stocks = set()

        # --- Build correlation graph ---
        edge_cor = self._corr_builder.build(stocks, date_str, expert_stocks).to(self.device)

        # --- Build input features (CRITICAL + H1 fix: pass ms_lstm and records) ---
        try:
            x, kept_stocks = _build_input_features(
                stocks, date_str, self.device,
                ms_lstm=ms_lstm, expert_records=records,
            )
        except _DataError:
            return _empty_predictions(stocks, date_str, "dualgat")

        # --- H3 fix: warn when model not fitted ---
        if self._edge_index_ind is None or self._edge_index_ind.shape[1] == 0:
            logger.warning(
                "Industry graph not built. Call fit() before predict() for "
                "meaningful predictions. Returning zero-filled fallback."
            )
            return _empty_predictions(stocks, date_str, "dualgat")

        with torch.no_grad():
            preds = self.model(x, self._edge_index_ind, edge_cor).cpu().numpy()

        df = pd.DataFrame({
            "stock": kept_stocks,
            "date": date_str,
            "predicted_return": preds,
            "signal_source": "dualgat",
        })

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
                "in_dim": self.in_dim,
                "hidden": self.hidden,
                "out_dim": self.out_dim,
                "heads": self.heads,
                "dropout": self.dropout,
                "edge_index_ind": self._edge_index_ind.cpu() if self._edge_index_ind is not None else None,
            },
            path,
        )
        logger.info(f"Model saved to {path}")

    def load(self, path: str | Path) -> None:
        """Load model state from disk."""
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.in_dim = checkpoint["in_dim"]
        self.hidden = checkpoint["hidden"]
        self.out_dim = checkpoint["out_dim"]
        self.heads = checkpoint["heads"]
        self.dropout = checkpoint["dropout"]

        self.model = DualGATModel(
            in_dim=self.in_dim,
            hidden=self.hidden,
            out_dim=self.out_dim,
            heads=self.heads,
            dropout=self.dropout,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        if checkpoint.get("edge_index_ind") is not None:
            self._edge_index_ind = checkpoint["edge_index_ind"].to(self.device)

        logger.info(f"Model loaded from {path}")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evaluate_ic(self, dates: list[str], stocks: list[str], ms_lstm) -> float:
        """Compute mean IC over validation dates."""
        self.model.eval()
        ics = []
        for date_str in dates:
            try:
                x, targets, edge_cor = _build_day_tensors_dualgat(
                    stocks, date_str, ms_lstm, self._corr_builder, self.device
                )
            except _DataError:
                continue

            if x is None or len(targets) < 3:
                continue

            with torch.no_grad():
                preds = self.model(x, self._edge_index_ind, edge_cor)
                ic = 1.0 - _ic_loss(preds, targets).item()
                ics.append(ic)

        return float(np.mean(ics)) if ics else 0.0

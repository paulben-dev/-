"""Dual Graph Attention Network for stock return prediction.

Architecture (from DualGAT paper):
  Two graph structures (industry + correlation) with 2-hop GATConv
  and learnable dual-graph attention fusion. Trained with IC loss
  on top of frozen MS-LSTM features.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime, timedelta

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
)


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

"""Dual Graph Attention Network for stock return prediction.

Architecture (from DualGAT paper):
  Two graph structures (industry + correlation) with 2-hop GATConv
  and learnable dual-graph attention fusion. Trained with IC loss
  on top of frozen MS-LSTM features.
"""
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta

from src.db import schema as db
from config import (
    CORR_WINDOW_DAYS,
    CORR_THRESHOLD_NORMAL,
    CORR_THRESHOLD_EXPERT,
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

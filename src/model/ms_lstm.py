"""Multi-Scale LSTM model for stock return prediction.

Architecture (from DualGAT paper):
  5 LSTM branches at strides [1, 2, 4, 8, 16] process 30-day OHLCV windows.
  Last hidden states are mean-pooled, concatenated with expert features,
  and passed through a 2-layer MLP to produce scalar return predictions.

Trained with cross-sectional IC loss: 1 - Pearson_correlation(pred, actual).
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime, timedelta

from src.data.models import ExpertRecord
from src.db import schema as db
from src.model.signal import transform_expert_signal, compute_expert_availability
from config import (
    MSLSTM_HIDDEN_DIM,
    MSLSTM_NUM_SCALES,
    MSLSTM_DROPOUT,
    MSLSTM_LEARNING_RATE,
    MSLSTM_WEIGHT_DECAY,
    MSLSTM_EPOCHS,
    MSLSTM_EARLY_STOP_PATIENCE,
    MSLSTM_SEQUENCE_LENGTH,
    MSLSTM_MODEL_PATH,
)

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

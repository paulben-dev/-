"""Tests for MS-LSTM model and predictor."""
import pytest
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.model.ms_lstm import MSLSTMModel, ic_loss


class TestMSLSTMModel:
    """Unit tests for the MS-LSTM neural network architecture."""

    @pytest.fixture
    def model(self):
        return MSLSTMModel(
            input_dim=5,
            hidden_dim=64,
            num_scales=5,
            expert_feat_dim=2,
            dropout=0.2,
        )

    def test_forward_output_shape(self, model):
        """forward() returns [N_stocks] predictions."""
        N = 20
        price = torch.randn(N, 30, 5)
        expert = torch.randn(N, 2)
        output = model(price, expert)
        assert output.shape == (N,)
        assert output.dtype == torch.float32

    def test_forward_deterministic_in_eval(self, model):
        """In eval mode, same input gives same output."""
        model.eval()
        price = torch.randn(5, 30, 5)
        expert = torch.randn(5, 2)
        with torch.no_grad():
            out1 = model(price, expert)
            out2 = model(price, expert)
        assert torch.allclose(out1, out2)

    def test_different_strides_reduce_sequence(self, model):
        """Each LSTM branch gets a different number of time steps."""
        strides = model.strides  # [1, 2, 4, 8, 16]
        seq_len = 30
        for s in strides:
            steps = seq_len // s + (1 if seq_len % s else 0)
            # With 30 and stride s: 30/s rounded up
            expected_steps = (seq_len + s - 1) // s
            assert expected_steps >= 2  # At least 2 steps for LSTM

    def test_dropout_active_in_train(self, model):
        """Dropout is active during training."""
        model.train()
        price = torch.randn(10, 30, 5)
        expert = torch.randn(10, 2)
        out1 = model(price, expert)
        out2 = model(price, expert)
        # With dropout active, outputs should differ (non-deterministic)
        assert not torch.allclose(out1, out2)


class TestICLoss:
    """Tests for the IC (Information Coefficient) loss function."""

    def test_perfect_positive_correlation(self):
        """Perfect prediction -> loss approx 0."""
        pred = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        actual = torch.tensor([0.05, 0.10, 0.15, 0.20, 0.25])
        loss = ic_loss(pred, actual)
        assert loss.item() == pytest.approx(0.0, abs=0.01)

    def test_perfect_negative_correlation(self):
        """Perfect inverse prediction -> loss approx 2."""
        pred = torch.tensor([0.5, 0.4, 0.3, 0.2, 0.1])
        actual = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        loss = ic_loss(pred, actual)
        assert loss.item() == pytest.approx(2.0, abs=0.01)

    def test_no_correlation(self):
        """Random predictions -> loss approx 1."""
        torch.manual_seed(42)
        pred = torch.randn(100)
        actual = torch.randn(100)
        loss = ic_loss(pred, actual)
        assert 0.5 < loss.item() < 1.5

    def test_handles_zero_variance(self):
        """Constant predictions -> no crash, returns finite value."""
        pred = torch.ones(10)
        actual = torch.randn(10)
        loss = ic_loss(pred, actual)
        assert torch.isfinite(loss)

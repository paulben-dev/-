"""Tests for MS-LSTM model and predictor."""
import numpy as np
import pandas as pd
import pytest
import torch

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
        """Each LSTM branch receives a correctly-stride-sampled input sequence."""
        seq_len = 30
        N = 4
        price = torch.randn(N, seq_len, model.input_dim)
        expert = torch.randn(N, model.expert_feat_dim)

        captured_steps = {}
        hooks = []

        def make_hook(idx):
            def hook(module, input_, output_):
                captured_steps[idx] = input_[0].shape[1]  # sequence length
            return hook

        for i, lstm in enumerate(model.lstms):
            hooks.append(lstm.register_forward_hook(make_hook(i)))

        try:
            _ = model(price, expert)
        finally:
            for h in hooks:
                h.remove()

        for i, stride in enumerate(model.strides):
            expected_steps = (seq_len + stride - 1) // stride  # ceil(seq_len / stride)
            assert captured_steps[i] == expected_steps, (
                f"Branch {i} (stride={stride}) got {captured_steps[i]} steps, "
                f"expected {expected_steps}"
            )

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


class TestMSLSTMPredictor:
    """Tests for the MSLSTMPredictor training/prediction wrapper."""

    @pytest.fixture
    def predictor(self):
        from src.model.ms_lstm import MSLSTMPredictor
        return MSLSTMPredictor(hidden_dim=16, num_scales=3)

    @pytest.fixture
    def seeded_predictor(self):
        """Predictor with fixed seed for reproducibility."""
        torch.manual_seed(42)
        np.random.seed(42)
        from src.model.ms_lstm import MSLSTMPredictor
        return MSLSTMPredictor(hidden_dim=16, num_scales=3)

    def test_predict_returns_dataframe(self, predictor, prepopulated_db):
        """predict() returns DataFrame matching BaselinePredictor format."""
        stocks = ["AAPL", "MSFT"]
        df = predictor.predict(stocks, "2024-06-15")
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["stock", "date", "predicted_return", "signal_source"]
        assert len(df) == 2
        assert df["signal_source"].iloc[0] == "ms_lstm"

    def test_predict_empty_stocks(self, predictor):
        """predict() handles empty stock list."""
        df = predictor.predict([], "2024-06-15")
        assert len(df) == 0

    def test_save_and_load_roundtrip(self, predictor, tmp_path):
        """Model survives save->load roundtrip and produces identical predictions."""
        path = tmp_path / "test_model.pt"
        predictor.save(path)
        assert path.exists()

        # Load into a new predictor
        from src.model.ms_lstm import MSLSTMPredictor
        predictor2 = MSLSTMPredictor()
        predictor2.load(path)

        # Both should produce same predictions
        price = torch.randn(5, 30, 5)
        expert = torch.randn(5, 2)
        predictor.model.eval()
        predictor2.model.eval()
        with torch.no_grad():
            out1 = predictor.model(price, expert)
            out2 = predictor2.model(price, expert)
        assert torch.allclose(out1, out2)

    def test_fit_runs_one_epoch(self, seeded_predictor, prepopulated_db):
        """fit() completes one epoch on tiny dataset without error."""
        stocks = ["AAPL", "MSFT"]
        history = seeded_predictor.fit(
            stocks=stocks,
            start_date="2024-05-15",
            end_date="2024-06-15",
            epochs=1,
            lr=1e-3,
        )
        assert "train_loss" in history
        assert len(history["train_loss"]) == 1
        assert isinstance(history["train_loss"][0], float)

    def test_predict_after_fit(self, seeded_predictor, prepopulated_db):
        """predict() works after a minimal fit."""
        stocks = ["AAPL", "MSFT"]
        seeded_predictor.fit(
            stocks=stocks,
            start_date="2024-05-15",
            end_date="2024-06-15",
            epochs=1,
            lr=1e-3,
        )
        df = seeded_predictor.predict(stocks, "2024-06-15")
        assert len(df) == 2
        assert df["signal_source"].iloc[0] == "ms_lstm"

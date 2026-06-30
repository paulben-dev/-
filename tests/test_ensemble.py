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
        """Meta-learner survives save->load roundtrip."""
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

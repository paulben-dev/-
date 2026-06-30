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

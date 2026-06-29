"""Tests for baseline predictor."""
import pytest
from datetime import datetime
from src.model.baseline import BaselinePredictor
from src.data.models import ExpertRecord


def test_predict_returns_dataframe(prepopulated_db):
    predictor = BaselinePredictor()
    df = predictor.predict(["AAPL", "MSFT", "GOOGL"], "2024-06-15")
    assert len(df) == 3
    assert list(df.columns) == ["stock", "date", "predicted_return", "signal_source"]
    assert sorted(df["stock"].tolist()) == sorted(["AAPL", "MSFT", "GOOGL"])


def test_predict_with_experts(prepopulated_db):
    """Test that expert signals are used when available."""
    predictor = BaselinePredictor()
    records = [
        ExpertRecord("u1", "AAPL", datetime(2024, 6, 15), 0.85, 0.70, "expert", "Bullish"),
    ]
    df = predictor.predict(["AAPL", "MSFT"], "2024-06-15", records)
    aapl = df[df["stock"] == "AAPL"].iloc[0]
    assert aapl["signal_source"] == "expert"
    msft = df[df["stock"] == "MSFT"].iloc[0]
    assert msft["signal_source"] == "momentum"


def test_predict_empty_stocks():
    predictor = BaselinePredictor()
    df = predictor.predict([], "2024-06-15")
    assert len(df) == 0


def test_predictions_are_normalized(prepopulated_db):
    predictor = BaselinePredictor()
    df = predictor.predict(["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"], "2024-06-15")
    assert abs(df["predicted_return"].mean()) < 1e-10
    assert abs(df["predicted_return"].std() - 1.0) < 0.1

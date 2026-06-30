"""End-to-end integration test for the full pipeline."""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys

from src.db import schema as db
from src.db.schema import init_db, insert_prices, insert_posts
from src.data.models import Price, Post
from src.expert.tracker import ExpertTracker
from src.model.baseline import BaselinePredictor
from src.model.signal import transform_expert_signal
from src.backtest.portfolio import construct_long_short


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def setup_integration_db(tmp_path, monkeypatch):
    """Set up database with sample data for integration test."""
    db_path = tmp_path / "integration.db"
    monkeypatch.setattr("src.db.schema.DB_PATH", db_path)
    monkeypatch.setattr("config.DB_PATH", db_path)
    import src.db.schema as schema_mod
    schema_mod._engine = None
    init_db()

    # Load sample prices
    prices_df = pd.read_csv(FIXTURES / "sample_prices.csv")
    prices = []
    for _, row in prices_df.iterrows():
        prices.append(Price(
            stock=row["stock"],
            date=datetime.strptime(row["date"], "%Y-%m-%d"),
            open=row["open"], high=row["high"], low=row["low"],
            close=row["close"], volume=row["volume"],
        ))
    insert_prices(prices)

    # Load sample posts
    posts_df = pd.read_csv(FIXTURES / "sample_posts.csv")
    posts = []
    for _, row in posts_df.iterrows():
        posts.append(Post(
            source=row["source"],
            user_id=row["user_id"],
            stock=row["stock"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            sentiment=row["sentiment"],
            content=row["content"],
        ))
    insert_posts(posts)

    yield


def test_full_pipeline_runs():
    """Verify the complete pipeline runs without errors."""
    stocks = ["AAPL", "MSFT", "GOOGL"]

    # Step 1: Expert tracing
    tracker = ExpertTracker()
    records = tracker.trace("2024-06-01")

    # May or may not find experts with limited data, but shouldn't crash
    assert isinstance(records, list)

    # Step 2: Signal transformation
    signals = transform_expert_signal(records, "2024-06-01")
    assert isinstance(signals, dict)

    # Step 3: Prediction
    predictor = BaselinePredictor()
    pred_df = predictor.predict(stocks, "2024-06-01", records)
    assert len(pred_df) == 3
    assert set(pred_df["stock"]) == set(stocks)

    # Step 4: Portfolio construction
    portfolio = construct_long_short(pred_df, quantile=0.33)
    assert len(portfolio["long"]) == 1  # 33% of 3
    assert len(portfolio["short"]) == 1

    # Step 5: Predictions are ordered
    preds = pred_df["predicted_return"].tolist()
    assert preds == sorted(preds, reverse=True)


def test_expert_tracing_with_real_data():
    """Test expert tracing with actual inserted posts."""
    tracker = ExpertTracker()
    records = tracker.trace("2024-06-01")
    assert isinstance(records, list)


def test_baseline_predictor_all_stocks_covered():
    """Every stock in the universe gets a prediction."""
    predictor = BaselinePredictor()
    stocks = ["AAPL", "MSFT", "GOOGL"]
    df = predictor.predict(stocks, "2024-06-01")
    assert len(df) == len(stocks)
    assert df["predicted_return"].notna().all()


class TestWalkForwardIntegration:
    """Integration smoke test for walk-forward + scanner pipeline."""

    def test_walkforward_scan_pipeline(self, prepopulated_db, tmp_path):
        """Walk-forward → scan pipeline runs end-to-end without errors."""
        import torch
        torch.manual_seed(42)
        np.random.seed(42)

        from src.backtest.walkforward import (
            WalkForwardConfig, run_walk_forward,
        )
        from src.backtest.scanner import (
            ParamSpec, build_param_grid, run_scan,
        )

        stocks = ["AAPL", "MSFT"]

        # Walk-forward in params mode.
        # May 1 – Jun 30 has 41 NYSE trading days, which is enough for
        # train_days=30 + validate_days=5 (the first window needs 35 days).
        wf_cfg = WalkForwardConfig(
            train_days=30,
            validate_days=5,
            step_days=5,
            mode="params",
            min_train_days=5,
        )
        wf_result = run_walk_forward(stocks, "2024-05-01", "2024-06-30", wf_cfg)
        assert len(wf_result.windows) > 0
        assert "sharpe_mean" in wf_result.summary

        # Parameter scan (hold-out mode, wf_config=None)
        specs = [ParamSpec("quantile", [0.10, 0.20])]
        grid = build_param_grid(specs)
        scan_df = run_scan(stocks, "2024-05-01", "2024-06-30", grid,
                           wf_config=None, metric="sharpe_ratio")
        assert len(scan_df) == 2
        # Results are sorted by metric descending
        assert scan_df["sharpe_ratio"].iloc[0] >= scan_df["sharpe_ratio"].iloc[1]

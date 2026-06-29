"""Tests for the expert tracker algorithm."""
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.expert.tracker import ExpertTracker
from src.data.models import ExpertRecord
from src.db import schema as db


@pytest.fixture
def tracker():
    return ExpertTracker()


def make_post_dict(user_id, stock, timestamp, sentiment):
    return {
        "user_id": user_id,
        "stock": stock,
        "timestamp": timestamp,
        "sentiment": sentiment,
        "source": "stocktwits",
        "content": "",
    }


class TestFilterDailyLatest:
    def test_keeps_latest_per_user_stock(self, tracker):
        posts = [
            make_post_dict("u1", "AAPL", "2024-06-15T10:00:00", "Bullish"),
            make_post_dict("u1", "AAPL", "2024-06-15T15:00:00", "Bearish"),
            make_post_dict("u1", "MSFT", "2024-06-15T12:00:00", "Bullish"),
        ]
        result = tracker._filter_daily_latest(posts)
        assert len(result) == 2
        aapl_post = [p for p in result if p["stock"] == "AAPL"][0]
        assert aapl_post["sentiment"] == "Bearish"  # Later post kept

    def test_different_users_same_stock_kept_separate(self, tracker):
        posts = [
            make_post_dict("u1", "AAPL", "2024-06-15T10:00:00", "Bullish"),
            make_post_dict("u2", "AAPL", "2024-06-15T11:00:00", "Bearish"),
        ]
        result = tracker._filter_daily_latest(posts)
        assert len(result) == 2


class TestCheckPrediction:
    def test_bullish_correct(self, tracker):
        """Bullish prediction + price went up = correct."""
        mock_prices = {
            "AAPL": [
                {"date": "2024-06-15", "close": 185.0},
                {"date": "2024-06-16", "close": 190.0},
            ]
        }
        with patch.object(db, "get_prices", return_value=mock_prices):
            result = tracker._check_prediction("AAPL", "2024-06-15", "Bullish")
            assert result == 1

    def test_bullish_wrong(self, tracker):
        """Bullish prediction + price went down = wrong."""
        mock_prices = {
            "AAPL": [
                {"date": "2024-06-15", "close": 190.0},
                {"date": "2024-06-16", "close": 185.0},
            ]
        }
        with patch.object(db, "get_prices", return_value=mock_prices):
            result = tracker._check_prediction("AAPL", "2024-06-15", "Bullish")
            assert result == 0

    def test_bearish_correct(self, tracker):
        """Bearish prediction + price went down = correct."""
        mock_prices = {
            "AAPL": [
                {"date": "2024-06-15", "close": 190.0},
                {"date": "2024-06-16", "close": 185.0},
            ]
        }
        with patch.object(db, "get_prices", return_value=mock_prices):
            result = tracker._check_prediction("AAPL", "2024-06-15", "Bearish")
            assert result == 1

    def test_bearish_wrong(self, tracker):
        """Bearish prediction + price went up = wrong."""
        mock_prices = {
            "AAPL": [
                {"date": "2024-06-15", "close": 185.0},
                {"date": "2024-06-16", "close": 190.0},
            ]
        }
        with patch.object(db, "get_prices", return_value=mock_prices):
            result = tracker._check_prediction("AAPL", "2024-06-15", "Bearish")
            assert result == 0

    def test_no_price_data(self, tracker):
        """Missing price data returns None."""
        with patch.object(db, "get_prices", return_value={}):
            result = tracker._check_prediction("AAPL", "2024-06-15", "Bullish")
            assert result is None

    def test_missing_next_day(self, tracker):
        """Only one price point returns None."""
        mock_prices = {
            "AAPL": [
                {"date": "2024-06-15", "close": 185.0},
            ]
        }
        with patch.object(db, "get_prices", return_value=mock_prices):
            result = tracker._check_prediction("AAPL", "2024-06-15", "Bullish")
            assert result is None

    def test_neutral_sentiment(self, tracker):
        """Neutral sentiment returns None (not handled)."""
        mock_prices = {
            "AAPL": [
                {"date": "2024-06-15", "close": 185.0},
                {"date": "2024-06-16", "close": 190.0},
            ]
        }
        with patch.object(db, "get_prices", return_value=mock_prices):
            result = tracker._check_prediction("AAPL", "2024-06-15", "Neutral")
            assert result is None


class TestComputeAccuracy:
    def test_all_correct(self, tracker):
        """All predictions correct returns 1.0."""
        with patch.object(tracker, "_check_prediction", return_value=1):
            posts = [make_post_dict("u1", "AAPL", "2024-06-15", "Bullish") for _ in range(10)]
            assert tracker._compute_accuracy(posts) == 1.0

    def test_all_wrong(self, tracker):
        """All predictions wrong returns 0.0."""
        with patch.object(tracker, "_check_prediction", return_value=0):
            posts = [make_post_dict("u1", "AAPL", "2024-06-15", "Bullish") for _ in range(10)]
            assert tracker._compute_accuracy(posts) == 0.0

    def test_mixed_accuracy(self, tracker):
        """Mixed predictions return proportional accuracy."""
        values = [1, 1, 1, 0, 0]  # 3 correct out of 5 = 0.6
        with patch.object(tracker, "_check_prediction", side_effect=values):
            posts = [make_post_dict("u1", "AAPL", f"2024-06-{15-i:02d}", "Bullish") for i in range(5)]
            assert tracker._compute_accuracy(posts) == 0.6

    def test_no_verifiable_posts(self, tracker):
        """When no predictions can be verified, returns 0.5 (default)."""
        with patch.object(tracker, "_check_prediction", return_value=None):
            posts = [make_post_dict("u1", "AAPL", "2024-06-15", "Bullish")]
            assert tracker._compute_accuracy(posts) == 0.5


class TestEvaluateUser:
    def test_insufficient_recent_posts(self, tracker):
        with patch.object(db, "get_user_history", return_value=[]):
            post_data = make_post_dict("u1", "AAPL", "2024-06-15T10:00:00", "Bullish")
            result = tracker._evaluate_user("u1", post_data, datetime(2024, 6, 15))
            assert result is None

    def test_expert_classification(self, tracker):
        target = datetime(2024, 6, 15)
        recent = [
            make_post_dict("u1", "AAPL", f"2024-06-{(14-i):02d}T10:00:00", "Bullish")
            for i in range(20)
        ]
        long_posts = recent * 5  # 100 posts

        with patch.object(db, "get_user_history") as mock_hist:
            mock_hist.side_effect = [recent, long_posts]
            with patch.object(tracker, "_compute_accuracy", return_value=0.85):
                post_data = make_post_dict("u1", "AAPL", "2024-06-15T10:00:00", "Bullish")
                result = tracker._evaluate_user("u1", post_data, target)

        assert result is not None
        assert result.expert_type == "expert"
        assert result.accuracy_recent == 0.85
        assert result.predicted_direction == "Bullish"

    def test_inverse_expert_classification(self, tracker):
        target = datetime(2024, 6, 15)
        recent = [
            make_post_dict("u1", "AAPL", f"2024-06-{(14-i):02d}T10:00:00", "Bearish")
            for i in range(20)
        ]
        long_posts = recent * 5

        with patch.object(db, "get_user_history") as mock_hist:
            mock_hist.side_effect = [recent, long_posts]
            with patch.object(tracker, "_compute_accuracy", return_value=0.10):
                post_data = make_post_dict("u1", "AAPL", "2024-06-15T10:00:00", "Bearish")
                result = tracker._evaluate_user("u1", post_data, target)

        assert result is not None
        assert result.expert_type == "inverse_expert"
        # Inverse expert with Bearish post → predicted direction flips to Bullish
        assert result.predicted_direction == "Bullish"

    def test_insufficient_days_diversity(self, tracker):
        """Not enough unique trading days returns None."""
        target = datetime(2024, 6, 15)
        # All 20 posts on the same day
        recent = [
            make_post_dict("u1", "AAPL", "2024-06-14T10:00:00", "Bullish")
            for _ in range(20)
        ]
        with patch.object(db, "get_user_history", return_value=recent):
            post_data = make_post_dict("u1", "AAPL", "2024-06-15T10:00:00", "Bullish")
            result = tracker._evaluate_user("u1", post_data, target)
            assert result is None

    def test_no_expert_when_below_thresholds(self, tracker):
        """Accuracy below thresholds should not classify as expert or inverse."""
        target = datetime(2024, 6, 15)
        recent = [
            make_post_dict("u1", "AAPL", f"2024-06-{(14-i):02d}T10:00:00", "Bullish")
            for i in range(20)
        ]
        long_posts = recent * 5

        with patch.object(db, "get_user_history") as mock_hist:
            mock_hist.side_effect = [recent, long_posts]
            # 0.60 is below 0.80 recent threshold but above 0.35 inverse threshold
            with patch.object(tracker, "_compute_accuracy", return_value=0.60):
                post_data = make_post_dict("u1", "AAPL", "2024-06-15T10:00:00", "Bullish")
                result = tracker._evaluate_user("u1", post_data, target)

        assert result is None


class TestTrace:
    def test_empty_posts(self, tracker):
        with patch.object(db, "get_posts_for_date", return_value=[]):
            result = tracker.trace("2024-06-15")
            assert result == []

    def test_trace_with_expert(self, tracker):
        """Integration-style test of the full trace pipeline."""
        posts = [
            make_post_dict("u1", "AAPL", "2024-06-15T10:00:00", "Bullish"),
        ]
        recent = [
            make_post_dict("u1", "AAPL", f"2024-06-{(14-i):02d}T10:00:00", "Bullish")
            for i in range(20)
        ]
        long_posts = recent * 5

        with patch.object(db, "get_posts_for_date", return_value=posts):
            with patch.object(db, "get_user_history") as mock_hist:
                mock_hist.side_effect = [recent, long_posts]
                with patch.object(tracker, "_compute_accuracy", return_value=0.85):
                    with patch.object(db, "insert_expert_records") as mock_insert:
                        result = tracker.trace("2024-06-15")

        assert len(result) == 1
        assert result[0].user_id == "u1"
        assert result[0].expert_type == "expert"
        mock_insert.assert_called_once()

    def test_trace_skips_spammer(self, tracker):
        """Users posting on >10 unique stocks in one day are skipped."""
        posts = []
        for i in range(11):
            posts.append(
                make_post_dict("u1", f"STOCK{i}", "2024-06-15T10:00:00", "Bullish")
            )
        with patch.object(db, "get_posts_for_date", return_value=posts):
            with patch.object(db, "get_user_history", return_value=[]):
                with patch.object(db, "insert_expert_records") as mock_insert:
                    result = tracker.trace("2024-06-15")

        assert len(result) == 0
        mock_insert.assert_called_once_with([])

    def test_trace_skips_users_with_insufficient_history(self, tracker):
        """Users without enough recent posts are silently skipped."""
        posts = [
            make_post_dict("u1", "AAPL", "2024-06-15T10:00:00", "Bullish"),
        ]
        with patch.object(db, "get_posts_for_date", return_value=posts):
            with patch.object(db, "get_user_history", return_value=[]):
                with patch.object(db, "insert_expert_records") as mock_insert:
                    result = tracker.trace("2024-06-15")

        assert len(result) == 0
        mock_insert.assert_called_once_with([])

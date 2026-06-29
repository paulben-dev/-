"""Tests for sentiment analysis module."""
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
from src.data.models import Post
from src.expert.sentiment import SentimentRouter, FinBERTSentiment, LABEL_MAP


class TestSentimentRouter:
    def test_stocktwits_post_passthrough(self):
        router = SentimentRouter()
        post = Post("stocktwits", "user1", "AAPL", datetime(2024, 6, 15), "Bullish")
        result = router.label_post(post)
        assert result.sentiment == "Bullish"

    def test_reddit_post_gets_labeled(self):
        mock_finbert = MagicMock()
        mock_finbert.analyze.return_value = "Bullish"
        router = SentimentRouter(finbert=mock_finbert)
        post = Post("reddit", "user1", "AAPL", datetime(2024, 6, 15), "Neutral", "AAPL is great!")
        result = router.label_post(post)
        assert result.sentiment == "Bullish"
        mock_finbert.analyze.assert_called_once()

    def test_label_posts_batch(self):
        mock_finbert = MagicMock()
        mock_finbert.analyze.return_value = "Bearish"
        router = SentimentRouter(finbert=mock_finbert)
        posts = [
            Post("reddit", "u1", "AAPL", datetime(2024, 6, 15), "Neutral", "bad"),
            Post("stocktwits", "u2", "MSFT", datetime(2024, 6, 15), "Bullish"),
        ]
        results = router.label_posts(posts)
        assert results[0].sentiment == "Bearish"
        assert results[1].sentiment == "Bullish"
        assert mock_finbert.analyze.call_count == 1


class TestFinBERTSentiment:
    def test_empty_text_returns_neutral(self):
        finbert = FinBERTSentiment()
        result = finbert.analyze("")
        assert result == "Neutral"

    def test_none_text_returns_neutral(self):
        finbert = FinBERTSentiment()
        result = finbert.analyze(None)
        assert result == "Neutral"

    def test_label_map(self):
        assert LABEL_MAP[0] == "Neutral"
        assert LABEL_MAP[1] == "Bullish"
        assert LABEL_MAP[2] == "Bearish"

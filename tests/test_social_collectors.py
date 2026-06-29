"""Tests for StockTwits and Reddit collectors."""
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
from src.data.stocktwits import StockTwitsCollector
from src.data.reddit import RedditCollector


class TestStockTwitsCollector:
    def test_collect_posts_success(self):
        collector = StockTwitsCollector(post_limit=5)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "messages": [
                {
                    "id": 1,
                    "body": "AAPL to the moon!",
                    "created_at": "2024-06-15T14:30:00Z",
                    "user": {"id": 123, "username": "trader1"},
                    "entities": {"sentiment": {"basic": "Bullish"}},
                }
            ]
        }
        mock_resp.raise_for_status.return_value = None

        with patch("src.data.stocktwits.requests.get", return_value=mock_resp):
            posts = collector.collect_social_posts(["AAPL"], "2024-06-15")

        assert len(posts) == 1
        assert posts[0].stock == "AAPL"
        assert posts[0].sentiment == "Bullish"
        assert posts[0].user_id == "123"
        assert posts[0].source == "stocktwits"

    def test_collect_posts_date_filter(self):
        collector = StockTwitsCollector(post_limit=5)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "messages": [
                {
                    "id": 1,
                    "body": "Old post",
                    "created_at": "2024-06-10T10:00:00Z",
                    "user": {"id": 1, "username": "old"},
                    "entities": {"sentiment": {"basic": "Bullish"}},
                }
            ]
        }
        mock_resp.raise_for_status.return_value = None

        with patch("src.data.stocktwits.requests.get", return_value=mock_resp):
            posts = collector.collect_social_posts(["AAPL"], "2024-06-15")

        assert len(posts) == 0  # Filtered out by date

    def test_api_error_handled_gracefully(self):
        collector = StockTwitsCollector()
        with patch("src.data.stocktwits.requests.get", side_effect=Exception("Connection error")):
            posts = collector.collect_social_posts(["AAPL"])
        assert posts == []


class TestRedditCollector:
    def test_extract_stock_from_text(self):
        collector = RedditCollector()
        text = "I think AAPL is going up this week, but TSLA looks risky"
        stock = collector._extract_stock_from_text(text, ["AAPL", "MSFT", "TSLA"])
        assert stock == "AAPL"  # First match

    def test_no_stock_found(self):
        collector = RedditCollector()
        text = "The market is crazy today"
        stock = collector._extract_stock_from_text(text, ["AAPL", "MSFT"])
        assert stock is None

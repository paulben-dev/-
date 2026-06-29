"""Tests for YFinance collector."""
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
import pandas as pd
from src.data.yfinance import YFinanceCollector


@pytest.fixture
def collector():
    return YFinanceCollector()


def test_collect_prices_mock(collector):
    mock_hist = MagicMock()
    mock_hist.empty = False
    mock_hist.iterrows.return_value = [
        (pd.Timestamp("2024-06-15"), pd.Series({
            "Open": 185.0, "High": 187.0, "Low": 184.0,
            "Close": 186.5, "Volume": 50000000
        }))
    ]

    mock_ticker = MagicMock()
    mock_ticker.history.return_value = mock_hist

    with patch("src.data.yfinance.yf.Tickers") as mock_tickers:
        mock_tickers.return_value.tickers = {"AAPL": mock_ticker}
        prices = collector.collect_prices(["AAPL"], "2024-06-01", "2024-06-16")

    assert len(prices) == 1
    assert prices[0].stock == "AAPL"
    assert prices[0].close == 186.5
    assert prices[0].volume == 50000000


def test_collect_prices_empty_history(collector):
    mock_hist = MagicMock()
    mock_hist.empty = True
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = mock_hist

    with patch("src.data.yfinance.yf.Tickers") as mock_tickers:
        mock_tickers.return_value.tickers = {"AAPL": mock_ticker}
        prices = collector.collect_prices(["AAPL"], "2024-06-01", "2024-06-16")

    assert len(prices) == 0


def test_collect_social_posts_returns_empty(collector):
    posts = collector.collect_social_posts(["AAPL"], "2024-06-15")
    assert posts == []

"""Yahoo Finance data collector using yfinance library."""
import logging
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
from src.data.base import BaseCollector
from src.data.models import Price, Post

logger = logging.getLogger(__name__)


class YFinanceCollector(BaseCollector):
    """Collects price and fundamental data from Yahoo Finance."""

    def collect_prices(self, stocks: list[str], start_date: str, end_date: str) -> list[Price]:
        """Download OHLCV data for a list of stocks."""
        prices = []
        for i in range(0, len(stocks), 20):
            batch = stocks[i:i + 20]
            tickers = yf.Tickers(" ".join(batch))
            for stock in batch:
                try:
                    ticker = tickers.tickers.get(stock)
                    if ticker is None:
                        logger.warning(f"No data for {stock}")
                        continue
                    hist = ticker.history(start=start_date, end=end_date)
                    if hist.empty:
                        logger.warning(f"Empty history for {stock}")
                        continue
                    for idx, row in hist.iterrows():
                        prices.append(Price(
                            stock=stock,
                            date=idx.to_pydatetime(),
                            open=float(row["Open"]),
                            high=float(row["High"]),
                            low=float(row["Low"]),
                            close=float(row["Close"]),
                            volume=int(row["Volume"]),
                        ))
                except Exception as e:
                    logger.error(f"Error fetching {stock}: {e}")
        logger.info(f"Collected {len(prices)} price records for {len(stocks)} stocks")
        return prices

    def collect_fundamentals(self, stocks: list[str]) -> pd.DataFrame:
        """Fetch key fundamental metrics for stocks."""
        records = []
        for stock in stocks:
            try:
                ticker = yf.Ticker(stock)
                info = ticker.info or {}
                records.append({
                    "stock": stock,
                    "market_cap": info.get("marketCap"),
                    "pe_ratio": info.get("trailingPE"),
                    "pb_ratio": info.get("priceToBook"),
                    "roe": info.get("returnOnEquity"),
                    "debt_to_equity": info.get("debtToEquity"),
                    "sector": info.get("sector", ""),
                    "industry": info.get("industry", ""),
                })
            except Exception as e:
                logger.error(f"Error fetching fundamentals for {stock}: {e}")
        return pd.DataFrame(records)

    def collect_social_posts(self, stocks: list[str], date: str | None = None) -> list[Post]:
        """YFinance does not provide social posts."""
        return []

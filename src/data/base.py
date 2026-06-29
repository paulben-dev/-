"""Abstract base class for data collectors."""
from abc import ABC, abstractmethod
from src.data.models import Post, Price


class BaseCollector(ABC):
    """Interface for all data collectors."""

    @abstractmethod
    def collect_prices(self, stocks: list[str], start_date: str, end_date: str) -> list[Price]:
        """Fetch OHLCV price data."""

    @abstractmethod
    def collect_social_posts(self, stocks: list[str], date: str) -> list[Post]:
        """Fetch social media posts for stocks on a given date."""

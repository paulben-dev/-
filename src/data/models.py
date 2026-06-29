"""Shared data models for the stock prediction system."""
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Post:
    """A social media post about a stock."""
    source: str           # "stocktwits" or "reddit"
    user_id: str          # Unique user identifier
    stock: str            # Ticker symbol
    timestamp: datetime   # When the post was made
    sentiment: str        # "Bullish", "Bearish", or "Neutral"
    content: str = ""     # Raw text content (for FinBERT)

    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)


@dataclass
class Price:
    """OHLCV price data for a stock on a single day."""
    stock: str
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

    def __post_init__(self):
        if isinstance(self.date, str):
            self.date = datetime.fromisoformat(self.date)


@dataclass
class ExpertRecord:
    """Expert classification result for a user on a given day."""
    user_id: str
    stock: str
    date: datetime
    accuracy_recent: float
    accuracy_long: float
    expert_type: str      # "expert", "inverse_expert", or "none"
    predicted_direction: str  # "Bullish" or "Bearish" (from user's post)

    def __post_init__(self):
        if isinstance(self.date, str):
            self.date = datetime.fromisoformat(self.date)

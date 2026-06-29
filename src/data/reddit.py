"""Reddit data collector using PRAW (Python Reddit API Wrapper)."""
import logging
import os
from datetime import datetime
import praw
from src.data.base import BaseCollector
from src.data.models import Post

logger = logging.getLogger(__name__)


class RedditCollector(BaseCollector):
    """Collects stock-related posts from Reddit subreddits."""

    def __init__(self, subreddits: list[str] | None = None, post_limit: int = 100):
        self.subreddits = subreddits or ["wallstreetbets", "stocks"]
        self.post_limit = post_limit
        self._reddit = None

    @property
    def reddit(self) -> praw.Reddit:
        """Lazy-initialize PRAW client."""
        if self._reddit is None:
            self._reddit = praw.Reddit(
                client_id=os.getenv("REDDIT_CLIENT_ID", "your_client_id"),
                client_secret=os.getenv("REDDIT_CLIENT_SECRET", "your_client_secret"),
                user_agent=os.getenv("REDDIT_USER_AGENT", "dualgat-stock-predictor/0.1"),
            )
        return self._reddit

    def _extract_stock_from_text(self, text: str, stocks: list[str]) -> str | None:
        """Simple ticker extraction from text."""
        import re
        words = set(re.findall(r'\b[A-Z]{1,5}\b', text))
        for stock in stocks:
            if stock in words:
                return stock
        return None

    def collect_social_posts(self, stocks: list[str], date: str | None = None) -> list[Post]:
        """Fetch Reddit posts mentioning target stocks."""
        posts = []
        target_date = datetime.fromisoformat(date).date() if date else None
        for subreddit_name in self.subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                for submission in subreddit.new(limit=self.post_limit):
                    created = datetime.fromtimestamp(submission.created_utc)
                    if target_date and created.date() != target_date:
                        continue
                    text = f"{submission.title} {submission.selftext}"
                    stock = self._extract_stock_from_text(text, stocks)
                    if stock is None:
                        continue
                    posts.append(Post(
                        source="reddit",
                        user_id=f"reddit_{submission.author.name if submission.author else 'unknown'}",
                        stock=stock,
                        timestamp=created,
                        sentiment="Neutral",  # FinBERT will re-label these
                        content=text[:2000],
                    ))
            except Exception as e:
                logger.warning(f"Reddit error for r/{subreddit_name}: {e}")
        logger.info(f"Collected {len(posts)} Reddit posts")
        return posts

    def collect_prices(self, stocks: list[str], start_date: str, end_date: str) -> list:
        """Reddit does not provide price data."""
        return []

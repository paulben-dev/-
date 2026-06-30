"""StockTwits data collector using the free API."""
import logging
from datetime import datetime
import requests
from src.data.base import BaseCollector
from src.data.models import Post

logger = logging.getLogger(__name__)

STOCKTWITS_API = "https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"


class StockTwitsCollector(BaseCollector):
    """Collects self-labeled sentiment posts from StockTwits."""

    def __init__(self, post_limit: int = 50):
        self.post_limit = post_limit

    def collect_social_posts(self, stocks: list[str], date: str | None = None) -> list[Post]:
        """Fetch recent StockTwits posts for given stocks."""
        posts = []
        target_date = datetime.fromisoformat(date).date() if date else None
        for stock in stocks:
            try:
                resp = requests.get(
                    STOCKTWITS_API.format(ticker=stock),
                    params={"limit": self.post_limit},
                    timeout=3,
                )
                resp.raise_for_status()
                data = resp.json()
                for msg in data.get("messages", []):
                    created = datetime.fromisoformat(msg["created_at"].replace("Z", "+00:00"))
                    if target_date and created.date() != target_date:
                        continue
                    sentiment_raw = msg.get("entities", {}).get("sentiment")
                    if sentiment_raw:
                        sentiment = sentiment_raw.get("basic", "Neutral").capitalize()
                    else:
                        sentiment = "Neutral"
                    posts.append(Post(
                        source="stocktwits",
                        user_id=str(msg["user"]["id"]),
                        stock=stock,
                        timestamp=created,
                        sentiment=sentiment,
                        content=msg.get("body", ""),
                    ))
            except requests.RequestException as e:
                logger.warning(f"StockTwits error for {stock}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error processing StockTwits for {stock}: {e}")
        logger.info(f"Collected {len(posts)} StockTwits posts")
        return posts

    def collect_prices(self, stocks: list[str], start_date: str, end_date: str) -> list:
        """StockTwits does not provide price data."""
        return []

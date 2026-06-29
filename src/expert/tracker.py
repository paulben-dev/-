"""Expert tracing system — Algorithm 1 from the DualGAT paper.

Implements the two-stage evaluation for identifying experts and inverse experts
from social media posts, as described in:
"Unleashing Expert Opinion from Social Media for Stock Prediction"
"""
import logging
from datetime import datetime, timedelta
from collections import defaultdict

from src.data.models import ExpertRecord
from src.db import schema as db
from config import (
    EXPERT_RECENT_N,
    EXPERT_MIN_DAYS,
    EXPERT_RECENT_THRESHOLD,
    EXPERT_LONG_THRESHOLD,
    EXPERT_LONG_WINDOW_DAYS,
)

logger = logging.getLogger(__name__)


class ExpertTracker:
    """Identifies true experts and inverse experts from social media posts.

    Algorithm 1 from: "Unleashing Expert Opinion from Social Media for Stock Prediction"

    Two-stage evaluation:
      1. Recent performance (last N posts, requiring M unique trading days)
      2. Long-term performance (past T=2 years window)

    Users with high accuracy in both windows are classified as experts;
    users with consistently low accuracy (worse than random) are inverse experts.
    """

    def __init__(self):
        self.recent_n = EXPERT_RECENT_N
        self.min_days = EXPERT_MIN_DAYS
        self.recent_threshold = EXPERT_RECENT_THRESHOLD
        self.long_threshold = EXPERT_LONG_THRESHOLD
        self.long_window = EXPERT_LONG_WINDOW_DAYS

    def trace(self, date_str: str) -> list[ExpertRecord]:
        """Run expert tracing for a given trading day.

        Args:
            date_str: Date string in YYYY-MM-DD format.

        Returns:
            List of ExpertRecord for users identified as experts/inverse experts.
        """
        target_date = datetime.fromisoformat(date_str)
        posts = db.get_posts_for_date(date_str)

        if not posts:
            logger.info(f"No posts found for {date_str}")
            return []

        # Step 1: Filter — keep only latest post per user-stock pair on this day
        filtered = self._filter_daily_latest(posts)

        # Step 2: Get unique users who posted
        user_ids = set(p["user_id"] for p in filtered)

        records = []
        for user_id in user_ids:
            user_posts_today = [p for p in filtered if p["user_id"] == user_id]

            # Focus check: skip users posting on too many stocks in one day
            unique_stocks = set(p["stock"] for p in user_posts_today)
            if len(unique_stocks) > 10:
                continue

            for post_data in user_posts_today:
                record = self._evaluate_user(user_id, post_data, target_date)
                if record is not None:
                    records.append(record)

        logger.info(f"Traced {len(records)} expert records for {date_str}")
        db.insert_expert_records(records)
        return records

    def _filter_daily_latest(self, posts: list[dict]) -> list[dict]:
        """Keep only the latest post per user-stock pair on the day."""
        groups = defaultdict(list)
        for p in posts:
            key = (p["user_id"], p["stock"])
            groups[key].append(p)
        return [max(g, key=lambda x: x["timestamp"]) for g in groups.values()]

    def _evaluate_user(
        self, user_id: str, post_data: dict, target_date: datetime
    ) -> ExpertRecord | None:
        """Two-stage evaluation: recent performance + long-term performance.

        Stage 1: Recent performance (last N posts before target date)
        Stage 2: Long-term performance (past T=2 years window)
        Stage 3: Classify as expert, inverse expert, or none.
        """
        date_str = target_date.strftime("%Y-%m-%d")

        # Stage 1: Recent performance (last N=20 posts before today)
        recent_posts = db.get_user_history(user_id, date_str, limit=self.recent_n)
        if len(recent_posts) < self.recent_n:
            return None  # Not enough recent posts

        # Check minimum unique trading days
        unique_days = set(p["timestamp"][:10] for p in recent_posts)
        if len(unique_days) < self.min_days:
            return None  # Not enough trading day diversity

        recent_accuracy = self._compute_accuracy(recent_posts)

        # Stage 2: Long-term performance (past T=2 years)
        long_start = (target_date - timedelta(days=self.long_window)).strftime(
            "%Y-%m-%d"
        )
        long_posts = db.get_user_history(user_id, date_str)

        # Filter to window
        long_posts = [p for p in long_posts if p["timestamp"][:10] >= long_start]
        if len(long_posts) < 10:
            return None  # Not enough long-term history

        long_accuracy = self._compute_accuracy(long_posts)

        # Stage 3: Classify
        expert_type = "none"
        if (
            recent_accuracy >= self.recent_threshold
            and long_accuracy >= self.long_threshold
        ):
            expert_type = "expert"
        elif recent_accuracy <= (1 - self.recent_threshold) and long_accuracy <= (
            1 - self.long_threshold
        ):
            expert_type = "inverse_expert"

        if expert_type == "none":
            return None

        # Determine predicted direction (invert for inverse experts)
        direction = post_data["sentiment"]
        if expert_type == "inverse_expert":
            direction = "Bearish" if direction == "Bullish" else "Bullish"

        return ExpertRecord(
            user_id=user_id,
            stock=post_data["stock"],
            date=target_date,
            accuracy_recent=round(recent_accuracy, 4),
            accuracy_long=round(long_accuracy, 4),
            expert_type=expert_type,
            predicted_direction=direction,
        )

    def _compute_accuracy(self, posts: list[dict]) -> float:
        """Compute prediction accuracy from a list of posts.

        A prediction is correct if:
        - Bullish post → stock price rose next trading day
        - Bearish post → stock price fell next trading day
        """
        correct = 0
        total = 0
        for p in posts:
            result = self._check_prediction(
                p["stock"], p["timestamp"][:10], p["sentiment"]
            )
            if result is not None:
                correct += result
                total += 1
        return correct / total if total > 0 else 0.5

    def _check_prediction(
        self, stock: str, date_str: str, sentiment: str
    ) -> int | None:
        """Check if a prediction was correct.

        Looks up price data for the prediction date and finds the next
        trading day to determine if the direction was correct.

        Returns 1 (correct), 0 (wrong), or None (insufficient data).
        """
        # Extend end date by 7 calendar days to capture next trading day
        # (handles weekends and holidays)
        end_date = (
            datetime.fromisoformat(date_str) + timedelta(days=7)
        ).strftime("%Y-%m-%d")
        prices = db.get_prices([stock], date_str, end_date)
        stock_prices = prices.get(stock, [])
        if len(stock_prices) < 2:
            return None

        # Find today's price and the next trading day's price
        today_price = None
        tomorrow_price = None
        for p in stock_prices:
            if p["date"] == date_str:
                today_price = p
            elif p["date"] > date_str and tomorrow_price is None:
                tomorrow_price = p

        if today_price is None or tomorrow_price is None:
            return None

        price_up = tomorrow_price["close"] > today_price["close"]

        if sentiment == "Bullish":
            return 1 if price_up else 0
        elif sentiment == "Bearish":
            return 1 if not price_up else 0
        return None

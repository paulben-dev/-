# tests/conftest.py
"""Shared test fixtures."""
import pytest
from datetime import datetime, timedelta
from src.db.schema import init_db, insert_prices, insert_posts
from src.data.models import Price, Post


@pytest.fixture
def prepopulated_db(tmp_path, monkeypatch):
    """Database pre-populated with sample price and post data."""
    db_path = tmp_path / "test_conftest.db"
    monkeypatch.setattr("src.db.schema.DB_PATH", db_path)
    monkeypatch.setattr("config.DB_PATH", db_path)
    import src.db.schema as schema_mod
    schema_mod._engine = None
    init_db()

    # Insert sample prices: 44 days from May 1 to June 13 + specific June 14/15
    import random
    random.seed(42)
    prices = []

    base_aapl = 170.0
    base_msft = 400.0

    # Generate prices for every day from May 1 to June 13
    current = datetime(2024, 5, 1)
    end_fill = datetime(2024, 6, 14)
    while current < end_fill:
        close_aapl = base_aapl + random.uniform(-2, 2)
        close_msft = base_msft + random.uniform(-5, 5)
        prices.append(Price("AAPL", current, close_aapl - 1, close_aapl + 1, close_aapl - 2, close_aapl, 50000000))
        prices.append(Price("MSFT", current, close_msft - 2, close_msft + 2, close_msft - 3, close_msft, 25000000))
        base_aapl = close_aapl
        base_msft = close_msft
        current += timedelta(days=1)

    # Overwrite June 14-15 with specific test values
    prices.append(Price("AAPL", datetime(2024, 6, 14), 184.0, 186.0, 183.0, 185.5, 50000000))
    prices.append(Price("AAPL", datetime(2024, 6, 15), 185.5, 187.0, 185.0, 186.5, 48000000))
    prices.append(Price("MSFT", datetime(2024, 6, 14), 414.0, 416.0, 413.0, 415.5, 25000000))
    prices.append(Price("MSFT", datetime(2024, 6, 15), 415.5, 419.0, 415.0, 418.0, 24000000))

    insert_prices(prices)

    # Insert sample posts (enough for expert tracing: 20+ posts over 5+ days)
    posts = []
    for i in range(25):
        posts.append(Post(
            "stocktwits", "expert_user", "AAPL",
            datetime(2024, 6, 15, 10, i), "Bullish", "Going up!"
        ))
    for i in range(25):
        posts.append(Post(
            "stocktwits", "inverse_user", "MSFT",
            datetime(2024, 6, 15, 11, i), "Bearish", "Going down!"
        ))
    insert_posts(posts)
    yield

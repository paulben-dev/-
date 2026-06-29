"""Tests for database schema and CRUD operations."""
import pytest
from datetime import datetime
from src.db.schema import init_db, get_db, insert_posts, get_posts_for_date, insert_prices
from src.data.models import Post, Price


@pytest.fixture(autouse=True)
def setup_db(tmp_path, monkeypatch):
    """Use a temporary database for each test."""
    db_path = tmp_path / "test.db"
    monkeypatch.setattr("src.db.schema.DB_PATH", db_path)
    global _engine
    import src.db.schema as schema_mod
    schema_mod._engine = None
    init_db()
    yield
    schema_mod._engine = None


def test_init_db_creates_tables(setup_db):
    engine = get_db()
    with engine.connect() as conn:
        tables = conn.execute(
            __import__("sqlalchemy").text(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        ).fetchall()
    table_names = [t[0] for t in tables]
    assert "posts" in table_names
    assert "prices" in table_names
    assert "expert_records" in table_names


def test_insert_and_retrieve_posts(setup_db):
    post = Post(
        source="stocktwits",
        user_id="user123",
        stock="AAPL",
        timestamp=datetime(2024, 6, 15, 14, 30),
        sentiment="Bullish",
        content="AAPL looks great!",
    )
    insert_posts([post])
    results = get_posts_for_date("2024-06-15")
    assert len(results) == 1
    assert results[0]["user_id"] == "user123"
    assert results[0]["sentiment"] == "Bullish"


def test_insert_posts_deduplicates(setup_db):
    post = Post("stocktwits", "user123", "AAPL", datetime(2024, 6, 15), "Bullish")
    insert_posts([post])
    insert_posts([post])
    results = get_posts_for_date("2024-06-15")
    assert len(results) == 1


def test_insert_prices(setup_db):
    price = Price("AAPL", datetime(2024, 6, 15), 185.0, 187.0, 184.0, 186.5, 50000000)
    insert_prices([price])
    from src.db.schema import get_prices
    results = get_prices(["AAPL"], "2024-06-15", "2024-06-15")
    assert "AAPL" in results
    assert results["AAPL"][0]["close"] == 186.5

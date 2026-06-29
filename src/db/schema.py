"""SQLite database schema and operations."""
import sqlalchemy as sa
from sqlalchemy import create_engine, text
from pathlib import Path
from config import DB_PATH
from src.data.models import Post, Price, ExpertRecord

_engine = None


def get_db() -> sa.Engine:
    """Get or create the database engine singleton."""
    global _engine
    if _engine is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
    return _engine


def init_db():
    """Create all tables if they don't exist."""
    engine = get_db()
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                user_id TEXT NOT NULL,
                stock TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                sentiment TEXT NOT NULL,
                content TEXT DEFAULT '',
                created_at TEXT DEFAULT (datetime('now')),
                UNIQUE(source, user_id, stock, timestamp)
            )
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_posts_user_stock
            ON posts(user_id, stock, timestamp)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_posts_timestamp
            ON posts(timestamp)
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stock TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume INTEGER NOT NULL,
                UNIQUE(stock, date)
            )
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_prices_stock_date
            ON prices(stock, date)
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS expert_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                stock TEXT NOT NULL,
                date TEXT NOT NULL,
                accuracy_recent REAL NOT NULL,
                accuracy_long REAL NOT NULL,
                expert_type TEXT NOT NULL,
                predicted_direction TEXT NOT NULL,
                UNIQUE(user_id, stock, date)
            )
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_expert_date
            ON expert_records(date)
        """))


def insert_posts(posts: list[Post]):
    """Insert posts into the database, ignoring duplicates."""
    if not posts:
        return
    engine = get_db()
    rows = [
        {
            "source": p.source,
            "user_id": p.user_id,
            "stock": p.stock,
            "timestamp": p.timestamp.isoformat(),
            "sentiment": p.sentiment,
            "content": p.content,
        }
        for p in posts
    ]
    with engine.begin() as conn:
        conn.execute(
            text("""INSERT OR IGNORE INTO posts (source, user_id, stock, timestamp, sentiment, content)
                    VALUES (:source, :user_id, :stock, :timestamp, :sentiment, :content)"""),
            rows,
        )


def insert_prices(prices: list[Price]):
    """Insert price data, replacing on conflict."""
    if not prices:
        return
    engine = get_db()
    rows = [
        {
            "stock": p.stock,
            "date": p.date.strftime("%Y-%m-%d"),
            "open": p.open,
            "high": p.high,
            "low": p.low,
            "close": p.close,
            "volume": p.volume,
        }
        for p in prices
    ]
    with engine.begin() as conn:
        conn.execute(
            text("""INSERT OR REPLACE INTO prices (stock, date, open, high, low, close, volume)
                    VALUES (:stock, :date, :open, :high, :low, :close, :volume)"""),
            rows,
        )


def insert_expert_records(records: list[ExpertRecord]):
    """Insert expert records, replacing on conflict."""
    if not records:
        return
    engine = get_db()
    rows = [
        {
            "user_id": r.user_id,
            "stock": r.stock,
            "date": r.date.strftime("%Y-%m-%d"),
            "accuracy_recent": r.accuracy_recent,
            "accuracy_long": r.accuracy_long,
            "expert_type": r.expert_type,
            "predicted_direction": r.predicted_direction,
        }
        for r in records
    ]
    with engine.begin() as conn:
        conn.execute(
            text("""INSERT OR REPLACE INTO expert_records
                    (user_id, stock, date, accuracy_recent, accuracy_long, expert_type, predicted_direction)
                    VALUES (:user_id, :stock, :date, :accuracy_recent, :accuracy_long, :expert_type, :predicted_direction)"""),
            rows,
        )


def get_posts_for_date(date_str: str) -> list[dict]:
    """Get all posts for a specific date."""
    engine = get_db()
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT * FROM posts WHERE date(timestamp) = :date"),
            {"date": date_str},
        )
        return [dict(row._mapping) for row in result]


def get_user_history(user_id: str, before_date: str, limit: int | None = None) -> list[dict]:
    """Get a user's post history before a given date, ordered by recency."""
    engine = get_db()
    query = """SELECT * FROM posts
                WHERE user_id = :uid AND date(timestamp) < :before
                ORDER BY timestamp DESC"""
    if limit is not None:
        query += f" LIMIT {limit}"
    with engine.connect() as conn:
        result = conn.execute(
            text(query),
            {"uid": user_id, "before": before_date},
        )
        return [dict(row._mapping) for row in result]


def get_prices(stocks: list[str], start_date: str, end_date: str) -> dict[str, list[dict]]:
    """Get price history for stocks in a date range, grouped by stock."""
    engine = get_db()
    placeholders = ",".join([f":s{i}" for i in range(len(stocks))])
    params = {f"s{i}": s for i, s in enumerate(stocks)}
    params["start"] = start_date
    params["end"] = end_date
    with engine.connect() as conn:
        result = conn.execute(
            text(f"""SELECT * FROM prices
                     WHERE stock IN ({placeholders})
                     AND date BETWEEN :start AND :end
                     ORDER BY stock, date"""),
            params,
        )
        rows = [dict(row._mapping) for row in result]
    grouped = {}
    for row in rows:
        grouped.setdefault(row["stock"], []).append(row)
    return grouped


def get_expert_records(start_date: str, end_date: str) -> list[dict]:
    """Get expert records in a date range."""
    engine = get_db()
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT * FROM expert_records WHERE date BETWEEN :start AND :end ORDER BY date"),
            {"start": start_date, "end": end_date},
        )
        return [dict(row._mapping) for row in result]

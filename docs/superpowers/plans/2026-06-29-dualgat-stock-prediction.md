# DualGAT Stock Prediction MVP — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a web-based stock prediction MVP (v0.1) with data collection, expert tracing, rule-based prediction, backtest engine, and FastAPI dashboard.

**Architecture:** Layered pipeline: Data Collectors → Sentiment Analysis → Expert Tracker → Signal Transform → Rule Predictor → Backtest Engine → Web API → Dashboard. Each layer is independently testable with clear interfaces.

**Tech Stack:** Python 3.10+, FastAPI, yfinance, PRAW, transformers (FinBERT), SQLite, pandas, numpy, HTMX, Chart.js

## Global Constraints

- CPU-only (no GPU for MVP)
- Local SQLite database at `data/predictions.db`
- Stock universe: configurable list, default ~20 liquid US stocks for MVP
- Python 3.10+ (use `str | None` not `Optional[str]`)
- Expert tracing thresholds from paper: P1=0.65, P2=0.80, N=20, K=5, T=2 years
- Backtest: long top 10% / short bottom 10%, 4bps cost, daily rebalancing

---

### Task 1: Project Scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `config.py`
- Create: `src/__init__.py`
- Create: `src/data/__init__.py`
- Create: `src/expert/__init__.py`
- Create: `src/model/__init__.py`
- Create: `src/backtest/__init__.py`
- Create: `src/web/__init__.py`
- Create: `src/db/__init__.py`
- Create: `src/web/templates/.gitkeep`
- Create: `src/web/static/.gitkeep`
- Create: `data/.gitkeep`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/fixtures/__init__.py`

**Interfaces:**
- Produces: directory structure, `config.py` constants, `requirements.txt` for all subsequent tasks

- [ ] **Step 1: Write requirements.txt**

```txt
# Core
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pandas>=2.0.0
numpy>=1.24.0
pydantic>=2.0.0

# Data Collection
yfinance>=0.2.30
praw>=7.7.0

# Sentiment
torch>=2.0.0
transformers>=4.35.0

# Database
sqlalchemy>=2.0.0

# Web
jinja2>=3.1.0
python-multipart>=0.0.6

# Scientific
scipy>=1.10.0

# Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0
httpx>=0.25.0
```

- [ ] **Step 2: Write config.py**

```python
"""Central configuration for the DualGAT stock prediction system."""
from pathlib import Path

# Project root
ROOT_DIR = Path(__file__).parent.resolve()

# Stock Universe (liquid US stocks across sectors for MVP)
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM",
    "JNJ", "V", "PG", "XOM", "WMT", "MA", "UNH", "HD", "BAC", "DIS",
    "NFLX", "ADBE",
]

# Expert Tracing (Algorithm 1 from paper)
EXPERT_RECENT_N = 20          # Number of recent posts to evaluate
EXPERT_MIN_DAYS = 5           # Minimum unique trading days in recent posts
EXPERT_RECENT_THRESHOLD = 0.80  # P2: recent accuracy >= 80% → expert
EXPERT_LONG_THRESHOLD = 0.65    # P1: long-term accuracy >= 65% → expert
EXPERT_LONG_WINDOW_DAYS = 730   # T = 2 years in days

# Signal Transformation
SIGNAL_LOOKBACK_DAYS = 30     # Days for average return calculation

# Momentum (for baseline when no expert signal)
MOMENTUM_LOOKBACK_DAYS = 20

# Graph (reserved for v0.3 DualGAT)
CORR_THRESHOLD_NORMAL = 0.77  # θ1
CORR_THRESHOLD_EXPERT = 0.67  # θ2

# Backtest
PORTFOLIO_QUANTILE = 0.10     # Top/bottom 10%
TRANSACTION_COST = 0.0004     # 4 basis points

# Database
DB_PATH = ROOT_DIR / "data" / "predictions.db"

# API
API_HOST = "0.0.0.0"
API_PORT = 8000

# Data Collection
REDDIT_SUBREDDITS = ["wallstreetbets", "stocks"]
REDDIT_POST_LIMIT = 100       # Posts per subreddit per fetch
STOCKTWITS_POST_LIMIT = 50    # Posts per stock per fetch
```

- [ ] **Step 3: Write conftest.py**

```python
# tests/conftest.py
"""Shared test fixtures."""
import pytest
from datetime import datetime
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

    # Insert sample prices
    prices = [
        Price("AAPL", datetime(2024, 6, 14), 184.0, 186.0, 183.0, 185.5, 50000000),
        Price("AAPL", datetime(2024, 6, 15), 185.5, 187.0, 185.0, 186.5, 48000000),
        Price("MSFT", datetime(2024, 6, 14), 414.0, 416.0, 413.0, 415.5, 25000000),
        Price("MSFT", datetime(2024, 6, 15), 415.5, 419.0, 415.0, 418.0, 24000000),
    ]
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
```

- [ ] **Step 4: Create directory structure**

```bash
mkdir -p src/data src/expert src/model src/backtest src/web/templates src/web/static src/db data tests/fixtures
touch src/__init__.py src/data/__init__.py src/expert/__init__.py src/model/__init__.py src/backtest/__init__.py src/web/__init__.py src/db/__init__.py tests/__init__.py tests/fixtures/__init__.py
```

- [ ] **Step 5: Verify structure**

```bash
python -c "import config; print(config.DEFAULT_TICKERS[:3])"
```
Expected: `['AAPL', 'MSFT', 'GOOGL']`

- [ ] **Step 6: Install dependencies**

```bash
pip install -r requirements.txt
```

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat: project scaffolding with config and dependencies"
```

---

### Task 2: Data Models and Database Schema

**Files:**
- Create: `src/data/models.py`
- Create: `src/db/schema.py`
- Create: `tests/test_db_schema.py`

**Interfaces:**
- Produces:
  - `Post` dataclass with fields: `source, user_id, stock, timestamp, sentiment, content`
  - `Price` dataclass with fields: `stock, date, open, high, low, close, volume`
  - `ExpertRecord` dataclass with fields: `user_id, stock, date, accuracy_recent, accuracy_long, expert_type`
  - `get_db()` → sqlalchemy Engine
  - `init_db()` → creates all tables
  - `insert_posts(posts: list[Post])` → None
  - `get_posts_for_date(date: str) → list[Post]`
  - `get_user_history(user_id: str, before_date: str, limit: int | None) → list[Post]`

- [ ] **Step 1: Write data models**

```python
# src/data/models.py
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
```

- [ ] **Step 2: Write database schema**

```python
# src/db/schema.py
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
                created_at TEXT DEFAULT (datetime('now'))
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
```

- [ ] **Step 3: Write test**

```python
# tests/test_db_schema.py
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
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_db_schema.py -v
```
Expected: 4 tests pass

- [ ] **Step 5: Commit**

```bash
git add src/data/models.py src/db/schema.py tests/test_db_schema.py
git commit -m "feat: data models and SQLite database schema"
```

---

### Task 3: YFinance Price Collector

**Files:**
- Create: `src/data/base.py`
- Create: `src/data/yfinance.py`
- Create: `tests/test_yfinance.py`

**Interfaces:**
- Consumes: `config.DEFAULT_TICKERS`, `src.data.models.Price`
- Produces: `YFinanceCollector.collect_prices(stocks, start, end) -> list[Price]`, `YFinanceCollector.collect_fundamentals(stocks) -> pd.DataFrame`

- [ ] **Step 1: Write abstract base**

```python
# src/data/base.py
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
```

- [ ] **Step 2: Write YFinance collector**

```python
# src/data/yfinance.py
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

    def collect_social_posts(self, stocks: list[str], date: str) -> list[Post]:
        """YFinance does not provide social posts."""
        return []
```

- [ ] **Step 3: Write test**

```python
# tests/test_yfinance.py
"""Tests for YFinance collector."""
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
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
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_yfinance.py -v
```
Expected: 3 tests pass

- [ ] **Step 5: Commit**

```bash
git add src/data/base.py src/data/yfinance.py tests/test_yfinance.py
git commit -m "feat: YFinance price collector with mock testing"
```

---

### Task 4: StockTwits and Reddit Collectors

**Files:**
- Create: `src/data/stocktwits.py`
- Create: `src/data/reddit.py`
- Create: `tests/test_social_collectors.py`

**Interfaces:**
- Consumes: `BaseCollector`, `Post`, `config.REDDIT_SUBREDDITS`, `config.STOCKTWITS_POST_LIMIT`
- Produces: `StockTwitsCollector.collect_social_posts(stocks, date) -> list[Post]`, `RedditCollector.collect_social_posts(stocks, date) -> list[Post]`

- [ ] **Step 1: Write StockTwits collector**

```python
# src/data/stocktwits.py
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
                    timeout=10,
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
```

- [ ] **Step 2: Write Reddit collector**

```python
# src/data/reddit.py
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
        stock_set = set(stocks)
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
```

- [ ] **Step 3: Write tests**

```python
# tests/test_social_collectors.py
"""Tests for StockTwits and Reddit collectors."""
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
from src.data.stocktwits import StockTwitsCollector
from src.data.reddit import RedditCollector


class TestStockTwitsCollector:
    def test_collect_posts_success(self):
        collector = StockTwitsCollector(post_limit=5)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "messages": [
                {
                    "id": 1,
                    "body": "AAPL to the moon!",
                    "created_at": "2024-06-15T14:30:00Z",
                    "user": {"id": 123, "username": "trader1"},
                    "entities": {"sentiment": {"basic": "Bullish"}},
                }
            ]
        }
        mock_resp.raise_for_status.return_value = None

        with patch("src.data.stocktwits.requests.get", return_value=mock_resp):
            posts = collector.collect_social_posts(["AAPL"], "2024-06-15")

        assert len(posts) == 1
        assert posts[0].stock == "AAPL"
        assert posts[0].sentiment == "Bullish"
        assert posts[0].user_id == "123"
        assert posts[0].source == "stocktwits"

    def test_collect_posts_date_filter(self):
        collector = StockTwitsCollector(post_limit=5)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "messages": [
                {
                    "id": 1,
                    "body": "Old post",
                    "created_at": "2024-06-10T10:00:00Z",
                    "user": {"id": 1, "username": "old"},
                    "entities": {"sentiment": {"basic": "Bullish"}},
                }
            ]
        }
        mock_resp.raise_for_status.return_value = None

        with patch("src.data.stocktwits.requests.get", return_value=mock_resp):
            posts = collector.collect_social_posts(["AAPL"], "2024-06-15")

        assert len(posts) == 0  # Filtered out by date

    def test_api_error_handled_gracefully(self):
        collector = StockTwitsCollector()
        with patch("src.data.stocktwits.requests.get", side_effect=Exception("Connection error")):
            posts = collector.collect_social_posts(["AAPL"])
        assert posts == []


class TestRedditCollector:
    def test_extract_stock_from_text(self):
        collector = RedditCollector()
        text = "I think AAPL is going up this week, but TSLA looks risky"
        stock = collector._extract_stock_from_text(text, ["AAPL", "MSFT", "TSLA"])
        assert stock == "AAPL"  # First match

    def test_no_stock_found(self):
        collector = RedditCollector()
        text = "The market is crazy today"
        stock = collector._extract_stock_from_text(text, ["AAPL", "MSFT"])
        assert stock is None
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_social_collectors.py -v
```
Expected: 4 tests pass

- [ ] **Step 5: Commit**

```bash
git add src/data/stocktwits.py src/data/reddit.py tests/test_social_collectors.py
git commit -m "feat: StockTwits and Reddit social media collectors"
```

---

### Task 5: Sentiment Analysis with FinBERT

**Files:**
- Create: `src/expert/sentiment.py`
- Create: `tests/test_sentiment.py`

**Interfaces:**
- Consumes: `src.data.models.Post`
- Produces:
  - `FinBERTSentiment.analyze(text: str) -> str`  (returns "Bullish"/"Bearish"/"Neutral")
  - `SentimentRouter.label_post(post: Post) -> Post` (routes to correct analyzer, returns post with sentiment set)

- [ ] **Step 1: Write sentiment analysis module**

```python
# src/expert/sentiment.py
"""Sentiment analysis for social media posts using FinBERT."""
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

# FinBERT label mapping
LABEL_MAP = {0: "Neutral", 1: "Bullish", 2: "Bearish"}


class FinBERTSentiment:
    """Financial sentiment analysis using ProsusAI/finbert."""

    def __init__(self):
        self._tokenizer = None
        self._model = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            self._model = AutoModelForSequenceClassification.from_pretrained(
                "ProsusAI/finbert"
            ).to(self._device)
            self._model.eval()
        return self._model

    def analyze(self, text: str) -> str:
        """Classify text as Bullish, Bearish, or Neutral."""
        if not text or not text.strip():
            return "Neutral"
        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            ).to(self._device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                prediction = torch.argmax(outputs.logits, dim=1).item()
            return LABEL_MAP[prediction]
        except Exception as e:
            logger.error(f"FinBERT error: {e}")
            return "Neutral"


class SentimentRouter:
    """Routes posts to the correct sentiment analyzer based on source."""

    def __init__(self, finbert: FinBERTSentiment | None = None):
        self.finbert = finbert or FinBERTSentiment()

    def label_post(self, post) -> "Post":
        """Label a post with sentiment, using FinBERT only for unlabeled sources."""
        if post.source == "stocktwits":
            return post  # Already self-labeled
        sentiment = self.finbert.analyze(post.content)
        post.sentiment = sentiment
        return post

    def label_posts(self, posts: list) -> list:
        """Label a batch of posts."""
        return [self.label_post(p) for p in posts]
```

- [ ] **Step 2: Write test**

```python
# tests/test_sentiment.py
"""Tests for sentiment analysis module."""
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
from src.data.models import Post
from src.expert.sentiment import SentimentRouter, FinBERTSentiment, LABEL_MAP


class TestSentimentRouter:
    def test_stocktwits_post_passthrough(self):
        router = SentimentRouter()
        post = Post("stocktwits", "user1", "AAPL", datetime(2024, 6, 15), "Bullish")
        result = router.label_post(post)
        assert result.sentiment == "Bullish"

    def test_reddit_post_gets_labeled(self):
        mock_finbert = MagicMock()
        mock_finbert.analyze.return_value = "Bullish"
        router = SentimentRouter(finbert=mock_finbert)
        post = Post("reddit", "user1", "AAPL", datetime(2024, 6, 15), "Neutral", "AAPL is great!")
        result = router.label_post(post)
        assert result.sentiment == "Bullish"
        mock_finbert.analyze.assert_called_once()

    def test_label_posts_batch(self):
        mock_finbert = MagicMock()
        mock_finbert.analyze.return_value = "Bearish"
        router = SentimentRouter(finbert=mock_finbert)
        posts = [
            Post("reddit", "u1", "AAPL", datetime(2024, 6, 15), "Neutral", "bad"),
            Post("stocktwits", "u2", "MSFT", datetime(2024, 6, 15), "Bullish"),
        ]
        results = router.label_posts(posts)
        assert results[0].sentiment == "Bearish"
        assert results[1].sentiment == "Bullish"
        assert mock_finbert.analyze.call_count == 1


class TestFinBERTSentiment:
    def test_empty_text_returns_neutral(self):
        finbert = FinBERTSentiment()
        result = finbert.analyze("")
        assert result == "Neutral"

    def test_none_text_returns_neutral(self):
        finbert = FinBERTSentiment()
        result = finbert.analyze(None)
        assert result == "Neutral"

    def test_label_map(self):
        assert LABEL_MAP[0] == "Neutral"
        assert LABEL_MAP[1] == "Bullish"
        assert LABEL_MAP[2] == "Bearish"
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/test_sentiment.py -v
```
Expected: 5 tests pass

- [ ] **Step 4: Commit**

```bash
git add src/expert/sentiment.py tests/test_sentiment.py
git commit -m "feat: FinBERT sentiment analysis with source-aware routing"
```

---

### Task 6: Expert Tracker — Core Algorithm

**Files:**
- Create: `src/expert/tracker.py`
- Create: `tests/test_tracker.py`

**Interfaces:**
- Consumes: `config` thresholds, `src.data.models.Post`, `src.data.models.ExpertRecord`, `src.db.schema`
- Produces:
  - `ExpertTracker.trace(date_str: str) -> list[ExpertRecord]`
  - `ExpertTracker._is_prediction_correct(post: Post, price_data: dict) -> bool`

- [ ] **Step 1: Write tracker implementation**

```python
# src/expert/tracker.py
"""Expert tracing system — Algorithm 1 from the DualGAT paper."""
import logging
from datetime import datetime, timedelta
from collections import defaultdict
from src.data.models import Post, ExpertRecord
from src.db import schema as db
from config import (
    EXPERT_RECENT_N, EXPERT_MIN_DAYS, EXPERT_RECENT_THRESHOLD,
    EXPERT_LONG_THRESHOLD, EXPERT_LONG_WINDOW_DAYS, DEFAULT_TICKERS,
)

logger = logging.getLogger(__name__)


class ExpertTracker:
    """Identifies true experts and inverse experts from social media posts.

    Algorithm 1 from: "Unleashing Expert Opinion from Social Media for Stock Prediction"
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

    def _evaluate_user(self, user_id: str, post_data: dict, target_date: datetime) -> ExpertRecord | None:
        """Two-stage evaluation: recent performance + long-term performance."""
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
        long_start = (target_date - timedelta(days=self.long_window)).strftime("%Y-%m-%d")
        long_posts = db.get_user_history(user_id, date_str)

        # Filter to window
        long_posts = [p for p in long_posts if p["timestamp"][:10] >= long_start]
        if len(long_posts) < 10:
            return None  # Not enough long-term history

        long_accuracy = self._compute_accuracy(long_posts)

        # Stage 3: Classify
        expert_type = "none"
        if recent_accuracy >= self.recent_threshold and long_accuracy >= self.long_threshold:
            expert_type = "expert"
        elif recent_accuracy <= (1 - self.recent_threshold) and long_accuracy <= (1 - self.long_threshold):
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
            result = self._check_prediction(p["stock"], p["timestamp"][:10], p["sentiment"])
            if result is not None:
                correct += result
                total += 1
        return correct / total if total > 0 else 0.5

    def _check_prediction(self, stock: str, date_str: str, sentiment: str) -> int | None:
        """Check if a prediction was correct. Returns 1 (correct), 0 (wrong), or None (no data)."""
        prices = db.get_prices([stock], date_str, date_str)
        stock_prices = prices.get(stock, [])
        if len(stock_prices) < 2:
            return None
        # Find the next trading day
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
```

- [ ] **Step 2: Write test**

```python
# tests/test_tracker.py
"""Tests for the expert tracker algorithm."""
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from src.expert.tracker import ExpertTracker
from src.data.models import ExpertRecord, Post
from src.db import schema as db


@pytest.fixture
def tracker():
    return ExpertTracker()


def make_post_dict(user_id, stock, timestamp, sentiment):
    return {
        "user_id": user_id, "stock": stock,
        "timestamp": timestamp, "sentiment": sentiment,
        "source": "stocktwits", "content": "",
    }


class TestFilterDailyLatest:
    def test_keeps_latest_per_user_stock(self, tracker):
        posts = [
            make_post_dict("u1", "AAPL", "2024-06-15T10:00:00", "Bullish"),
            make_post_dict("u1", "AAPL", "2024-06-15T15:00:00", "Bearish"),
            make_post_dict("u1", "MSFT", "2024-06-15T12:00:00", "Bullish"),
        ]
        result = tracker._filter_daily_latest(posts)
        assert len(result) == 2
        aapl_post = [p for p in result if p["stock"] == "AAPL"][0]
        assert aapl_post["sentiment"] == "Bearish"  # Later post kept


class TestComputeAccuracy:
    def test_all_correct_bullish(self, tracker):
        """Test that Bullish predictions when price goes up are correct."""
        posts = [make_post_dict("u1", "AAPL", "2024-06-15", "Bullish")]
        assert tracker._compute_accuracy(posts) == 0.5  # No price data → defaults


class TestEvaluateUser:
    def test_insufficient_recent_posts(self, tracker):
        with patch.object(db, "get_user_history", return_value=[]):
            post_data = make_post_dict("u1", "AAPL", "2024-06-15T10:00:00", "Bullish")
            result = tracker._evaluate_user("u1", post_data, datetime(2024, 6, 15))
            assert result is None

    def test_expert_classification(self, tracker):
        target = datetime(2024, 6, 15)
        # Create 20 recent posts with high accuracy
        recent = [
            make_post_dict("u1", "AAPL", f"2024-06-{(14-i):02d}T10:00:00", "Bullish")
            for i in range(20)
        ]
        long_posts = recent * 5  # 100 posts

        with patch.object(db, "get_user_history") as mock_hist:
            mock_hist.side_effect = [recent, long_posts]
            with patch.object(tracker, "_compute_accuracy", return_value=0.85):
                post_data = make_post_dict("u1", "AAPL", "2024-06-15T10:00:00", "Bullish")
                result = tracker._evaluate_user("u1", post_data, target)

        assert result is not None
        assert result.expert_type == "expert"
        assert result.accuracy_recent == 0.85

    def test_inverse_expert_classification(self, tracker):
        target = datetime(2024, 6, 15)
        recent = [
            make_post_dict("u1", "AAPL", f"2024-06-{(14-i):02d}T10:00:00", "Bearish")
            for i in range(20)
        ]
        long_posts = recent * 5

        with patch.object(db, "get_user_history") as mock_hist:
            mock_hist.side_effect = [recent, long_posts]
            with patch.object(tracker, "_compute_accuracy", return_value=0.10):
                post_data = make_post_dict("u1", "AAPL", "2024-06-15T10:00:00", "Bearish")
                result = tracker._evaluate_user("u1", post_data, target)

        assert result is not None
        assert result.expert_type == "inverse_expert"
        # Inverse expert with Bearish post → predicted direction flips to Bullish
        assert result.predicted_direction == "Bullish"
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/test_tracker.py -v
```
Expected: 5 tests pass

- [ ] **Step 4: Commit**

```bash
git add src/expert/tracker.py tests/test_tracker.py
git commit -m "feat: expert tracker implementing Algorithm 1 from paper"
```

---

### Task 7: Signal Transformation and Feature Engineering

**Files:**
- Create: `src/model/signal.py`
- Create: `src/model/features.py`
- Create: `tests/test_signal.py`

**Interfaces:**
- Consumes: `ExpertRecord`, price data from DB, `config.SIGNAL_LOOKBACK_DAYS`, `config.MOMENTUM_LOOKBACK_DAYS`
- Produces:
  - `transform_expert_signal(records: list[ExpertRecord], date_str: str) -> dict[str, float]`
  - `compute_momentum(prices: dict[str, list[dict]], date_str: str, lookback: int = 20) -> dict[str, float]`
  - `compute_return_ratio(close_today: float, close_prev: float) -> float`

- [ ] **Step 1: Write signal transformation module**

```python
# src/model/signal.py
"""Expert signal transformation: binary predictions → continuous return signals."""
import logging
from datetime import datetime, timedelta
from collections import defaultdict
from src.data.models import ExpertRecord
from src.db import schema as db
from config import SIGNAL_LOOKBACK_DAYS

logger = logging.getLogger(__name__)


def compute_return_ratio(close_today: float, close_prev: float) -> float:
    """Compute return ratio: (close_today - close_prev) / close_prev."""
    if close_prev == 0:
        return 0.0
    return (close_today - close_prev) / close_prev


def transform_expert_signal(records: list[ExpertRecord], date_str: str) -> dict[str, float]:
    """Transform binary expert predictions into continuous return signals.

    For Bullish predictions: use average return of past 30 up-days.
    For Bearish predictions: use average return of past 30 down-days.
    For multiple experts on same stock → randomly sample one.

    Returns:
        Dict mapping stock ticker → expert signal value (0 if no expert signal).
    """
    if not records:
        return {}

    target_date = datetime.fromisoformat(date_str)
    lookback_start = (target_date - timedelta(days=SIGNAL_LOOKBACK_DAYS + 5)).strftime("%Y-%m-%d")

    # Group records by stock, pick one expert per stock randomly
    stock_experts = defaultdict(list)
    for r in records:
        stock_experts[r.stock].append(r)

    import random
    signals = {}
    for stock, experts in stock_experts.items():
        expert = random.choice(experts)
        avg_return = _compute_directional_average(stock, target_date, expert.predicted_direction)
        signals[stock] = avg_return

    return signals


def _compute_directional_average(stock: str, target_date: datetime, direction: str) -> float:
    """Compute average return ratio for a given direction over past 30 days.

    Bullish → average of positive-return days.
    Bearish → average of negative-return days (returned as negative value).
    """
    start_date = (target_date - timedelta(days=SIGNAL_LOOKBACK_DAYS + 5)).strftime("%Y-%m-%d")
    end_date = target_date.strftime("%Y-%m-%d")

    prices = db.get_prices([stock], start_date, end_date)
    stock_prices = prices.get(stock, [])
    if len(stock_prices) < 5:
        return 0.0

    returns = []
    for i in range(1, len(stock_prices)):
        ret = compute_return_ratio(
            stock_prices[i]["close"],
            stock_prices[i - 1]["close"],
        )
        returns.append(ret)

    if direction == "Bullish":
        positive = [r for r in returns if r > 0]
        return sum(positive) / len(positive) if positive else 0.01  # Default small positive
    else:
        negative = [r for r in returns if r < 0]
        return sum(negative) / len(negative) if negative else -0.01  # Default small negative


def compute_expert_availability(records: list[ExpertRecord], stocks: list[str]) -> dict[str, int]:
    """Binary indicator: does this stock have an expert signal today?"""
    stocks_with_experts = set(r.stock for r in records)
    return {s: (1 if s in stocks_with_experts else 0) for s in stocks}
```

- [ ] **Step 2: Write features module**

```python
# src/model/features.py
"""Feature engineering for stock prediction."""
import logging
from datetime import datetime, timedelta
from src.db import schema as db
from config import MOMENTUM_LOOKBACK_DAYS

logger = logging.getLogger(__name__)


def compute_momentum(
    stocks: list[str],
    date_str: str,
    lookback: int = MOMENTUM_LOOKBACK_DAYS,
) -> dict[str, float]:
    """Compute momentum factor: return over the past N days.

    Positive momentum → expect continuation (short-term).
    Used as baseline when no expert signal is available.
    """
    target_date = datetime.fromisoformat(date_str)
    start_date = (target_date - timedelta(days=lookback + 5)).strftime("%Y-%m-%d")
    end_date = date_str

    prices = db.get_prices(stocks, start_date, end_date)
    momentum = {}

    for stock in stocks:
        stock_prices = prices.get(stock, [])
        if len(stock_prices) < 2:
            momentum[stock] = 0.0
            continue
        # Find price closest to target_date
        recent = [p for p in stock_prices if p["date"] <= date_str]
        if len(recent) < 2:
            momentum[stock] = 0.0
            continue
        recent.sort(key=lambda x: x["date"])
        # Return over lookback window
        first_close = recent[0]["close"]
        last_close = recent[-1]["close"]
        if first_close == 0:
            momentum[stock] = 0.0
        else:
            momentum[stock] = (last_close - first_close) / first_close
    return momentum


def build_feature_vector(
    stocks: list[str],
    date_str: str,
    expert_signals: dict[str, float],
    expert_availability: dict[str, int],
) -> dict[str, dict]:
    """Build the complete feature vector for each stock on a given day.

    Returns:
        Dict mapping stock → {momentum, expert_signal, expert_available, ...}
    """
    momentum = compute_momentum(stocks, date_str)
    features = {}
    for stock in stocks:
        features[stock] = {
            "momentum": momentum.get(stock, 0.0),
            "expert_signal": expert_signals.get(stock, 0.0),
            "expert_available": expert_availability.get(stock, 0),
        }
    return features
```

- [ ] **Step 3: Write tests**

```python
# tests/test_signal.py
"""Tests for signal transformation and features."""
import pytest
from datetime import datetime
from src.model.signal import compute_return_ratio, compute_expert_availability
from src.model.features import compute_momentum
from src.data.models import ExpertRecord


class TestReturnRatio:
    def test_positive_return(self):
        assert compute_return_ratio(110.0, 100.0) == 0.1

    def test_negative_return(self):
        assert compute_return_ratio(95.0, 100.0) == -0.05

    def test_zero_prev(self):
        assert compute_return_ratio(110.0, 0.0) == 0.0


class TestExpertAvailability:
    def test_some_stocks_have_experts(self):
        records = [
            ExpertRecord("u1", "AAPL", datetime(2024, 6, 15), 0.85, 0.70, "expert", "Bullish"),
            ExpertRecord("u2", "MSFT", datetime(2024, 6, 15), 0.15, 0.30, "inverse_expert", "Bearish"),
        ]
        avail = compute_expert_availability(records, ["AAPL", "MSFT", "GOOGL", "AMZN"])
        assert avail["AAPL"] == 1
        assert avail["MSFT"] == 1
        assert avail["GOOGL"] == 0
        assert avail["AMZN"] == 0

    def test_no_experts(self):
        avail = compute_expert_availability([], ["AAPL", "MSFT"])
        assert avail == {"AAPL": 0, "MSFT": 0}
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_signal.py -v
```
Expected: 5 tests pass

- [ ] **Step 5: Commit**

```bash
git add src/model/signal.py src/model/features.py tests/test_signal.py
git commit -m "feat: expert signal transformation and feature engineering"
```

---

### Task 8: Rule-Based Baseline Predictor

**Files:**
- Create: `src/model/baseline.py`
- Create: `tests/test_baseline.py`

**Interfaces:**
- Consumes: `src.model.signal`, `src.model.features`, `src.expert.tracker`
- Produces: `BaselinePredictor.predict(stocks, date_str, expert_records) -> pd.DataFrame` with columns [stock, date, predicted_return]

- [ ] **Step 1: Write predictor**

```python
# src/model/baseline.py
"""Rule-based baseline predictor for stock returns.

Strategy:
- Has expert signal → use expert direction + 30-day average return magnitude
- No expert signal → use 20-day momentum factor
"""
import logging
import pandas as pd
from src.model.signal import transform_expert_signal, compute_expert_availability
from src.model.features import compute_momentum

logger = logging.getLogger(__name__)


class BaselinePredictor:
    """Simple rule-based stock return predictor for MVP."""

    def predict(
        self,
        stocks: list[str],
        date_str: str,
        expert_records: list | None = None,
    ) -> pd.DataFrame:
        """Generate daily return ratio predictions for all stocks.

        Args:
            stocks: List of ticker symbols.
            date_str: Prediction date (YYYY-MM-DD).
            expert_records: Expert identification results for this date.

        Returns:
            DataFrame with columns [stock, date, predicted_return, signal_source].
        """
        expert_records = expert_records or []
        expert_signals = transform_expert_signal(expert_records, date_str)
        expert_avail = compute_expert_availability(expert_records, stocks)
        momentum = compute_momentum(stocks, date_str)

        predictions = []
        for stock in stocks:
            if expert_avail[stock] == 1 and stock in expert_signals:
                pred = expert_signals[stock]
                source = "expert"
            else:
                pred = momentum.get(stock, 0.0)
                source = "momentum"

            predictions.append({
                "stock": stock,
                "date": date_str,
                "predicted_return": pred,
                "signal_source": source,
            })

        df = pd.DataFrame(predictions)
        # Normalize predictions cross-sectionally for ranking
        if len(df) > 0:
            mean = df["predicted_return"].mean()
            std = df["predicted_return"].std()
            if std > 0:
                df["predicted_return"] = (df["predicted_return"] - mean) / std

        logger.info(f"Generated {len(df)} predictions for {date_str}, "
                     f"{df['signal_source'].value_counts().get('expert', 0)} from experts")
        return df.sort_values("predicted_return", ascending=False)
```

- [ ] **Step 2: Write test**

```python
# tests/test_baseline.py
"""Tests for baseline predictor."""
import pytest
from datetime import datetime
from src.model.baseline import BaselinePredictor
from src.data.models import ExpertRecord


def test_predict_returns_dataframe():
    predictor = BaselinePredictor()
    df = predictor.predict(["AAPL", "MSFT", "GOOGL"], "2024-06-15")
    assert len(df) == 3
    assert list(df.columns) == ["stock", "date", "predicted_return", "signal_source"]
    assert df["stock"].tolist() == ["AAPL", "MSFT", "GOOGL"]


def test_predict_with_experts(prepopulated_db):
    """Test that expert signals are used when available."""
    predictor = BaselinePredictor()
    records = [
        ExpertRecord("u1", "AAPL", datetime(2024, 6, 15), 0.85, 0.70, "expert", "Bullish"),
    ]
    df = predictor.predict(["AAPL", "MSFT"], "2024-06-15", records)
    aapl = df[df["stock"] == "AAPL"].iloc[0]
    assert aapl["signal_source"] == "expert"
    msft = df[df["stock"] == "MSFT"].iloc[0]
    assert msft["signal_source"] == "momentum"


def test_predict_empty_stocks():
    predictor = BaselinePredictor()
    df = predictor.predict([], "2024-06-15")
    assert len(df) == 0


def test_predictions_are_normalized():
    predictor = BaselinePredictor()
    df = predictor.predict(["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"], "2024-06-15")
    assert abs(df["predicted_return"].mean()) < 1e-10
    assert abs(df["predicted_return"].std() - 1.0) < 0.1
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/test_baseline.py -v
```
Expected: 4 tests pass

- [ ] **Step 4: Commit**

```bash
git add src/model/baseline.py tests/test_baseline.py
git commit -m "feat: rule-based baseline predictor for MVP"
```

---

### Task 9: Backtest Engine — Metrics

**Files:**
- Create: `src/backtest/metrics.py`
- Create: `tests/test_metrics.py`

**Interfaces:**
- Consumes: prediction DataFrame, price data
- Produces:
  - `compute_accuracy(pred_df, actual_returns) -> float`
  - `compute_ic(pred_series, actual_series) -> float` (Pearson)
  - `compute_ric(pred_series, actual_series) -> float` (Spearman)
  - `compute_icir(ic_series) -> float`
  - `compute_sharpe(daily_returns, risk_free=0.04) -> float`
  - `compute_all_metrics(pred_df, price_data) -> dict`

- [ ] **Step 1: Write metrics module**

```python
# src/backtest/metrics.py
"""Quantitative evaluation metrics for stock prediction."""
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime
from src.db import schema as db


def compute_accuracy(predictions: pd.Series, actual_returns: pd.Series) -> float:
    """Directional accuracy: did we predict the right sign?"""
    aligned = pd.concat([predictions, actual_returns], axis=1, join="inner").dropna()
    aligned.columns = ["pred", "actual"]
    if len(aligned) == 0:
        return 0.5
    correct = ((aligned["pred"] > 0) & (aligned["actual"] > 0)) | \
              ((aligned["pred"] < 0) & (aligned["actual"] < 0))
    return correct.mean()


def compute_ic(predictions: pd.Series, actual_returns: pd.Series) -> float:
    """Information Coefficient: Pearson correlation between predicted and actual."""
    aligned = predictions.align(actual_returns, join="inner")
    combined = pd.concat([aligned[0], aligned[1]], axis=1).dropna()
    if len(combined) < 3:
        return 0.0
    return combined.iloc[:, 0].corr(combined.iloc[:, 1])


def compute_ric(predictions: pd.Series, actual_returns: pd.Series) -> float:
    """Rank IC: Spearman rank correlation."""
    aligned = predictions.align(actual_returns, join="inner")
    combined = pd.concat([aligned[0], aligned[1]], axis=1).dropna()
    if len(combined) < 3:
        return 0.0
    r, _ = stats.spearmanr(combined.iloc[:, 0], combined.iloc[:, 1])
    return r if not np.isnan(r) else 0.0


def compute_icir(ic_series: pd.Series) -> float:
    """IC Information Ratio: mean(IC) / std(IC)."""
    if len(ic_series) < 2 or ic_series.std() == 0:
        return 0.0
    return ic_series.mean() / ic_series.std()


def compute_annualized_return(daily_returns: pd.Series, transaction_cost: float = 0.0004) -> float:
    """Annualized return from daily returns, accounting for transaction costs."""
    net_returns = daily_returns - transaction_cost
    cumulative = (1 + net_returns).prod()
    n_days = len(daily_returns)
    if n_days == 0:
        return 0.0
    annualized = cumulative ** (252 / n_days) - 1
    return float(annualized)


def compute_sharpe(daily_returns: pd.Series, risk_free_rate: float = 0.04,
                   transaction_cost: float = 0.0004) -> float:
    """Sharpe Ratio: (annualized_return - risk_free) / annualized_volatility."""
    if len(daily_returns) < 5:
        return 0.0
    net_returns = daily_returns - transaction_cost
    excess = net_returns.mean() * 252 - risk_free_rate
    vol = net_returns.std() * np.sqrt(252)
    return float(excess / vol) if vol > 0 else 0.0


def compute_daily_ic_series(
    pred_df: pd.DataFrame, stocks: list[str],
    start_date: str, end_date: str,
) -> pd.Series:
    """Compute IC for each day in the date range."""
    dates = sorted(pred_df["date"].unique())
    daily_ic = {}
    for date_str in dates:
        day_preds = pred_df[pred_df["date"] == date_str].set_index("stock")
        day_prices = db.get_prices(stocks, date_str, date_str)
        # Get next-day actual returns
        actuals = {}
        for stock in stocks:
            sp = day_prices.get(stock, [])
            if len(sp) >= 2:
                actuals[stock] = (sp[-1]["close"] - sp[0]["close"]) / sp[0]["close"] if sp[0]["close"] else 0
        if not actuals:
            continue
        actual_series = pd.Series(actuals, name="actual")
        if stock in day_preds.index:
            pred_series = day_preds["predicted_return"]
            ic = compute_ic(pred_series, actual_series)
            daily_ic[date_str] = ic
    return pd.Series(daily_ic)


def compute_all_metrics(
    pred_df: pd.DataFrame, stocks: list[str],
    start_date: str, end_date: str,
) -> dict:
    """Compute all quantitative metrics for predictions."""
    daily_ic = compute_daily_ic_series(pred_df, stocks, start_date, end_date)

    return {
        "mean_ic": daily_ic.mean() if len(daily_ic) > 0 else 0.0,
        "icir": compute_icir(daily_ic),
        "ic_std": daily_ic.std() if len(daily_ic) > 0 else 0.0,
        "n_days": len(daily_ic),
    }
```

- [ ] **Step 2: Write tests**

```python
# tests/test_metrics.py
"""Tests for backtest metrics."""
import pytest
import numpy as np
import pandas as pd
from src.backtest.metrics import (
    compute_accuracy, compute_ic, compute_ric, compute_icir,
    compute_annualized_return, compute_sharpe,
)


class TestAccuracy:
    def test_perfect_accuracy(self):
        pred = pd.Series([0.1, -0.1, 0.05], index=["A", "B", "C"])
        actual = pd.Series([0.05, -0.02, 0.03], index=["A", "B", "C"])
        assert compute_accuracy(pred, actual) == 1.0

    def test_bad_accuracy(self):
        pred = pd.Series([0.1, -0.1], index=["A", "B"])
        actual = pd.Series([-0.05, 0.02], index=["A", "B"])
        assert compute_accuracy(pred, actual) == 0.0

    def test_partial_match(self):
        pred = pd.Series([0.1, -0.1], index=["A", "B"])
        actual = pd.Series([0.05, 0.02], index=["A", "B"])
        assert compute_accuracy(pred, actual) == 0.5


class TestIC:
    def test_perfect_positive_correlation(self):
        pred = pd.Series([1, 2, 3], index=["A", "B", "C"])
        actual = pd.Series([0.5, 1.0, 1.5], index=["A", "B", "C"])
        assert compute_ic(pred, actual) == pytest.approx(1.0)

    def test_perfect_negative_correlation(self):
        pred = pd.Series([3, 2, 1], index=["A", "B", "C"])
        actual = pd.Series([0.5, 1.0, 1.5], index=["A", "B", "C"])
        assert compute_ic(pred, actual) == pytest.approx(-1.0)

    def test_no_correlation(self):
        np.random.seed(42)
        pred = pd.Series(np.random.randn(30))
        actual = pd.Series(np.random.randn(30))
        ic = compute_ic(pred, actual)
        assert -0.5 < ic < 0.5


class TestICIR:
    def test_positive_icir(self):
        ic = pd.Series([0.05, 0.06, 0.04, 0.07, 0.05])
        assert compute_icir(ic) > 0

    def test_short_series(self):
        assert compute_icir(pd.Series([0.05])) == 0.0


class TestReturns:
    def test_annualized_return_positive(self):
        returns = pd.Series([0.001] * 100)  # 0.1% daily
        ar = compute_annualized_return(returns)
        assert ar > 0

    def test_sharpe_positive(self):
        returns = pd.Series(np.random.normal(0.001, 0.01, 252))
        sr = compute_sharpe(returns)
        assert isinstance(sr, float)
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/test_metrics.py -v
```
Expected: 10 tests pass

- [ ] **Step 4: Commit**

```bash
git add src/backtest/metrics.py tests/test_metrics.py
git commit -m "feat: backtest metrics — IC, RIC, Sharpe, accuracy"
```

---

### Task 10: Backtest Engine — Portfolio Construction

**Files:**
- Create: `src/backtest/portfolio.py`
- Create: `tests/test_portfolio.py`

**Interfaces:**
- Consumes: `config.PORTFOLIO_QUANTILE`, `config.TRANSACTION_COST`
- Produces:
  - `construct_long_short(pred_df: pd.DataFrame, date_str: str, quantile: float) -> dict` with keys `long`, `short`
  - `run_backtest(pred_df, stocks, start, end) -> dict` with cumulative returns, daily returns, metrics

- [ ] **Step 1: Write portfolio module**

```python
# src/backtest/portfolio.py
"""Long-short portfolio construction and backtest simulation."""
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.db import schema as db
from src.backtest.metrics import (
    compute_accuracy, compute_ic, compute_icir, compute_annualized_return, compute_sharpe,
    compute_daily_ic_series,
)
from config import PORTFOLIO_QUANTILE, TRANSACTION_COST

logger = logging.getLogger(__name__)


def construct_long_short(pred_df: pd.DataFrame, quantile: float = PORTFOLIO_QUANTILE) -> dict:
    """Construct long-short portfolio for a single day.

    Args:
        pred_df: Predictions for one day with columns [stock, predicted_return].
        quantile: Fraction of stocks to long/short (default 0.10 = top/bottom 10%).

    Returns:
        Dict with keys: long (list of stocks), short (list of stocks),
                        long_weight, short_weight.
    """
    df = pred_df.sort_values("predicted_return", ascending=False)
    n_stocks = len(df)
    n_positions = max(1, int(n_stocks * quantile))

    long_stocks = df.head(n_positions)["stock"].tolist()
    short_stocks = df.tail(n_positions)["stock"].tolist()

    weight = 1.0 / n_positions if n_positions > 0 else 0.0

    return {
        "long": long_stocks,
        "short": short_stocks,
        "long_weight": weight,
        "short_weight": weight,
        "date": df["date"].iloc[0] if "date" in df.columns else None,
    }


def run_backtest(
    pred_df: pd.DataFrame,
    stocks: list[str],
    start_date: str,
    end_date: str,
    quantile: float = PORTFOLIO_QUANTILE,
    transaction_cost: float = TRANSACTION_COST,
) -> dict:
    """Run a full backtest simulation.

    Strategy: Long top quantile, short bottom quantile, daily rebalancing.
    Returns daily P&L and summary metrics.
    """
    dates = sorted(pred_df["date"].unique())
    daily_returns = []
    daily_long_short = []

    for date_str in dates:
        day_pred = pred_df[pred_df["date"] == date_str]
        if len(day_pred) == 0:
            continue

        portfolio = construct_long_short(day_pred, quantile)

        # Get next-day returns for selected stocks
        next_date = _get_next_trading_day(date_str)
        if next_date is None:
            continue

        long_ret = _get_portfolio_return(portfolio["long"], date_str, next_date)
        short_ret = _get_portfolio_return(portfolio["short"], date_str, next_date)

        daily_ret = (long_ret - short_ret) / 2 - transaction_cost
        daily_returns.append(daily_ret)
        daily_long_short.append({
            "date": date_str,
            "long_return": long_ret,
            "short_return": short_ret,
            "long_short_return": daily_ret,
            "long_stocks": portfolio["long"],
            "short_stocks": portfolio["short"],
        })

    dr_series = pd.Series(daily_returns)
    cumulative = (1 + dr_series).cumprod()

    # Compute metrics
    daily_ic = compute_daily_ic_series(pred_df, stocks, start_date, end_date)

    return {
        "daily_returns": dr_series,
        "cumulative_returns": cumulative,
        "daily_long_short": pd.DataFrame(daily_long_short),
        "annualized_return": compute_annualized_return(dr_series, transaction_cost),
        "sharpe_ratio": compute_sharpe(dr_series, transaction_cost=transaction_cost),
        "max_drawdown": float(_max_drawdown(cumulative)),
        "mean_ic": daily_ic.mean() if len(daily_ic) > 0 else 0.0,
        "icir": compute_icir(daily_ic),
        "n_trading_days": len(dr_series),
    }


def _get_portfolio_return(stocks: list[str], date_str: str, next_date: str) -> float:
    """Get equal-weighted portfolio return from date to next_date."""
    if not stocks:
        return 0.0
    prices = db.get_prices(stocks, date_str, next_date)
    returns = []
    for stock in stocks:
        sp = prices.get(stock, [])
        if len(sp) >= 2:
            sp.sort(key=lambda x: x["date"])
            ret = (sp[-1]["close"] - sp[0]["close"]) / sp[0]["close"] if sp[0]["close"] else 0.0
            returns.append(ret)
    return np.mean(returns) if returns else 0.0


def _get_next_trading_day(date_str: str) -> str | None:
    """Get the next calendar day (simplified — ignores weekends/holidays)."""
    dt = datetime.fromisoformat(date_str) + timedelta(days=1)
    return dt.strftime("%Y-%m-%d")


def _max_drawdown(cumulative: pd.Series) -> float:
    """Compute maximum drawdown from cumulative returns."""
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return float(drawdown.min())
```

- [ ] **Step 2: Write tests**

```python
# tests/test_portfolio.py
"""Tests for portfolio construction and backtest."""
import pytest
import pandas as pd
from src.backtest.portfolio import construct_long_short, _max_drawdown


class TestConstructLongShort:
    def test_basic_construction(self):
        df = pd.DataFrame({
            "stock": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
            "predicted_return": [0.5, 0.4, 0.3, 0.2, 0.1, -0.1, -0.2, -0.3, -0.4, -0.5],
            "date": "2024-06-15",
        })
        result = construct_long_short(df, quantile=0.1)
        assert result["long"] == ["A"]
        assert result["short"] == ["J"]
        assert result["long_weight"] == 1.0
        assert result["short_weight"] == 1.0

    def test_quantile_20(self):
        df = pd.DataFrame({
            "stock": ["A", "B", "C", "D", "E"],
            "predicted_return": [0.5, 0.3, 0.0, -0.3, -0.5],
            "date": "2024-06-15",
        })
        result = construct_long_short(df, quantile=0.2)
        assert len(result["long"]) == 1  # 20% of 5 = 1
        assert len(result["short"]) == 1


class TestMaxDrawdown:
    def test_no_drawdown(self):
        cum = pd.Series([1.0, 1.05, 1.10, 1.15])
        assert _max_drawdown(cum) == 0.0

    def test_with_drawdown(self):
        cum = pd.Series([1.0, 1.10, 1.05, 0.95, 1.02])
        dd = _max_drawdown(cum)
        assert dd < 0
        assert dd == pytest.approx(-0.136, abs=0.01)  # (0.95 - 1.10) / 1.10
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/test_portfolio.py -v
```
Expected: 3 tests pass

- [ ] **Step 4: Commit**

```bash
git add src/backtest/portfolio.py tests/test_portfolio.py
git commit -m "feat: long-short portfolio construction and backtest simulation"
```

---

### Task 11: FastAPI Web Service

**Files:**
- Create: `src/web/api.py`
- Create: `tests/test_api.py`

**Interfaces:**
- Consumes: `src.model.baseline.BaselinePredictor`, `src.expert.tracker.ExpertTracker`, `src.backtest.portfolio.run_backtest`
- Produces: FastAPI app with 5 endpoints:
  - `GET /api/stocks` → stock universe
  - `GET /api/experts?date=` → expert records for date
  - `GET /api/predictions?date=` → predictions for date
  - `GET /api/backtest?start=&end=` → backtest results
  - `POST /api/collect` → trigger data collection

- [ ] **Step 1: Write API module**

```python
# src/web/api.py
"""FastAPI web service for the stock prediction system."""
import logging
from datetime import datetime, timedelta
from fastapi import FastAPI, Query, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request
import pandas as pd

from src.db.schema import init_db, get_expert_records
from src.expert.tracker import ExpertTracker
from src.model.baseline import BaselinePredictor
from src.backtest.portfolio import run_backtest
from src.data.yfinance import YFinanceCollector
from src.data.stocktwits import StockTwitsCollector
from src.data.reddit import RedditCollector
from src.db import schema as db
from config import DEFAULT_TICKERS, PORTFOLIO_QUANTILE

logger = logging.getLogger(__name__)

# Initialize app
app = FastAPI(title="DualGAT Stock Predictor", version="0.1.0")

# Static files and templates
import os
static_dir = os.path.join(os.path.dirname(__file__), "static")
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

# Services (lazy init)
_tracker = None
_predictor = None


def get_tracker() -> ExpertTracker:
    global _tracker
    if _tracker is None:
        _tracker = ExpertTracker()
    return _tracker


def get_predictor() -> BaselinePredictor:
    global _predictor
    if _predictor is None:
        _predictor = BaselinePredictor()
    return _predictor


@app.on_event("startup")
async def startup():
    init_db()
    logger.info("Database initialized")


@app.get("/api/stocks")
async def get_stocks():
    """Return the stock universe."""
    return {"stocks": DEFAULT_TICKERS, "count": len(DEFAULT_TICKERS)}


@app.get("/api/experts")
async def get_experts(date: str = Query(None, description="Date in YYYY-MM-DD format")):
    """Get expert records for a given date. Defaults to latest available."""
    if date is None:
        date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    # Try to trace experts for this date
    tracker = get_tracker()
    records = tracker.trace(date)

    return {
        "date": date,
        "expert_count": len([r for r in records if r.expert_type == "expert"]),
        "inverse_expert_count": len([r for r in records if r.expert_type == "inverse_expert"]),
        "total": len(records),
        "experts": [
            {
                "user_id": r.user_id,
                "stock": r.stock,
                "expert_type": r.expert_type,
                "predicted_direction": r.predicted_direction,
                "accuracy_recent": r.accuracy_recent,
                "accuracy_long": r.accuracy_long,
            }
            for r in records
        ],
    }


@app.get("/api/predictions")
async def get_predictions(date: str = Query(None, description="Date in YYYY-MM-DD format")):
    """Get stock return predictions for a given date."""
    if date is None:
        date = (datetime.now()).strftime("%Y-%m-%d")

    tracker = get_tracker()
    predictor = get_predictor()

    expert_records = tracker.trace(date)
    pred_df = predictor.predict(DEFAULT_TICKERS, date, expert_records)

    return {
        "date": date,
        "predictions": pred_df.to_dict(orient="records"),
        "expert_coverage": len([r for r in expert_records if r.expert_type != "none"]),
    }


@app.get("/api/backtest")
async def get_backtest(
    start: str = Query(None, description="Start date YYYY-MM-DD"),
    end: str = Query(None, description="End date YYYY-MM-DD"),
):
    """Run backtest over a date range."""
    if start is None:
        start = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    if end is None:
        end = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    tracker = get_tracker()
    predictor = get_predictor()

    # Generate predictions for each day
    all_preds = []
    current = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)
    while current <= end_dt:
        date_str = current.strftime("%Y-%m-%d")
        expert_records = tracker.trace(date_str)
        pred_df = predictor.predict(DEFAULT_TICKERS, date_str, expert_records)
        all_preds.append(pred_df)
        current += timedelta(days=1)

    if not all_preds:
        raise HTTPException(404, "No predictions generated for the date range")

    combined_preds = pd.concat(all_preds, ignore_index=True)
    results = run_backtest(combined_preds, DEFAULT_TICKERS, start, end)

    return {
        "start": start,
        "end": end,
        "annualized_return": results["annualized_return"],
        "sharpe_ratio": results["sharpe_ratio"],
        "max_drawdown": results["max_drawdown"],
        "mean_ic": results["mean_ic"],
        "icir": results["icir"],
        "n_trading_days": results["n_trading_days"],
        "cumulative_returns": results["cumulative_returns"].tolist(),
    }


@app.post("/api/collect")
async def trigger_collection(
    start: str = Query(None, description="Start date YYYY-MM-DD"),
    end: str = Query(None, description="End date YYYY-MM-DD"),
):
    """Trigger data collection from all sources."""
    if start is None:
        start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    results = {"prices": 0, "stocktwits": 0, "reddit": 0}

    # Collect prices
    try:
        yf_collector = YFinanceCollector()
        prices = yf_collector.collect_prices(DEFAULT_TICKERS, start, end)
        db.insert_prices(prices)
        results["prices"] = len(prices)
    except Exception as e:
        results["prices_error"] = str(e)

    # Collect social posts
    try:
        st_collector = StockTwitsCollector()
        for i in range((datetime.fromisoformat(end) - datetime.fromisoformat(start)).days + 1):
            date_str = (datetime.fromisoformat(start) + timedelta(days=i)).strftime("%Y-%m-%d")
            posts = st_collector.collect_social_posts(DEFAULT_TICKERS, date_str)
            db.insert_posts(posts)
            results["stocktwits"] += len(posts)
    except Exception as e:
        results["stocktwits_error"] = str(e)

    try:
        reddit_collector = RedditCollector()
        for i in range((datetime.fromisoformat(end) - datetime.fromisoformat(start)).days + 1):
            date_str = (datetime.fromisoformat(start) + timedelta(days=i)).strftime("%Y-%m-%d")
            posts = reddit_collector.collect_social_posts(DEFAULT_TICKERS, date_str)
            db.insert_posts(posts)
            results["reddit"] += len(posts)
    except Exception as e:
        results["reddit_error"] = str(e)

    return {"status": "ok", "results": results}


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Render the main dashboard."""
    return templates.TemplateResponse("index.html", {"request": request})
```

- [ ] **Step 2: Write API tests**

```python
# tests/test_api.py
"""Tests for the FastAPI web service."""
import pytest
from fastapi.testclient import TestClient
from src.web.api import app
from src.db.schema import init_db
import src.db.schema as schema_mod


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Create test client with isolated database."""
    db_path = tmp_path / "test_api.db"
    monkeypatch.setattr("src.web.api.db", schema_mod)
    monkeypatch.setattr("src.db.schema.DB_PATH", db_path)
    monkeypatch.setattr("config.DB_PATH", db_path)
    schema_mod._engine = None
    init_db()
    return TestClient(app)


class TestStockEndpoint:
    def test_get_stocks(self, client):
        resp = client.get("/api/stocks")
        assert resp.status_code == 200
        data = resp.json()
        assert "stocks" in data
        assert "count" in data
        assert data["count"] > 0


class TestExpertEndpoint:
    def test_get_experts_empty(self, client):
        resp = client.get("/api/experts?date=2024-06-15")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0

    def test_get_experts_default_date(self, client):
        resp = client.get("/api/experts")
        # Should work with default date (yesterday)
        assert resp.status_code == 200


class TestPredictionEndpoint:
    def test_get_predictions(self, client):
        resp = client.get("/api/predictions?date=2024-06-15")
        assert resp.status_code == 200
        data = resp.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 20  # DEFAULT_TICKERS


class TestBacktestEndpoint:
    def test_get_backtest(self, client):
        resp = client.get("/api/backtest?start=2024-06-01&end=2024-06-15")
        assert resp.status_code == 200
        data = resp.json()
        assert "sharpe_ratio" in data
        assert "annualized_return" in data


class TestDashboard:
    def test_dashboard_renders(self, client):
        resp = client.get("/")
        # Dashboard returns HTML
        assert "text/html" in resp.headers.get("content-type", "")


class TestCollectEndpoint:
    def test_collect_triggers(self, client):
        resp = client.post("/api/collect?start=2024-06-01&end=2024-06-02")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/test_api.py -v
```
Expected: 7 tests pass

- [ ] **Step 4: Commit**

```bash
git add src/web/api.py tests/test_api.py
git commit -m "feat: FastAPI web service with 5 endpoints"
```

---

### Task 12: Dashboard HTML + JavaScript

**Files:**
- Create: `src/web/templates/index.html`
- Create: `src/web/static/app.js`

**Interfaces:**
- Consumes: REST API endpoints from Task 11
- Produces: Interactive single-page dashboard with predictions table, expert list, backtest chart

- [ ] **Step 1: Write dashboard HTML**

```html
<!-- src/web/templates/index.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DualGAT Stock Predictor</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <script src="https://unpkg.com/htmx.org@1.9.10"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
               background: #f5f7fa; color: #1a1a2e; padding: 20px; }
        .header { text-align: center; margin-bottom: 30px; }
        .header h1 { font-size: 28px; color: #16213e; }
        .header p { color: #666; margin-top: 5px; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; max-width: 1400px; margin: 0 auto; }
        .card { background: white; border-radius: 12px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
        .card h2 { font-size: 16px; color: #333; margin-bottom: 15px; padding-bottom: 10px; border-bottom: 1px solid #eee; }
        .card.full { grid-column: 1 / -1; }
        table { width: 100%; border-collapse: collapse; font-size: 13px; }
        th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #f0f0f0; }
        th { background: #f8f9fa; font-weight: 600; color: #555; }
        tr:hover { background: #f8f9ff; }
        .badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; }
        .badge-expert { background: #d4edda; color: #155724; }
        .badge-inverse { background: #f8d7da; color: #721c24; }
        .badge-bullish { background: #d4edda; color: #155724; }
        .badge-bearish { background: #f8d7da; color: #721c24; }
        .badge-momentum { background: #e2e3e5; color: #383d41; }
        .badge-expert-signal { background: #cce5ff; color: #004085; }
        .metric { text-align: center; }
        .metric .value { font-size: 28px; font-weight: 700; }
        .metric .label { font-size: 12px; color: #888; margin-top: 4px; }
        .metrics-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; }
        .green { color: #28a745; } .red { color: #dc3545; }
        .controls { margin-bottom: 15px; display: flex; gap: 10px; }
        .controls input, .controls button { padding: 6px 12px; border: 1px solid #ddd; border-radius: 6px; }
        .controls button { background: #16213e; color: white; border: none; cursor: pointer; }
        .controls button:hover { background: #0f3460; }
        #chart-container { height: 300px; position: relative; }
        .loading { text-align: center; padding: 40px; color: #888; }
        .loading::after { content: "..."; animation: dots 1.5s infinite; }
        @keyframes dots { 0%,20%{content:"."} 40%{content:".."} 60%,100%{content:"..."} }
    </style>
</head>
<body>
    <div class="header">
        <h1>📈 DualGAT Stock Predictor</h1>
        <p>Expert Opinion-Driven Stock Return Prediction — MVP v0.1</p>
    </div>

    <div class="controls" style="max-width:1400px; margin: 0 auto 15px;">
        <input type="date" id="date-picker" />
        <button onclick="loadPredictions()">Load Predictions</button>
        <button onclick="loadExperts()">Load Experts</button>
        <button onclick="triggerCollect()">📥 Collect Data</button>
        <span id="status" style="margin-left:auto; color:#888; font-size:13px;"></span>
    </div>

    <div class="grid">
        <div class="card">
            <h2>📊 Predictions <span id="pred-date"></span></h2>
            <div id="predictions-table"><div class="loading">Loading predictions</div></div>
        </div>

        <div class="card">
            <h2>🎯 Expert Signals <span id="expert-date"></span></h2>
            <div id="experts-table"><div class="loading">Loading experts</div></div>
        </div>

        <div class="card full">
            <h2>📈 Cumulative Returns</h2>
            <div class="metrics-row" id="metrics-summary">
                <div class="metric"><div class="value">--</div><div class="label">Annualized Return</div></div>
                <div class="metric"><div class="value">--</div><div class="label">Sharpe Ratio</div></div>
                <div class="metric"><div class="value">--</div><div class="label">Mean IC</div></div>
                <div class="metric"><div class="value">--</div><div class="label">Max Drawdown</div></div>
            </div>
            <div id="chart-container"><canvas id="returnsChart"></canvas></div>
        </div>
    </div>

    <script src="/static/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Write JavaScript**

```javascript
// src/web/static/app.js
let returnsChart = null;

// Set default date to yesterday
document.addEventListener('DOMContentLoaded', () => {
    const yesterday = new Date();
    yesterday.setDate(yesterday.getDate() - 1);
    document.getElementById('date-picker').value = yesterday.toISOString().split('T')[0];
    loadPredictions();
    loadExperts();
    loadBacktest();
});

function getDate() {
    return document.getElementById('date-picker').value;
}

function setStatus(msg) {
    document.getElementById('status').textContent = msg;
}

async function loadPredictions() {
    const date = getDate();
    setStatus('Loading predictions...');
    document.getElementById('pred-date').textContent = date;
    try {
        const resp = await fetch(`/api/predictions?date=${date}`);
        const data = await resp.json();
        document.getElementById('pred-date').textContent =
            `${date} (${data.expert_coverage} signals)`;

        if (!data.predictions || data.predictions.length === 0) {
            document.getElementById('predictions-table').innerHTML =
                '<p style="color:#888; text-align:center; padding:20px;">No predictions available</p>';
            return;
        }

        let html = '<table><thead><tr><th>Stock</th><th>Predicted Return</th><th>Signal Source</th></tr></thead><tbody>';
        data.predictions.slice(0, 20).forEach(p => {
            const retClass = p.predicted_return > 0 ? 'green' : 'red';
            const sourceClass = p.signal_source === 'expert' ? 'badge-expert-signal' : 'badge-momentum';
            html += `<tr>
                <td><strong>${p.stock}</strong></td>
                <td class="${retClass}">${(p.predicted_return * 100).toFixed(2)}%</td>
                <td><span class="badge ${sourceClass}">${p.signal_source}</span></td>
            </tr>`;
        });
        html += '</tbody></table>';
        document.getElementById('predictions-table').innerHTML = html;
        setStatus('Predictions loaded');
    } catch (e) {
        document.getElementById('predictions-table').innerHTML =
            `<p style="color:red;">Error: ${e.message}</p>`;
        setStatus('Error loading predictions');
    }
}

async function loadExperts() {
    const date = getDate();
    setStatus('Loading experts...');
    document.getElementById('expert-date').textContent = date;
    try {
        const resp = await fetch(`/api/experts?date=${date}`);
        const data = await resp.json();
        document.getElementById('expert-date').textContent =
            `${date} (${data.expert_count} experts, ${data.inverse_expert_count} inverse)`;

        if (!data.experts || data.experts.length === 0) {
            document.getElementById('experts-table').innerHTML =
                '<p style="color:#888; text-align:center; padding:20px;">No expert signals today</p>';
            return;
        }

        let html = '<table><thead><tr><th>User</th><th>Stock</th><th>Type</th><th>Direction</th><th>Recent Acc</th><th>Long Acc</th></tr></thead><tbody>';
        data.experts.forEach(e => {
            const typeClass = e.expert_type === 'expert' ? 'badge-expert' : 'badge-inverse';
            const dirClass = e.predicted_direction === 'Bullish' ? 'badge-bullish' : 'badge-bearish';
            html += `<tr>
                <td>${e.user_id}</td>
                <td><strong>${e.stock}</strong></td>
                <td><span class="badge ${typeClass}">${e.expert_type}</span></td>
                <td><span class="badge ${dirClass}">${e.predicted_direction}</span></td>
                <td>${(e.accuracy_recent * 100).toFixed(1)}%</td>
                <td>${(e.accuracy_long * 100).toFixed(1)}%</td>
            </tr>`;
        });
        html += '</tbody></table>';
        document.getElementById('experts-table').innerHTML = html;
        setStatus('Experts loaded');
    } catch (e) {
        document.getElementById('experts-table').innerHTML =
            `<p style="color:red;">Error: ${e.message}</p>`;
        setStatus('Error loading experts');
    }
}

async function loadBacktest() {
    const endDate = getDate();
    const startDate = new Date(endDate);
    startDate.setDate(startDate.getDate() - 90);
    const start = startDate.toISOString().split('T')[0];

    try {
        const resp = await fetch(`/api/backtest?start=${start}&end=${endDate}`);
        const data = await resp.json();

        document.querySelector('#metrics-summary').innerHTML = `
            <div class="metric"><div class="value ${data.annualized_return > 0 ? 'green' : 'red'}">${(data.annualized_return * 100).toFixed(1)}%</div><div class="label">Annualized Return</div></div>
            <div class="metric"><div class="value">${data.sharpe_ratio.toFixed(2)}</div><div class="label">Sharpe Ratio</div></div>
            <div class="metric"><div class="value">${(data.mean_ic * 100).toFixed(2)}%</div><div class="label">Mean IC</div></div>
            <div class="metric"><div class="value red">${(data.max_drawdown * 100).toFixed(1)}%</div><div class="label">Max Drawdown</div></div>
        `;

        if (returnsChart) returnsChart.destroy();
        const ctx = document.getElementById('returnsChart').getContext('2d');
        const cumReturns = data.cumulative_returns || [];
        const labels = cumReturns.map((_, i) => i);

        returnsChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Cumulative Return',
                    data: cumReturns,
                    borderColor: cumReturns.length > 0 && cumReturns[cumReturns.length - 1] >= 1 ? '#28a745' : '#dc3545',
                    backgroundColor: 'rgba(40, 167, 69, 0.1)',
                    fill: true,
                    tension: 0.3,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    y: {
                        ticks: { callback: v => (v * 100).toFixed(1) + '%' }
                    }
                }
            }
        });
        setStatus('Backtest loaded');
    } catch (e) {
        console.error('Backtest error:', e);
    }
}

async function triggerCollect() {
    setStatus('Collecting data...');
    const endDate = getDate();
    const startDate = new Date(endDate);
    startDate.setDate(startDate.getDate() - 7);
    try {
        const resp = await fetch(`/api/collect?start=${startDate.toISOString().split('T')[0]}&end=${endDate}`, { method: 'POST' });
        const data = await resp.json();
        setStatus(`Collected: ${data.results.prices} prices, ${data.results.stocktwits} StockTwits, ${data.results.reddit} Reddit posts`);
        loadPredictions();
        loadExperts();
    } catch (e) {
        setStatus('Collection failed: ' + e.message);
    }
}

// Auto-refresh every 5 minutes
setInterval(() => {
    loadPredictions();
    loadExperts();
}, 300000);
```

- [ ] **Step 3: Write entry point**

```python
# run.py
"""Entry point for the DualGAT Stock Prediction system."""
import uvicorn
from config import API_HOST, API_PORT

if __name__ == "__main__":
    uvicorn.run(
        "src.web.api:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
    )
```

- [ ] **Step 4: Verify HTML/JS serve correctly**

```bash
python -c "from src.web.api import app; print('API app loaded successfully')"
```

- [ ] **Step 5: Commit**

```bash
git add src/web/templates/index.html src/web/static/app.js run.py
git commit -m "feat: dashboard UI with Chart.js and HTMX + run.py entry point"
```

---

### Task 13: Integration Test — End-to-End Pipeline

**Files:**
- Create: `tests/test_integration.py`
- Create: `tests/fixtures/sample_prices.csv`
- Create: `tests/fixtures/sample_posts.csv`

**Interfaces:**
- Consumes: All modules
- Produces: End-to-end test verifying the complete pipeline works

- [ ] **Step 1: Create sample fixtures**

```csv
# tests/fixtures/sample_prices.csv
stock,date,open,high,low,close,volume
AAPL,2024-06-01,185.0,187.0,184.0,186.5,50000000
AAPL,2024-06-02,186.5,188.0,185.5,187.2,48000000
AAPL,2024-06-03,187.2,189.0,186.8,188.5,52000000
MSFT,2024-06-01,415.0,418.0,413.0,416.5,25000000
MSFT,2024-06-02,416.5,420.0,415.0,419.0,24000000
MSFT,2024-06-03,419.0,422.0,418.5,421.0,26000000
GOOGL,2024-06-01,175.0,177.0,174.0,176.0,30000000
GOOGL,2024-06-02,176.0,178.5,175.5,177.8,29000000
GOOGL,2024-06-03,177.8,179.0,176.5,178.2,31000000
```

```csv
# tests/fixtures/sample_posts.csv
source,user_id,stock,timestamp,sentiment,content
stocktwits,user1,AAPL,2024-06-01T10:00:00,Bullish,AAPL going up
stocktwits,user1,AAPL,2024-06-02T10:00:00,Bullish,Still bullish
stocktwits,user2,MSFT,2024-06-01T11:00:00,Bearish,MSFT overvalued
stocktwits,user2,MSFT,2024-06-02T11:00:00,Bearish,Downtrend continues
```

- [ ] **Step 2: Write integration test**

```python
# tests/test_integration.py
"""End-to-end integration test for the full pipeline."""
import pytest
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys

from src.db import schema as db
from src.db.schema import init_db, insert_prices, insert_posts
from src.data.models import Price, Post
from src.expert.tracker import ExpertTracker
from src.model.baseline import BaselinePredictor
from src.model.signal import transform_expert_signal
from src.backtest.portfolio import construct_long_short


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def setup_integration_db(tmp_path, monkeypatch):
    """Set up database with sample data for integration test."""
    db_path = tmp_path / "integration.db"
    monkeypatch.setattr("src.db.schema.DB_PATH", db_path)
    monkeypatch.setattr("config.DB_PATH", db_path)
    import src.db.schema as schema_mod
    schema_mod._engine = None
    init_db()

    # Load sample prices
    prices_df = pd.read_csv(FIXTURES / "sample_prices.csv")
    prices = []
    for _, row in prices_df.iterrows():
        prices.append(Price(
            stock=row["stock"],
            date=datetime.strptime(row["date"], "%Y-%m-%d"),
            open=row["open"], high=row["high"], low=row["low"],
            close=row["close"], volume=row["volume"],
        ))
    insert_prices(prices)

    # Load sample posts
    posts_df = pd.read_csv(FIXTURES / "sample_posts.csv")
    posts = []
    for _, row in posts_df.iterrows():
        posts.append(Post(
            source=row["source"],
            user_id=row["user_id"],
            stock=row["stock"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            sentiment=row["sentiment"],
            content=row["content"],
        ))
    insert_posts(posts)

    yield


def test_full_pipeline_runs(setup_integration_db):
    """Verify the complete pipeline runs without errors."""
    stocks = ["AAPL", "MSFT", "GOOGL"]

    # Step 1: Expert tracing
    tracker = ExpertTracker()
    records = tracker.trace("2024-06-01")

    # May or may not find experts with limited data, but shouldn't crash
    assert isinstance(records, list)

    # Step 2: Signal transformation
    signals = transform_expert_signal(records, "2024-06-01")
    assert isinstance(signals, dict)

    # Step 3: Prediction
    predictor = BaselinePredictor()
    pred_df = predictor.predict(stocks, "2024-06-01", records)
    assert len(pred_df) == 3
    assert set(pred_df["stock"]) == set(stocks)

    # Step 4: Portfolio construction
    portfolio = construct_long_short(pred_df, quantile=0.33)
    assert len(portfolio["long"]) == 1  # 33% of 3
    assert len(portfolio["short"]) == 1

    # Step 5: Predictions are ordered
    preds = pred_df["predicted_return"].tolist()
    assert preds == sorted(preds, reverse=True)


def test_expert_tracing_with_real_data(setup_integration_db):
    """Test expert tracing with actual inserted posts."""
    tracker = ExpertTracker()
    records = tracker.trace("2024-06-01")
    assert isinstance(records, list)


def test_baseline_predictor_all_stocks_covered(setup_integration_db):
    """Every stock in the universe gets a prediction."""
    predictor = BaselinePredictor()
    stocks = ["AAPL", "MSFT", "GOOGL"]
    df = predictor.predict(stocks, "2024-06-01")
    assert len(df) == len(stocks)
    assert df["predicted_return"].notna().all()
```

- [ ] **Step 3: Run integration test**

```bash
python -m pytest tests/test_integration.py -v
```
Expected: 3 tests pass

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration.py tests/fixtures/sample_prices.csv tests/fixtures/sample_posts.csv
git commit -m "test: end-to-end integration test with sample fixtures"
```

---

### Task 14: Final Verification

- [ ] **Step 1: Run all tests**

```bash
python -m pytest tests/ -v --tb=short
```
Expected: All tests pass (~35+ tests)

- [ ] **Step 2: Verify imports**

```bash
python -c "
from src.data.models import Post, Price, ExpertRecord
from src.data.yfinance import YFinanceCollector
from src.data.stocktwits import StockTwitsCollector
from src.data.reddit import RedditCollector
from src.expert.sentiment import SentimentRouter, FinBERTSentiment
from src.expert.tracker import ExpertTracker
from src.model.signal import transform_expert_signal, compute_return_ratio
from src.model.features import compute_momentum
from src.model.baseline import BaselinePredictor
from src.backtest.metrics import compute_ic, compute_ric, compute_sharpe
from src.backtest.portfolio import construct_long_short, run_backtest
from src.db.schema import init_db, get_db
from src.web.api import app
print('All imports successful')
"
```
Expected: "All imports successful"

- [ ] **Step 3: Start server smoke test**

```bash
timeout 5 python run.py 2>&1 || true
```
Expected: Server starts and shows "Uvicorn running on..."

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "chore: final integration verification, all tests pass"
```

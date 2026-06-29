# DualGAT Stock Prediction System — Design Spec

**Date:** 2026-06-29
**Paper:** Unleashing Expert Opinion from Social Media for Stock Prediction (arXiv:2504.10078v2)
**Target:** MVP (v0.1) → v0.2 (MS-LSTM) → v0.3 (DualGAT)

---

## Overview

A web-based stock prediction system that identifies expert traders from social media and propagates their signals across related stocks using graph neural networks.

### MVP Scope (v0.1)
- Data collection from yfinance, StockTwits, Reddit
- FinBERT sentiment analysis for unlabeled posts
- Expert tracing (Algorithm 1 from paper)
- Rule-based baseline predictor (no DL)
- FastAPI web service with dashboard
- Backtest evaluation (IC, RIC, Sharpe, cumulative returns)

### Later Phases
- v0.2: MS-LSTM temporal pre-training model
- v0.3: DualGAT graph propagation model

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Web Dashboard                      │
│              (FastAPI + HTMX + Chart.js)              │
├─────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌───────────────────┐  │
│  │ Collector│  │ Expert   │  │  Predictor         │  │
│  │          │→ │ Tracker  │→ │  (Rule→MS-LSTM     │  │
│  │ yfinance │  │          │  │   →DualGAT)        │  │
│  │ Reddit   │  │Algorithm1│  │                    │  │
│  │StockTwits│  │          │  │                    │  │
│  └──────────┘  └──────────┘  └─────────┬──────────┘  │
│                                        │              │
│  ┌────────────────────────────────────┐│              │
│  │          Backtest Engine           │◄──────────────│
│  │  IC / RIC / ICIR / Sharpe / AR    │               │
│  └────────────────────────────────────┘              │
└─────────────────────────────────────────────────────┘
```

---

## Data Flow

1. **Collection**: Daily cron/scheduled job fetches price data (yfinance), StockTwits posts, and Reddit posts for configured stock universe
2. **Sentiment**: StockTwits posts use self-labeled sentiment; Reddit posts run through FinBERT for Bullish/Bearish/Neutral classification
3. **Expert Tracing**: Algorithm 1 runs on collected posts, outputs expert/inverse_expert labels and predicted direction per stock-day
4. **Signal Transform**: Binary expert predictions → continuous return signals using 30-day historical average returns
5. **Prediction**: Rule-based model (MVP) or MS-LSTM/DualGAT (later) produces daily return ratio predictions
6. **Backtest**: Long-short portfolio construction (top/bottom 10%), compute metrics
7. **API/Web**: Expose predictions, expert list, backtest results via REST API and dashboard

---

## Component Design

### 1. Data Collection (`src/data/`)

**Interface:**
```python
class DataCollector:
    def fetch_prices(stocks: list[str], start: str, end: str) -> pd.DataFrame
    def fetch_fundamentals(stocks: list[str]) -> pd.DataFrame
    def fetch_social_posts(stocks: list[str], date: str) -> list[Post]
```

**Implementations:**
- `YFinanceCollector`: Wraps yfinance for OHLCV + fundamentals
- `StockTwitsCollector`: StockTwits free API for self-labeled posts
- `RedditCollector`: PRAW for r/wallstreetbets, r/stocks

**Storage:** SQLite with tables: `prices`, `fundamentals`, `posts`, `experts`

### 2. Sentiment Analysis (`src/expert/sentiment.py`)

- `FinBERTSentiment`: ProsusAI/finbert via transformers, maps to Bullish/Bearish/Neutral
- `StockTwitsSentiment`: Pass-through for self-labeled data
- `SentimentRouter`: Routes to correct analyzer based on source

### 3. Expert Tracker (`src/expert/tracker.py`)

Full implementation of Algorithm 1:

```
Algorithm: Expert Tracing for Date d
- Filter: Keep latest post per user-stock pair on day d
- For each user i who posted on day d:
  1. Recent: Last N=20 posts, require ≥K=5 unique trading days
     - Expert if recent_accuracy ≥ P2 (80%)
     - Inverse Expert if recent_accuracy ≤ 1-P2 (20%)
  2. Long-term: All posts in past T=2 years
     - Expert required: long_term_accuracy ≥ P1 (65%)
     - Inverse Expert required: long_term_accuracy ≤ 1-P1 (35%)
  3. Focus: Skip users posting on too many stocks in one day
```

**Key classes:**
- `ExpertTracker`: Main orchestrator
- `UserHistory`: Per-user historical accuracy tracking
- `PostFilter`: Bot/spam detection (interval-based)

### 4. Prediction Engine (`src/model/`)

**MVP (v0.1) — Rule-Based:**
- Has expert signal: use direction + 30-day average return magnitude
- No expert signal: use momentum factor (past 20-day return)
- Output: cross-sectional return ratio for next trading day

**v0.2 — MS-LSTM:**
- Multi-scale LSTM: N LSTM branches at different temporal scales (1, 2, 4, 8, 16 day strides)
- Average final hidden states → MLP → return prediction
- IC Loss function (maximize Information Coefficient)
- Input: OHLCV + fundamental features, L=30 day window

**v0.3 — DualGAT:**
- Input: MS-LSTM output + expert_availability (binary) + expert_signal (continuous)
- Industry graph: GICS sector-based adjacency
- Correlation graph: 30-day price correlation, θ1=0.77, θ2=0.67
- 2-hop GAT with dual-graph attention fusion
- Output: refined return ratio per stock

### 5. Backtest Engine (`src/backtest/`)

**Metrics:**
- Accuracy (ACC): direction prediction correctness
- Information Coefficient (IC): Pearson correlation
- Rank IC (RIC): Spearman rank correlation
- ICIR: IC / std(IC)
- Annualized Return (AR): long-short portfolio
- Sharpe Ratio (SR): risk-adjusted return

**Portfolio construction:**
- Long top 10% predicted returns
- Short bottom 10% predicted returns
- 4 bps transaction cost
- Daily rebalancing

### 6. Web Service (`src/web/`)

**API Endpoints:**
- `GET /api/predictions?date=YYYY-MM-DD` — daily predictions
- `GET /api/experts` — current expert list with accuracy stats
- `GET /api/backtest?start=YYYY-MM-DD&end=YYYY-MM-DD` — backtest results
- `POST /api/collect` — trigger data collection
- `GET /api/stocks` — stock universe

**Dashboard:**
- Stock prediction table (sortable, filterable)
- Expert leaderboard
- Cumulative returns chart (Chart.js)
- IC/RIC over time chart
- Single-page app with HTMX for interactivity

---

## Directory Structure

```
calss4/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── base.py           # Abstract DataCollector
│   │   ├── yfinance.py       # YFinanceCollector
│   │   ├── stocktwits.py     # StockTwitsCollector
│   │   ├── reddit.py         # RedditCollector
│   │   └── models.py         # Post, Price dataclasses
│   ├── expert/
│   │   ├── __init__.py
│   │   ├── tracker.py        # ExpertTracker (Algorithm 1)
│   │   ├── sentiment.py      # FinBERT + StockTwits sentiment
│   │   └── history.py        # UserHistory persistence
│   ├── model/
│   │   ├── __init__.py
│   │   ├── baseline.py       # Rule-based predictor (MVP)
│   │   ├── features.py       # Feature engineering
│   │   └── signal.py         # Expert signal transformation
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── metrics.py        # IC, RIC, Sharpe, AR
│   │   └── portfolio.py      # Long-short portfolio construction
│   ├── web/
│   │   ├── __init__.py
│   │   ├── api.py            # FastAPI app
│   │   ├── templates/
│   │   │   └── index.html    # Dashboard template
│   │   └── static/
│   │       └── app.js        # Chart.js + HTMX logic
│   └── db/
│       ├── __init__.py
│       └── schema.py         # SQLite schema + migrations
├── config.py                 # Configuration (tickers, thresholds, dates)
├── run.py                    # Entry point
├── requirements.txt
└── docs/
    └── superpowers/
        └── specs/
            └── 2026-06-29-dualgat-stock-prediction-design.md
```

---

## Configuration (`config.py`)

```python
# Stock Universe
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", ...]  # ~100 stocks

# Expert Tracing (from paper)
EXPERT_RECENT_N = 20
EXPERT_MIN_DAYS = 5
EXPERT_RECENT_THRESHOLD = 0.80    # P2
EXPERT_LONG_THRESHOLD = 0.65      # P1
EXPERT_LONG_WINDOW_YEARS = 2

# Signal Transform
SIGNAL_LOOKBACK_DAYS = 30

# Graph (for v0.3)
CORR_THRESHOLD_NORMAL = 0.77      # θ1
CORR_THRESHOLD_EXPERT = 0.67      # θ2

# Backtest
PORTFOLIO_QUANTILE = 0.10
TRANSACTION_COST_BPS = 0.0004

# Data paths
DB_PATH = "data/stocktwits.db"
```

---

## Testing Strategy

- **Unit tests**: Each collector, sentiment analyzer, tracker filter, metric function
- **Integration tests**: Full pipeline on a small stock subset (5 tickers, 30 days)
- **Smoke tests**: API endpoints return correct shape
- **Fixtures**: Sample posts, prices in test fixtures directory

---

## Non-Goals (v0.1)

- Real-time streaming data (daily batch is sufficient)
- User authentication/authorization
- Production deployment (single-user local server)
- GPU acceleration (CPU-only for MVP; GPU support in v0.2+)
- Paper's full StockTwits dataset replication

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

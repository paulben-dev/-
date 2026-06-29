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

# Demo mode — set True when using < 2 years of data
DEMO_MODE = True

# Expert Tracing (Algorithm 1 from paper)
# Paper values: N=20, K=5, P2=0.80, P1=0.65, T=730 days
# Demo values are relaxed for shorter data windows
EXPERT_RECENT_N = 15 if DEMO_MODE else 20
EXPERT_MIN_DAYS = 3 if DEMO_MODE else 5
EXPERT_RECENT_THRESHOLD = 0.65 if DEMO_MODE else 0.80  # P2
EXPERT_LONG_THRESHOLD = 0.55 if DEMO_MODE else 0.65    # P1
EXPERT_LONG_WINDOW_DAYS = 60 if DEMO_MODE else 730     # T

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

# MS-LSTM (v0.2)
MSLSTM_HIDDEN_DIM = 64
MSLSTM_NUM_SCALES = 5          # strides: 1, 2, 4, 8, 16
MSLSTM_DROPOUT = 0.2
MSLSTM_LEARNING_RATE = 1e-3
MSLSTM_WEIGHT_DECAY = 1e-5
MSLSTM_EPOCHS = 100
MSLSTM_EARLY_STOP_PATIENCE = 10
MSLSTM_SEQUENCE_LENGTH = 30    # Lookback window in trading days
MSLSTM_MODEL_PATH = ROOT_DIR / "data" / "ms_lstm_model.pt"

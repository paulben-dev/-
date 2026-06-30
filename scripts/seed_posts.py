"""Seed DB with simulated social posts aligned to actual price data.

Uses existing real price data to generate posts where:
  - Good users (~78% accuracy) → become "experts"
  - Bad users (~22% accuracy) → become "inverse_experts"
Only inserts posts (no price overwrites).
"""
import random
from datetime import datetime, timedelta
from src.data.models import Post
from src.db import schema as db
from config import DEFAULT_TICKERS

random.seed(42)

GOOD = [
    ("trader_alpha", 0.78, ["AAPL", "MSFT", "NVDA", "GOOGL"]),
    ("stock_wizard", 0.82, ["AMZN", "META", "NFLX", "ADBE"]),
    ("market_guru", 0.75, ["TSLA", "JPM", "BAC", "V"]),
    ("tech_analyst", 0.80, ["AAPL", "NVDA", "MSFT", "GOOGL"]),
    ("value_hunter", 0.72, ["JNJ", "PG", "WMT", "XOM"]),
    ("quant_mind", 0.85, ["MA", "HD", "UNH", "DIS"]),
]

BAD = [
    ("fomo_kid", 0.25, ["TSLA", "NVDA", "AAPL", "META"]),
    ("panic_seller", 0.20, ["AMZN", "MSFT", "GOOGL", "NFLX"]),
    ("meme_trader99", 0.30, ["DIS", "TSLA", "BAC", "JPM"]),
    ("bag_holder", 0.18, ["WMT", "XOM", "PG", "JNJ"]),
]

NOISE = [
    ("casual_investor", 0.50, ["AAPL", "GOOGL"]),
    ("random_poster", 0.48, ["MSFT", "TSLA"]),
]

BULL_MSGS = [
    "${s} looking strong! 🚀", "Adding more {s}, great setup",
    "{s} to the moon! 🌙", "Bullish on {s}, PT raised",
    "Just bought {s}, conviction play", "{s} breakout confirmed",
]
BEAR_MSGS = [
    "${s} overvalued here 📉", "Trimming {s} position",
    "{s} heading lower IMO", "Bearish on {s}, reducing risk",
    "Sold my {s}, too risky", "{s} breakdown on volume",
]

def get_next_move(stock, date_str, cache):
    """Return 1 if stock rose next trading day, 0 if fell, None if N/A."""
    prices = cache.get(stock, [])
    for i, p in enumerate(prices):
        if p["date"] == date_str and i + 1 < len(prices):
            return 1 if prices[i + 1]["close"] > p["close"] else 0
    return None

def sentiment(accuracy, actual):
    """Generate sentiment matching user accuracy against actual direction."""
    if actual is None:
        return random.choice(["Bullish", "Bearish"])
    correct = random.random() < accuracy
    if actual == 1:  # price went up
        return "Bullish" if correct else "Bearish"
    else:            # price went down
        return "Bearish" if correct else "Bullish"

print("Loading real price data...")
db.init_db()
end = datetime.now()
cache = db.get_prices(DEFAULT_TICKERS, (end - timedelta(days=180)).strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

all_dates = sorted({p["date"] for sp in cache.values() for p in sp})
post_dates = all_dates[-45:-2]  # Skip last 2 (need next-day price)
print(f"Generating posts for {len(post_dates)} trading days ({post_dates[0]} to {post_dates[-1]})")

posts = []
for date_str in post_dates:
    for user_list, activity in [(GOOD, 0.65), (BAD, 0.55), (NOISE, 0.40)]:
        for name, acc, stocks in user_list:
            if random.random() > activity:
                continue
            for stock in random.sample(stocks, k=min(random.randint(1, 2), len(stocks))):
                move = get_next_move(stock, date_str, cache)
                sent = sentiment(acc, move)
                msg = random.choice(BULL_MSGS if sent == "Bullish" else BEAR_MSGS).replace("{s}", stock)
                h = min(random.randint(8, 20), 23)
                posts.append(Post(
                    source=random.choice(["reddit", "stocktwits"]),
                    user_id=name, stock=stock,
                    timestamp=datetime.fromisoformat(date_str).replace(hour=h, minute=random.randint(0, 59)),
                    sentiment=sent, content=msg,
                ))

db.insert_posts(posts)

# Verify
from sqlalchemy import text
with db.get_db().connect() as conn:
    cnt = conn.execute(text("SELECT count(*) FROM posts")).scalar()
    print(f"Done! Total posts in DB: {cnt}")
    for u in GOOD + BAD + NOISE:
        n = conn.execute(text("SELECT count(*) FROM posts WHERE user_id=:u"), {"u": u[0]}).scalar()
        label = "🟢expert" if u[1] >= 0.7 else ("🔴inverse" if u[1] <= 0.3 else "⚪noise")
        print(f"  {label} {u[0]}: {n} posts")

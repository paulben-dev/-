#!/usr/bin/env python3
"""Generate 90 days of simulated market + social media data for demo."""
import random
import numpy as np
from datetime import datetime, timedelta
from src.db.schema import init_db, insert_prices, insert_posts
from src.data.models import Price, Post

random.seed(42)
np.random.seed(42)

# --- Config ---
STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM",
    "JNJ", "V", "PG", "XOM", "WMT", "MA", "UNH", "HD", "BAC", "DIS",
    "NFLX", "ADBE",
]
START_DATE = datetime(2024, 1, 1)
N_DAYS = 90  # ~3 months of trading data

# Expert user IDs
EXPERT_USERS = [f"expert_{i}" for i in range(1, 6)]
INVERSE_USERS = [f"inverse_{i}" for i in range(1, 4)]
NOISE_USERS = [f"noise_{i}" for i in range(1, 15)]
BOT_USERS = [f"bot_{i}" for i in range(1, 3)]

init_db()

# --- Generate Price Data ---
print("Generating price data...")
prices = []
base_prices = {
    "AAPL": 185, "MSFT": 375, "GOOGL": 140, "AMZN": 150, "NVDA": 500,
    "META": 350, "TSLA": 250, "JPM": 170, "JNJ": 155, "V": 260,
    "PG": 145, "XOM": 100, "WMT": 165, "MA": 420, "UNH": 530,
    "HD": 350, "BAC": 33, "DIS": 90, "NFLX": 480, "ADBE": 580,
}
volatility = {s: np.random.uniform(0.008, 0.025) for s in STOCKS}
trend = {s: np.random.uniform(-0.0003, 0.0008) for s in STOCKS}
correlation = np.random.uniform(0.3, 0.7, (len(STOCKS), len(STOCKS)))
np.fill_diagonal(correlation, 1.0)

# Generate correlated returns
returns_matrix = np.random.multivariate_normal(
    [trend[s] for s in STOCKS],
    np.diag([volatility[s] for s in STOCKS]) @ correlation @ np.diag([volatility[s] for s in STOCKS]),
    N_DAYS,
)

current_prices = {s: base_prices[s] for s in STOCKS}
for day_idx in range(N_DAYS):
    date = START_DATE + timedelta(days=day_idx)
    if date.weekday() >= 5:  # Skip weekends
        continue
    for stock_idx, stock in enumerate(STOCKS):
        ret = returns_matrix[day_idx, stock_idx]
        close = current_prices[stock] * (1 + ret)
        open_p = current_prices[stock] * (1 + np.random.uniform(-0.005, 0.005))
        high = max(open_p, close) * (1 + abs(np.random.normal(0, 0.005)))
        low = min(open_p, close) * (1 - abs(np.random.normal(0, 0.005)))
        volume = int(abs(np.random.normal(5_000_000, 2_000_000)))
        prices.append(Price(stock, date, round(open_p, 2), round(high, 2), round(low, 2), round(close, 2), volume))
        current_prices[stock] = close

insert_prices(prices)
print(f"  Inserted {len(prices)} price records")

# --- Generate Social Media Posts ---
print("Generating social media posts...")
posts = []
trading_dates = sorted(set(p.date for p in prices))
trade_date_strs = [d.strftime("%Y-%m-%d") for d in trading_dates]

for date in trading_dates:
    # Skip the last date — we need tomorrow's price for alignment
    if date == trading_dates[-1]:
        continue
    for stock in STOCKS:
        # Get TOMORROW's price movement (what the tracker evaluates against)
        stock_prices_today = [p for p in prices if p.stock == stock and p.date == date]
        next_date_prices = [p for p in prices if p.stock == stock and p.date > date]
        if len(stock_prices_today) == 0 or len(next_date_prices) == 0:
            continue
        today_close = stock_prices_today[0].close
        tomorrow_close = next_date_prices[0].close
        price_up_tomorrow = tomorrow_close > today_close

        # 1. Expert users (correct ~75% of the time) — specialize in 5 stocks each
        expert_stocks = {eid: STOCKS[i*4:(i+1)*4] for i, eid in enumerate(EXPERT_USERS)}
        for expert_id in EXPERT_USERS:
            if stock not in expert_stocks[expert_id]:
                continue
            if random.random() < 0.9:  # 90% chance expert posts about their specialty stocks
                correct = random.random() < 0.82  # Expert: ~82% correct
                sentiment = "Bullish" if (correct and price_up_tomorrow) or (not correct and not price_up_tomorrow) else "Bearish"
                posts.append(Post(
                    source="stocktwits",
                    user_id=expert_id,
                    stock=stock,
                    timestamp=datetime(date.year, date.month, date.day, 10 + hash(expert_id) % 8, random.randint(0, 59)),
                    sentiment=sentiment,
                    content=f"Analysis suggests {stock} will go {'up' if sentiment == 'Bullish' else 'down'}",
                ))

        # 2. Inverse expert users (wrong ~80% of the time) — specialize in 5 stocks
        inv_stocks = {iid: STOCKS[i*7:(i+1)*7] for i, iid in enumerate(INVERSE_USERS)}
        for inv_id in INVERSE_USERS:
            if stock not in inv_stocks[inv_id]:
                continue
            if random.random() < 0.85:
                correct = random.random() < 0.20  # Only 20% correct
                sentiment = "Bullish" if (correct and price_up_tomorrow) or (not correct and not price_up_tomorrow) else "Bearish"
                posts.append(Post(
                    source="stocktwits",
                    user_id=inv_id,
                    stock=stock,
                    timestamp=datetime(date.year, date.month, date.day, 11 + hash(inv_id) % 8, random.randint(0, 59)),
                    sentiment=sentiment,
                    content=f"I predict {stock} is going {'up' if sentiment == 'Bullish' else 'down'}",
                ))

        # 3. Noise users (random) — keep low rate
        for noise_id in random.sample(NOISE_USERS, 3):
            if random.random() < 0.10:
                sentiment = random.choice(["Bullish", "Bearish"])
                posts.append(Post(
                    source="stocktwits",
                    user_id=noise_id,
                    stock=stock,
                    timestamp=datetime(date.year, date.month, date.day, 12 + hash(noise_id) % 6, random.randint(0, 59)),
                    sentiment=sentiment,
                    content=f"{stock} is {'great' if sentiment == 'Bullish' else 'terrible'}!",
                ))

        # 4. Reddit posts (need sentiment from FinBERT)
        reddit_users = [f"reddit_user_{i}" for i in range(1, 6)]
        for ru in random.sample(reddit_users, 2):
            if random.random() < 0.1:
                tone = "bullish" if random.random() < 0.5 else "bearish"
                posts.append(Post(
                    source="reddit",
                    user_id=ru,
                    stock=stock,
                    timestamp=datetime(date.year, date.month, date.day, 14, random.randint(0, 59)),
                    sentiment="Neutral",
                    content=f"${stock} is a {'strong buy' if tone == 'bullish' else 'sell'} based on technical analysis and market conditions",
                ))

insert_posts(posts)
print(f"  Inserted {len(posts)} social media posts")
print(f"\n✓ Done! Database seeded at data/predictions.db")
print(f"  - {len(prices)} price records across {len(STOCKS)} stocks")
print(f"  - {len(posts)} posts from {len(EXPERT_USERS)} experts, {len(INVERSE_USERS)} inverse experts, {len(NOISE_USERS)} noise users")
print(f"  - Date range: {trade_date_strs[0]} to {trade_date_strs[-1]}")
print(f"\nNow visit http://localhost:8000 and try:")
print(f"  1. Set date to {trade_date_strs[-1]} and click 'Load Predictions'")
print(f"  2. Visit /api/experts?date={trade_date_strs[-1]}")
print(f"  3. Visit /api/backtest?start={trade_date_strs[20]}&end={trade_date_strs[-1]}")

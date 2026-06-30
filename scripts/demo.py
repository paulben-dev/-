#!/usr/bin/env python3
"""Demo: run the full pipeline with relaxed parameters for 90-day data."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# ── Monkey-patch config for demo (90 days of data, not 2 years) ──
import config
config.EXPERT_LONG_WINDOW_DAYS = 60   # Paper: 730 days; Demo: 60 days
config.EXPERT_RECENT_N = 10           # Paper: 20; Demo: 10
config.EXPERT_MIN_DAYS = 3            # Paper: 5; Demo: 3
config.EXPERT_RECENT_THRESHOLD = 0.70 # Paper: 0.80; Demo: 0.70
config.EXPERT_LONG_THRESHOLD = 0.60   # Paper: 0.65; Demo: 0.60

from src.expert.tracker import ExpertTracker
from src.model.baseline import BaselinePredictor
from src.backtest.portfolio import run_backtest
from src.db.schema import init_db, get_posts_for_date, get_prices
from config import DEFAULT_TICKERS
from datetime import datetime, timedelta

init_db()

# ── Find the latest date with data ──
sample = get_posts_for_date("2024-03-28")
if not sample:
    print("No posts found. Run scripts/seed_data.py first.")
    sys.exit(1)

date = "2024-03-28"
print(f"=== DualGAT Demo: {date} ===\n")

# ── Step 1: Expert Tracing ──
print("[1/4] Tracing experts...")
tracker = ExpertTracker()
records = tracker.trace(date)

experts = [r for r in records if r.expert_type == "expert"]
inverse = [r for r in records if r.expert_type == "inverse_expert"]
print(f"  Found {len(experts)} experts + {len(inverse)} inverse experts")

if records:
    print(f"\n  Top Expert Signals:")
    for r in sorted(records, key=lambda x: -x.accuracy_recent)[:10]:
        icon = "🟢" if r.expert_type == "expert" else "🔴"
        print(f"  {icon} {r.user_id:20s} | {r.stock:5s} | {r.predicted_direction:8s} | "
              f"recent={r.accuracy_recent:.0%} long={r.accuracy_long:.0%}")

# ── Step 2: Predictions ──
print(f"\n[2/4] Generating predictions...")
predictor = BaselinePredictor()
pred_df = predictor.predict(DEFAULT_TICKERS, date, records)
expert_count = (pred_df["signal_source"] == "expert").sum()
print(f"  Predictions: {len(pred_df)} stocks ({expert_count} from experts, {len(pred_df) - expert_count} from momentum)")
print(f"\n  Top 5 Buys:")
for _, row in pred_df.head(5).iterrows():
    print(f"  📈 {row['stock']:5s} | {row['predicted_return']:+.3f} | source: {row['signal_source']}")

# ── Step 3: Backtest ──
print(f"\n[3/4] Running backtest (2024-02-01 to 2024-03-28)...")
all_preds = []
current = datetime(2024, 2, 1)
end = datetime(2024, 3, 28)
while current <= end:
    ds = current.strftime("%Y-%m-%d")
    recs = tracker.trace(ds)
    pdf = predictor.predict(DEFAULT_TICKERS, ds, recs)
    all_preds.append(pdf)
    current += timedelta(days=1)

import pandas as pd
combined = pd.concat(all_preds, ignore_index=True)
result = run_backtest(combined, DEFAULT_TICKERS, "2024-02-01", "2024-03-28")

print(f"\n  📊 Backtest Results:")
print(f"  Annualized Return: {result['annualized_return']:+.1%}")
print(f"  Sharpe Ratio:      {result['sharpe_ratio']:.2f}")
print(f"  Max Drawdown:      {result['max_drawdown']:.1%}")
print(f"  Mean IC:           {result['mean_ic']:.4f}")
print(f"  ICIR:              {result['icir']:.2f}")
print(f"  Trading Days:      {result['n_trading_days']}")

# ── Step 4: Compare Expert vs Momentum signals ──
print(f"\n[4/4] Signal quality comparison:")
expert_preds = pred_df[pred_df["signal_source"] == "expert"]
momentum_preds = pred_df[pred_df["signal_source"] == "momentum"]
print(f"  Expert-driven predictions:  {len(expert_preds)} stocks")
print(f"  Momentum-driven predictions: {len(momentum_preds)} stocks")

print(f"\n✓ Demo complete! Dashboard available at http://localhost:8000")

#!/usr/bin/env python3
"""Train MS-LSTM model and compare against rule-based baseline.

Usage:
    python3 scripts/train_ms_lstm.py [--start YYYY-MM-DD] [--end YYYY-MM-DD]
"""
import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DEFAULT_TICKERS, MSLSTM_MODEL_PATH
from src.db.schema import init_db, get_prices
from src.model.baseline import BaselinePredictor
from src.model.ms_lstm import MSLSTMPredictor, _get_trading_dates
from src.expert.tracker import ExpertTracker
from src.backtest.metrics import compute_ic

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("train_ms_lstm")


def main():
    parser = argparse.ArgumentParser(description="Train MS-LSTM model")
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs")
    parser.add_argument("--stocks", type=int, default=10, help="Number of stocks to train on")
    args = parser.parse_args()

    # Default date range: last 6 months
    if args.end is None:
        args.end = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    if args.start is None:
        start_dt = datetime.fromisoformat(args.end) - timedelta(days=180)
        args.start = start_dt.strftime("%Y-%m-%d")

    stocks = DEFAULT_TICKERS[: args.stocks]
    epochs = args.epochs or 100

    logger.info(f"Training MS-LSTM on {len(stocks)} stocks from {args.start} to {args.end}")
    logger.info(f"Stocks: {stocks}")

    # Initialize database
    init_db()

    # --- Train MS-LSTM ---
    logger.info("Training MS-LSTM model...")
    ms_lstm = MSLSTMPredictor()
    history = ms_lstm.fit(
        stocks=stocks,
        start_date=args.start,
        end_date=args.end,
        epochs=epochs,
    )

    if not history["train_loss"]:
        logger.error("Training produced no results. Check that price data exists.")
        sys.exit(1)

    logger.info(
        f"Training complete. Best epoch: {history['best_epoch']}, "
        f"Best val IC: {max(history['val_ic']):.4f}"
    )

    # Save model
    ms_lstm.save(MSLSTM_MODEL_PATH)

    # --- Evaluate vs Baseline ---
    logger.info("Comparing MS-LSTM vs rule-based baseline...")
    trading_dates = _get_trading_dates(stocks, args.start, args.end)

    # Get validation dates (last 20%)
    split = int(len(trading_dates) * 0.8)
    val_dates = trading_dates[split:]

    baseline = BaselinePredictor()
    tracker = ExpertTracker()

    ms_ics = []
    bl_ics = []
    for date_str in val_dates:
        records = tracker.trace(date_str)

        # MS-LSTM predictions
        ms_preds = ms_lstm.predict(stocks, date_str)
        # Baseline predictions
        bl_preds = baseline.predict(stocks, date_str, records)

        # Actual returns
        prices = get_prices(stocks, date_str, date_str)
        actuals = {}
        for stock in stocks:
            sp = prices.get(stock, [])
            if len(sp) >= 2:
                sp.sort(key=lambda x: x["date"])
                for i, p in enumerate(sp):
                    if p["date"] == date_str and i > 0:
                        prev = sp[i - 1]["close"]
                        curr = p["close"]
                        actuals[stock] = (curr - prev) / prev if prev else 0.0

        if len(actuals) < 3:
            continue

        # Build aligned series
        actual_series = pd.Series(actuals)
        ms_series = ms_preds.set_index("stock")["predicted_return"]
        bl_series = bl_preds.set_index("stock")["predicted_return"]

        ms_ic = compute_ic(ms_series, actual_series)
        bl_ic = compute_ic(bl_series, actual_series)
        ms_ics.append(ms_ic)
        bl_ics.append(bl_ic)

    ms_mean_ic = np.mean(ms_ics) if ms_ics else 0.0
    bl_mean_ic = np.mean(bl_ics) if bl_ics else 0.0

    logger.info("=" * 50)
    logger.info(f"MS-LSTM  mean IC: {ms_mean_ic:.4f}  ({len(ms_ics)} days)")
    logger.info(f"Baseline mean IC: {bl_mean_ic:.4f}  ({len(bl_ics)} days)")
    logger.info(f"Difference:       {ms_mean_ic - bl_mean_ic:+.4f}")
    logger.info("=" * 50)

    if ms_mean_ic > bl_mean_ic:
        logger.info("✅ MS-LSTM outperforms baseline!")
    else:
        logger.info("⚠️  MS-LSTM does not beat baseline. Consider more data or tuning.")


if __name__ == "__main__":
    main()

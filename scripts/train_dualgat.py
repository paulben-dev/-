#!/usr/bin/env python3
"""Train DualGAT model and compare against MS-LSTM and baseline.

Usage:
    python3 scripts/train_dualgat.py [--start YYYY-MM-DD] [--end YYYY-MM-DD]
"""
import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DEFAULT_TICKERS, MSLSTM_MODEL_PATH, DUALGAT_MODEL_PATH
from src.db.schema import init_db, get_prices
from src.model.baseline import BaselinePredictor
from src.model.ms_lstm import MSLSTMPredictor
from src.model.dualgat import DualGATPredictor, _get_trading_dates
from src.expert.tracker import ExpertTracker
from src.backtest.metrics import compute_ic

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("train_dualgat")


def main():
    parser = argparse.ArgumentParser(description="Train DualGAT model")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs")
    parser.add_argument("--stocks", type=int, default=10, help="Number of stocks")
    parser.add_argument("--ms-lstm", default=str(MSLSTM_MODEL_PATH), help="MS-LSTM model path")
    args = parser.parse_args()

    if args.end is None:
        args.end = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    if args.start is None:
        args.start = (datetime.fromisoformat(args.end) - timedelta(days=180)).strftime("%Y-%m-%d")

    stocks = DEFAULT_TICKERS[: args.stocks]
    epochs = args.epochs or 100

    logger.info(f"Training DualGAT on {len(stocks)} stocks from {args.start} to {args.end}")
    logger.info(f"MS-LSTM model: {args.ms_lstm}")

    init_db()

    # Check MS-LSTM model exists
    if not Path(args.ms_lstm).exists():
        logger.error(f"MS-LSTM model not found at {args.ms_lstm}. Train MS-LSTM first.")
        sys.exit(1)

    # Train DualGAT
    logger.info("Training DualGAT...")
    dualgat = DualGATPredictor()
    history = dualgat.fit(
        stocks=stocks,
        start_date=args.start,
        end_date=args.end,
        ms_lstm_path=args.ms_lstm,
        epochs=epochs,
    )

    if not history["train_loss"]:
        logger.error("Training produced no results.")
        sys.exit(1)

    logger.info(
        f"Training complete. Best epoch: {history['best_epoch']}, "
        f"Best val IC: {max(history['val_ic']):.4f}"
    )
    dualgat.save(DUALGAT_MODEL_PATH)

    # Evaluate: DualGAT vs MS-LSTM vs Baseline
    logger.info("Comparing DualGAT vs MS-LSTM vs Baseline...")
    trading_dates = _get_trading_dates(stocks, args.start, args.end)
    split = int(len(trading_dates) * 0.8)
    val_dates = trading_dates[split:]

    ms_lstm = MSLSTMPredictor()
    ms_lstm.load(args.ms_lstm)
    baseline = BaselinePredictor()
    tracker = ExpertTracker()

    dg_ics, ms_ics, bl_ics = [], [], []
    for date_str in val_dates:
        records = tracker.trace(date_str)

        dg_preds = dualgat.predict(stocks, date_str)
        ms_preds = ms_lstm.predict(stocks, date_str)
        bl_preds = baseline.predict(stocks, date_str, records)

        yesterday_str = (datetime.fromisoformat(date_str) - timedelta(days=5)).strftime("%Y-%m-%d")
        prices = get_prices(stocks, yesterday_str, date_str)
        actuals = {}
        for stock in stocks:
            sp = prices.get(stock, [])
            if len(sp) >= 2:
                sp.sort(key=lambda x: x["date"])
                prev_close = sp[0]["close"]
                curr_close = sp[-1]["close"]
                if prev_close and prev_close != 0:
                    actuals[stock] = (curr_close - prev_close) / prev_close

        if len(actuals) < 3:
            continue

        actual_series = pd.Series(actuals)
        dg_series = dg_preds.set_index("stock")["predicted_return"]
        ms_series = ms_preds.set_index("stock")["predicted_return"]
        bl_series = bl_preds.set_index("stock")["predicted_return"]

        dg_ics.append(compute_ic(dg_series, actual_series))
        ms_ics.append(compute_ic(ms_series, actual_series))
        bl_ics.append(compute_ic(bl_series, actual_series))

    def _safe_mean(seq):
        return np.mean(seq) if seq else float("nan")

    logger.info("=" * 50)
    logger.info(f"DualGAT  mean IC: {_safe_mean(dg_ics):.4f}  ({len(dg_ics)} days)")
    logger.info(f"MS-LSTM  mean IC: {_safe_mean(ms_ics):.4f}  ({len(ms_ics)} days)")
    logger.info(f"Baseline mean IC: {_safe_mean(bl_ics):.4f}  ({len(bl_ics)} days)")
    logger.info("=" * 50)

    dg_mean = _safe_mean(dg_ics)
    ms_mean = _safe_mean(ms_ics)
    if not np.isnan(dg_mean) and not np.isnan(ms_mean) and dg_mean > ms_mean:
        logger.info("DualGAT outperforms MS-LSTM!")
    else:
        logger.info("DualGAT does not beat MS-LSTM. Consider tuning.")


if __name__ == "__main__":
    main()

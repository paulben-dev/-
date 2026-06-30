#!/usr/bin/env python3
"""Train ensemble meta-learner and save both weighted and meta models.

Usage:
    python3 scripts/train_ensemble.py [--start YYYY-MM-DD] [--end YYYY-MM-DD]

If --meta flag is passed, also trains the meta-learner MLP.
Otherwise saves the weighted-average config only.
"""
import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    DEFAULT_TICKERS,
    MSLSTM_MODEL_PATH,
    DUALGAT_MODEL_PATH,
    ENSEMBLE_MODEL_PATH,
    ENSEMBLE_META_PATH,
)
from src.db.schema import init_db, get_prices
from src.model.baseline import BaselinePredictor
from src.model.ms_lstm import MSLSTMPredictor
from src.model.dualgat import DualGATPredictor, _get_trading_dates
from src.model.ensemble import EnsemblePredictor
from src.expert.tracker import ExpertTracker
from src.backtest.metrics import compute_ic

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("train_ensemble")


def main():
    parser = argparse.ArgumentParser(description="Train ensemble model")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--meta", action="store_true", help="Train meta-learner MLP")
    parser.add_argument("--stocks", type=int, default=10)
    args = parser.parse_args()

    if args.end is None:
        args.end = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    if args.start is None:
        args.start = (datetime.fromisoformat(args.end) - timedelta(days=180)).strftime("%Y-%m-%d")

    stocks = DEFAULT_TICKERS[: args.stocks]
    logger.info(f"Training ensemble on {len(stocks)} stocks from {args.start} to {args.end}")

    init_db()

    # Load sub-models
    logger.info("Loading sub-models...")
    baseline = BaselinePredictor()

    if not MSLSTM_MODEL_PATH.exists():
        logger.error(f"MS-LSTM model not found at {MSLSTM_MODEL_PATH}")
        sys.exit(1)
    ms_lstm = MSLSTMPredictor()
    ms_lstm.load(MSLSTM_MODEL_PATH)

    if not DUALGAT_MODEL_PATH.exists():
        logger.error(f"DualGAT model not found at {DUALGAT_MODEL_PATH}")
        sys.exit(1)
    dualgat = DualGATPredictor()
    dualgat.load(DUALGAT_MODEL_PATH)

    tracker = ExpertTracker()

    # Save weighted-average ensemble
    logger.info("Saving weighted-average ensemble...")
    ensemble = EnsemblePredictor(strategy="weighted")
    ensemble.save(ENSEMBLE_MODEL_PATH)

    # Train meta-learner if requested
    if args.meta:
        logger.info("Training meta-learner MLP...")
        meta = EnsemblePredictor(strategy="meta")
        history = meta.fit_meta(
            stocks=stocks,
            start_date=args.start,
            end_date=args.end,
            baseline=baseline,
            ms_lstm=ms_lstm,
            dualgat=dualgat,
        )
        if history["train_loss"]:
            logger.info(f"Meta training done. Best epoch: {history['best_epoch']}, "
                        f"Best val IC: {max(history['val_ic']):.4f}")
            meta.save(ENSEMBLE_META_PATH)
        else:
            logger.warning("Meta training produced no results — saving weighted ensemble as meta fallback")
            meta.save(ENSEMBLE_META_PATH)

    # Evaluate all 4 models
    logger.info("Comparing all models...")
    trading_dates = _get_trading_dates(stocks, args.start, args.end)
    split = int(len(trading_dates) * 0.8)
    val_dates = trading_dates[split:]

    # Re-load ensemble (weighted)
    ensemble_w = EnsemblePredictor(strategy="weighted")
    if ENSEMBLE_MODEL_PATH.exists():
        ensemble_w.load(ENSEMBLE_MODEL_PATH)

    ensemble_m = None
    if args.meta and ENSEMBLE_META_PATH.exists():
        ensemble_m = EnsemblePredictor(strategy="meta")
        ensemble_m.load(ENSEMBLE_META_PATH)

    results = {"baseline": [], "ms_lstm": [], "dualgat": [], "ensemble_w": []}
    if ensemble_m:
        results["ensemble_m"] = []

    for date_str in val_dates:
        records = tracker.trace(date_str)

        bl_df = baseline.predict(stocks, date_str, records)
        ms_df = ms_lstm.predict(stocks, date_str)
        dg_df = dualgat.predict(stocks, date_str)
        ew_df = ensemble_w.predict(stocks, date_str, bl_df, ms_df, dg_df)

        prices = get_prices(stocks, date_str, date_str)
        actuals = {}
        for stock in stocks:
            sp = prices.get(stock, [])
            sp_sorted = sorted(sp, key=lambda x: x["date"])
            for i, p in enumerate(sp_sorted):
                if p["date"] == date_str and i > 0:
                    prev = sp_sorted[i - 1]["close"]
                    curr = p["close"]
                    if prev > 0:
                        actuals[stock] = (curr - prev) / prev
                    break

        if len(actuals) < 3:
            continue

        import pandas as pd
        actual_series = pd.Series(actuals)
        for label, df in [("baseline", bl_df), ("ms_lstm", ms_df), ("dualgat", dg_df), ("ensemble_w", ew_df)]:
            pred_series = df.set_index("stock")["predicted_return"]
            results[label].append(compute_ic(pred_series, actual_series))

        if ensemble_m:
            em_df = ensemble_m.predict(stocks, date_str, bl_df, ms_df, dg_df)
            em_series = em_df.set_index("stock")["predicted_return"]
            results["ensemble_m"].append(compute_ic(em_series, actual_series))

    logger.info("=" * 55)
    for label, ics in results.items():
        if ics:
            logger.info(f"  {label:15s}  mean IC: {np.mean(ics):.4f}  ({len(ics)} days)")
    logger.info("=" * 55)

    best_label = max(results, key=lambda k: np.mean(results[k]) if results[k] else -999)
    logger.info(f"Best model: {best_label}")


if __name__ == "__main__":
    main()

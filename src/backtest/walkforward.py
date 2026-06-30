"""Walk-forward validation engine.

Rolling-window framework for out-of-sample model evaluation.
Supports two modes:
  - "full": Retrain models within each training window.
  - "params": Use existing models, vary only backtest parameters.
"""
from dataclasses import dataclass, field
import logging
import numpy as np
import pandas as pd

from config import (
    WF_TRAIN_DAYS,
    WF_VALIDATE_DAYS,
    WF_STEP_DAYS,
    WF_MODE,
    WF_MIN_TRAIN_DAYS,
    MSLSTM_MODEL_PATH,
    DUALGAT_MODEL_PATH,
    ENSEMBLE_MODEL_PATH,
)

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    """Walk-forward validation parameters (all in trading days)."""
    train_days: int = WF_TRAIN_DAYS
    validate_days: int = WF_VALIDATE_DAYS
    step_days: int = WF_STEP_DAYS
    mode: str = WF_MODE           # "full" | "params"
    min_train_days: int = WF_MIN_TRAIN_DAYS


@dataclass
class WalkForwardResult:
    """Results from a walk-forward validation run."""
    windows: list[dict] = field(default_factory=list)
    oos_predictions: pd.DataFrame | None = None
    summary: dict = field(default_factory=dict)


def run_walk_forward(
    stocks: list[str],
    start_date: str,
    end_date: str,
    config: WalkForwardConfig,
    param_grid: dict | None = None,
) -> WalkForwardResult:
    """Run walk-forward validation over the date range.

    Args:
        stocks: Ticker symbols.
        start_date / end_date: Overall date range (YYYY-MM-DD).
        config: Walk-forward parameters.
        param_grid: For "params" mode, dict of {param_name: value} to
                    pass through to run_backtest.

    Returns:
        WalkForwardResult with per-window metrics and concatenated OOS predictions.
    """
    from src.backtest.calendar import trading_days_between
    from src.backtest.portfolio import run_backtest

    all_td = trading_days_between(start_date, end_date)
    if len(all_td) < config.min_train_days + config.validate_days:
        logger.warning(
            f"Insufficient trading days ({len(all_td)}) for "
            f"min_train={config.min_train_days}+val={config.validate_days}"
        )
        return WalkForwardResult()

    windows = []
    oos_preds_list = []

    idx = 0
    window_id = 1

    while idx + config.train_days + config.validate_days <= len(all_td):
        train_start = all_td[idx]
        train_end = all_td[idx + config.train_days - 1]
        val_start = all_td[idx + config.train_days]
        val_end_idx = idx + config.train_days + config.validate_days - 1
        val_end = all_td[val_end_idx]

        window_info = {
            "window_id": window_id,
            "train_start": train_start,
            "train_end": train_end,
            "val_start": val_start,
            "val_end": val_end,
        }

        try:
            # Generate predictions for validation window
            preds = _generate_predictions(stocks, val_start, val_end, config.mode,
                                          train_start, train_end)

            if preds is not None and len(preds) > 0:
                # Run backtest on validation window
                backtest_kwargs = param_grid or {}
                bt_result = run_backtest(preds, stocks, val_start, val_end,
                                         **backtest_kwargs)
                window_info.update({
                    "sharpe_ratio": bt_result.get("sharpe_ratio", 0.0),
                    "mean_ic": bt_result.get("mean_ic", 0.0),
                    "annualized_return": bt_result.get("annualized_return", 0.0),
                    "max_drawdown": bt_result.get("max_drawdown", 0.0),
                    "icir": bt_result.get("icir", 0.0),
                    "n_trading_days": bt_result.get("n_trading_days", 0),
                })

                preds["window_id"] = window_id
                oos_preds_list.append(preds)

        except Exception as e:
            logger.warning(f"Window {window_id} failed: {e}")
            window_info["error"] = str(e)

        windows.append(window_info)

        idx += config.step_days
        window_id += 1

    # Build summary
    sharpe_vals = [w.get("sharpe_ratio", 0.0) for w in windows if "sharpe_ratio" in w]
    ic_vals = [w.get("mean_ic", 0.0) for w in windows if "mean_ic" in w]

    n_failed = sum(1 for w in windows if "error" in w)
    summary = {
        "n_windows": len(windows),
        "n_successful": len(windows) - n_failed,
        "n_failed": n_failed,
        "sharpe_mean": float(np.mean(sharpe_vals)) if sharpe_vals else 0.0,
        "sharpe_std": float(np.std(sharpe_vals)) if sharpe_vals else 0.0,
        "mean_ic_mean": float(np.mean(ic_vals)) if ic_vals else 0.0,
        "mean_ic_std": float(np.std(ic_vals)) if ic_vals else 0.0,
    }

    oos_df = pd.concat(oos_preds_list, ignore_index=True) if oos_preds_list else None

    return WalkForwardResult(
        windows=windows,
        oos_predictions=oos_df,
        summary=summary,
    )


def _generate_predictions(
    stocks: list[str],
    start: str,
    end: str,
    mode: str,
    train_start: str | None = None,
    train_end: str | None = None,
) -> pd.DataFrame | None:
    """Generate predictions for a date range.

    In "full" mode, retrains MS-LSTM/DualGAT/Ensemble on the training window
    before predicting on the validation window.
    In "params" mode, uses existing model instances.
    """
    from src.backtest.calendar import trading_days_between
    from src.model.baseline import BaselinePredictor
    from src.expert.tracker import ExpertTracker

    dates = trading_days_between(start, end)
    if not dates:
        return None

    if mode == "full" and train_start and train_end:
        # Retrain models on training window
        _retrain_models(stocks, train_start, train_end)

    # Use default predictors (which load from disk, or Baseline)
    from src.model.ms_lstm import MSLSTMPredictor
    from src.model.dualgat import DualGATPredictor
    from src.model.ensemble import EnsemblePredictor

    baseline = BaselinePredictor()
    tracker = ExpertTracker()

    # Try loading MS-LSTM
    ms_lstm = None
    try:
        ms_lstm = MSLSTMPredictor()
        ms_lstm.load(MSLSTM_MODEL_PATH)
    except Exception:
        pass

    # Try loading DualGAT
    dualgat = None
    try:
        dualgat = DualGATPredictor()
        dualgat.load(DUALGAT_MODEL_PATH)
    except Exception:
        pass

    # Try loading Ensemble
    ensemble = None
    try:
        ensemble = EnsemblePredictor()
        ensemble.load(ENSEMBLE_MODEL_PATH)
    except Exception:
        pass

    all_preds = []
    for date_str in dates:
        expert_records = tracker.trace(date_str)

        if ensemble is not None:
            bl_df = baseline.predict(stocks, date_str, expert_records)
            ms_df = ms_lstm.predict(stocks, date_str) if ms_lstm else bl_df.copy()
            dg_df = dualgat.predict(stocks, date_str) if dualgat else bl_df.copy()
            pred_df = ensemble.predict(stocks, date_str, bl_df, ms_df, dg_df)
        elif dualgat is not None:
            pred_df = dualgat.predict(stocks, date_str)
        elif ms_lstm is not None:
            pred_df = ms_lstm.predict(stocks, date_str)
        else:
            pred_df = baseline.predict(stocks, date_str, expert_records)

        all_preds.append(pred_df)

    return pd.concat(all_preds, ignore_index=True) if all_preds else None


def _retrain_models(stocks: list[str], train_start: str, train_end: str) -> None:
    """Retrain MS-LSTM, DualGAT, and Ensemble on the given window."""
    logger.info(f"Retraining models on window [{train_start}, {train_end}]")

    try:
        # MS-LSTM
        from src.model.ms_lstm import MSLSTMPredictor
        ms = MSLSTMPredictor()
        ms.fit(stocks, train_start, train_end)
        ms.save(MSLSTM_MODEL_PATH)
    except Exception as e:
        logger.warning(f"MS-LSTM retrain failed: {e}")

    try:
        # DualGAT (requires pre-trained MS-LSTM)
        from src.model.dualgat import DualGATPredictor
        dg = DualGATPredictor()
        dg.fit(stocks, train_start, train_end, ms_lstm_path=str(MSLSTM_MODEL_PATH))
        dg.save(DUALGAT_MODEL_PATH)
    except Exception as e:
        logger.warning(f"DualGAT retrain failed: {e}")

    try:
        # Ensemble meta-learner
        from src.model.baseline import BaselinePredictor
        from src.model.ms_lstm import MSLSTMPredictor
        from src.model.dualgat import DualGATPredictor
        from src.model.ensemble import EnsemblePredictor

        baseline = BaselinePredictor()
        ms = MSLSTMPredictor()
        ms.load(MSLSTM_MODEL_PATH)
        dg = DualGATPredictor()
        dg.load(DUALGAT_MODEL_PATH)

        ensemble = EnsemblePredictor(strategy="meta")
        ensemble.fit_meta(stocks, train_start, train_end, baseline, ms, dg, epochs=30)
        ensemble.save(ENSEMBLE_MODEL_PATH)
    except Exception as e:
        logger.warning(f"Ensemble retrain failed: {e}")

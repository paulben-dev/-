"""Rule-based baseline predictor for stock returns.

Strategy:
- Has expert signal → use expert direction + 30-day average return magnitude
- No expert signal → use 20-day momentum factor
"""
import logging
import pandas as pd
from src.model.signal import transform_expert_signal, compute_expert_availability
from src.model.features import compute_momentum

logger = logging.getLogger(__name__)


class BaselinePredictor:
    """Simple rule-based stock return predictor for MVP."""

    def predict(
        self,
        stocks: list[str],
        date_str: str,
        expert_records: list | None = None,
    ) -> pd.DataFrame:
        """Generate daily return ratio predictions for all stocks.

        Args:
            stocks: List of ticker symbols.
            date_str: Prediction date (YYYY-MM-DD).
            expert_records: Expert identification results for this date.

        Returns:
            DataFrame with columns [stock, date, predicted_return, signal_source].
        """
        if not stocks:
            df = pd.DataFrame(columns=["stock", "date", "predicted_return", "signal_source"])
            logger.info(f"Generated 0 predictions for {date_str}")
            return df

        expert_records = expert_records or []
        expert_signals = transform_expert_signal(expert_records, date_str)
        expert_avail = compute_expert_availability(expert_records, stocks)
        momentum = compute_momentum(stocks, date_str)

        predictions = []
        for stock in stocks:
            if expert_avail[stock] == 1 and stock in expert_signals:
                pred = expert_signals[stock]
                source = "expert"
            else:
                pred = momentum.get(stock, 0.0)
                source = "momentum"

            predictions.append({
                "stock": stock,
                "date": date_str,
                "predicted_return": pred,
                "signal_source": source,
            })

        df = pd.DataFrame(predictions)
        # Normalize predictions cross-sectionally for ranking
        if len(df) > 0:
            mean = df["predicted_return"].mean()
            std = df["predicted_return"].std()
            if std > 0:
                df["predicted_return"] = (df["predicted_return"] - mean) / std

        if len(df) > 0:
            expert_count = df["signal_source"].value_counts().get("expert", 0)
            logger.info(f"Generated {len(df)} predictions for {date_str}, {expert_count} from experts")
        else:
            logger.info(f"Generated 0 predictions for {date_str}")
        return df.sort_values("predicted_return", ascending=False)

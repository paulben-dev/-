"""Slippage and market-impact cost model.

Simplified Almgren-Chriss: fixed commission + volume-proportional impact.
"""
from dataclasses import dataclass
from config import SLIPPAGE_FIXED_COST, SLIPPAGE_IMPACT_FACTOR, SLIPPAGE_VOLUME_FRACTION


@dataclass
class SlippageConfig:
    """Slippage model parameters.

    Attributes:
        fixed_cost: Fixed commission as fraction of trade value (e.g. 0.0004 = 4bp).
        impact_factor: Coefficient for price impact relative to participation rate.
        daily_volume_fraction: Assumed maximum fraction of daily volume we can trade.
    """
    fixed_cost: float = SLIPPAGE_FIXED_COST
    impact_factor: float = SLIPPAGE_IMPACT_FACTOR
    daily_volume_fraction: float = SLIPPAGE_VOLUME_FRACTION


def estimate_slippage(
    weight: float,
    daily_volume: int,
    close_price: float,
    config: SlippageConfig,
) -> float:
    """Estimate total slippage cost as fraction of trade value.

    Formula:
        cost = fixed_cost
             + impact_factor * |weight| / (daily_volume_frac * daily_volume)

    When daily_volume is zero or the participation denominator is zero,
    only the fixed_cost is charged.

    Args:
        weight: Portfolio weight of the trade (sign ignored, magnitude matters).
        daily_volume: Reported daily volume in shares.
        close_price: Closing price per share (not used in simplified formula,
                     kept for future extensions).
        config: Slippage model parameters.

    Returns:
        Slippage cost as a fraction of trade value.
    """
    cost = config.fixed_cost
    abs_weight = abs(weight)

    if abs_weight == 0.0:
        return 0.0

    denominator = config.daily_volume_fraction * daily_volume
    if denominator > 0:
        cost += config.impact_factor * abs_weight / denominator

    return cost

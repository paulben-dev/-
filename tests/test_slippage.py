"""Tests for slippage / market-impact model."""
import pytest
from src.backtest.slippage import SlippageConfig, estimate_slippage


class TestEstimateSlippage:
    @pytest.fixture
    def config(self):
        return SlippageConfig()

    def test_zero_trade_zero_slippage(self, config):
        """Zero-weight trade incurs zero slippage."""
        cost = estimate_slippage(0.0, 1_000_000, 100.0, config)
        assert cost == 0.0

    def test_slippage_increases_with_trade_size(self, config):
        """Larger trades incur more slippage."""
        small = estimate_slippage(0.01, 1_000_000, 100.0, config)
        large = estimate_slippage(0.05, 1_000_000, 100.0, config)
        assert large > small

    def test_minimum_cost_is_fixed_cost(self, config):
        """Even tiny trades pay at least the fixed commission."""
        cost = estimate_slippage(0.0001, 10_000_000, 100.0, config)
        assert cost >= config.fixed_cost

    def test_slippage_decreases_with_volume(self, config):
        """Higher daily volume means lower impact."""
        low_vol = estimate_slippage(0.01, 100_000, 100.0, config)
        high_vol = estimate_slippage(0.01, 10_000_000, 100.0, config)
        assert low_vol > high_vol

    def test_negative_weight_same_as_positive(self, config):
        """Short and long positions have same slippage for equal |weight|."""
        long_cost = estimate_slippage(0.01, 1_000_000, 100.0, config)
        short_cost = estimate_slippage(-0.01, 1_000_000, 100.0, config)
        assert long_cost == short_cost

    def test_zero_volume_returns_fixed_cost(self, config):
        """Zero reported volume still returns fixed_cost (no division by zero)."""
        cost = estimate_slippage(0.01, 0, 100.0, config)
        assert cost >= config.fixed_cost
        assert cost == config.fixed_cost

    def test_custom_config(self):
        """Custom SlippageConfig values are respected."""
        cfg = SlippageConfig(fixed_cost=0.001, impact_factor=0.5, daily_volume_fraction=0.02)
        cost = estimate_slippage(0.01, 1_000_000, 100.0, cfg)
        assert cost > 0.001  # impact adds on top of fixed

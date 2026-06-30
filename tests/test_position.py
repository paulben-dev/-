"""Tests for risk-aware position sizing."""
import pytest
import numpy as np
from src.backtest.position import PositionConfig, size_positions, GICS_SECTORS


class TestPositionConfig:
    def test_defaults_values(self):
        """Default config values match POSITION_* constants (vol target and
        single-stock cap are non-zero, i.e. enabled by default)."""
        cfg = PositionConfig()
        assert cfg.target_vol == 0.15
        assert cfg.max_single_weight == 0.05
        assert cfg.sector_neutral is False
        assert cfg.max_turnover == 1.0


class TestGicsSectors:
    def test_all_default_tickers_mapped(self):
        """All 20 DEFAULT_TICKERS have GICS sector assignments."""
        from config import DEFAULT_TICKERS
        for t in DEFAULT_TICKERS:
            assert t in GICS_SECTORS, f"{t} missing from GICS_SECTORS"


class TestSizePositions:
    @pytest.fixture
    def stocks(self):
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

    @pytest.fixture
    def preds(self):
        return np.array([0.05, 0.03, -0.01, 0.02, 0.04])

    @pytest.fixture
    def prices(self):
        """Recent close prices for vol computation (5 days each, high vol)."""
        return {
            "AAPL":  [175.0, 182.0, 168.0, 185.0, 178.0],
            "MSFT":  [380.0, 395.0, 370.0, 405.0, 388.0],
            "GOOGL": [135.0, 142.0, 130.0, 145.0, 138.0],
            "AMZN":  [170.0, 178.0, 165.0, 182.0, 175.0],
            "NVDA":  [780.0, 820.0, 760.0, 830.0, 795.0],
        }

    def test_no_constraints_returns_equal_weights(self, stocks, preds, prices):
        """With all constraints disabled, long/short get equal absolute weights."""
        cfg = PositionConfig(target_vol=0.0, max_single_weight=0.0,
                             sector_neutral=False, max_turnover=0.0)
        weights = size_positions(stocks, preds, {}, prices, cfg)
        # 5 stocks, top/bottom 10% = 1 long, 1 short (with quantile=0.10)
        assert len(weights) > 0
        assert abs(sum(weights.values())) < 0.01  # long ≈ short

    def test_max_single_weight_enforced(self, stocks, preds, prices):
        """No single position exceeds the cap."""
        cfg = PositionConfig(target_vol=0.0, max_single_weight=0.03,
                             sector_neutral=False, max_turnover=0.0)
        weights = size_positions(stocks, preds, {}, prices, cfg)
        for w in weights.values():
            assert abs(w) <= 0.03 + 1e-10

    def test_sector_neutral_applied(self):
        """When sector_neutral=True, long/short exposure is equalized per
        GICS sector.  Sectors with only one side are left untouched."""
        # 8-stock universe: IT has AAPL/MSFT long + NVDA short => overlap
        stocks_sn = ["AAPL", "MSFT", "GOOGL", "META", "JPM",
                     "JNJ", "NVDA", "XOM"]
        preds_sn = np.array([0.10, 0.08, 0.06, 0.04, 0.02,
                             -0.01, -0.04, -0.06], dtype=float)

        cfg = PositionConfig(target_vol=0.0, max_single_weight=0.0,
                             sector_neutral=True, max_turnover=0.0)
        weights = size_positions(stocks_sn, preds_sn, {}, {},
                                 cfg, quantile=0.25)

        # Aggregate long / short exposure per sector
        sectors: dict[str, dict[str, float]] = {}
        for s, w in weights.items():
            sec = GICS_SECTORS.get(s, "Unknown")
            sectors.setdefault(sec, {"long": 0.0, "short": 0.0})
            if w > 0:
                sectors[sec]["long"] += w
            else:
                sectors[sec]["short"] += abs(w)

        # IT: AAPL + MSFT (long) vs NVDA (short) => should be balanced
        it = sectors.get("Information Technology", {})
        assert it.get("long", 0) > 0, "Expected IT long exposure"
        assert it.get("short", 0) > 0, "Expected IT short exposure"
        assert abs(it["long"] - it["short"]) < 1e-10, \
            f"IT not neutral: long={it['long']}, short={it['short']}"

        # Energy (XOM short only) should be left as-is
        energy = sectors.get("Energy", {})
        assert energy.get("short", 0) > 0, "Expected Energy short exposure"
        assert energy.get("long", 0) == 0, "Energy should have no long leg"

    def test_turnover_constraint_enforced(self, stocks, preds, prices):
        """Turnover between consecutive days is limited."""
        cfg = PositionConfig(target_vol=0.0, max_single_weight=0.0,
                             sector_neutral=False, max_turnover=0.5)
        prev_weights = {"AAPL": 0.10, "MSFT": -0.10}
        weights = size_positions(stocks, preds, prev_weights, prices, cfg)
        # Total change in |weight| should not exceed max_turnover
        turnover = sum(abs(weights.get(s, 0.0) - prev_weights.get(s, 0.0))
                       for s in set(stocks) | set(prev_weights)) / 2
        assert turnover <= 0.5 + 1e-10

    def test_vol_scaling_reduces_weights(self, stocks, preds, prices):
        """When target_vol is set, weights are scaled down from raw."""
        cfg_no_vol = PositionConfig(target_vol=0.0, max_single_weight=0.0,
                                     sector_neutral=False, max_turnover=0.0)
        cfg_vol = PositionConfig(target_vol=0.10, max_single_weight=0.0,
                                  sector_neutral=False, max_turnover=0.0)
        w_no_vol = size_positions(stocks, preds, {}, prices, cfg_no_vol)
        w_vol = size_positions(stocks, preds, {}, prices, cfg_vol)
        no_vol_sum = sum(abs(v) for v in w_no_vol.values())
        vol_sum = sum(abs(v) for v in w_vol.values())
        # Vol-constrained weights should be <= unconstrained
        assert vol_sum <= no_vol_sum + 1e-10

    def test_empty_stocks_returns_empty(self):
        """Empty stock list returns empty dict."""
        cfg = PositionConfig()
        result = size_positions([], np.array([]), {}, {}, cfg)
        assert result == {}

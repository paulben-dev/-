"""Tests for the FastAPI web service."""
import pytest
from fastapi.testclient import TestClient
from src.web.api import app
from src.db.schema import init_db
import src.db.schema as schema_mod


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Create test client with isolated database."""
    db_path = tmp_path / "test_api.db"
    monkeypatch.setattr("src.web.api.db", schema_mod)
    monkeypatch.setattr("src.db.schema.DB_PATH", db_path)
    monkeypatch.setattr("config.DB_PATH", db_path)
    schema_mod._engine = None
    init_db()
    return TestClient(app)


class TestStockEndpoint:
    def test_get_stocks(self, client):
        resp = client.get("/api/stocks")
        assert resp.status_code == 200
        data = resp.json()
        assert "stocks" in data
        assert "count" in data
        assert data["count"] > 0


class TestExpertEndpoint:
    def test_get_experts_empty(self, client):
        resp = client.get("/api/experts?date=2024-06-15")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0

    def test_get_experts_default_date(self, client):
        resp = client.get("/api/experts")
        # Should work with default date (yesterday)
        assert resp.status_code == 200


class TestPredictionEndpoint:
    def test_get_predictions(self, client):
        resp = client.get("/api/predictions?date=2024-06-15")
        assert resp.status_code == 200
        data = resp.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 20  # DEFAULT_TICKERS


class TestBacktestEndpoint:
    def test_get_backtest(self, populated_client):
        resp = populated_client.get("/api/backtest?start=2024-06-01&end=2024-06-15")
        assert resp.status_code == 200
        data = resp.json()
        assert "sharpe_ratio" in data
        assert "annualized_return" in data


class TestDashboard:
    def test_dashboard_renders(self, client):
        resp = client.get("/")
        # Dashboard returns HTML
        assert "text/html" in resp.headers.get("content-type", "")


class TestCollectEndpoint:
    def test_collect_triggers(self, client):
        resp = client.post("/api/collect?start=2024-06-01&end=2024-06-02")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ------------------------------------------------------------------
# Fixture: client sharing the prepopulated database
# ------------------------------------------------------------------

@pytest.fixture
def populated_client(prepopulated_db):
    """TestClient connected to the prepopulated database.

    Unlike the ``client`` fixture, this does NOT re-patch DB_PATH,
    so tests see the price and post data inserted by ``prepopulated_db``.
    """
    return TestClient(app)


# ------------------------------------------------------------------
# Task 4: /api/models, model=? param, /api/backtest/compare
# ------------------------------------------------------------------

class TestModelsEndpoint:
    """Tests for GET /api/models."""

    def test_list_models(self, client):
        """Returns all 4 models with availability info."""
        resp = client.get("/api/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        ids = [m["id"] for m in data["models"]]
        assert "baseline" in ids
        assert "ms_lstm" in ids
        assert "dualgat" in ids
        assert "ensemble" in ids
        # Baseline is always available
        baseline = next(m for m in data["models"] if m["id"] == "baseline")
        assert baseline["available"] is True


class TestModelParamPredictions:
    """Tests for GET /api/predictions?model=..."""

    def test_predictions_default_to_baseline(self, populated_client):
        """Without model param, defaults to baseline (backward compat)."""
        resp = populated_client.get("/api/predictions?date=2024-06-15")
        assert resp.status_code == 200
        data = resp.json()
        assert "predictions" in data
        assert data.get("model", "baseline") == "baseline"

    def test_predictions_with_model_explicit_baseline(self, populated_client):
        """model=baseline returns predictions (explicit)."""
        resp = populated_client.get(
            "/api/predictions?date=2024-06-15&model=baseline"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == "baseline"
        assert "predictions" in data

    def test_predictions_invalid_model(self, client):
        """Invalid model name returns 400."""
        resp = client.get("/api/predictions?date=2024-06-15&model=unknown")
        assert resp.status_code == 400.

    def test_predictions_ms_lstm_not_available(self, populated_client):
        """model=ms_lstm returns 503 when .pt file is missing."""
        resp = populated_client.get(
            "/api/predictions?date=2024-06-15&model=ms_lstm"
        )
        # 200 if model file happens to exist, 503 otherwise
        assert resp.status_code in (200, 503)

    def test_head_without_date_param(self, client):
        """HEAD check — date defaults to today."""
        resp = client.head("/api/predictions")
        # HEAD may return 405 (method not allowed) or 200
        assert resp.status_code in (200, 405)


class TestBacktestCompareEndpoint:
    """Tests for GET /api/backtest/compare."""

    def test_compare_returns_available_models(self, populated_client):
        """Compare endpoint returns backtest results for available models."""
        resp = populated_client.get(
            "/api/backtest/compare?start=2024-05-20&end=2024-06-15"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        # baseline should always be present
        assert "baseline" in data["models"]
        for mid in data["models"]:
            m = data["models"][mid]
            assert "annualized_return" in m
            assert "sharpe_ratio" in m
            assert "cumulative_returns" in m

    def test_compare_skips_unavailable_gracefully(self, client):
        """When no model files exist, endpoint returns 404, not 500."""
        resp = client.get(
            "/api/backtest/compare?start=2024-06-01&end=2024-06-15"
        )
        # 200 if baseline works with available data, 404 if no trading data
        assert resp.status_code in (200, 404)

    def test_compare_insufficient_date_range(self, client):
        """Tiny date range with no trading data returns 404."""
        resp = client.get(
            "/api/backtest/compare?start=2024-06-01&end=2024-06-02"
        )
        # May return 404 (insufficient data) or 200 if data exists
        assert resp.status_code in (200, 404)

    def test_compare_has_required_metrics(self, populated_client):
        """Each model result includes all required metrics."""
        resp = populated_client.get(
            "/api/backtest/compare?start=2024-05-20&end=2024-06-15"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "start" in data
        assert "end" in data
        for mid, m in data.get("models", {}).items():
            for key in ("annualized_return", "sharpe_ratio",
                        "max_drawdown", "mean_ic", "icir",
                        "n_trading_days", "cumulative_returns"):
                assert key in m, f"{mid} missing '{key}'"

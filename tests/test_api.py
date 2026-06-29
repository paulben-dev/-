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
    def test_get_backtest(self, client):
        resp = client.get("/api/backtest?start=2024-06-01&end=2024-06-15")
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

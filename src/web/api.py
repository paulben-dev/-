"""FastAPI web service for the stock prediction system."""
import logging
import os
from datetime import datetime, timedelta

import pandas as pd
from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.db.schema import init_db, get_expert_records
from src.expert.tracker import ExpertTracker
from src.model.baseline import BaselinePredictor
from src.backtest.portfolio import run_backtest
from src.data.yfinance import YFinanceCollector
from src.data.stocktwits import StockTwitsCollector
from src.data.reddit import RedditCollector
from src.db import schema as db
from config import DEFAULT_TICKERS, PORTFOLIO_QUANTILE, API_HOST, API_PORT

logger = logging.getLogger(__name__)

# Initialize app
app = FastAPI(title="DualGAT Stock Predictor", version="0.1.0")

# Static files and templates
static_dir = os.path.join(os.path.dirname(__file__), "static")
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)
# Disable template caching to avoid Jinja2 cache-key issue with request context
templates.env.cache = None

# Services (lazy init)
_tracker = None
_predictor = None


def get_tracker() -> ExpertTracker:
    global _tracker
    if _tracker is None:
        _tracker = ExpertTracker()
    return _tracker


def get_predictor() -> BaselinePredictor:
    global _predictor
    if _predictor is None:
        _predictor = BaselinePredictor()
    return _predictor


@app.on_event("startup")
async def startup():
    init_db()
    logger.info("Database initialized")


@app.get("/api/stocks")
async def get_stocks():
    """Return the stock universe."""
    return {"stocks": DEFAULT_TICKERS, "count": len(DEFAULT_TICKERS)}


@app.get("/api/experts")
async def get_experts(date: str = Query(None, description="Date in YYYY-MM-DD format")):
    """Get expert records for a given date. Defaults to latest available."""
    if date is None:
        date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    # Try to trace experts for this date
    tracker = get_tracker()
    records = tracker.trace(date)

    return {
        "date": date,
        "expert_count": len([r for r in records if r.expert_type == "expert"]),
        "inverse_expert_count": len([r for r in records if r.expert_type == "inverse_expert"]),
        "total": len(records),
        "experts": [
            {
                "user_id": r.user_id,
                "stock": r.stock,
                "expert_type": r.expert_type,
                "predicted_direction": r.predicted_direction,
                "accuracy_recent": r.accuracy_recent,
                "accuracy_long": r.accuracy_long,
            }
            for r in records
        ],
    }


@app.get("/api/predictions")
async def get_predictions(date: str = Query(None, description="Date in YYYY-MM-DD format")):
    """Get stock return predictions for a given date."""
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    tracker = get_tracker()
    predictor = get_predictor()

    expert_records = tracker.trace(date)
    pred_df = predictor.predict(DEFAULT_TICKERS, date, expert_records)

    return {
        "date": date,
        "predictions": pred_df.to_dict(orient="records"),
        "expert_coverage": len([r for r in expert_records if r.expert_type != "none"]),
    }


@app.get("/api/backtest")
async def get_backtest(
    start: str = Query(None, description="Start date YYYY-MM-DD"),
    end: str = Query(None, description="End date YYYY-MM-DD"),
):
    """Run backtest over a date range."""
    if start is None:
        start = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    if end is None:
        end = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    tracker = get_tracker()
    predictor = get_predictor()

    # Collect all trading dates with price data to avoid weekends/holidays
    from src.db import schema as db_schema
    all_prices = db_schema.get_prices(DEFAULT_TICKERS, start, end)
    trading_dates = set()
    for stock_prices in all_prices.values():
        for p in stock_prices:
            trading_dates.add(p["date"])
    trading_dates = sorted(trading_dates)

    # Generate predictions for each trading day
    all_preds = []
    for date_str in trading_dates:
        if date_str < start or date_str > end:
            continue
        expert_records = tracker.trace(date_str)
        pred_df = predictor.predict(DEFAULT_TICKERS, date_str, expert_records)
        all_preds.append(pred_df)

    if not all_preds:
        raise HTTPException(404, "No predictions generated for the date range")

    combined_preds = pd.concat(all_preds, ignore_index=True)
    results = run_backtest(combined_preds, DEFAULT_TICKERS, start, end)

    return {
        "start": start,
        "end": end,
        "annualized_return": results["annualized_return"],
        "sharpe_ratio": results["sharpe_ratio"],
        "max_drawdown": results["max_drawdown"],
        "mean_ic": results["mean_ic"],
        "icir": results["icir"],
        "n_trading_days": results["n_trading_days"],
        "cumulative_returns": results["cumulative_returns"].tolist(),
    }


@app.post("/api/collect")
async def trigger_collection(
    start: str = Query(None, description="Start date YYYY-MM-DD"),
    end: str = Query(None, description="End date YYYY-MM-DD"),
):
    """Trigger data collection from all sources."""
    if start is None:
        start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    results = {"prices": 0, "stocktwits": 0, "reddit": 0}

    # Collect prices
    try:
        yf_collector = YFinanceCollector()
        prices = yf_collector.collect_prices(DEFAULT_TICKERS, start, end)
        db.insert_prices(prices)
        results["prices"] = len(prices)
    except Exception as e:
        results["prices_error"] = str(e)

    # Collect social posts
    try:
        st_collector = StockTwitsCollector()
        start_dt = datetime.fromisoformat(start)
        end_dt = datetime.fromisoformat(end)
        for i in range((end_dt - start_dt).days + 1):
            date_str = (start_dt + timedelta(days=i)).strftime("%Y-%m-%d")
            posts = st_collector.collect_social_posts(DEFAULT_TICKERS, date_str)
            db.insert_posts(posts)
            results["stocktwits"] += len(posts)
    except Exception as e:
        results["stocktwits_error"] = str(e)

    try:
        reddit_collector = RedditCollector()
        start_dt = datetime.fromisoformat(start)
        end_dt = datetime.fromisoformat(end)
        for i in range((end_dt - start_dt).days + 1):
            date_str = (start_dt + timedelta(days=i)).strftime("%Y-%m-%d")
            posts = reddit_collector.collect_social_posts(DEFAULT_TICKERS, date_str)
            db.insert_posts(posts)
            results["reddit"] += len(posts)
    except Exception as e:
        results["reddit_error"] = str(e)

    return {"status": "ok", "results": results}


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Render the English dashboard."""
    return templates.TemplateResponse(request, "index.html")


@app.get("/zh", response_class=HTMLResponse)
async def dashboard_zh(request: Request):
    """Render the Chinese dashboard."""
    return templates.TemplateResponse(request, "index_zh.html")

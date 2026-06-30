"""FastAPI web service for the stock prediction system."""
import asyncio
import json
import logging
import os
import queue
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import pandas as pd
from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
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

# Thread pool for blocking I/O (data collection, etc.)
_executor = ThreadPoolExecutor(max_workers=2)

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


def _do_collect(
    start: str,
    end: str,
    stocks: list[str],
    progress_callback: callable = None,
) -> dict:
    """Run blocking data collection (called from thread pool).

    If progress_callback is provided, it is called with dicts like:
        {"step": "prices", "msg": "...", "status": "running|ok|error"}
    """
    results = {"prices": 0, "stocktwits": 0, "reddit": 0}
    _emit = progress_callback or (lambda _evt: None)

    # ── Collect prices ──
    _emit({"step": "prices", "msg": "🔍 正在获取价格数据...", "status": "running"})
    try:
        yf_collector = YFinanceCollector()
        prices = yf_collector.collect_prices(stocks, start, end)
        db.insert_prices(prices)
        results["prices"] = len(prices)
        _emit({"step": "prices", "msg": f"✅ 已采集 {len(prices)} 条价格数据", "status": "ok"})
    except Exception as e:
        results["prices_error"] = str(e)
        _emit({"step": "prices", "msg": f"❌ 价格采集失败: {e}", "status": "error"})

    # ── Collect StockTwits ──
    _emit({"step": "stocktwits", "msg": "📡 正在获取 StockTwits 帖子...", "status": "running"})
    try:
        st_collector = StockTwitsCollector()
        all_st_posts = []
        batch_size = 5
        for i in range(0, len(stocks), batch_size):
            batch = stocks[i:i + batch_size]
            posts = st_collector.collect_social_posts(batch)
            all_st_posts.extend(posts)
            done = min(i + batch_size, len(stocks))
            _emit({"step": "stocktwits", "msg": f"📡 StockTwits: {done}/{len(stocks)} 只股票...", "status": "running"})
        if all_st_posts:
            db.insert_posts(all_st_posts)
        results["stocktwits"] = len(all_st_posts)
        _emit({"step": "stocktwits", "msg": f"✅ 已采集 {len(all_st_posts)} 条 StockTwits 帖子", "status": "ok"})
    except Exception as e:
        results["stocktwits_error"] = str(e)
        _emit({"step": "stocktwits", "msg": f"⚠️ StockTwits 跳过: {e}", "status": "error"})

    # ── Collect Reddit ──
    _emit({"step": "reddit", "msg": "📡 正在获取 Reddit 帖子...", "status": "running"})
    try:
        reddit_collector = RedditCollector()
        all_rdt_posts = []
        for i in range(0, len(stocks), batch_size):
            batch = stocks[i:i + batch_size]
            posts = reddit_collector.collect_social_posts(batch)
            all_rdt_posts.extend(posts)
            done = min(i + batch_size, len(stocks))
            _emit({"step": "reddit", "msg": f"📡 Reddit: {done}/{len(stocks)} 只股票...", "status": "running"})
        if all_rdt_posts:
            db.insert_posts(all_rdt_posts)
        results["reddit"] = len(all_rdt_posts)
        _emit({"step": "reddit", "msg": f"✅ 已采集 {len(all_rdt_posts)} 条 Reddit 帖子", "status": "ok"})
    except Exception as e:
        results["reddit_error"] = str(e)
        _emit({"step": "reddit", "msg": f"⚠️ Reddit 跳过: {e}", "status": "error"})

    # ── Done ──
    summary = f"价格 {results['prices']} 条"
    if results.get("stocktwits"):
        summary += f", StockTwits {results['stocktwits']} 条"
    if results.get("reddit"):
        summary += f", Reddit {results['reddit']} 条"
    _emit({"step": "done", "msg": f"🎉 采集完成 — {summary}", "status": "done"})

    return results


@app.post("/api/collect")
async def trigger_collection(
    start: str = Query(None, description="Start date YYYY-MM-DD"),
    end: str = Query(None, description="End date YYYY-MM-DD"),
):
    """Trigger data collection from all sources (runs in background thread)."""
    if start is None:
        start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        _executor, _do_collect, start, end, DEFAULT_TICKERS
    )

    return {"status": "ok", "results": results}


@app.get("/api/collect/stream")
async def trigger_collection_stream(
    start: str = Query(None, description="Start date YYYY-MM-DD"),
    end: str = Query(None, description="End date YYYY-MM-DD"),
):
    """Trigger data collection with SSE progress streaming."""
    if start is None:
        start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    q: queue.Queue = queue.Queue()

    def on_progress(evt: dict):
        q.put(evt)

    loop = asyncio.get_event_loop()

    async def generate():
        # Kick off collection in a background thread
        future = loop.run_in_executor(
            _executor, _do_collect, start, end, DEFAULT_TICKERS, on_progress
        )

        done = False
        while not done:
            try:
                evt = q.get_nowait()
                yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"
                if evt.get("step") == "done":
                    done = True
            except queue.Empty:
                await asyncio.sleep(0.2)

        await future  # Ensure thread completes cleanly

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Render the English dashboard."""
    return templates.TemplateResponse(request, "index.html")


@app.get("/zh", response_class=HTMLResponse)
async def dashboard_zh(request: Request):
    """Render the Chinese dashboard."""
    return templates.TemplateResponse(request, "index_zh.html")

# API

## Purpose
FastAPI REST API with 20+ endpoints for portfolio monitoring, trade management, risk status, ML model info, news analysis, and Prometheus metrics. Telegram webhook integration.

## Layer
Layer 6 -- API / Dashboard. Can import from all layers 0-5.

## Key Files
- `v1/router.py` -- Main API router aggregating all sub-routers
- `v1/system.py` -- Health check, readiness, version endpoints
- `v1/portfolio.py` -- Portfolio state, equity, positions endpoints
- `v1/trades.py` -- Trade history, recent trades, P&L endpoints
- `v1/signals.py` -- Active signals, signal history endpoints
- `v1/risk.py` -- Circuit breaker status, drawdown, exposure endpoints
- `v1/ml.py` -- ML model status, feature importance, predictions endpoints
- `v1/news.py` -- News feed, sentiment analysis endpoints
- `v1/auth.py` -- API key authentication middleware
- `v1/telegram.py` -- Telegram webhook router for bot commands
- `alerts.py` -- TelegramAlerter with priority queue, rate limiting (20 msg/min), batching (moved from core/ in Phase 22)
- `telegram_bot.py` -- TelegramBotHandler for webhook-based commands: /status, /breakers, /stop, /kill, /gonogo (moved from core/ in Phase 22)
- `metrics.py` -- Prometheus MetricsCollector: portfolio equity, trade counts, slippage histograms, drawdown gauges, signal counters

## Public API
- `api_router` -- FastAPI APIRouter with all v1 endpoints mounted
- `MetricsCollector` -- static methods for recording Prometheus metrics (set_portfolio_equity, record_trade, etc.)
- `TelegramAlerter` -- alert dispatch (no-op when token is empty)
- `TelegramBotHandler` -- webhook command handler

## Contracts
- Input: HTTP requests with API key auth, JSON request bodies
- Output: JSON responses following REST conventions, Prometheus /metrics endpoint
- Invariants: All endpoints require API key authentication (via `v1/auth.py`). Telegram webhook validates `telegram_webhook_secret`. Metrics use label dimensions: market, period, strategy.

## Testing
- Test location: `tests/unit/test_api_*.py` (test_api_health.py, test_api_portfolio.py, test_api_trades.py, test_api_signals_risk.py, test_api_system.py, test_api_auth.py)
- Run: `uv run pytest tests/unit/test_api_health.py tests/unit/test_api_portfolio.py tests/unit/test_api_system.py -v`

## Common Patterns
- Each domain has its own sub-router file in `v1/`
- Routers are assembled in `v1/router.py` via `include_router()`
- Telegram webhook is conditionally mounted in `main.py` based on config
- MetricsCollector uses static methods so callers need no instance
- FastAPI dependency injection used for database sessions (`get_db()`)

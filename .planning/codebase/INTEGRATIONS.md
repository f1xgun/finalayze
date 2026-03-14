# External Integrations

**Analysis Date:** 2026-03-14

## APIs & External Services

**Brokers (Execution):**
- Alpaca (US equities) - REST + WebSocket
  - SDK: `alpaca-py` 0.43.2+
  - Auth: `FINALAYZE_ALPACA_API_KEY`, `FINALAYZE_ALPACA_SECRET_KEY` (env vars)
  - Paper/Live: `FINALAYZE_ALPACA_PAPER=true` for paper trading, false for live
  - Implementation: `src/finalayze/execution/alpaca_broker.py`
  - Max portfolio: `FINALAYZE_ALPACA_MAX_PORTFOLIO_VALUE` (default 10,000 USD)

- Tinkoff Invest / T-Bank (MOEX equities & bonds)
  - SDK: `t-tech-investments` 0.2.0+ (custom T-Bank index)
  - Auth: `FINALAYZE_TINKOFF_TOKEN` (gRPC token)
  - Endpoints: Production `invest-public-api.tbank.ru:443`, Sandbox `sandbox-invest-public-api.tbank.ru:443`
  - Implementation: `src/finalayze/execution/tinkoff_broker.py`
  - Max portfolio: `FINALAYZE_TINKOFF_MAX_PORTFOLIO_VALUE` (default 500,000 RUB)
  - Note: SDK defaults to old `tinkoff.ru` domain; code overrides to `tbank.ru`

**Market Data (Candles & OHLCV):**
- Alpaca REST (via alpaca-py)
  - US equities, derivatives
  - Market hours: 9:30-16:00 ET

- Tinkoff Invest gRPC (via t-tech-investments)
  - MOEX equities, indices, bonds
  - Market hours: 10:00-18:45 MSK
  - Implementation: `src/finalayze/data/fetchers/tinkoff_data.py`
  - Candle intervals: 1m, 1h, 1d (via `CandleInterval` enum)

- Yahoo Finance (yfinance 0.2.50)
  - US equities, indices, forex
  - Implementation: `src/finalayze/data/fetchers/yfinance.py`
  - NOTE: Cannot fetch MOEX tickers (SBER, GAZP, LKOH, etc.)

- Finnhub REST (finnhub-python 2.4.20)
  - US equities candles via `/stock/candle` endpoint
  - API: `https://finnhub.io/api/v1`
  - Auth: `FINALAYZE_FINNHUB_API_KEY`
  - Implementation: `src/finalayze/data/fetchers/finnhub.py`
  - Supports timeframes: 1m, 1h, 1d

- MOEX ISS REST (MOEX Information & Statistical Server)
  - MOEX index candles, market turnover
  - API: `https://iss.moex.com/iss`
  - Implementation: `src/finalayze/data/fetchers/moex_iss.py`
  - Pagination: 100 rows/page, chunk-by-year for multi-year fetches

**News & Sentiment:**
- NewsAPI (`newsapi.org/v2/everything`)
  - Auth: `FINALAYZE_NEWSAPI_API_KEY`
  - Implementation: `src/finalayze/data/fetchers/newsapi.py`
  - Endpoint: `https://newsapi.org/v2/everything`
  - Returns: List of NewsArticle objects with text/source

**FX Rates & Macro:**
- Central Bank of Russia (CBR) REST XML API
  - FX rates via `/scripts/XML_dynamic.asp` (REST)
  - Key rate via `/DailyInfoWebServ/DailyInfo.asmx` (SOAP over HTTP)
  - Implementation: `src/finalayze/data/fetchers/cbr.py`
  - Currencies: USD, EUR vs RUB
  - Handles pre-2014 DST and timezone conversion

## Data Storage

**Databases:**
- PostgreSQL 13+ (primary)
  - Connection: `FINALAYZE_DATABASE_URL` (e.g., `postgresql+asyncpg://user:pass@host:5432/dbname`)
  - Pool settings: `db_pool_size=10`, `db_max_overflow=5`, `db_pool_timeout=30`, `db_pool_recycle=1800`
  - Async driver: asyncpg 0.30.0+
  - ORM: SQLAlchemy 2.0+ with `sqlalchemy[asyncio]`
  - Migrations: Alembic 1.14.0+
  - Implementation: `src/finalayze/core/db.py` (engine + session factory)
  - Models: `src/finalayze/core/models.py` (SQLAlchemy declarative models)

**File Storage:**
- Local filesystem (backtests, model artifacts)
  - Test results: `results/iterations/`
  - Models: `models/<segment>/` (xgb.pkl, lgbm.pkl, catboost.pkl, calibrator.pkl, meta_learner.pkl)
  - Configuration: `config/` (settings, segments, logging, universes)

**Caching:**
- Redis 6+ (primary cache)
  - URL: `FINALAYZE_REDIS_URL` (default: `redis://localhost:6379/0`)
  - Use cases:
    - Sentiment score cache: `_sentiment_cache` in trading loop (thread-safe dict with lock)
    - Event bus: Redis Streams for async event broadcasting
    - Health check cache: Avoids hammering DB/Redis on every request
  - Implementation: `src/finalayze/core/events.py` (EventBus with Redis Streams)
  - Client: `redis.asyncio` 5.2.0+ (async Redis)

## Authentication & Identity

**LLM Providers (Pluggable):**
- Default: OpenRouter (free-tier models)
  - API: `https://openrouter.ai/api/v1`
  - Auth: `FINALAYZE_LLM_API_KEY`
  - Model: `FINALAYZE_LLM_MODEL` (default: `meta-llama/llama-3.1-8b-instruct:free`)

- Alternative: OpenAI GPT
  - API: `https://api.openai.com/v1`
  - Auth: `FINALAYZE_LLM_API_KEY`
  - Model: `FINALAYZE_LLM_MODEL`

- Alternative: Anthropic Claude
  - Auth: `FINALAYZE_ANTHROPIC_API_KEY` (independent from llm_api_key)
  - Model: `FINALAYZE_LLM_MODEL`

- Provider selection: `FINALAYZE_LLM_PROVIDER` (openrouter|openai|anthropic)
- Implementation: `src/finalayze/analysis/llm_client.py` (factory pattern, provider-agnostic interface)
- Caching: SHA-256 LRU cache (1000 entries max) to avoid duplicate LLM calls

**API Keys for Analysis:**
- Alpaca API key: `FINALAYZE_ALPACA_API_KEY` (execution)
- Tinkoff token: `FINALAYZE_TINKOFF_TOKEN` (execution + data)
- Finnhub key: `FINALAYZE_FINNHUB_API_KEY` (market data)
- NewsAPI key: `FINALAYZE_NEWSAPI_API_KEY` (news)
- LLM API key: `FINALAYZE_LLM_API_KEY` or `FINALAYZE_ANTHROPIC_API_KEY` (analysis)

**API Authentication (FastAPI):**
- Simple key-based: `FINALAYZE_API_KEY` (header: `X-API-Key`)
- Real mode token: `FINALAYZE_REAL_TOKEN` (required to enable REAL mode via API)
- Implementation: `src/finalayze/api/v1/auth.py`

## Monitoring & Observability

**Error Tracking:**
- None detected in primary system (errors logged via structlog)
- Note: External integrations (Sentry, DataDog, etc.) not wired

**Logs:**
- structlog 24.4.0+ with context preservation
- Output: Structured JSON (local) or stdout (production)
- Configuration: `config/logging.py` (mode-aware)
- Per-module loggers: `_log = structlog.get_logger()` pattern throughout

**Metrics (Prometheus):**
- Export: `/metrics` endpoint (Prometheus text format)
- Instrumentator: `prometheus-fastapi-instrumentator` 7.0+
- Metrics exported: `src/finalayze/api/metrics.py`
  - Portfolio equity (USD), P&L, drawdown
  - Open positions count
  - Circuit breaker level (gauge)
  - Trade counts, slippage (bps), fill latency (seconds)
  - Order rejections by reason
- Note: No authentication on `/metrics` (internal network only in production)

**Alerts (Telegram):**
- Bot API: `https://api.telegram.org/bot<TOKEN>/sendMessage`
- Auth: `FINALAYZE_TELEGRAM_BOT_TOKEN`, `FINALAYZE_TELEGRAM_CHAT_ID`
- Implementation: `src/finalayze/core/alerts.py` (TelegramAlerter)
- Fire-and-forget: HTTP errors are logged, never propagate to trading loop
- Events: Trade fills, rejections, circuit breaker trips, daily summaries

## CI/CD & Deployment

**Hosting:**
- Local: Docker container (see `docker/Dockerfile`)
- Cloud: Any platform supporting Python 3.12 + PostgreSQL + Redis
- Example: AWS EC2 + RDS + ElastiCache, or GCP Cloud Run + Cloud SQL + Memorystore

**CI Pipeline:**
- Not detected in primary codebase (manual testing expected)
- Linting: `ruff check .`
- Formatting: `ruff format --check`
- Type checking: `mypy src/`
- Tests: `pytest --cov=src/finalayze --cov-fail-under=50`

**Pre-Commit Hooks:**
- Config: `pre-commit` 4.0.0+ (configured in `pyproject.toml`)
- Hooks: Linting, formatting, type checking (ruff + mypy)

## Environment Configuration

**Required env vars (REAL mode):**
- `FINALAYZE_DATABASE_URL` - PostgreSQL connection string
- `FINALAYZE_ALPACA_API_KEY`, `FINALAYZE_ALPACA_SECRET_KEY` - Alpaca execution
- `FINALAYZE_TINKOFF_TOKEN` - Tinkoff Invest (MOEX)
- `FINALAYZE_LLM_API_KEY` or `FINALAYZE_ANTHROPIC_API_KEY` - LLM provider
- `FINALAYZE_REAL_CONFIRMED=true` - Safety switch for REAL mode

**Optional env vars:**
- `FINALAYZE_MODE` - WorkMode (debug|test|sandbox|real, default: debug)
- `FINALAYZE_REDIS_URL` - Redis connection (default: redis://localhost:6379/0)
- `FINALAYZE_FINNHUB_API_KEY` - Finnhub market data
- `FINALAYZE_NEWSAPI_API_KEY` - NewsAPI articles
- `FINALAYZE_TELEGRAM_BOT_TOKEN`, `FINALAYZE_TELEGRAM_CHAT_ID` - Telegram alerts
- `FINALAYZE_LLM_PROVIDER` - openrouter|openai|anthropic (default: openrouter)
- `FINALAYZE_LLM_MODEL` - Model name (default: meta-llama/llama-3.1-8b-instruct:free)
- `FINALAYZE_ML_ENABLED` - Enable ML ensemble (default: false)
- `FINALAYZE_BOND_CYCLE_ENABLED` - Enable bond portfolio layer (default: true)
- Market limits: `FINALAYZE_ALPACA_MAX_PORTFOLIO_VALUE`, `FINALAYZE_TINKOFF_MAX_PORTFOLIO_VALUE`
- Risk params: `FINALAYZE_KELLY_FRACTION`, `FINALAYZE_STOP_LOSS_ATR_MULTIPLIER`, `FINALAYZE_CIRCUIT_BREAKER_L1/2/3`
- Cycle timings: `FINALAYZE_NEWS_CYCLE_MINUTES`, `FINALAYZE_STRATEGY_CYCLE_MINUTES`, `FINALAYZE_DAILY_RESET_HOUR_UTC`

**Secrets location:**
- Local development: `.env` file (git-ignored)
- Production: Environment variables (set by deployment platform or secrets manager)
- Note: `.env*` files are never committed; no credentials in repo

## Webhooks & Callbacks

**Incoming:**
- None detected (system pulls data, not push-based)

**Outgoing:**
- Telegram message posting (fire-and-forget, not event-driven)
- No webhook integrations to external systems

---

*Integration audit: 2026-03-14*

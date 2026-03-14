# Technology Stack

**Analysis Date:** 2026-03-14

## Languages

**Primary:**
- Python 3.12 - All application, analysis, and testing code; strict type checking via mypy

**Secondary:**
- YAML - Strategy presets and configuration files (`src/finalayze/strategies/presets/*.yaml`)

## Runtime

**Environment:**
- CPython 3.12 (constraint: `>=3.12,<3.14` to avoid resolution issues with gRPC dependencies)
- Platform: Unix-like only (`sys_platform != 'win32'` via uv)

**Package Manager:**
- uv 0.10.4+ (Python package management and locking)
- Lockfile: `uv.lock` (committed)
- Build backend: `uv_build` 0.10.4

## Frameworks

**Core:**
- FastAPI 0.115.0+ - REST API server (Layer 6)
- Uvicorn 0.34.0+ - ASGI server for FastAPI

**Database:**
- SQLAlchemy 2.0.36+ - Async ORM, configured for async sessions via `sqlalchemy[asyncio]`
- asyncpg 0.30.0+ - PostgreSQL driver for async queries
- Alembic 1.14.0+ - Database schema migrations

**Data & Analysis:**
- pandas 2.2.0+ - Data manipulation and time series
- polars 1.17.0+ - High-performance data frames (parallel option)
- pandas-ta 0.3.14b1 - Technical analysis indicators (ATR, RSI, MACD, ADX, etc.)
- numpy 1.26.0+ - Numerical computing
- statsmodels 0.14.0+ - Statistical models and tests

**ML Models & Training:**
- scikit-learn 1.5.0+ - Preprocessing, feature selection, cross-validation
- xgboost 2.1.0+ - Gradient boosting (primary ensemble component)
- lightgbm 4.5.0+ - Gradient boosting (secondary ensemble component)
- catboost 1.2.0+ - Gradient boosting (tertiary ensemble component)
- torch 2.5.0+ - PyTorch for neural networks (future phase)
- optuna 4.7.0+ - Hyperparameter tuning with overfitting guardrails

**Testing:**
- pytest 8.3.0+ - Test runner
- pytest-asyncio 0.25.0+ - Async test support
- pytest-cov 6.0.0+ - Coverage reporting (threshold: 50%)
- pytest-mock 3.14.0+ - Mocking framework
- respx 0.22.0+ - HTTP mocking for integration tests

**Monitoring & Logging:**
- structlog 24.4.0+ - Structured logging with context preservation
- prometheus-client 0.20+ - Prometheus metrics export
- prometheus-fastapi-instrumentator 7.0+ - Automatic HTTP instrumentation

**Build & Code Quality:**
- ruff 0.9.0+ - Linter and formatter (line-length: 100, strict mode)
- mypy 1.14.0+ - Static type checker (strict mode: `disallow_untyped_defs=true`)
- pre-commit 4.0.0+ - Git pre-commit hooks

**Brokers & Market Data:**
- alpaca-py 0.43.2+ - US equity trading via Alpaca REST API + WebSocket
- t-tech-investments 0.2.0+ - Tinkoff Invest gRPC SDK for MOEX (T-Bank custom index)
- yfinance 0.2.50+ - Yahoo Finance candle data (US equities, indices, FX only)

**LLM & AI:**
- anthropic 0.42.0+ - Claude API client (async capable)
- openai 1.50.0+ - OpenAI GPT API client (async capable, used for OpenAI and OpenRouter)

**News & External Data:**
- finnhub-python 2.4.20+ - Finnhub REST API for market data
- httpx 0.28.0+ - Async HTTP client (preferred over requests)
- lxml 6.0.2+ - XML/HTML parsing (CBR API response parsing)
- PyWavelets 1.6.0+ - Wavelet transforms for signal processing

**Task Scheduling & Caching:**
- APScheduler 3.10.4+ - Background task scheduling (trading loop cycles)
- Celery 5.4.0+ - Distributed task queue (optional, wired but not primary)
- redis 5.2.0+ - Redis client for caching, event bus, sentiment cache
- redis[asyncio] - Async Redis connection support

**Dashboard & Visualization:**
- streamlit 1.41.0+ - Web-based dashboard for performance monitoring
- altair 6.0.0+ (via streamlit) - Declarative visualization grammar

**Utilities:**
- pydantic 2.10.0+ - Data validation (Pydantic v2)
- pydantic-settings 2.7.0+ - Environment-based settings (Pydantic v2)
- python-dotenv 1.0.1+ - `.env` file loading for local development
- python-dateutil 2.9.0+ - Date/time utilities and timezone handling
- pyyaml 6.0.2+ - YAML parsing
- hmmlearn 0.3.3+ - Hidden Markov Models (regime detection)
- arch 7.0+ - Autoregressive conditional heteroscedasticity (volatility)

## Configuration

**Environment:**
- `.env` file at project root (loaded via pydantic-settings)
- Env var prefix: `FINALAYZE_` (all settings prefixed, e.g., `FINALAYZE_ALPACA_API_KEY`)
- Work modes: debug, test, sandbox, real (controls credential requirement)

**Key Configuration Files:**
- `config/settings.py` - Pydantic settings with validation and mode-specific defaults
- `config/modes.py` - WorkMode enum (debug, test, sandbox, real)
- `config/segments.py` - Market segment definitions (us_tech, us_broad, us_healthcare, us_finance, ru_blue_chips, ru_energy)
- `config/logging.py` - structlog configuration
- `.ruff.toml` - Linter/formatter rules (100 char line length, strict selection)
- `pyproject.toml` - Full project metadata, dependencies, tool configs

**Build Configuration:**
- `pyproject.toml` - Poetry/uv manifest with dependency groups
  - Core deps: FastAPI, SQLAlchemy, pandas, ML libraries, brokers, LLM clients
  - Optional dev deps: pytest, mypy, ruff, pre-commit

**Custom Package Index:**
- T-Bank gRPC SDK via `[[tool.uv.index]]`:
  - URL: `https://opensource.tbank.ru/api/v4/projects/238/packages/pypi/simple`
  - Package: `t-tech-investments` (explicit=true, only for this index)

## Platform Requirements

**Development:**
- Python 3.12
- PostgreSQL 13+ or compatible (for local testing)
- Redis 6+ (for caching and event bus)
- OpenSSL/TLS support (for gRPC SSL verification)
- Certificates: `certs/grpc_roots.pem` (T-Bank Russian CA root, optional but recommended)

**Production:**
- PostgreSQL 13+ (TimescaleDB recommended for time-series)
- Redis 6+ (session/cache store, event streaming)
- Minimal: 2 CPU cores, 2GB RAM for live trading
- Recommended: 4 CPU cores, 4GB RAM for backtesting + live

**Deployment Target:**
- Docker container (see `docker/Dockerfile`)
- Cloud: Any platform supporting Python 3.12 + PostgreSQL + Redis (AWS, GCP, Azure, Heroku, etc.)

---

*Stack analysis: 2026-03-14*

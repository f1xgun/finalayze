# Architecture Overview

## System Purpose

Finalayze is an AI-powered multi-market stock trading system that:

1. **Ingests** news (global and regional), social sentiment, and market data
   from multiple sources.
2. **Analyzes** using LLMs (Claude Sonnet) for sentiment scoring and
   fact-checking, plus traditional technical indicators.
3. **Predicts** price movements using an ML ensemble (XGBoost + LightGBM + LSTM)
   trained per segment.
4. **Executes** trades via Alpaca (US markets) and Tinkoff Invest (MOEX).

## High-Level Component Map

```
+-------------------+     +-------------------+     +-------------------+
|   Data Ingestion  |     |   News Pipeline   |     |  Market Data Feed |
|   (data/)         |     |   (analysis/)     |     |  (markets/)       |
+--------+----------+     +--------+----------+     +--------+----------+
         |                         |                          |
         v                         v                          v
+--------+-------------------------+---------------------------+--------+
|                         Event Bus / Message Layer                     |
|                         (core/events.py)                             |
+--------+-------------------------+---------------------------+--------+
         |                         |                          |
         v                         v                          v
+--------+----------+     +--------+----------+     +--------+----------+
|   ML Pipeline     |     |  Strategy Engine  |     |  Risk Manager     |
|   (ml/)           |     |  (strategies/)    |     |  (risk/)          |
+--------+----------+     +--------+----------+     +--------+----------+
         |                         |                          |
         v                         v                          v
+--------+-------------------------+---------------------------+--------+
|                         Execution Layer                               |
|                         (execution/)                                  |
+---+-------------------------------------------------------------------+
    |                                          |
    v                                          v
+---+---------------+              +-----------+-------+
|  Alpaca Broker    |              |  Tinkoff Broker   |
|  (US Markets)     |              |  (MOEX)           |
+-------------------+              +-------------------+
```

## Dependency Layers

See [DEPENDENCY_LAYERS.md](DEPENDENCY_LAYERS.md) for the full layering rules.

```
Layer 0: Types & Schemas       core/
Layer 1: Configuration          config/
Layer 2: Data / Repository      data/, markets/
Layer 3: Analysis / ML          analysis/, ml/
Layer 4: Strategy / Risk        strategies/, risk/
Layer 5: Execution              execution/
Layer 6: API / Dashboard        api/, dashboard/
```

## Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.12 |
| Web framework | FastAPI |
| Database | PostgreSQL + TimescaleDB |
| Cache | Redis |
| ORM | SQLAlchemy 2.0 (async) |
| ML - Gradient Boosting | XGBoost, LightGBM |
| ML - Deep Learning | PyTorch (LSTM) |
| Technical Analysis | pandas-ta |
| LLM | Claude Sonnet (via Anthropic API) |
| US Broker | Alpaca |
| MOEX Broker | Tinkoff Invest |
| Dashboard | Streamlit |
| Package Manager | uv |
| Linter/Formatter | ruff |
| Type Checker | mypy (strict) |
| Testing | pytest, pytest-cov, pytest-asyncio |

## Work Modes

The system supports four operational modes, controlled by `config/modes.py`:

| Mode | Broker | Data Source | Purpose |
|---|---|---|---|
| `debug` | Mock | Fixtures | Local development, verbose logging |
| `sandbox` | Alpaca Paper / Tinkoff Sandbox | Live feeds | Paper trading validation |
| `test` | Simulated | Historical | Automated integration testing, backtesting |
| `real` | Alpaca Live / Tinkoff Live | Live feeds | Production trading |

## Key Design Principles

1. **Layer isolation** -- imports flow downward only, never upward.
2. **Async-first** -- all I/O operations use async/await.
3. **Configuration-driven** -- no magic numbers; everything in config.
4. **Mode-aware** -- every component respects the active work mode.
5. **Segment-oriented** -- ML models and strategies are scoped per segment.
6. **Fail-safe** -- risk checks gate every order before execution.
7. **Observable** -- structured logging, metrics, health checks on every layer.

## Data Storage

- **PostgreSQL + TimescaleDB**: OHLCV data (hypertables), trade history,
  portfolio snapshots, news articles, sentiment scores.
- **Redis**: real-time caches (order book snapshots, latest prices, rate
  limiter counters, session state).
- **File system**: ML model artifacts, backtest results, configuration YAML.

## External Integrations

| Integration | Protocol | Purpose |
|---|---|---|
| Alpaca | REST + WebSocket | US market data and order execution |
| Tinkoff Invest | gRPC + REST | MOEX market data and order execution |
| Anthropic (Claude) | REST | News sentiment analysis, fact-checking |
| News APIs | REST | News article ingestion |
| Social APIs | REST | Social sentiment data |

## Phase 1 Implemented Components

The following components have been implemented as of Phase 1 (2026-02-22). All
carry strict mypy typing and are covered by unit tests.

### Layer 0: Types & Schemas (`core/`)

| File | Description |
|------|-------------|
| `core/schemas.py` | Pydantic v2 schemas: `Candle`, `Signal`, `TradeResult`, `PortfolioState`, `BacktestResult`, `SignalDirection` |
| `core/exceptions.py` | Domain exception hierarchy — 12 classes across data, broker, risk, strategy, and config domains |
| `core/models.py` | SQLAlchemy 2.0 async ORM models: `Market`, `Segment`, `Instrument`, `Candle`, `Signal`, `Order`, `PortfolioSnapshot`, `CurrencyRate` |
| `core/modes.py` | `WorkMode` enum + `ModeManager` with real-mode safety guard |
| `core/clock.py` | `ClockBase` ABC, `RealClock`, `SimulatedClock` (controllable time for backtesting) |
| `core/events.py` | `EventBus` backed by Redis Streams; `MarketDataEvent`, `SignalEvent` |
| `core/db.py` | Async SQLAlchemy engine/session factory stub |

### Layer 1: Configuration (`config/`)

| File | Description |
|------|-------------|
| `config/settings.py` | Pydantic `Settings` with validation for all external services |
| `config/modes.py` | Mode-aware config resolver |
| `config/segments.py` | Segment definitions (us_tech, us_broad, …) |
| `config/logging.py` | structlog setup: JSON output, per-mode log levels |

### Layer 2: Data & Markets (`data/`, `markets/`)

| File | Description |
|------|-------------|
| `data/fetchers/base.py` | `FetcherBase` ABC |
| `data/fetchers/yfinance.py` | `YFinanceFetcher` — multi-level column fix, UTC normalization |
| `data/fetchers/finnhub.py` | `FinnhubFetcher` — OHLCV candles, `RateLimitError` on HTTP 429 |
| `data/rate_limiter.py` | `RateLimiter` — token bucket with async `acquire()` |
| `data/normalizer.py` | `DataNormalizer` — OHLCV schema validation, batch normalization |
| `markets/registry.py` | `MarketRegistry` — US + MOEX market definitions, lookup by ID |
| `markets/schedule.py` | `MarketSchedule` — US 09:30-16:00 ET, MOEX weekday guards, `is_open()` |

### Layer 4: Strategies & Risk (`strategies/`, `risk/`)

| File | Description |
|------|-------------|
| `strategies/base.py` | `BaseStrategy` ABC with segment awareness |
| `strategies/momentum.py` | `MomentumStrategy` — RSI + MACD, configurable per-segment YAML params |
| `strategies/mean_reversion.py` | `MeanReversionStrategy` — Bollinger Bands, per-segment params |
| `strategies/combiner.py` | `StrategyCombiner` — weighted ensemble, YAML preset loader |
| `strategies/presets/us_tech.yaml` | Strategy parameters for US tech segment |
| `strategies/presets/us_broad.yaml` | Strategy parameters for US broad market segment |
| `risk/position_sizer.py` | Half-Kelly position sizer (`Decimal`-safe) |
| `risk/stop_loss.py` | ATR-based stop-loss calculator (pure `Decimal`) |
| `risk/pre_trade_check.py` | 11-check pre-trade pipeline (cash, allocation, position limits, …) |

### Layer 5: Execution (`execution/`)

| File | Description |
|------|-------------|
| `execution/broker_base.py` | `BrokerBase` ABC |
| `execution/simulated_broker.py` | `SimulatedBroker` — fills at next candle open, stop-loss monitoring, portfolio tracking |

### Backtest

| File | Description |
|------|-------------|
| `backtest/engine.py` | `BacktestEngine` — historical candle replay with signal + risk pipeline |
| `backtest/performance.py` | `PerformanceAnalyzer` — Sharpe ratio, max drawdown, win rate, profit factor |
| `scripts/run_backtest.py` | CLI runner for single-symbol backtests |
| `scripts/seed_historical_data.py` | yfinance-based historical data seeder with per-symbol error handling |

### Layer 6: API (`api/`)

| File | Description |
|------|-------------|
| `main.py` | FastAPI application entry point, CORS middleware |
| `api/v1/system.py` | `GET /api/v1/health`, `GET /api/v1/mode`, `POST /api/v1/mode` |

### Infrastructure

| File | Description |
|------|-------------|
| `alembic/` | Alembic migration setup; initial migration creates all 8 tables + TimescaleDB hypertable for candles |
| `.github/workflows/` | GitHub Actions CI: lint (ruff), type-check (mypy), test (pytest) |

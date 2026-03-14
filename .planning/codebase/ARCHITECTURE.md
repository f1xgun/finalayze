# Architecture

**Analysis Date:** 2026-03-14

## Pattern Overview

**Overall:** Layered event-driven trading system with strict dependency isolation and mode-aware execution.

**Key Characteristics:**
- 6-layer strict dependency hierarchy (downward imports only, never upward)
- Event-driven via Redis Streams for market data, signals, and execution events
- Segment-oriented ML models (per market/strategy pool) with per-symbol trading
- Mode-aware runtime (debug/sandbox/test/real) with safety gates
- APScheduler-based live trading loop with separate news/strategy/reset cycles
- Backtest engine that replays historical candles with identical signal/risk pipeline

## Layers

**Layer 0: Types & Schemas (`core/schemas.py`, `core/exceptions.py`, `core/modes.py`)**
- Purpose: Pure data definitions, enums, exceptions — zero external I/O
- Location: `src/finalayze/core/`
- Contains: Pydantic models (`Candle`, `Signal`, `PortfolioState`, `TradeResult`), `SignalDirection` enum, `PortfolioLayer` enum, exception hierarchy (12 domain-specific error classes), `WorkMode` enum
- Depends on: Standard library + Pydantic only
- Used by: All upper layers

**Layer 1: Configuration (`config/`)**
- Purpose: Load and validate environment variables + configuration files
- Location: `config/` (at project root, imported via `sys.path` manipulation)
- Contains: `Settings` (Pydantic BaseSettings), `WorkMode` + `ModeManager` runtime, segment registry, logging setup
- Depends on: Layer 0 + Pydantic
- Used by: Layer 2+ to access global settings

**Layer 2: Data & Repository (`data/`, `markets/`)**
- Purpose: Market data fetchers, normalization, caching; instrument registry; exchange schedules
- Location: `src/finalayze/data/fetchers/`, `src/finalayze/markets/`
- Contains:
  - `FetcherBase` ABC + implementations: `YFinanceFetcher`, `TinkoffFetcher`, `FinnhubFetcher`, `NewsApiFetcher`, `CBRFetcher`, `MOEXISSFetcher`
  - `DataNormalizer` (OHLCV validation, UTC normalization)
  - `RedisCache`, `RateLimiter` (token bucket)
  - `InstrumentRegistry`, `MarketRegistry` (instrument lookup by symbol/FIGI)
  - `MarketSchedule` (US 09:30-16:00 ET, MOEX weekday guards)
  - `FXRateService`, `CurrencyConverter` (cross-asset FX handling)
  - `MacroCacheService` (CBR rates, commodity indices)
- Depends on: Layers 0, 1 + external APIs (yfinance, Tinkoff gRPC, etc.)
- Used by: Layer 3+ for market data ingestion

**Layer 3: Analysis & ML (`analysis/`, `ml/`)**
- Purpose: News sentiment analysis (Claude Sonnet LLM), feature engineering, ML model training/inference
- Location: `src/finalayze/analysis/`, `src/finalayze/ml/`
- Contains:
  - `NewsAnalyzer` (raw article → Claude analysis → sentiment + entity extraction)
  - `EventClassifier` (classifies news events: M&A, dividend, earnings, etc.)
  - `ImpactEstimator` (estimates market impact from news)
  - `FactChecker` (validates claim accuracy)
  - `LLMClient` (Claude Sonnet wrapper + async prompt caching)
  - ML feature engineering: `TechnicalFeatures` (45 indicators via pandas-ta), `MultiTimeframeFeatures`, `CorporateActionsFeatures`
  - ML models: XGBoost, LightGBM, CatBoost, LSTM (stacking ensemble with meta-learner)
  - Training pipeline: walk-forward splitter, sequential bootstrapping, trend-scanning labels, conformal calibration, quality gates
  - `MLModelRegistry` (load/verify models, manage segment-specific weights)
- Depends on: Layers 0, 1, 2 + pandas, numpy, scikit-learn, xgboost, lightgbm, torch, anthropic
- Used by: Layer 4 strategies for ML-assisted signal generation

**Layer 4: Strategy & Risk (`strategies/`, `risk/`)**
- Purpose: Trading signal generation, risk management, position sizing, pre-trade checks
- Location: `src/finalayze/strategies/`, `src/finalayze/risk/`
- Contains:
  - **Strategies** (`BaseStrategy` ABC):
    - `MomentumStrategy` (RSI + MACD)
    - `MeanReversionStrategy` (Bollinger Bands)
    - `DualMomentumStrategy` (cross-asset)
    - `RSI2ConnorsStrategy` (reversal)
    - `OUMeanReversionStrategy` (Ornstein-Uhlenbeck)
    - `PairsStrategy` (cointegration-based)
    - `EventDrivenStrategy` (news-triggered, disabled)
    - `MLStrategy` (ML ensemble wrapper)
    - `DividendGapStrategy` (corporate action arbitrage, MOEX)
    - `BondDurationRotationStrategy` (fixed income layer)
    - `BondCarryStrategy` (MOEX bonds)
    - `CBRCalendarStrategy` (macro calendar overlay)
  - **Signal Combiner** (`StrategyCombiner`):
    - YAML preset loading per segment
    - ADX(14) regime routing: momentum strategies on trending (ADX > 35), mean-reversion on choppy (ADX < 15)
    - Confidence aggregation + exit logic
    - Turn-of-month seasonality boost
    - Hierarchical Allocation (HRP) for dynamic weighting
  - **Risk Management**:
    - `PositionSizingPipeline` (Half-Kelly + vol targeting + 6-step risk gates)
    - `PreTradeChecker` (11-check pipeline: cash, allocation, position limits, volatility caps, exposure checks)
    - `CircuitBreaker` (L1=5%, L2=10%, L3=15% daily DD) + cross-market circuit breaker
    - `DV01Sizing` (fixed income DV01-aware sizing)
    - `VolTargetingStep` (target portfolio vol ± 5%)
    - `EVTStep`, `CopulaStep` (extreme value theory + tail correlation)
    - `RegimeStep` (HMM regime-aware position caps)
    - `KellyStep` (half-Kelly with rolling win rate / avg win ratio)
    - `MetaLabelStep` (ML confidence gate for position size modulation)
  - **Stop Loss**:
    - ATR-based trailing stops (strategy-specific multipliers in `backtest/config.py`)
    - Chandelier exit (ATR + high/low tracking)
    - Grace bar: skip stop-loss check on fill candle
    - Catastrophic drop circuit (15% intraday forces exit even on grace bar)
  - **Drawdown Monitoring** (daily reset, cumulative DD limits)
- Depends on: Layers 0, 1, 2, 3 + pandas-ta, scipy, statsmodels
- Used by: Layer 5 execution and Layer 6 live trading loop

**Layer 5: Execution (`execution/`)**
- Purpose: Order routing, broker abstraction, fill reconciliation, slippage tracking
- Location: `src/finalayze/execution/`
- Contains:
  - `BrokerBase` ABC (abstract `submit_order()`, `get_portfolio()`, `get_positions()`)
  - `AlpacaBroker` (Alpaca REST API, paper + live modes)
  - `TinkoffBroker` (T-Bank gRPC API for MOEX, sandbox + live, with sandboxed `AsyncSandboxClient`)
  - `SimulatedBroker` (for backtests, fills at next candle open, tracks stops)
  - `BrokerRouter` (dispatches orders to correct broker based on symbol market ID)
  - `RetryPolicy` (exponential backoff for transient failures)
  - Fill reconciliation (slippage, commission tracking)
- Depends on: Layers 0, 1, 2, 3, 4 + alpaca-trade-api, t-tech-investments (Tinkoff SDK)
- Used by: Layer 6 for order submission

**Layer 6: API, Dashboard & Orchestration (`api/`, `dashboard/`, `core/trading_loop.py`)**
- Purpose: REST endpoints, WebSocket feeds, Streamlit dashboard, live trading orchestration
- Location: `src/finalayze/api/v1/`, `src/finalayze/dashboard/`, `src/finalayze/main.py`, `src/finalayze/core/trading_loop.py`
- Contains:
  - **FastAPI endpoints** (`api/v1/router.py` + sub-routers):
    - `system.py`: Health, mode management, metrics
    - `portfolio.py`: Portfolio state, positions, snapshots
    - `trades.py`: Trade history, P&L
    - `signals.py`: Last signals per symbol/strategy
    - `risk.py`: Exposure limits, circuit breaker status, drawdown
    - `news.py`: Recent articles, sentiment scores
    - `ml.py`: Model metadata, feature importance, retraining status
    - `auth.py`: API key validation middleware
  - **Streamlit dashboard** (`dashboard/`): multi-page with portfolio performance, strategy signals, P&L charts, live positions
  - **TradingLoop** (APScheduler-based orchestrator):
    - `_news_cycle` (30 min): fetch news, analyze sentiment, cache results
    - `_strategy_cycle` (60 min): for each symbol, generate signal → apply circuit breaker → submit orders
    - `_daily_reset` (00:00 UTC): reset circuit breakers, send daily P&L alerts via Telegram
    - Runs live risk checks before submission
    - Thread-safe sentiment cache with locking
  - **Main app** (`main.py`): FastAPI factory, CORS middleware, Prometheus metrics
- Depends on: Layers 0–5 + fastapi, structlog, prometheus-client, streamlit, apscheduler
- Used by: External clients via REST/WebSocket, dashboard users

## Data Flow

**Market Data Flow:**
1. Alpaca WebSocket or Tinkoff gRPC → `DataNormalizer` (UTC, schema validation)
2. → Redis cache + TimescaleDB hypertable
3. → Event Bus (`MARKET_UPDATE` events)
4. → Strategy Engine + ML Pipeline + Caches

**Signal Generation Flow:**
1. Strategy generates signal from candles + sentiment → `Signal` (direction, confidence, features)
2. `StrategyCombiner` aggregates per-segment weights (YAML presets)
3. ADX regime routing gates momentum vs. mean-reversion pools
4. Combined confidence vs. threshold → `StrategyCombiner.generate_combined_signal()`

**Order Submission Flow:**
1. Combined signal → Risk Pipeline (position sizing, pre-trade checks, circuit breaker)
2. Approved `OrderRequest` → `BrokerRouter`
3. Dispatches to `AlpacaBroker` (US) or `TinkoffBroker` (MOEX) or `SimulatedBroker` (backtest)
4. Fill recorded → Portfolio updated → `TRADE_EXECUTED` event

**State Management:**
- **Portfolio**: in-memory dict of positions + cash, updated on fill
- **Circuit Breaker**: in-memory tracking of drawdown per day/week/month
- **Sentiment Cache**: thread-safe dict, locked during updates (news cycle updates cache, strategy cycle reads)
- **Persistence**: PostgreSQL (trade log, portfolio snapshots), TimescaleDB (OHLCV hypertable), Redis (real-time caches)

## Key Abstractions

**BaseStrategy:**
- Purpose: Abstract interface for all trading signal generators
- Examples: `MomentumStrategy`, `MeanReversionStrategy`, `BondDurationRotationStrategy` in `src/finalayze/strategies/`
- Pattern: Subclasses override `generate_signal()` which takes candles + segment ID → returns `Signal | None`

**BrokerBase:**
- Purpose: Abstract interface for order submission + portfolio tracking
- Examples: `AlpacaBroker`, `TinkoffBroker`, `SimulatedBroker` in `src/finalayze/execution/`
- Pattern: Live brokers submit via REST/gRPC; simulated broker tracks fills at next candle open

**StrategyCombiner:**
- Purpose: Weighted ensemble of strategies with regime-aware gating
- File: `src/finalayze/strategies/combiner.py`
- Pattern: Loads YAML presets per segment; applies ADX routing; aggregates confidences; checks min thresholds

**PositionSizingPipeline:**
- Purpose: 6-step position size computation (Kelly → Vol Target → Regime → EVT → Copula → Hard Caps)
- File: `src/finalayze/risk/position_sizing_pipeline.py`
- Pattern: Each step is a named step with input/output context; pipeline is composable

**BacktestEngine:**
- Purpose: Replay historical candles with identical signal + risk pipeline as live
- File: `src/finalayze/backtest/engine.py`
- Pattern: Iterates per-symbol candles in timestamp order; calls strategy → applies risk → tracks fills in simulated broker

## Entry Points

**Live Trading Loop:**
- Location: `src/finalayze/core/trading_loop.py::TradingLoop`
- Triggers: APScheduler (periodic news/strategy/reset cycles)
- Responsibilities: Orchestrate news analysis → signal generation → risk checks → order submission → alerts

**Backtest Script:**
- Location: `scripts/run_iteration.py`
- Triggers: CLI invocation (e.g., `uv run python scripts/run_iteration.py --segment us_tech`)
- Responsibilities: Load historical candles → instantiate strategies + risk config → run backtest engine → output metrics

**FastAPI Server:**
- Location: `src/finalayze/main.py::app`
- Triggers: ASGI server (e.g., Uvicorn)
- Responsibilities: REST endpoint routing, Prometheus metrics, API authentication

**Streamlit Dashboard:**
- Location: `src/finalayze/dashboard/app.py`
- Triggers: `streamlit run src/finalayze/dashboard/app.py`
- Responsibilities: Real-time portfolio visualization, strategy signals, P&L charts

## Error Handling

**Strategy:** Try-except around signal generation; return `None` if computation fails (safe default: HOLD)

**Patterns:**
- `DataFetchError` (L2): raised by fetchers on HTTP errors / malformed data
- `BrokerError` (L5): raised by broker on order submission failures
- `RiskCheckFailed` (L4): raised by pre-trade checker when checks don't pass (not an exception but a rejection)
- `ModeError` (L0): raised when attempting to transition to REAL mode without confirmation
- Structured logging via `structlog`: all errors logged with context (symbol, strategy, bar index)

## Cross-Cutting Concerns

**Logging:** `structlog` with JSON output in prod, pretty-print in debug. Per-mode log level (DEBUG in debug/sandbox, INFO in test/real). `setup_logging()` called at module level before any logger creation.

**Validation:** Pydantic v2 models enforce schema at layer boundaries. OHLCV candles validated for UTC timezone, positive volume, OHLC ordering.

**Authentication:** API key validation in `api/v1/auth.py` via `X-API-Key` header; mode confirmation via `FINALAYZE_REAL_CONFIRMED=true` env var for REAL mode.

---

*Architecture analysis: 2026-03-14*

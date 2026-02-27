# Roadmap

## Phase Overview

| Phase | Name | Duration | Status |
|-------|------|----------|--------|
| 0 | Code Quality Foundation | 2-3 days | COMPLETE |
| 1 | Foundation, US Market & Sandbox | Weeks 1-5 | COMPLETE |
| 2 | MOEX + Tinkoff, Intelligence & Test Mode | Weeks 6-10 | COMPLETE |
| 3 | Hardening & Advanced | Weeks 11-14 | NOT STARTED |
| 4 | Real Trading & Optimization | Weeks 15-18 | NOT STARTED |

## Phase 0: Code Quality Foundation (COMPLETE)

- [x] Project initialization with uv
- [x] pyproject.toml with full dependency spec
- [x] ruff configuration (full rule set)
- [x] mypy configuration (strict mode)
- [x] Pre-commit hooks
- [x] Docker Compose (PostgreSQL + TimescaleDB + Redis)
- [x] Documentation skeleton (all docs/ files)
- [x] CLAUDE.md, WORKFLOW.md, CHANGELOG.md
- [x] .env.example
- [x] .gitignore
- [x] Git initialized

## Phase 1: Foundation, US Market & Sandbox -- COMPLETE

See [PHASE_1.md](PHASE_1.md) for detailed execution plan.

- [x] Database schema + Alembic migrations
- [x] Core framework (mode manager, event bus, clock, logging, exception hierarchy)
- [x] Markets framework (registry, segments, schedule)
- [x] Data ingestion (yfinance + Finnhub fetcher, rate limiter, normalizer)
- [x] Strategies (momentum + mean reversion + combiner + YAML presets)
- [x] Backtest engine + simulated broker
- [x] Risk management (position sizing, stop-loss, pre-trade checks)
- [x] FastAPI skeleton (health + mode endpoints)
- [ ] Instrument registry (IN PROGRESS)
- [ ] Currency conversion stub (IN PROGRESS)
- [ ] docker-compose integration test

### Phase 1 Backtest Vertical Slice -- COMPLETE (2026-02-22)

Implemented bottom-up through the full stack:
- Layer 0: Pydantic schemas (Candle, Signal, TradeResult, PortfolioState, BacktestResult)
- Layer 0: Exception hierarchy (12 classes)
- Layer 1: Config (settings, modes, segments, structlog logging)
- Layer 2: SQLAlchemy ORM models + Alembic migration
- Layer 2: MarketRegistry (US + MOEX), MarketSchedule
- Layer 2: YFinanceFetcher + FinnhubFetcher, RateLimiter, DataNormalizer
- Layer 4: MomentumStrategy (RSI+MACD), MeanReversionStrategy (Bollinger Bands), StrategyCombiner
- Layer 4: Half-Kelly position sizing, ATR stop-loss, pre-trade checks (11 checks)
- Layer 5: SimulatedBroker (fills at next open, stop-loss monitoring)
- Layer 6: FastAPI app, GET /api/v1/health, GET+POST /api/v1/mode
- Core: ModeManager, Clock (RealClock + SimulatedClock), EventBus (Redis Streams)
- Backtest: BacktestEngine + PerformanceAnalyzer
- CLI: scripts/run_backtest.py, scripts/seed_historical_data.py
- 349 unit tests, 95.64% coverage

## Phase 2: MOEX + Tinkoff, Intelligence & Test Mode -- COMPLETE (2026-02-23)

PRs: #90 (Track A — Intelligence), #91 (Track B — MOEX). 489 tests, 93% coverage.

- [x] Tinkoff gRPC integration (`TinkoffFetcher` via t-tech-investments SDK)
- [x] 8 MOEX instruments with FIGI identifiers (SBER, GAZP, LKOH, GMKN, YNDX, NVTK, ROSN, VTBR)
- [x] Abstract LLM client (OpenRouter default, OpenAI, Anthropic) + cache + retry
- [x] News analysis pipeline (NewsApiFetcher, NewsAnalyzer EN/RU, EventClassifier, ImpactEstimator)
- [x] ML pipeline scaffold (XGBoost + LightGBM per segment, EnsembleModel, MLModelRegistry)
- [x] Event-driven strategy (reads YAML min_sentiment threshold per segment)
- [x] AlpacaBroker (paper/live via alpaca-py)
- [x] TinkoffBroker (sandbox/live via t-tech-investments, lot-size aware)
- [x] BrokerRouter (dispatches orders by market_id)
- [x] Alembic migration 002 (news_articles + sentiment_scores tables)
- [ ] MOEX historical data seeding (deferred to Phase 3)
- [ ] Test mode controller (deferred to Phase 3)
- [ ] Streamlit dashboard (deferred to Phase 3)

## Phase 3: Hardening & Advanced

- [ ] Per-market circuit breakers
- [ ] Cross-market risk management
- [ ] LSTM per segment
- [ ] Pairs trading
- [ ] Alerting (Slack/Discord/Telegram)
- [ ] Integration + E2E tests

## Phase 4: Real Trading & Optimization

### Track A: Strategy & Backtest Hardening

See [2026-02-27-phase4-track-a-design.md](2026-02-27-phase4-track-a-design.md).

Driven by expert consensus (quant trader, swing trader, risk officer) after
backtest validation of the MomentumStrategy regime lookback fix.

- [ ] PR A-1: Backtest infrastructure (trailing stops, transaction costs, benchmark, circuit breakers, batch runner)
- [ ] PR A-2: Signal quality (trend filter, signal state machine, ADX filter, volume filter)
- [ ] PR A-3: Risk calibration (rolling Kelly, daily/weekly loss limits, confidence calibration)
- [ ] PR A-4: Statistical validation (walk-forward, 50-symbol universe, Monte Carlo, OOS pipeline)

### Track B: Observability & Dashboard

See [2026-02-27-phase4-track-b-design.md](2026-02-27-phase4-track-b-design.md)
and [2026-02-27-phase4-track-b-plan.md](2026-02-27-phase4-track-b-plan.md).

- [ ] PR B-1: Core API (all endpoints + X-API-Key auth + Alembic migration 003)
- [ ] PR B-2: Streamlit dashboard (5 pages, calls B-1 API)
- [ ] PR B-3: Prometheus metrics + Alertmanager rules

### Track C: Production Go-Live

Depends on Track A (validation pass) + Track B (monitoring ready).

- [ ] Live broker integration testing (Alpaca paper → live, Tinkoff sandbox → live)
- [ ] Performance optimization (Polars, caching)
- [ ] 6-month paper trading validation period
- [ ] Production Docker deployment
- [ ] Go/No-Go decision based on Track A acceptance criteria

# Roadmap

## Phase Overview

| Phase | Name | Duration | Status |
|-------|------|----------|--------|
| 0 | Code Quality Foundation | 2-3 days | COMPLETE |
| 1 | Foundation, US Market & Sandbox | Weeks 1-5 | IN PROGRESS (core + backtest stack complete) |
| 2 | MOEX + Tinkoff, Intelligence & Test Mode | Weeks 6-10 | NOT STARTED |
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

## Phase 1: Foundation, US Market & Sandbox -- IN PROGRESS

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

## Phase 2: MOEX + Tinkoff, Intelligence & Test Mode

- [ ] Tinkoff gRPC integration
- [ ] MOEX historical data seeding
- [ ] Claude news analysis pipeline (EN + RU)
- [ ] ML pipeline (XGBoost + LightGBM per segment)
- [ ] Event-driven strategy
- [ ] Broker router (Alpaca + Tinkoff)
- [ ] Test mode (paper trading on both markets)
- [ ] Streamlit dashboard

## Phase 3: Hardening & Advanced

- [ ] Per-market circuit breakers
- [ ] Cross-market risk management
- [ ] LSTM per segment
- [ ] Pairs trading
- [ ] Alerting (Slack/Discord/Telegram)
- [ ] Integration + E2E tests

## Phase 4: Real Trading & Optimization

- [ ] Live broker integrations
- [ ] Performance optimization (Polars, caching)
- [ ] Prometheus + Grafana monitoring
- [ ] Walk-forward optimization
- [ ] Production deployment

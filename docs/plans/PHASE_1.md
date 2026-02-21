# Phase 1: Foundation, US Market & Sandbox

**Duration:** Weeks 1-5
**Goal:** Core infrastructure, US market data, 2 strategies, working backtester, segment framework.

## Week 1: Project Setup

| Task | Status | Details |
|------|--------|---------|
| Database schema | NOT STARTED | SQLAlchemy models for markets, segments, instruments, candles, news, signals, orders, portfolio_snapshots, currency_rates |
| Alembic setup | NOT STARTED | Initial migration with all core tables; TimescaleDB hypertables |
| Pydantic settings | NOT STARTED | Multi-market config with validation |

## Week 1-2: Core Framework

| Task | Status | Details |
|------|--------|---------|
| Mode manager | NOT STARTED | WorkMode enum, mode switching logic, safety checks for real mode |
| Redis Streams event bus | NOT STARTED | Publish/subscribe for market_data, news, signals, execution events |
| Clock abstraction | NOT STARTED | Real clock vs simulated clock for sandbox mode |
| Structured logging | NOT STARTED | structlog setup, per-mode log levels, JSON output |
| Exception hierarchy | NOT STARTED | Domain exceptions (already stubbed in core/exceptions.py) |
| FastAPI skeleton | NOT STARTED | Health endpoint, mode endpoint, CORS, middleware |

## Week 2: Markets Framework

| Task | Status | Details |
|------|--------|---------|
| Market registry | NOT STARTED | US + MOEX definitions, lookup by ID |
| Segment definitions | NOT STARTED | Load from config, DB override support |
| Instrument registry | NOT STARTED | Symbol lookup, FIGI mapping for MOEX |
| Currency conversion stub | NOT STARTED | USD/RUB conversion (stub for Phase 1, real in Phase 2) |
| Trading hours scheduler | NOT STARTED | Per-market open/close times, is_market_open() |
| YAML preset loader | NOT STARTED | Load strategy params from presets/ directory |

## Week 3: Data Ingestion (US)

| Task | Status | Details |
|------|--------|---------|
| Abstract fetcher interface | NOT STARTED | Base class for all data fetchers |
| Finnhub fetcher | NOT STARTED | OHLCV candles + news articles |
| yfinance fallback | NOT STARTED | Fallback data source for US stocks |
| Rate limiter | NOT STARTED | Per-source rate limiting (token bucket) |
| Data normalizer | NOT STARTED | Normalize candle format across sources |
| Historical data loader | NOT STARTED | Seed script for 2 years of US stock data |

## Week 3-4: Strategies

| Task | Status | Details |
|------|--------|---------|
| BaseStrategy ABC | NOT STARTED | Abstract base with segment awareness |
| pandas-ta integration | NOT STARTED | Technical indicator computation |
| Momentum strategy | NOT STARTED | RSI + MACD with per-segment params |
| Mean reversion strategy | NOT STARTED | Bollinger Bands with per-segment params |
| Strategy combiner | NOT STARTED | Per-segment weighted ensemble |
| YAML presets for us_tech, us_broad | DONE | Created in Phase 0 |

## Week 4-5: Backtest + Risk

| Task | Status | Details |
|------|--------|---------|
| Historical replay engine | NOT STARTED | Multi-market event replay |
| Simulated broker | NOT STARTED | Fill simulation based on historical prices |
| Performance analyzer | NOT STARTED | Sharpe, drawdown, win rate per segment |
| Backtest CLI runner | NOT STARTED | Script to run backtests from command line |
| Half-Kelly position sizer | NOT STARTED | Position sizing with Kelly fraction |
| ATR stop-loss | NOT STARTED | ATR-based stop-loss calculation |
| Portfolio constraints | NOT STARTED | Max positions, max allocation, cash reserve |
| Pre-trade check pipeline | NOT STARTED | 11-check pipeline (basic subset for Phase 1) |
| Drawdown calculator | NOT STARTED | Running drawdown tracking |

## Acceptance Criteria

- [ ] `docker-compose up` starts PG (TimescaleDB), Redis, app
- [ ] Markets + segments tables populated for US
- [ ] 2 years data loaded for AAPL, MSFT, GOOGL, AMZN, SPY
- [ ] Strategies run with `us_tech` params vs `us_broad` params and produce different signals
- [ ] Backtest produces per-segment metrics
- [ ] Risk checks fire correctly
- [ ] 80%+ unit test coverage on strategies + risk

## Documentation Updates After Phase 1

- [ ] `docs/architecture/OVERVIEW.md` -- actual architecture with implemented components
- [ ] `docs/design/STRATEGIES.md` -- momentum + mean reversion details
- [ ] `docs/design/SEGMENTS.md` -- segment system as built
- [ ] `docs/quality/GRADES.md` -- grade each implemented module
- [ ] `docs/plans/ROADMAP.md` -- mark Phase 1 tasks complete
- [ ] `CHANGELOG.md` -- all Phase 1 changes

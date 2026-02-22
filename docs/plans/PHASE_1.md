# Phase 1: Foundation, US Market & Sandbox

**Duration:** Weeks 1-5
**Goal:** Core infrastructure, US market data, 2 strategies, working backtester, segment framework.

## Week 1: Project Setup

| Task | Status | Details |
|------|--------|---------|
| Database schema | DONE | SQLAlchemy models in core/models.py for markets, segments, instruments, candles, signals, orders, portfolio_snapshots, currency_rates |
| Alembic setup | DONE | Initial migration with all core tables; TimescaleDB hypertables |
| Pydantic settings | DONE | Multi-market config with validation in config/settings.py |

## Week 1-2: Core Framework

| Task | Status | Details |
|------|--------|---------|
| Mode manager | DONE | WorkMode enum, mode switching logic, safety checks for real mode (src/finalayze/core/modes.py) |
| Redis Streams event bus | DONE | Publish/subscribe for market_data, news, signals, execution events (src/finalayze/core/events.py) |
| Clock abstraction | DONE | Real clock vs simulated clock for sandbox mode (src/finalayze/core/clock.py) |
| Structured logging | DONE | structlog setup, per-mode log levels, JSON output (config/logging.py) |
| Exception hierarchy | DONE | Domain exceptions — 12 classes (src/finalayze/core/exceptions.py) |
| FastAPI skeleton | DONE | Health endpoint, mode endpoint, CORS, middleware (src/finalayze/main.py, src/finalayze/api/v1/system.py) |

## Week 2: Markets Framework

| Task | Status | Details |
|------|--------|---------|
| Market registry | DONE | US + MOEX definitions, lookup by ID (src/finalayze/markets/registry.py) |
| Segment definitions | DONE | Load from config, DB override support (config/segments.py) |
| Instrument registry | DONE | Symbol lookup with InstrumentNotFoundError, 7 default US instruments (src/finalayze/markets/instruments.py) |
| Currency conversion stub | DONE | USD/RUB stub with rate validation, Decimal arithmetic, fallback rate (src/finalayze/markets/currency.py) |
| Trading hours scheduler | DONE | Per-market open/close times, is_market_open(); US 09:30-16:00 ET + MOEX weekday guards (src/finalayze/markets/schedule.py) |
| YAML preset loader | DONE | Strategy params loaded from presets/ directory via strategies/combiner.py |

## Week 3: Data Ingestion (US)

| Task | Status | Details |
|------|--------|---------|
| Abstract fetcher interface | DONE | Base class for all data fetchers (src/finalayze/data/fetchers/base.py) |
| Finnhub fetcher | DONE | OHLCV candles + RateLimitError on 429 (src/finalayze/data/fetchers/finnhub.py) |
| yfinance fallback | DONE | Fallback data source for US stocks; multi-level column fix, UTC normalization (src/finalayze/data/fetchers/yfinance.py) |
| Rate limiter | DONE | Per-source rate limiting (token bucket, async acquire) (src/finalayze/data/rate_limiter.py) |
| Data normalizer | DONE | Normalize OHLCV format across sources, batch mode (src/finalayze/data/normalizer.py) |
| Historical data loader | DONE | Seed script for US stock data (scripts/seed_historical_data.py) |

## Week 3-4: Strategies

| Task | Status | Details |
|------|--------|---------|
| BaseStrategy ABC | DONE | Abstract base with segment awareness (src/finalayze/strategies/base.py) |
| pandas-ta integration | DONE | Technical indicator computation integrated |
| Momentum strategy | DONE | RSI + MACD with per-segment params (src/finalayze/strategies/momentum.py) |
| Mean reversion strategy | DONE | Bollinger Bands with per-segment params (src/finalayze/strategies/mean_reversion.py) |
| Strategy combiner | DONE | Per-segment weighted ensemble (src/finalayze/strategies/combiner.py) |
| YAML presets for us_tech, us_broad | DONE | Created in Phase 0; us_tech.yaml + us_broad.yaml in strategies/presets/ |

## Week 4-5: Backtest + Risk

| Task | Status | Details |
|------|--------|---------|
| Historical replay engine | DONE | Multi-market event replay (src/finalayze/backtest/engine.py) |
| Simulated broker | DONE | Fill simulation: fills at candle open, stop-loss monitoring, portfolio tracking (src/finalayze/execution/simulated_broker.py) |
| Performance analyzer | DONE | Sharpe, drawdown, win rate per segment (src/finalayze/backtest/performance.py) |
| Backtest CLI runner | DONE | Script to run backtests from command line (scripts/run_backtest.py) |
| Half-Kelly position sizer | DONE | Position sizing with Kelly fraction (src/finalayze/risk/position_sizer.py) |
| ATR stop-loss | DONE | ATR-based stop-loss calculation, pure Decimal arithmetic (src/finalayze/risk/stop_loss.py) |
| Portfolio constraints | DONE | Max positions, max allocation, cash reserve (src/finalayze/risk/pre_trade_check.py) |
| Pre-trade check pipeline | DONE | 11-check pipeline implemented |
| Drawdown calculator | DONE | Running drawdown tracking built into performance.py |

## Acceptance Criteria

- [ ] `docker-compose up` starts PG (TimescaleDB), Redis, app
- [x] Markets + segments tables populated for US
- [ ] 2 years data loaded for AAPL, MSFT, GOOGL, AMZN, SPY
- [x] Strategies run with `us_tech` params vs `us_broad` params and produce different signals
- [x] Backtest produces per-segment metrics
- [x] Risk checks fire correctly
- [x] 80%+ unit test coverage on strategies + risk (95.64% overall)

## Documentation Updates After Phase 1

- [x] `docs/architecture/OVERVIEW.md` -- actual architecture with implemented components
- [ ] `docs/design/STRATEGIES.md` -- momentum + mean reversion details
- [ ] `docs/design/SEGMENTS.md` -- segment system as built
- [x] `docs/quality/GRADES.md` -- grade each implemented module
- [x] `docs/plans/ROADMAP.md` -- mark Phase 1 tasks complete
- [x] `CHANGELOG.md` -- all Phase 1 changes

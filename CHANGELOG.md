# Changelog

All notable changes to the Finalayze project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — Phase 1: Foundation, US Market & Sandbox

### Added
- Core: exception hierarchy (12 classes), ModeManager (WorkMode enum, real-mode guard), Clock abstraction (RealClock + SimulatedClock), Redis Streams EventBus (MarketDataEvent, SignalEvent)
- Data: FinnhubFetcher (OHLCV + RateLimitError on 429), YFinanceFetcher (multi-level column fix, UTC normalization), RateLimiter (token bucket, async acquire), DataNormalizer (OHLCV validation, batch mode)
- Markets: MarketRegistry (US + MOEX), MarketSchedule (US 09:30-16:00 ET + MOEX weekday guards), CurrencyConverter stub (USD/RUB)
- Strategies: MomentumStrategy (RSI + MACD, per-segment YAML), MeanReversionStrategy (Bollinger Bands), StrategyCombiner (weighted ensemble), YAML presets for us_tech + us_broad
- Risk: Half-Kelly position sizer, ATR stop-loss (pure Decimal), pre-trade check pipeline (11 checks)
- Execution: SimulatedBroker (fill at candle open, stop-loss, portfolio tracking), BrokerBase ABC
- Backtest: BacktestEngine (historical replay), PerformanceAnalyzer (Sharpe, drawdown, win rate), scripts/run_backtest.py CLI
- API: FastAPI app (GET /api/v1/health, GET+POST /api/v1/mode), CORS middleware
- Config: structlog setup (JSON, per-mode log level), segment definitions, Pydantic settings
- CI: GitHub Actions (lint, typecheck, test), Claude Code GitHub App integration
- Scripts: scripts/seed_historical_data.py (yfinance-based, per-symbol error handling)
- Pydantic schemas: Candle, Signal, TradeResult, PortfolioState, BacktestResult (Layer 0)
- SQLAlchemy ORM models for markets, segments, instruments, candles, signals, orders (Layer 2)
- Alembic initial migration with TimescaleDB hypertable for candles (Layer 2)
- 349 unit tests, 95.64% coverage across all implemented modules

## [0.0.1] - 2026-02-21

### Added

- Initial project scaffolding and repository setup.
- `pyproject.toml` with full dependency specification (Python 3.12, FastAPI,
  SQLAlchemy 2.0 async, XGBoost, LightGBM, PyTorch, pandas-ta, ruff, mypy, pytest).
- Package manager: uv with lockfile.
- Directory structure: `src/`, `tests/`, `config/`, `docs/`, `scripts/`,
  `alembic/`, `docker/`.
- Pre-commit hooks configuration (`.pre-commit-config.yaml`).
- Environment template (`.env.example`).
- `.gitignore` for Python, IDE, and environment files.
- Full documentation suite in `docs/` covering architecture, design, quality,
  operations, and plans.
- Phase 0 (Code Quality Foundation) completed: ruff, mypy, pre-commit, project
  structure established.

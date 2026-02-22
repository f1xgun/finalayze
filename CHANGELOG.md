# Changelog

All notable changes to the Finalayze project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Pydantic schemas: Candle, Signal, TradeResult, PortfolioState, BacktestResult (Layer 0)
- SQLAlchemy ORM models for markets, segments, instruments, candles, signals, orders (Layer 2)
- Alembic initial migration with TimescaleDB hypertable for candles (Layer 2)
- MarketRegistry with US and MOEX definitions (Layer 2)
- YFinanceFetcher for historical OHLCV data (Layer 2)
- MomentumStrategy using RSI+MACD with per-segment YAML parameters (Layer 4)
- Half-Kelly position sizing, ATR stop-loss, pre-trade risk checks (Layer 4)
- SimulatedBroker: fills at next candle open, stop-loss monitoring (Layer 5)
- BacktestEngine: full candle iteration with signal processing and risk management
- PerformanceAnalyzer: Sharpe ratio, max drawdown, win rate, profit factor
- CLI runner: scripts/run_backtest.py for single-symbol backtest
- 135 unit tests, 93%+ coverage across all new modules

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

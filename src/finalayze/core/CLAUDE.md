# Core

## Purpose
Foundation layer providing shared schemas, exceptions, event bus, work modes, database access, and validation logging.

## Layer
Layer 0 -- Types & Schemas. Zero project imports allowed (only Pydantic, stdlib).

## Moved Modules (Phase 22)
- `trading_loop.py` -> `orchestration/trading_loop.py` (Layer 5)
- `bond_cycle.py` -> `orchestration/bond_cycle.py` (Layer 5)
- `alerts.py` -> `api/alerts.py` (Layer 6)
- `telegram_bot.py` -> `api/telegram_bot.py` (Layer 6)

Backward-compatible shim modules remain in core/ for existing imports.

## Key Files
- `schemas.py` -- Pydantic models: Candle, Signal, TradeResult, PortfolioState, BacktestResult, NewsArticle, SentimentResult, BondInfo, MarketContext, IterationMetrics, LayerConfig
- `exceptions.py` -- Exception hierarchy rooted at FinalayzeError (ConfigurationError, DataFetchError, BrokerError, PredictionError, etc.)
- `modes.py` -- WorkMode enum (DEBUG/SANDBOX/TEST/REAL) and ModeManager
- `events.py` -- Redis Streams event bus (EventBus with XADD/XREAD/consumer groups)
- `db.py` -- SQLAlchemy 2.0 async engine/session factory, `get_db()` FastAPI dependency
- `models.py` -- SQLAlchemy ORM models
- `clock.py` -- Time abstraction for testability
- `validation_logger.py` -- Structured cycle logging for validation
- `kill_switch.py` -- Emergency shutdown orchestrator (Layer 0/6 boundary)
- `layer_ledger.py` -- Per-layer cash, positions, drawdown tracking
- `bond_math.py` -- Bond math utilities (YTM, duration, convexity)

## Public API
- `Signal`, `Candle`, `TradeResult`, `PortfolioState`, `BacktestResult` -- core data types
- `SignalDirection` -- BUY/SELL/HOLD enum
- `MarketContext`, `MoexMarketData` -- ambient market data containers
- `FinalayzeError` and subclasses -- exception hierarchy
- `WorkMode`, `ModeManager` -- operating mode control
- `EventBus` -- async Redis Streams pub/sub

## Contracts
- Input: All timestamps must be UTC-aware (validated by Pydantic validators)
- Output: All schemas are frozen Pydantic models (immutable after creation)
- Invariants: confidence in [0.0, 1.0], sentiment in [-1.0, 1.0], volume >= 0. Exception names end with `Error` (ruff N818)

## Testing
- Test location: `tests/unit/core/`, `tests/unit/test_core.py`, `tests/unit/test_events.py`
- Run: `uv run pytest tests/unit/core/ tests/unit/test_core.py -v`

## Common Patterns
- `from __future__ import annotations` in every file
- Use `TYPE_CHECKING` guard for imports from upper layers
- All Pydantic models use `model_config = ConfigDict(frozen=True)`
- Decimal for monetary values, float for probabilities/ratios

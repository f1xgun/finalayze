# Phase 1 -- Backtest Pipeline Vertical Slice Design

**Date:** 2026-02-22
**Status:** Approved
**Goal:** Run `uv run python scripts/run_backtest.py --symbol AAPL --segment us_tech` and get a performance report.

## Approach

Bottom-up vertical slice through the full stack:
- Build from data layer up (schemas -> DB -> fetcher -> strategy -> risk -> broker -> backtest)
- Each layer testable independently with TDD (RED-GREEN-REFACTOR)
- Full PostgreSQL + TimescaleDB from the start (docker-compose required)

## Layer 0: Pydantic Schemas & Core Types

**File:** `src/finalayze/core/schemas.py`

Shared immutable data types:

- `Candle` -- OHLCV: symbol, market_id, timeframe, timestamp, open, high, low, close, volume
  - Prices as `Decimal`, timestamp as UTC-aware `datetime`
- `Signal` -- strategy output: strategy_name, symbol, market_id, segment_id, direction (BUY/SELL/HOLD), confidence (0-1), features dict, reasoning str
- `TradeResult` -- execution result: signal_id, symbol, side, quantity, entry_price, exit_price, pnl, pnl_pct
- `PortfolioState` -- snapshot: cash (Decimal), positions dict, equity (Decimal), timestamp
- `BacktestResult` -- performance metrics: sharpe, max_drawdown, win_rate, profit_factor, total_return, trades count

Key decisions:
- `Decimal` for all money/price fields (never float)
- All timestamps UTC-aware (enforced by ruff DTZ)
- Frozen Pydantic models for immutability
- `SignalDirection` as StrEnum: BUY, SELL, HOLD

## Layer 2: DB Models + Alembic

**Files:** `src/finalayze/core/models.py`, `alembic/versions/001_initial.py`

SQLAlchemy async models:

- `MarketModel` -- id (PK), name, currency, timezone, open_time, close_time
- `SegmentModel` -- id (PK), market_id (FK), name, description, active_strategies (ARRAY), strategy_params (JSONB), ml_model_id, max_allocation_pct, news_languages (ARRAY)
- `InstrumentModel` -- (symbol, market_id) composite PK, segment_id (FK), name, figi, instrument_type, currency, lot_size, is_active
- `CandleModel` -- (symbol, market_id, timeframe, timestamp) composite PK, OHLCV columns, source. **TimescaleDB hypertable** on timestamp.
- `SignalModel` -- UUID PK, strategy_name, symbol, market_id, segment_id, direction, confidence, features (JSONB), reasoning, created_at, mode
- `OrderModel` -- UUID PK, signal_id (FK), broker, broker_order_id, symbol, market_id, side, order_type, quantity, limit_price, stop_price, currency, status, filled_quantity, filled_avg_price, submitted_at, filled_at, risk_checks (JSONB), mode

Alembic initial migration creates all tables + `SELECT create_hypertable('candles', 'timestamp')`.

## Layer 2: Market Registry

**File:** `src/finalayze/markets/registry.py`

- `MarketDefinition` frozen dataclass: id, name, currency, timezone, open_time, close_time
- `MarketRegistry` class with:
  - `get_market(market_id: str) -> MarketDefinition`
  - `list_markets() -> list[MarketDefinition]`
  - `is_market_open(market_id: str, at: datetime) -> bool`
- Pre-loaded with US and MOEX definitions
- `SegmentRegistry` loads from `config/segments.py` defaults + YAML presets

## Layer 2: yfinance Fetcher

**Files:** `src/finalayze/data/fetchers/base.py`, `src/finalayze/data/fetchers/yfinance.py`, `src/finalayze/data/store.py`

- `BaseFetcher` ABC: `async fetch_candles(symbol, start, end, timeframe) -> list[Candle]`
- `YFinanceFetcher` implements BaseFetcher, wraps `yfinance.download()`
- `DataStore` class: `async save_candles(candles)`, `async load_candles(symbol, market_id, start, end, timeframe) -> list[Candle]`
- Store uses SQLAlchemy async session, upserts on conflict

## Layer 4: Momentum Strategy

**File:** `src/finalayze/strategies/momentum.py`, `src/finalayze/strategies/base.py`

- `BaseStrategy` ABC:
  - `name() -> str`
  - `supported_segments() -> list[str]`
  - `generate_signal(symbol, candles, segment_config) -> Signal | None`
  - `get_parameters(segment_id) -> dict` -- loads from YAML preset
- `MomentumStrategy(BaseStrategy)`:
  - Computes RSI and MACD via pandas-ta
  - BUY: RSI < oversold AND MACD histogram crosses above zero
  - SELL: RSI > overbought AND MACD histogram crosses below zero
  - Confidence based on RSI distance from threshold + MACD histogram magnitude
  - Parameters from per-segment YAML presets (us_tech: RSI 14/30/70, us_broad: RSI 14/30/70 but different weights)

## Layer 4: Risk Management

**Files:** `src/finalayze/risk/position_sizer.py`, `src/finalayze/risk/stop_loss.py`, `src/finalayze/risk/pre_trade_check.py`

- **Half-Kelly position sizing:**
  ```
  f* = (win_rate * avg_win_ratio - (1 - win_rate)) / avg_win_ratio
  position_value = equity * f* * kelly_fraction(0.5)
  capped at max_position_pct (20%) of equity
  ```
- **ATR stop-loss:**
  ```
  stop_loss_price = entry_price - ATR(14) * stop_loss_atr_multiplier(2.0)
  ```
- **Pre-trade checks (basic subset for this slice):**
  1. Position size <= max (20% of portfolio)
  2. Cash sufficient for order
  3. Open positions < max_positions_per_market (10)

## Layer 5: Simulated Broker

**File:** `src/finalayze/execution/simulated_broker.py`, `src/finalayze/execution/broker_base.py`

- `BrokerBase` ABC: `submit_order(order) -> OrderResult`, `get_portfolio() -> PortfolioState`
- `SimulatedBroker`:
  - Fills market orders at next candle's open price
  - Tracks cash, positions, equity internally
  - Checks stop-loss on each candle (fills if low <= stop price)
  - No slippage model (fills at exact price)
  - No commission model

## Backtest Module

**Files:** `src/finalayze/backtest/engine.py`, `src/finalayze/backtest/performance.py`

- `BacktestEngine`:
  - Input: symbol, segment_id, date range, strategy instance, initial_cash
  - Loads candles from DB
  - Iterates daily:
    1. Check stop-losses on current candle
    2. Strategy generates signal from candle history up to current day
    3. Pre-trade risk checks
    4. Simulated broker fills
    5. Record portfolio snapshot
  - Returns: trades list + portfolio snapshots
- `PerformanceAnalyzer`:
  - Input: trades + snapshots
  - Computes: Sharpe ratio, max drawdown, win rate, profit factor, total return, trade count, avg win, avg loss
  - Returns `BacktestResult`

## CLI Runner

**File:** `scripts/run_backtest.py`

```
uv run python scripts/run_backtest.py --symbol AAPL --segment us_tech --start 2023-01-01 --end 2024-12-31
```

- Connects to DB (requires docker-compose up)
- Seeds candles from yfinance if not present
- Instantiates MomentumStrategy with us_tech params
- Runs BacktestEngine
- Prints performance table

## Deferred (NOT in this slice)

- Redis event bus (not needed for synchronous backtest)
- Mean reversion strategy (second strategy, next slice)
- Strategy combiner (needs 2+ strategies)
- FastAPI endpoints (UI layer, separate slice)
- Clock abstraction (live trading only)
- Structured logging integration
- Currency conversion (US-only in this slice)

## Testing Strategy

- Unit tests for each module (TDD: write test first, then implement)
- Target: 80%+ coverage on strategies + risk modules
- Integration test: full backtest run on test fixtures (small candle dataset)
- No external API calls in tests (mock yfinance responses)

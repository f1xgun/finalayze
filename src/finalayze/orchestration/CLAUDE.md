# Orchestration

## Purpose
Top-level trading orchestrators: APScheduler-based live trading loop and bond cycle processor.
Moved from core/ in Phase 22 (dependency layer cleanup) to enforce correct dependency layering.

## Layer
Layer 5 -- Orchestration. Imports from all lower layers (0-4) plus execution (L5).
Injected with L6 dependencies (MetricsCollector, TelegramAlerter) via constructor.

## Key Files
- `trading_loop.py` -- TradingLoop: APScheduler-based live loop with news_cycle, strategy_cycle, daily_reset
- `bond_cycle.py` -- BondCycleProcessor: bond trading across portfolio layers (OFZ carry, PK->PD rotation)

## Public API
- `TradingLoop` -- live trading orchestrator
- `BondCycleProcessor` -- bond cycle processor
- `BondCycleResult` -- dataclass result of a bond cycle run
- `apply_ofz_rotation` -- OFZ PK->PD rotation logic

## Contracts
- Input: Settings, fetchers, strategies, risk components injected via constructor
- Output: Executes trades via BrokerRouter, sends alerts via TelegramAlerter
- Invariants: All upper-layer imports are deferred (TYPE_CHECKING or inline) to avoid circular imports

## Testing
- Test location: `tests/unit/core/test_trading_loop.py`, `tests/unit/core/test_bond_cycle.py`
- Note: Tests import via `finalayze.core.*` re-exports (backward compatibility)
- Run: `uv run pytest tests/unit/core/test_trading_loop.py tests/unit/core/test_bond_cycle.py -v`

## Common Patterns
- `from __future__ import annotations` in every file
- Use `TYPE_CHECKING` guard for imports from upper layers (alerts, monitoring)
- Thread safety via threading.Lock for _sentiment_cache in TradingLoop

# Phase 22: Dependency Layer Cleanup - Context

**Gathered:** 2026-03-22
**Status:** Ready for planning

<domain>
## Phase Boundary

Extract orchestrator files (trading_loop.py, bond_cycle.py) from core/ to new orchestration/ module. Move telegram_bot.py and alerts.py to L6. Inject MetricsCollector via constructor. Document layer assignments for backtest/ and monitoring/. Remove or wire dead event bus streams. Handle stub API endpoints.

</domain>

<decisions>
## Implementation Decisions

### Claude's Discretion
All implementation choices are at Claude's discretion — pure infrastructure phase.
Key constraints from audit findings:
- LAYER-01: Create src/finalayze/orchestration/ with trading_loop.py and bond_cycle.py
- LAYER-02: Move telegram_bot.py to api/ or dashboard/, alerts.py to api/ or notifications/
- LAYER-03: TradingLoop.__init__ accepts Optional[MetricsCollector] parameter, no import from api.metrics
- LAYER-04: Add CLAUDE.md to backtest/ (Layer: cross-cutting test infra) and monitoring/ (Layer 6)
- DEAD-01: Remove STREAM_MARKET_DATA, STREAM_SIGNALS, STREAM_EXECUTION if no consumers; keep STREAM_COUPONS if bond_discovery still publishes
- DEAD-02: Stub endpoints return 501 Not Implemented with {"detail": "Not yet implemented"} instead of empty 200

### Critical Risks
- trading_loop.py has ~50 import sites across the codebase — all must be updated
- APScheduler job references may use module paths that change
- Tests import from core.trading_loop — must update all test imports
- main.py wires trading_loop — entry point must be updated

</decisions>

<code_context>
## Existing Code Insights

### Key Files to Modify
- `src/finalayze/core/trading_loop.py` → `src/finalayze/orchestration/trading_loop.py`
- `src/finalayze/core/bond_cycle.py` → `src/finalayze/orchestration/bond_cycle.py`
- `src/finalayze/core/telegram_bot.py` → `src/finalayze/api/telegram_bot.py`
- `src/finalayze/core/alerts.py` → `src/finalayze/api/alerts.py`
- `src/finalayze/core/events.py` — remove dead streams
- `src/finalayze/api/v1/*.py` — stub endpoints
- All files importing from core.trading_loop, core.bond_cycle, core.alerts, core.telegram_bot

### Established Patterns
- Each module has __init__.py with public API exports
- Each module has CLAUDE.md documenting layer, public API, contracts
- Deferred imports used extensively to avoid circular dependencies
- TYPE_CHECKING guards used for type-only imports

</code_context>

<specifics>
## Specific Ideas

No specific requirements — infrastructure phase.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

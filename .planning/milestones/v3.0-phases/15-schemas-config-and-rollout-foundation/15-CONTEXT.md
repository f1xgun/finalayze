# Phase 15: Schemas, Config, and Rollout Foundation - Context

**Gathered:** 2026-03-21
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase delivers the RolloutPhase enum, per-phase risk limit overrides in Settings, wiring into PreTradeChecker and CircuitBreaker, and a capital ladder validation script. Pure infrastructure — no UI, no dashboard, no monitoring.

</domain>

<decisions>
## Implementation Decisions

### Claude's Discretion
All implementation choices are at Claude's discretion — pure infrastructure phase.

Key constraints from codebase scout:
- PreTradeChecker already accepts `max_position_pct` and `max_positions_per_market` from Settings
- CircuitBreaker L1/L2/L3 thresholds already configurable via `settings.circuit_breaker_l1/l2/l3`
- `max_sector_concentration_pct` and `min_cash_reserve_pct` are hardcoded defaults in PreTradeChecker (0.40 and 0.20) — need to surface to Settings
- Cross-market breaker has a bug: uses `settings.max_cross_market_exposure_pct` (0.80) instead of `_DEFAULT_CROSS_HALT` (0.10)
- MOEX lot rounding happens at broker layer (`TinkoffBroker.submit_order`) — lot sizes from InstrumentRegistry
- No rollout concept exists in codebase — entirely new

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `config/settings.py` — Pydantic Settings with `FINALAYZE_` env prefix, already has risk fields
- `src/finalayze/risk/pre_trade_check.py` — PreTradeChecker with 14 checks, accepts risk params at init
- `src/finalayze/risk/circuit_breaker.py` — CircuitBreaker with L1/L2/L3 thresholds from Settings
- `src/finalayze/risk/position_sizing_pipeline.py` — Pipeline with KellyStep, HardCapsStep etc.
- `src/finalayze/execution/tinkoff_broker.py` — Lot size rounding at execution layer
- `src/finalayze/markets/instruments.py` — InstrumentRegistry with lot_size per instrument

### Established Patterns
- Settings loaded from env vars with `FINALAYZE_` prefix via Pydantic
- Risk params flow: Settings → TradingLoop → PreTradeChecker/CircuitBreaker at init
- StrEnum for enums (ruff UP042 convention)
- Sizing pipeline uses Decimal for monetary values, lot rounding at broker layer

### Integration Points
- `main.py:241` — CircuitBreaker instantiation from settings
- `trading_loop.py:171` — PreTradeChecker instantiation from settings
- `backtest/engine.py:253,653` — PreTradeChecker in backtest uses same settings

</code_context>

<specifics>
## Specific Ideas

No specific requirements — infrastructure phase.

Success criteria from ROADMAP:
1. FINALAYZE_ROLLOUT_PHASE=MINIMAL → 3% max position, 1% daily loss, 2% DD auto-stop
2. Switch to STANDARD/FULL adjusts limits without code changes
3. Capital ladder script validates MOEX lot sizes at 50K/150K/500K/2.5M RUB

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

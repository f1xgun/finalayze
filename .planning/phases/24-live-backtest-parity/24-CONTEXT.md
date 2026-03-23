# Phase 24: Live-Backtest Parity - Context

**Gathered:** 2026-03-23
**Status:** Ready for planning

<domain>
## Phase Boundary

Align live trading loop risk pipeline with backtest engine: wire PositionSizingPipeline, implement trailing stops, pass all 14 pre-trade check parameters, prevent same-cycle re-entry after stop-loss exit.

</domain>

<decisions>
## Implementation Decisions

### Claude's Discretion
All implementation choices are at Claude's discretion — infrastructure phase.
Key constraints from audit:
- PARITY-01: Live _build_order() must instantiate PositionSizingPipeline with same steps as backtest Engine
- PARITY-02: Live trailing stop needs persistent state across APScheduler cycles — store in _stop_loss_prices dict with high-water mark
- PARITY-03: Pass stop_loss_price, has_pending_order, regime_state, strategy_name, correlations to PreTradeChecker.check()
- PARITY-04: Maintain per-cycle _exited_symbols set, skip signal generation for symbols that had stop-loss exits this cycle
- Pipeline was designed for backtest — may need adapter for live context (portfolio equity, market data)

</decisions>

<code_context>
## Existing Code Insights

### Key Files
- `src/finalayze/orchestration/trading_loop.py:1486` — _build_order() with simplified Kelly sizing
- `src/finalayze/orchestration/trading_loop.py:1562-1577` — fixed stop-loss (no trailing)
- `src/finalayze/orchestration/trading_loop.py:1439-1454` — pre-trade check with missing params
- `src/finalayze/orchestration/trading_loop.py:1326` — stop-loss check before signal generation
- `src/finalayze/risk/position_sizing_pipeline.py` — full pipeline with 7+ steps
- `src/finalayze/execution/simulated_broker.py:170` — trailing stop reference implementation
- `src/finalayze/risk/pre_trade_check.py:144` — full 14-check interface

### Established Patterns
- PositionSizingPipeline accepts SizingContext with equity, volatility, regime, etc.
- SimulatedBroker trailing stop: activation threshold → ratcheting high-water mark → trigger on low
- PreTradeChecker.check() returns (approved: bool, reason: str)

</code_context>

<specifics>
## Specific Ideas

No specific requirements — infrastructure phase.

</specifics>

<deferred>
## Deferred Ideas

None.

</deferred>

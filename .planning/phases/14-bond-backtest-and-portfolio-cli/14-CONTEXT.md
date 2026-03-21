# Phase 14: Bond Backtest and Portfolio CLI - Context

**Gathered:** 2026-03-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Close remaining audit gaps: wire apply_ofz_rotation into BondBacktestEngine for bond backtest validation, implement real engine calls in run_portfolio_backtest.py CLI.

</domain>

<decisions>
## Implementation Decisions

### Claude's Discretion
All implementation choices are at Claude's discretion — pure infrastructure/gap-closure phase.

Key fixes:
1. BondBacktestEngine: integrate apply_ofz_rotation from bond_cycle.py so bond backtests reflect CBR cutting cycle rotation (CORE→STRATEGIC capital shift)
2. run_portfolio_backtest.py: replace stub _run_bond_backtest() and _run_equity_backtest() with real implementations that load data and run engines, producing actual PortfolioBacktestResult

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `apply_ofz_rotation` in `core/bond_cycle.py` — already implemented, just not connected to backtest
- `BondBacktestEngine` in `backtest/bond_engine.py` — runs independently
- `BacktestEngine` in `backtest/engine.py` — equity backtest
- `PortfolioBacktestOrchestrator` in `backtest/portfolio_orchestrator.py` — merges curves
- `run_iteration.py` — existing equity orchestration pattern to follow

### Integration Points
- `backtest/bond_engine.py` — add OFZ rotation call before layer processing
- `scripts/run_portfolio_backtest.py` — implement _run_bond_backtest(), _run_equity_backtest(), _extract_usdrub_series()

</code_context>

<specifics>
## Specific Ideas

No specific requirements — infrastructure phase.

</specifics>

<deferred>
## Deferred Ideas

None.

</deferred>

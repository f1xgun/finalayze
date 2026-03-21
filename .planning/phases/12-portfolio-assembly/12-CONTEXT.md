# Phase 12: Portfolio Assembly - Context

**Gathered:** 2026-03-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Create PortfolioBacktestOrchestrator that runs bond and equity engines jointly, merges equity curves with 40/60 OFZ/equity allocation, monthly rebalancing, and RUB crisis brake. Target walk-forward Sharpe >= +0.10 on the blended portfolio.

</domain>

<decisions>
## Implementation Decisions

### PortfolioBacktestOrchestrator Design
- Location: new file `src/finalayze/backtest/portfolio_orchestrator.py` at backtest layer
- Curve merging: run BondBacktestEngine and BacktestEngine independently, merge equity curves by date alignment and weighted sum (40% OFZ + 60% equity)
- Aggregate metrics: compute Sharpe/DD/PF on the merged portfolio curve using existing PerformanceAnalyzer
- Walk-forward: apply walk-forward validation on the merged portfolio curve (not individual engines)

### Portfolio Allocation & Rebalancing
- Allocation: static 40/60 split via initial capital allocation (OFZ gets 40% of total, equity gets 60%)
- Monthly rebalancing: at each month boundary, compare actual weights to 40/60 target; if drift > 5%, adjust next period's capital allocation
- RUB crisis brake: check USDRUB 20-bar return > 15% — if triggered, freeze equity allocation, shift new capital to 80/20 OFZ/equity until FX stabilizes
- Crisis brake data: USDRUB from MacroSnapshot/MOEX ISS — same data already available

### Walk-Forward Sharpe Target
- WF window: 12mo train + 6mo test (same as equity WF) applied to merged portfolio curve
- Sharpe measurement: annualized Sharpe on WF test windows averaged across folds
- If Sharpe < 0.10: report achieved Sharpe — aspirational target, not a hard gate for phase completion
- Bond carry contribution: OFZ carry (6-8% yield) provides base return that should lift blended Sharpe

### Claude's Discretion
- Internal PortfolioBacktestOrchestrator API design (run method signature, result dataclass)
- Test structure and fixture design for portfolio-level tests
- How to handle date alignment gaps between bond and equity curves
- Monthly rebalancing implementation details (exact drift calculation)
- Crisis brake cooldown period (how long to wait before reverting to 40/60)

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `BacktestEngine` in `backtest/engine.py` — equity backtesting, per-symbol iteration
- `BondBacktestEngine` in `backtest/bond_engine.py` — OFZ bond backtesting
- `PerformanceAnalyzer` in `backtest/performance.py` — Sharpe, DD, PF computation
- `PortfolioAggregator` in `backtest/portfolio_aggregator.py` — layer aggregation (bond-only currently)
- `WalkForwardOptimizer` in `backtest/walk_forward.py` — 12/6/3 month windows
- `IterationTracker` in `backtest/iteration_tracker.py` — metrics + gates
- `MacroSnapshot` with `usdrub` field — for crisis brake detection
- `run_iteration.py` — existing equity orchestration script

### Established Patterns
- Bond and equity engines are independent — no shared state
- PortfolioAggregator merges LayerResults by date alignment
- Walk-forward produces per-fold metrics, averaged for overall Sharpe
- Iteration comparison via IterationTracker gates (S1-S3, C1-C6)

### Integration Points
- `backtest/portfolio_orchestrator.py` — new file, orchestrates bond + equity engines
- `scripts/run_iteration.py` — needs portfolio mode for joint runs
- `backtest/performance.py` — may need portfolio-level metrics
- USDRUB candles — already fetched in `_compute_moex_sizing_data()`

</code_context>

<specifics>
## Specific Ideas

No specific requirements — standard orchestration using established patterns.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

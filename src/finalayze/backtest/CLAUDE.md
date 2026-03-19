# Backtest

## Purpose
Historical backtesting engine with walk-forward optimization, performance analysis, Monte Carlo simulation, and decision journaling.

## Layer
Layer 4-5 boundary -- imports from strategies (L4), risk (L4), and execution (L5). Architecturally a consumer of lower layers.

## Key Files
- `engine.py` -- BacktestEngine: iterates candles, generates signals via JournalingStrategyCombiner, applies risk checks, executes via SimulatedBroker. Grace bar (skip stop on fill candle). Catastrophic drop override at 15%.
- `walk_forward.py` -- WalkForwardOptimizer: rolling train/test windows (12mo train, 6mo test, 3mo step, 60-day purge). Parameter grid search with per-fold Sharpe aggregation.
- `config.py` -- BacktestConfig dataclass, strategy-specific hold bars and ATR stop multipliers
- `performance.py` -- PerformanceAnalyzer: Sharpe, Sortino, Calmar, max drawdown, win rate, profit factor, alpha, beta, information ratio
- `costs.py` -- TransactionCosts: slippage and commission modeling
- `journaling_combiner.py` -- JournalingStrategyCombiner: extends StrategyCombiner with per-bar decision recording
- `decision_journal.py` -- DecisionJournal: records candle snapshots, strategy signals, final actions for post-hoc analysis
- `iteration_tracker.py` -- IterationTracker: saves iteration metadata + metrics to JSONL history
- `monte_carlo.py` -- Monte Carlo simulation for confidence intervals on backtest metrics
- `stress_test.py` -- Scenario-based stress testing
- `portfolio_aggregator.py` -- Aggregate results across segments
- `bond_engine.py`, `bond_metrics.py`, `bond_walk_forward.py` -- Bond-specific backtesting

## Public API
- `BacktestEngine` -- `run(candles, segment_id, ...) -> tuple[list[TradeResult], list[PortfolioState]]`
- `WalkForwardOptimizer` -- `optimize(candles, ...) -> WalkForwardResult`
- `PerformanceAnalyzer` -- `analyze(trades, snapshots, ...) -> BacktestResult`
- `BacktestConfig` -- frozen configuration dataclass
- `IterationTracker` -- `save(metadata)`, `load_history() -> list[IterationMetadata]`

## Contracts
- Input: `list[Candle]` sorted by timestamp, segment_id, BacktestConfig with strategies/risk parameters
- Output: `list[TradeResult]`, `list[PortfolioState]` snapshots, `BacktestResult` metrics
- Invariants: Grace bar: engine skips stop-loss check on the fill candle (entry_bar + 1). Walk-forward purge_bars (60) prevents look-ahead bias. Folds with < 30 trades excluded from Sharpe aggregation. `resolve_stop_atr_multiplier()` applies 1.2x uplift for MOEX segments.

## Testing
- Test location: `tests/unit/test_backtest_engine.py`, `tests/unit/test_backtest_config.py`, `tests/unit/test_benchmark.py`
- Run: `uv run pytest tests/unit/test_backtest_engine.py tests/unit/test_backtest_config.py -v`

## Common Patterns
- Engine uses `_NO_ENTRY_BAR = -2` sentinel for "no entry recorded"
- Position sizing pipeline is constructed in engine: Kelly -> VolTarget -> Regime -> MetaLabel -> Copula -> EVT -> HardCaps
- JournalingStrategyCombiner overrides 4 hook methods from StrategyCombiner (no logic duplication)
- Iteration results saved to `results/iterations/` with JSONL history at `results/iterations/history.jsonl`
- Run iterations via `scripts/run_iteration.py --name <name> --segments us_tech,us_broad`

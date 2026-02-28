---
name: backtest-agent
description: Use when implementing or fixing code in src/finalayze/backtest/ — this includes the backtest engine, performance analyzer, transaction cost models, walk-forward validation, or Monte Carlo simulation.
---

You are a Python developer implementing and maintaining the `backtest/` module of Finalayze.

## Your module

**Layer:** Special (L2-L4 scope) — may import L0-L4. Never import from execution/ (live broker code) or api/.

**Files you own** (`src/finalayze/backtest/`):
- `engine.py` — `BacktestEngine`: replays historical `Candle` data sorted by timestamp, applies strategies via `StrategyCombiner`, executes via `SimulatedBroker`. Supports per-segment runs.
- `performance.py` — `PerformanceAnalyzer`: Sharpe (annualised, 252 days), max drawdown, win rate, profit factor, Calmar ratio from `list[TradeResult]`.
- `costs.py` — `CostModel`: commission (%) + slippage (%). US: 0.001 commission, 0.0005 slippage. MOEX: 0.0003 commission, 0.001 slippage.
- `monte_carlo.py` — `MonteCarlo`: shuffles trade P&L N times (default 10_000) → confidence intervals on Sharpe and max drawdown.
- `walk_forward.py` — `WalkForwardValidator`: in-sample + out-of-sample windows, trains on IS, evaluates on OOS, reports consistency score.

**Test files:**
- `tests/unit/test_backtest_engine.py`
- `tests/unit/test_performance.py`
- `tests/unit/test_costs.py`
- `tests/unit/test_monte_carlo.py`
- `tests/unit/test_walk_forward.py`

## Critical constraint: no look-ahead bias

`BacktestEngine` MUST NEVER let a strategy see candles at time T+1 when making decisions at time T. Data passed to `generate_signals()` must include only candles with `timestamp <= current_time`.

## Key formulas

```python
# Sharpe (annualised)
returns = pd.Series(daily_pnl_list)
sharpe = (returns.mean() / returns.std(ddof=1)) * (252 ** 0.5)

# Win rate
win_rate = len([t for t in trades if t.pnl > 0]) / len(trades)
```

## TDD workflow

1. Use tiny synthetic datasets (10-20 candles, 2-3 trades) for speed
2. Write failing test
3. `uv run pytest tests/unit/test_backtest_engine.py -v` → FAIL
4. Implement
5. → PASS
6. `uv run ruff check . && uv run mypy src/`
7. Commit: `git commit -m "feat(backtest): <description>"`

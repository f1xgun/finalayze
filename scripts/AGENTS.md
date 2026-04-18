# scripts/ — CLI Entry Points (Area Node)

Parent: [root AGENTS.md](../AGENTS.md)

Standalone Python CLIs for backtests, iterations, training, evaluation, and sandbox ops.
Scripts import the `finalayze` package; none of them should be imported **by** the package.

## Conventions

- Every script inserts the project root into `sys.path` before importing `config` (because
  `config/` sits at repo root, not under `src/`). Pattern:
  ```python
  from pathlib import Path
  import sys
  sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
  ```
- CLIs use `argparse` (not click/typer) for zero extra deps.
- Long-running scripts emit structlog events and flush stdout often.
- Outputs land in `results/` (iterations, evaluations) or `models/` (training).

## Script index

### Backtest / iteration (the most-used family)

| Script | Purpose |
|---|---|
| `run_iteration.py` | **Primary**. Runs a full named iteration across segments, saves to `results/iterations/`. Entry for the `backtest-iteration` skill. |
| `run_backtest.py` | Single-segment backtest without iteration metadata. |
| `run_batch_backtest.py` | Batch backtests across a grid of configs. |
| `run_portfolio_backtest.py` / `run_portfolio_iteration.py` | Multi-segment portfolio aggregation. |
| `run_comprehensive_backtest.py` | Full walk-forward + Monte Carlo + stress tests. |
| `run_strategy_isolation.py` | Run each strategy alone to get isolated Sharpe / PF. |
| `run_bond_iteration.py` | Bond-specific iteration (OFZ carry + rotation). |
| `compare_iterations.py` | Side-by-side metric diff between two iterations in history.jsonl. |
| `list_iterations.py` | Tabular summary of iteration history. |

### Evaluation / validation

| Script | Purpose |
|---|---|
| `run_evaluation.py`, `run_batch_evaluation.py` | Grading runs with evaluation-agent reports. |
| `run_moex_evaluation.py`, `run_moex_tinkoff_eval.py` | MOEX-focused evaluation. |
| `run_validation.py`, `run_sandbox_validation.py` | Sandbox paper-trade validation. |
| `validate_capital_ladder.py`, `validate_ofz_data.py` | Data-integrity checks. |
| `generate_validation_report.py` | HTML / JSON validation summary. |
| `smoke_test_sandbox.py`, `run_sandbox.py` | Sandbox smoke tests. |

### ML training

| Script | Purpose |
|---|---|
| `train_models.py` | US training. Key flags: `--segment`, `--walk-forward`, `--excess-returns`, `--sequential-bootstrap`, `--force-save`. |
| `train_moex_models.py` | MOEX-specific training (requires Tinkoff token). |
| `auto_ml_research.py` | Automated research loop (feature eng, ensemble weights, fold analysis). |
| `tune_hyperparams.py`, `tune_strategy_params.py` | Optuna tuning with overfitting guardrails. |
| `training/` | Shared training utilities for the above. |

### Data / ops

| Script | Purpose |
|---|---|
| `seed_historical_data.py` | Populate database with historical candles. |
| `fetch_moex_dividends.py` | Pull MOEX dividend history via Tinkoff + fallback YAML. |
| `build_event_data.py` | Assemble event dataset for event-driven strategy. |
| `derive_gate_thresholds.py` | Derive go/no-go thresholds from sandbox metrics. |
| `daily_review.py` | Post-market autonomous review pipeline (see `.claude/skills/daily-review`). |
| `test_pairs_cointegration.py` | Pairs-trading cointegration screen. |
| `run_interaction_test.py` | Cross-component interaction test harness. |

## When adding a new script

1. Place at `scripts/<verb>_<noun>.py`.
2. Add argparse entry point under `if __name__ == "__main__":`.
3. Add a row here.
4. If the script gates a GSD workflow (e.g. replaces a skill), document in `.claude/skills/`.

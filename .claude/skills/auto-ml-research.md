# Auto ML Research — Autonomous Experiment Loop

Run unattended ML experiments overnight.  The script generates hypotheses,
trains models, evaluates walk-forward quality gates, and keeps improvements.
Inspired by karpathy/autoresearch.

## When to Use

- User asks to "run ML experiments overnight" or "try feature combinations"
- After enabling ML for a new segment — find the best feature/hyperparameter config
- When you want to systematically evaluate feature importance via ablation
- When investigating whether simpler feature sets can match complex ones

## Quick Start

```bash
# Run all strategies on us_tech (ablation + efficiency + hyperparameter + random)
uv run python scripts/auto_ml_research.py --segment us_tech --strategy all

# Just feature ablation (drop features one at a time)
uv run python scripts/auto_ml_research.py --segment us_tech --strategy ablation

# Efficiency-driven feature selection (Pareto-optimal)
uv run python scripts/auto_ml_research.py --segment us_tech --strategy efficiency

# Hyperparameter perturbation
uv run python scripts/auto_ml_research.py --segment us_tech --strategy hyperparameter

# Limit to 20 experiments
uv run python scripts/auto_ml_research.py --segment us_tech --max-experiments 20
```

## Experiment Strategies

| Strategy | What It Does | When to Use |
|---|---|---|
| `ablation` | Drop each feature one-by-one | Find unnecessary features (simplification) |
| `efficiency` | Select by MI/complexity ratio | Find minimal effective feature set |
| `hyperparameter` | Perturb model params one-by-one | Tune XGB/LightGBM/CatBoost |
| `random_subset` | Random feature subsets | Explore feature space broadly |
| `all` | Run all strategies sequentially | Overnight unattended research |

## How It Works

1. **Data load** (one-time) — fetches candles, builds triple-barrier dataset
2. **Baseline** — standard MI feature selection, default hyperparams
3. **Experiment loop** — for each hypothesis:
   - Configure features / hyperparams per the strategy
   - Train XGBoost + LightGBM + CatBoost on each walk-forward fold
   - Evaluate 7 quality gates (accuracy, Brier, PF, signal count, etc.)
   - Compute composite score = 0.4*accuracy + 0.3*(1-brier) + 0.2*gate_pass + 0.1*(1-complexity)
   - Log result to JSONL
4. **Summary** — top experiments ranked by score

## Output

Results logged to `results/experiments/<segment>_experiment_log.jsonl`.

Each entry contains:
- `name`, `description`, `strategy` — what was tried
- `score` — composite score (higher = better)
- `avg_accuracy`, `avg_brier`, `avg_profit_factor` — key metrics
- `feature_count`, `features_used` — which features
- `complexity` — total/mean/max complexity, n_external, n_high_compute
- `gate_pass_rates` — per-gate pass rates across folds
- `status` — "keep" (gates passed), "discard" (gates failed), "crash"

## Simplicity Criterion

The script integrates feature complexity scoring from
`src/finalayze/ml/training/feature_complexity.py`:

- Each feature has a complexity score based on lookback, compute cost, and data dependency
- **Efficiency** = importance / complexity (higher = more signal per cost)
- The `efficiency` strategy uses Pareto-optimal selection: cheap informative features first
- Ablation experiments identify features that can be removed without quality loss

**Key principle**: "A small improvement that adds complexity is not worth it.
Removing something and getting equal or better results is a great outcome."

## Interpreting Results

After the run, look at the top experiments:

1. **Ablation winners** (status=keep after dropping a feature) → that feature is unnecessary.
   Remove it from the feature set for simpler, faster models.

2. **Efficiency winners** (fewer features, similar score) → simpler is better.
   Update `max_features` in training config.

3. **Hyperparameter winners** (better score) → update tuned_params.
   Save to `results/tuned_params/<segment>/`.

## Integration with Existing Skills

- Run `auto-ml-research` FIRST to find the best config
- Then run `ml-experiment` to train final models with that config
- Then run `backtest-iteration` to validate end-to-end trading impact
- Gate: `backtest-iteration` results must still pass iteration gates

## NEVER STOP Pattern

When running with `--strategy all`, the script processes all experiments
sequentially.  For true overnight autonomous mode, use Claude with `/loop`:

```
/loop 270s uv run python scripts/auto_ml_research.py --segment us_tech --strategy all
```

Or have Claude drive it interactively — read the JSONL after each run,
generate new hypotheses based on what worked, and queue the next batch.

# Weekly Strategy Deep Dive (Multi-Agent)

Multi-agent collaborative review of weekly trading performance. Three domain experts
analyze independently, then their findings are cross-fed for deliberation to produce
consensus findings and experiment proposals.

Uses parallel sub-agents for independent analysis, then a synthesis round where
each expert sees and challenges the others' findings.

## When to Use

- Every Saturday morning for scheduled weekly review
- When user says "weekly review", "deep dive", "strategy review"
- After a week with significant anomalies flagged by daily reviews

## Prerequisites

- Daily review data exists in `results/daily/` for the past week

## Instructions

### Step 1: Collect Weekly Data

```bash
cd /Users/f1xgun/finalayze

# Collect daily data for any missing days
uv run python scripts/daily_review.py --summary

# Run fresh walk-forward backtest on current parameters
uv run python scripts/run_iteration.py \
  --name "weekly-review-$(date +%Y%m%d)" \
  --description "Automated weekly review iteration" \
  --segments us_tech,us_broad
```

Read all daily reports from this week:
```
results/daily/{mon..fri}/raw_data.json
results/daily/{mon..fri}/analysis.md (if exists)
```

Read the latest iteration results from `results/iterations/history.jsonl`.

### Step 2: Prepare Context Brief

Create a shared context document that all teammates will read:

```markdown
# Weekly Context Brief: {week_start} - {week_end}

## Daily Snapshots
| Day | Signals | Fills | Fill% | MaxDD | Errors | Verdict |
|-----|---------|-------|-------|-------|--------|---------|
| Mon | ... | ... | ... | ... | ... | ... |
| ... | ... | ... | ... | ... | ... | ... |

## Week Totals
- Total signals: X | Total fills: Y | Fill rate: Z%
- Max drawdown (week): X% | Avg daily drawdown: Y%
- Errors: X | Circuit breaker triggers: Y

## Latest Backtest Iteration
- Name: weekly-review-YYYYMMDD
- WF Sharpe: X | Max DD: Y% | Trades: Z | Verdict: PASS/REJECT

## Strategy Performance (from backtest)
| Strategy | Sharpe | PF | Trades | Win% |
|----------|--------|-----|--------|------|
| ... | ... | ... | ... | ... |

## Active Anomalies from Daily Reviews
- [list all anomalies flagged this week]

## ML Status
- Enabled segments: [list]
- Model last trained: [date]
- Accuracy trend: [improving/stable/degrading]

## Open Action Items
- P0: [from daily reviews]
- P1: [from daily reviews]
```

Save to `results/weekly/{week}/context_brief.md`.

### Step 3: Round 1 — Independent Analysis (Parallel Sub-Agents)

Launch 3 sub-agents in parallel. Each reads the context brief independently.
Use the Agent tool with specialized sub-agent types.

**Agent 1 — quant-analyst**: Signal quality, regime analysis, parameter sensitivity,
experiment proposals. Reads: `results/weekly/{week}/context_brief.md`,
`src/finalayze/strategies/presets/*.yaml`, `results/iterations/history.jsonl`.

**Agent 2 — risk-officer**: Drawdown analysis, position sizing, circuit breaker review,
correlation risk, go-live readiness score. Reads: `results/weekly/{week}/context_brief.md`,
`config/gate_thresholds.yaml`, `src/finalayze/risk/pre_trade_check.py`.

**Agent 3 — ml-engineer**: Model drift, prediction accuracy, feature importance shifts,
retraining decision, data quality. Reads: `results/weekly/{week}/context_brief.md`,
`models/*/segment_meta.json`, `src/finalayze/ml/features/technical.py`.

Save each output to:
- `results/weekly/{week}/quant_analysis.md`
- `results/weekly/{week}/risk_analysis.md`
- `results/weekly/{week}/ml_analysis.md`

### Step 4: Round 2 — Deliberation (Cross-Review Sub-Agents)

Launch 3 more sub-agents, each reading ALL Round 1 outputs to challenge
and respond to the other experts' findings.

**Agent 4 — risk-officer** (deliberation): Read all three Round 1 analyses.
Task: Challenge quant's experiment proposals from risk perspective.
Flag any correlations or concentration risks quant missed. Score experiments
on risk-adjusted expected value. Save to `results/weekly/{week}/risk_response.md`.

**Agent 5 — quant-analyst** (deliberation): Read all three Round 1 analyses.
Task: Respond to risk officer's concerns. Evaluate ML engineer's retraining
recommendation — would fresh models change your experiment priorities?
Adjust experiment proposals based on feedback. Save to `results/weekly/{week}/quant_response.md`.

**Agent 6 — ml-engineer** (deliberation): Read all three Round 1 analyses.
Task: Assess whether quant's proposed parameter changes would invalidate
current ML models. Confirm or revise retraining recommendation based on
risk officer's drawdown findings. Save to `results/weekly/{week}/ml_response.md`.

### Step 5: Synthesize Consensus

Combine all teammate outputs into the weekly report:

```markdown
# Weekly Deep Dive: {week}

## Executive Summary
[2-3 sentence verdict: are we improving, degrading, or flat?]

## Performance Scorecard
| Metric | This Week | Last Week | Trend | Target |
|--------|-----------|-----------|-------|--------|
| WF Sharpe | X | Y | arrow | >0.5 |
| Max DD | X% | Y% | arrow | <2.27% |
| Fill Rate | X% | Y% | arrow | >95% |
| Trade Count | X | Y | arrow | >18/5d |

## Expert Analyses

### Quant Analyst Findings
[summary]

### Risk Officer Findings
[summary]

### ML Engineer Findings
[summary]

## Consensus Action Items
- **AGREED**: [items all experts agree on]
- **DEBATED**: [items with disagreement — note positions]
- **DEFERRED**: [items needing more data]

## Experiment Queue (prioritized)
1. [Experiment with highest expected value and risk assessment]
2. [Second experiment]

## Go-Live Readiness Score
- Current: X/100
- Last week: Y/100
- Blockers: [list]
- Estimated weeks to ready: N

## Decisions for Human Review
[Items that need human judgment — e.g., capital allocation, risk tolerance]
```

Save to `results/weekly/{week}/deep_dive.md`.

### Step 6: Route Actions

- Experiments → add to GSD backlog as candidate phases
- P0 fixes → GSD todos
- Retraining decision → if yes, trigger `/ml-experiment`
- Go-live blockers → update `docs/operations/GO_LIVE_CHECKLIST.md` status

## Token Cost Estimate

| Component | Est. Tokens |
|-----------|-------------|
| Context brief (shared) | ~5K |
| Round 1: 3 parallel agents | ~150-200K |
| Round 2: 3 deliberation agents | ~100-150K |
| Synthesis | ~30K |
| **Total** | **~300-400K** |

At Opus pricing (~$15/MTok input, $75/MTok output), expect ~$5-10 per weekly review.

## Budget Mode (Cheaper Alternative)

Skip Round 2 deliberation. Run only Round 1 (3 parallel agents) + synthesis.
Cost: ~150-200K tokens (~$3-5). Loses cross-expert challenge quality but
still provides 3 independent expert views.

To use: Set `budget_mode: true` in `config/pipelines.yaml` under `weekly_deep_dive`.

## Scheduling

Designed for weekly autonomous execution:
```
cron: "7 10 * * 6"   # Saturday 10:07 local time
prompt: "/weekly-deep-dive"
```

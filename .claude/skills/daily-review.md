# Daily Post-Market Review

Autonomous pipeline that collects, analyzes, and prioritizes trading day findings.
Run after market close to get actionable insights without manual log reading.

## When to Use

- After US market close (16:00 ET / 23:00 MSK) or MOEX close (18:50 MSK)
- When scheduled via cron trigger for autonomous daily review
- When user says "daily review", "what happened today", "analyze today's trading"

## Instructions

### Step 1: Collect Data

```bash
cd /Users/f1xgun/finalayze
uv run python scripts/daily_review.py --collect
```

Read the generated report from `results/daily/{date}/raw_data.json`.

If no cycle data exists (system wasn't trading), note this and skip to Step 4.

### Step 2: Analyze with Domain Experts

Launch sub-agents in parallel for multi-perspective analysis:

**Agent 1 — quant-analyst**: Read `raw_data.json` and analyze:
- Signal quality: how many signals converted to profitable trades?
- Strategy attribution: which strategies generated signals, which were profitable?
- Compare today's metrics to 7-day rolling average — any degradation?
- Check if any strategy shows consistent false signals (>3 consecutive losses)

**Agent 2 — risk-officer**: Read `raw_data.json` and analyze:
- Drawdown: is it trending up over the 7-day window?
- Circuit breaker triggers: root cause analysis if triggered
- Position sizing: were positions appropriately sized for volatility?
- Exposure: concentrated in any single instrument or sector?

**Agent 3 — ml-engineer** (only if ML is enabled for any segment):
- Model prediction accuracy for today's signals
- Feature drift: any features showing unusual distributions?
- Calibration: are confidence scores well-calibrated to actual outcomes?

### Step 3: Synthesize Findings

Combine expert analyses into a structured assessment:

```markdown
## Daily Review: {date}

### Performance Summary
| Metric | Today | 7d Avg | Delta | Status |
|--------|-------|--------|-------|--------|
| Signals | X | Y | Z% | OK/WARN |
| Fill Rate | X% | Y% | Z% | OK/WARN |
| Max Drawdown | X% | Y% | Z% | OK/WARN |
| Errors | X | Y | Z | OK/WARN |

### Key Findings
1. [Most important finding with evidence]
2. [Second finding]
3. [Third finding]

### Action Items
- **P0 (fix today):** [critical issues]
- **P1 (this week):** [important improvements]
- **P2 (backlog):** [nice-to-have optimizations]

### Experiment Proposals
- [Hypothesis → expected impact → how to test]
```

Save to `results/daily/{date}/analysis.md`.

### Step 4: Route Action Items

For each action item:
- **P0 items**: Create GSD todos via `/gsd:add-todo` with `[P0-DAILY]` prefix
- **P1 items**: Add to backlog via `/gsd:add-backlog` with context
- **P2 items**: Add to notes via `/gsd:note`

### Step 5: Update Trend Tracker

Append one-line summary to `results/daily/trend.csv`:
```
date,signals,fills,fill_rate,max_dd,errors,anomaly_count,verdict
```

Verdict: GOOD (no anomalies), WATCH (minor anomalies), ACTION (P0 items exist)

## Output Files

```
results/daily/{date}/
  raw_data.json     -- collected metrics (Step 1)
  analysis.md       -- synthesized analysis (Step 3)
results/daily/
  trend.csv         -- rolling trend data (Step 5)
```

## Scheduling

This skill is designed to run autonomously via scheduled trigger:
```
cron: "3 23 * * 1-5"   # Weekdays 23:03 MSK (after MOEX close)
prompt: "/daily-review"
```

For US-only review, schedule at 16:03 ET.

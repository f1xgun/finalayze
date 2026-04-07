# Go-Live Readiness Scorecard

Automated production readiness assessment. Checks each GO_LIVE_CHECKLIST criterion,
computes a readiness score, and tracks progress toward real trading.

## When to Use

- Every Sunday for scheduled readiness check
- Before proposing to move from sandbox to real trading
- When user asks "are we ready for production?", "go-live status"

## Instructions

### Step 1: Run Automated Checks

Execute each verifiable criterion and record pass/fail:

**Tests & Quality:**
```bash
cd /Users/f1xgun/finalayze

# Tests passing?
uv run pytest tests/ -x --tb=short -q 2>&1 | tail -5

# Lint clean?
uv run ruff check . --quiet 2>&1 | tail -3

# Type check?
uv run mypy src/ --no-error-summary 2>&1 | tail -5

# Test count
uv run pytest tests/ --collect-only -q 2>&1 | tail -1
```

**Backtest Gates:**
```bash
# Latest iteration verdict
tail -1 results/iterations/history.jsonl | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print(f'Verdict: {d[\"verdict\"]}  WF Sharpe: {d.get(\"wf_sharpe\",\"N/A\")}  Max DD: {d.get(\"wf_max_drawdown\",\"N/A\")}')"
```

**Sandbox Metrics (if DB available):**
```bash
# Check if sandbox has been running
ls -la results/validation/cycles.jsonl 2>/dev/null && \
  python3 -c "
import json
from pathlib import Path
from collections import defaultdict
lines = Path('results/validation/cycles.jsonl').read_text().strip().splitlines()
by_date = defaultdict(list)
for l in lines:
    d = json.loads(l)
    by_date[d['timestamp'][:10]].append(d)
print(f'Sandbox days: {len(by_date)}')
print(f'Total cycles: {len(lines)}')
if lines:
    last = json.loads(lines[-1])
    print(f'Last cycle: {last[\"timestamp\"]}')
    print(f'Max DD: {max(json.loads(l)[\"drawdown_pct\"] for l in lines):.2f}%')
"
```

**Configuration Check:**
```bash
# Check env vars (without printing values)
python3 -c "
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path('.env'))
checks = {
    'FINALAYZE_TINKOFF_TOKEN': bool(os.getenv('FINALAYZE_TINKOFF_TOKEN')),
    'FINALAYZE_LLM_API_KEY': bool(os.getenv('FINALAYZE_LLM_API_KEY')),
    'FINALAYZE_TELEGRAM_BOT_TOKEN': bool(os.getenv('FINALAYZE_TELEGRAM_BOT_TOKEN')),
    'FINALAYZE_TELEGRAM_CHAT_ID': bool(os.getenv('FINALAYZE_TELEGRAM_CHAT_ID')),
}
for k, v in checks.items():
    print(f'  {k}: {\"SET\" if v else \"MISSING\"} ')
"
```

**Docker Health:**
```bash
docker ps --filter name=finalayze-sandbox --format '{{.Names}}: {{.Status}}' 2>/dev/null
```

### Step 2: Compute Scorecard

Map each check to a criterion with weight:

| # | Criterion | Weight | Source | Auto-Check |
|---|-----------|--------|--------|------------|
| 1 | Tests green | 15 | pytest exit code | Yes |
| 2 | Lint clean | 5 | ruff exit code | Yes |
| 3 | Type check clean | 5 | mypy exit code | Yes |
| 4 | WF Sharpe > 0 (ru_* segments) | 15 | history.jsonl | Yes |
| 5 | Sandbox 5+ trading days | 15 | cycles.jsonl | Yes |
| 6 | Max DD < 2.27% | 10 | gate_thresholds | Yes |
| 7 | Fill rate > 95% | 5 | sandbox metrics | Yes |
| 8 | Circuit breakers configured | 5 | config check | Yes |
| 9 | Tinkoff token (real) set | 5 | env var check | Partial |
| 10 | Telegram alerts configured | 5 | env var check | Yes |
| 11 | Emergency procedures tested | 5 | Manual | No |
| 12 | Starting capital verified | 10 | Manual | No |

**Score = sum of passed weights / 100**

### Step 3: Generate Report

```markdown
# Go-Live Readiness Scorecard

**Date:** {date}
**Score:** {score}/100 ({verdict})
**Previous:** {prev_score}/100 ({delta})

## Automated Checks
| # | Criterion | Status | Details |
|---|-----------|--------|---------|
| 1 | Tests green | PASS/FAIL | 2325 tests, 0 failures |
| ... | ... | ... | ... |

## Manual Checks (require human verification)
| # | Criterion | Status | Last Verified |
|---|-----------|--------|---------------|
| 11 | Emergency procedures tested | PENDING | never |
| 12 | Starting capital verified | PENDING | never |

## Verdict
- **READY** (score >= 85 AND all critical checks pass)
- **ALMOST** (score >= 70, minor items remaining)
- **NOT READY** (score < 70 OR critical check fails)

## Blockers
1. [Critical items preventing go-live]

## Trend
| Date | Score | Delta | Notes |
|------|-------|-------|-------|
| {date} | X | +Y | ... |
| {prev} | X | +Y | ... |
```

Save to `results/readiness/{date}/scorecard.md`.

### Step 4: Append to History

Append one line to `results/readiness/history.csv`:
```
date,score,verdict,tests_pass,lint_pass,mypy_pass,wf_sharpe_pass,sandbox_days,max_dd_pass,blockers
```

### Step 5: Notify

If score changed significantly (>10 points) or verdict changed, flag for user attention.
If score >= 85 for 2 consecutive weeks, suggest scheduling go-live review.

## Scheduling

```
cron: "17 11 * * 0"   # Sunday 11:17 local time
prompt: "/go-live-scorecard"
```

## Output Files

```
results/readiness/{date}/scorecard.md
results/readiness/history.csv
```

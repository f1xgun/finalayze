# Autonomous Audit

Unattended end-of-day / end-of-week loop: sense recent anomalies, run a multi-agent R&D
diagnosis, then **auto-fix + merge only the safe class** (docs / tests / lockfile) on green CI,
and **escalate everything else to a human** via Telegram. Invoked headless from cron
(see `docs/operations/autonomous_audit.md`).

## When to Use

- Fired by cron via `scripts/run_autonomous_audit.sh` (weekdays after market close = light `daily`
  pass; weekend = full `weekly` sweep).
- When the operator says "run the autonomous audit" / "do the nightly R&D".

## Non-negotiable safety rules (read first)

1. **Real-money LIVE execution is a HARD STOP.** Never place a live order, never flip a broker to
   live/real, never set `FINALAYZE_REAL_CONFIRMED`. Diagnosis + sandbox only.
2. **Default-risky.** A fix may auto-merge ONLY if `scripts/audit_triage.py` classifies its REAL
   diff as `safe`. Anything touching `src/finalayze/` (strategy/risk/ML/execution/core money),
   `config/`, `alembic/`, `scripts/`, `.github/`, `docker/`, or `pyproject.toml` is **risky** →
   PR + escalate, never auto-merge.
3. **CLAUDE.md invariants still apply** (TDD; imports flow 0→6; MOEX=Tinkoff-only;
   `ruff`/`mypy` green). A risky fix that would change strategy/risk/backtest/ML economics needs a
   `backtest-iteration` cert — that is a human decision, so it is always escalated.
4. **Bounded.** Process at most the top 5 confirmed findings per run. Log anything dropped.
5. One run at a time (the cron wrapper holds an flock); never force-push or touch `main` directly.

## Instructions

### Step 1 — Sense + Diagnose (R&D agent team)

Run the diagnosis workflow (it senses logs/metrics, audits across dimensions, and adversarially
verifies every finding). Pass `mode` from the cron invocation (`daily` or `weekly`):

> Use the **Workflow** tool with `{ name: "nightly-audit", args: { mode: "<daily|weekly>" } }`.

It returns `{ mode, ops_summary, ops_findings, confirmed: [{severity, title, file, line,
evidence, recommendation, risk_hint}] }`. If `confirmed` is empty and there are no ops_findings,
write a brief "all clear" report (Step 5) and stop.

### Step 2 — Triage (decide)

Sort confirmed findings by severity. Take the top 5. For each, the `risk_hint` is only a hint —
the authoritative decision is made on the REAL diff in Step 3.

### Step 3 — Act per finding

For each finding, work on a fresh branch off `origin/main` in an isolated worktree
(`fix/auto-<short-slug>`), newest `origin/main` as base:

1. Implement the smallest correct fix. **Write a failing test first** (TDD).
2. Run gates locally: `uv run ruff check . && uv run ruff format --check . && uv run mypy src/finalayze/`
   plus the affected test modules. If they don't go green, do NOT open a PR — record the finding as
   "needs human" and move on.
3. Classify the REAL change:
   `uv run python scripts/audit_triage.py $(git diff --name-only origin/main...HEAD)`
   - **exit 0 (SAFE):** push, open a PR, and merge it (`gh pr merge --squash`) once CI is green.
   - **exit 2 (RISKY):** push, open a PR (do NOT merge), and add `needs-human-review`. Escalate in
     Step 4. This is the path for every strategy/risk/ML/money change.
4. Never merge a risky PR. Never bypass CI. If `gh pr merge --auto` is rejected, leave it for the
   human rather than admin-merging risky changes.

### Step 4 — Escalate risky items

For each risky/needs-human finding, send a Telegram alert (reuse the existing alerter):

```bash
uv run python scripts/notify_telegram.py --priority high \
  --title "Autonomous audit: human review needed" \
  --body "<severity> <title> — PR <url>. <one-line why it's risky>."
```

(If `scripts/notify_telegram.py` is absent, fall back to writing the escalation into the report and
log that Telegram was unavailable — never silently drop it.)

### Step 5 — Report

Write `docs/audit/auto/<YYYY-MM-DD>-<daily|weekly>.md`: ops summary, every confirmed finding with
its verdict (auto-merged / escalated / needs-human), PR links, and what was dropped by the top-5
cap. Then send a one-paragraph Telegram digest (counts: auto-merged, escalated, all-clear).

## Outputs

- 0+ auto-merged safe PRs (docs/tests/lockfile only).
- 0+ open risky PRs labelled `needs-human-review`, each with a Telegram escalation.
- A dated report under `docs/audit/auto/`.
- A Telegram digest.

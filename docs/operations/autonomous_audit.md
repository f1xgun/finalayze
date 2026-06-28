# Autonomous Audit Loop

Cron-driven, unattended end-of-day / end-of-week loop that **senses** recent anomalies,
runs a **multi-agent R&D diagnosis**, **auto-merges only the safe class** (docs / tests /
lockfile) on green CI, and **escalates everything else to a human** via Telegram.

## The loop: Sense → Diagnose → Decide → Act → Report

| Stage | What | Built from |
|---|---|---|
| Sense | recent `docker logs`, `/health`, `cycles.jsonl` vs `config/pipelines.yaml` thresholds | `nightly-audit.js` Sense phase (`live-monitor-agent`) |
| Diagnose | multi-dimension code audit, **every finding adversarially verified** | `.claude/workflows/nightly-audit.js` |
| Decide | per-fix risk class, **default-risky** | `scripts/audit_triage.py` |
| Act | safe → PR + auto-merge on green CI; risky → PR + escalate (never merge) | `.claude/skills/autonomous-audit.md` |
| Report | dated report + Telegram digest | skill Step 5 + `scripts/notify_telegram.py` |

## Safety model (why this is safe to run unattended)

- **Real-money LIVE execution is a hard stop** — the skill never trades live / never flips a broker.
- **Default-risky merge gate** — `scripts/audit_triage.py` auto-merges a change ONLY when *every*
  changed path is `docs/`, `tests/`, `*.md`, or `uv.lock`. Anything under `src/finalayze/`,
  `config/`, `alembic/`, `scripts/`, `.github/`, `docker/`, or `pyproject.toml` is **risky** → PR +
  Telegram escalation, never auto-merged. Unknown paths are risky too.
- **CI is still the gate** — safe PRs merge only on green CI; risky PRs are never admin-merged.
- **Bounded** — ≤ 5 findings per run; drops are logged.
- The permission flag (`--dangerously-skip-permissions`) only exists because an unattended no-TTY
  session can't answer prompts. The *real* guardrails are the four rules above, not the flag.

## Install (cron)

The runner is `scripts/run_autonomous_audit.sh [daily|weekly]`. Pick off-minutes (not :00/:30).
Times are the **host's local timezone**.

```cron
# Autonomous audit — weekdays after MOEX close (light daily pass)
13 23 * * 1-5  /Users/f1xgun/finalayze/scripts/run_autonomous_audit.sh daily   >> /Users/f1xgun/finalayze/results/auto-audit/cron.log 2>&1
# Autonomous audit — Saturday morning (full weekly sweep)
27 10 * * 6    /Users/f1xgun/finalayze/scripts/run_autonomous_audit.sh weekly  >> /Users/f1xgun/finalayze/results/auto-audit/cron.log 2>&1
```

Install with `crontab -e` (paste the two lines). Verify with `crontab -l`.

### Prerequisites

1. **Auth (headless):** `.env` must contain `CLAUDE_CODE_OAUTH_TOKEN` (already used by the Docker
   stack). The runner sources `.env`.
2. **Tools on PATH:** the runner prepends common locations; override with `AUTONOMOUS_AUDIT_PATH`
   if `claude` / `uv` / `gh` / `docker` live elsewhere. `gh` must be authenticated (`gh auth status`).
3. **Telegram (optional but recommended):** `FINALAYZE_TELEGRAM_BOT_TOKEN` +
   `FINALAYZE_TELEGRAM_CHAT_ID` in `.env` enable escalation; without them escalations land only in
   the report.
4. **Host must be awake** at the scheduled time. A sleeping laptop won't fire cron — run on an
   always-on host/server, or `caffeinate` the Mac, or use a **GitHub Actions `schedule:`** workflow
   that invokes the same skill (the repo is public, so Actions minutes are unlimited).

## Test it before trusting cron

```bash
# Dry sense+diagnose only (no fixes): run the workflow directly in an interactive session
#   Workflow { name: "nightly-audit", args: { mode: "daily" } }

# Full loop once, manually (will open/auto-merge safe PRs + escalate risky):
scripts/run_autonomous_audit.sh daily
tail -f results/auto-audit/*_daily.log
```

Start by reviewing a few `daily` runs' reports under `docs/audit/auto/` before relying on the
unattended auto-merge.

## Files

- `.claude/workflows/nightly-audit.js` — the R&D diagnosis workflow (agent team + verify).
- `.claude/skills/autonomous-audit.md` — the orchestration playbook (the brain).
- `scripts/audit_triage.py` — the default-risky safe/risky classifier (the safety core).
- `scripts/run_autonomous_audit.sh` — cron entrypoint (lock + headless `claude` + report).
- `scripts/notify_telegram.py` — escalation/digest notifier.

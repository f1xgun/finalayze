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

### Billing — subscription vs API (read this)

`claude -p` is a known billing footgun. Which account pays depends on the credential:

- **`CLAUDE_CODE_OAUTH_TOKEN`** (from `claude setup-token`) → draws on your **Max/Pro
  subscription** quota. This is what `.env` + the runner use.
- **`ANTHROPIC_API_KEY`** present in the environment → `claude -p` **silently** bills
  **pay-per-token API**, bypassing the subscription (claude-code #37686: one Max user hit
  **$1,800 in two days**; #43333: some versions route `-p` through API even with OAuth).

The runner defends against this: it `unset`s `ANTHROPIC_API_KEY`/`ANTHROPIC_AUTH_TOKEN` and
**refuses to run without `CLAUDE_CODE_OAUTH_TOKEN`**. After the **first** run, confirm usage
landed on the **claude.ai subscription** dashboard, *not* platform.claude.com (API). Also note:
on the subscription, a full weekly multi-agent audit (~dozens of agents, millions of tokens) eats
your interactive Max rate limits — consider a cheaper model for the audit agents, a smaller daily
fan-out, or deliberately using an API key with a **Console spend limit** if you want isolated,
predictable pipeline billing instead.

### Prerequisites

1. **Auth (headless):** `.env` must contain `CLAUDE_CODE_OAUTH_TOKEN` (already used by the Docker
   stack). The runner sources `.env`, drops any `ANTHROPIC_API_KEY`, and fails closed if the OAuth
   token is missing (so it can never silently fall back to API billing).
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

## GitHub Actions (recommended: subscription-only, always-on)

`.github/workflows/autonomous-audit.yml` runs the same skill in the cloud on a schedule —
**reliable timing** (no sleeping laptop) and **subscription billing** (`CLAUDE_CODE_OAUTH_TOKEN`
only; `ANTHROPIC_API_KEY` is never set, so `claude -p` can't fall back to pay-per-token API).

**Cloud limitation:** the runner cannot reach your local Docker stack, so **live log/metric
sensing is unavailable** there — the Actions run does code audit + gates + fix PRs. Use the host
cron (above) if you also want live-ops sensing.

### One-time setup

```bash
# 1) Generate a subscription OAuth token (uses your Max/Pro plan, NOT API billing):
claude setup-token            # prints a token

# 2) Add it as a repo secret (this is what makes Actions bill the subscription):
gh secret set CLAUDE_CODE_OAUTH_TOKEN     # paste the token

# 3) (Recommended) a PAT so bot PRs trigger CI + can auto-merge. A GITHUB_TOKEN-created
#    PR does NOT trigger other workflows, so without this the "merge on green CI" step
#    can't see CI. Scope: repo (contents + pull_requests).
gh secret set AUDIT_GH_PAT                 # paste a fine-grained PAT

# 4) (Optional) Telegram escalation:
gh secret set FINALAYZE_TELEGRAM_BOT_TOKEN
gh secret set FINALAYZE_TELEGRAM_CHAT_ID
```

Do **NOT** add `ANTHROPIC_API_KEY` as a secret/env — its mere presence flips `claude -p` to API
billing.

### Schedule + manual run

Cron in Actions is **UTC** (unlike the host crontab). The workflow ships with weekdays 20:13 UTC
(≈23:13 MSK) `daily` and Saturday 07:27 UTC (≈10:27 MSK) `weekly`. Trigger on demand to test:

```bash
gh workflow run "Autonomous Audit" -f mode=daily
gh run watch
```

After the first run, **confirm billing landed on the claude.ai subscription dashboard, not
platform.claude.com** (#43333 caveat).

## Files

- `.github/workflows/autonomous-audit.yml` — scheduled cloud run (subscription-only).
- `.claude/workflows/nightly-audit.js` — the R&D diagnosis workflow (agent team + verify).
- `.claude/skills/autonomous-audit.md` — the orchestration playbook (the brain).
- `scripts/audit_triage.py` — the default-risky safe/risky classifier (the safety core).
- `scripts/run_autonomous_audit.sh` — cron entrypoint (lock + headless `claude` + report).
- `scripts/notify_telegram.py` — escalation/digest notifier.

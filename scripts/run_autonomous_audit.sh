#!/usr/bin/env bash
# Cron entrypoint for the autonomous audit loop.
#
#   scripts/run_autonomous_audit.sh [daily|weekly]
#
# Launches a HEADLESS Claude Code session that runs the `autonomous-audit` skill:
# sense -> diagnose (nightly-audit workflow) -> auto-merge the safe class on green
# CI -> escalate everything else to Telegram. See docs/operations/autonomous_audit.md.
#
# Safety lives in the skill + scripts/audit_triage.py (default-risky) + CI, NOT in
# the permission flag: an unattended (no-TTY) session cannot answer prompts, so it
# runs with auto-approved tools. Never auto-merges risky changes; never trades live.
set -euo pipefail

MODE="${1:-daily}"
if [[ "$MODE" != "daily" && "$MODE" != "weekly" ]]; then
  echo "usage: $0 [daily|weekly]" >&2
  exit 64
fi

# Repo root = parent of this script's dir.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Cron has a minimal PATH; prepend the usual tool locations + allow override.
export PATH="${AUTONOMOUS_AUDIT_PATH:-/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:/usr/bin:/bin}:$PATH"

# Load .env so CLAUDE_CODE_OAUTH_TOKEN + FINALAYZE_TELEGRAM_* are available headless.
if [[ -f .env ]]; then set -a; . ./.env; set +a; fi

# ── Billing safety (claude-code issues #37686, #43333) ───────────────────────
# `claude -p` bills to the pay-per-token API account whenever ANTHROPIC_API_KEY is
# set in the environment -- silently bypassing the Max/Pro subscription (one user
# hit $1,800 in two days). Force subscription (OAuth) billing: drop any inherited
# API key and REQUIRE the OAuth token. (Verify the first run lands on the
# claude.ai subscription dashboard, not platform.claude.com -- #43333 reports some
# versions still route -p through API even with OAuth.)
unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN
if [[ -z "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]]; then
  echo "CLAUDE_CODE_OAUTH_TOKEN not set -- refusing to run (would risk pay-per-token API billing)" >&2
  exit 78
fi

LOG_DIR="results/auto-audit"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y-%m-%d_%H%M)"
LOG_FILE="$LOG_DIR/${STAMP}_${MODE}.log"

# Portable single-instance lock (mkdir is atomic on macOS + Linux; flock is Linux-only).
LOCK_DIR="$LOG_DIR/.lock"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "another autonomous-audit run holds the lock ($LOCK_DIR); exiting" | tee -a "$LOG_FILE"
  exit 0
fi
cleanup() { rmdir "$LOCK_DIR" 2>/dev/null || true; }
trap cleanup EXIT

notify_fail() {
  python3 scripts/notify_telegram.py --priority high \
    --title "Autonomous audit FAILED ($MODE)" \
    --body "Runner errored; see $LOG_FILE on the host." >/dev/null 2>&1 || true
}
trap 'notify_fail' ERR

PROMPT="Run the autonomous-audit skill with mode=${MODE}. Follow its safety rules exactly: \
diagnose via the nightly-audit workflow, auto-merge ONLY the safe class (docs/tests/uv.lock) on \
green CI, open PRs + Telegram-escalate everything risky, never trade live, write the dated report."

echo "[$(date -u +%FT%TZ)] autonomous-audit mode=$MODE starting" | tee -a "$LOG_FILE"

# Headless, non-interactive. --dangerously-skip-permissions is required for an unattended
# (no-TTY) session; the audit policy + audit_triage + CI are the actual guardrails.
claude -p "$PROMPT" \
  --dangerously-skip-permissions \
  >>"$LOG_FILE" 2>&1

echo "[$(date -u +%FT%TZ)] autonomous-audit mode=$MODE finished (exit 0)" | tee -a "$LOG_FILE"

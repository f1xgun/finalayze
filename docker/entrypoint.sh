#!/usr/bin/env bash
set -euo pipefail

# Mark Claude Code onboarding as complete so headless -p runs don't hang.
if [ ! -f "${HOME}/.claude.json" ]; then
    echo '{"hasCompletedOnboarding":true}' > "${HOME}/.claude.json"
fi

# Bootstrap a local git repo so meta-agent fix-spawn can create worktrees.
# The container has no .git (excluded from image), so we init one from the
# current source snapshot. Worktree changes stay local unless the operator
# explicitly copies them out.
if [ ! -d .git ]; then
    git init -q
    git config user.email "meta-agent@finalayze.local"
    git config user.name "Meta Agent"
    git add -A
    git commit -q -m "init: container snapshot for meta-agent fix worktrees"
fi

echo "Running database migrations..."
uv run alembic -c alembic/alembic.ini upgrade head

# Force single worker in sandbox mode (TradingLoop thread safety)
if [ "${FINALAYZE_MODE:-}" = "sandbox" ]; then
    UVICORN_WORKERS=1
    echo "Sandbox mode: forcing single uvicorn worker"
fi

echo "Starting Finalayze API server..."
exec uv run uvicorn finalayze.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers "${UVICORN_WORKERS:-2}"

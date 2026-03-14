#!/usr/bin/env bash
set -euo pipefail

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

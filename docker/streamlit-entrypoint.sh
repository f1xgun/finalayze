#!/usr/bin/env bash
set -euo pipefail

mkdir -p /home/finalayze/.streamlit

# Fail-closed: no weak 'admin' default (audit 2026-06-28). When DASHBOARD_PASSWORD
# is unset the password is empty and the dashboard refuses login ("Password not
# configured") until the operator sets a real one -- never a guessable default.
cat > /home/finalayze/.streamlit/secrets.toml <<TOML
password = "${DASHBOARD_PASSWORD:-}"
api_key = "${FINALAYZE_API_KEY:-}"
api_url = "${DASHBOARD_API_URL:-http://app:8000}"
TOML

exec uv run streamlit run src/finalayze/dashboard/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0

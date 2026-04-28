#!/usr/bin/env bash
set -euo pipefail

mkdir -p /home/finalayze/.streamlit

cat > /home/finalayze/.streamlit/secrets.toml <<TOML
password = "${DASHBOARD_PASSWORD:-admin}"
api_key = "${FINALAYZE_API_KEY:-}"
api_url = "${DASHBOARD_API_URL:-http://app:8000}"
TOML

exec uv run streamlit run src/finalayze/dashboard/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0

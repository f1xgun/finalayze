#!/usr/bin/env bash
# SessionStart hook: rebuild ast-index if stale
# Skips rebuild if index was updated within the last hour

set -euo pipefail

if ! command -v ast-index &>/dev/null; then
  echo "ast-index not installed. Run: brew tap defendend/ast-index && brew install ast-index" >&2
  exit 0
fi

# Check if index exists and was rebuilt recently (within 1 hour)
DB_PATH=$(ast-index db-path 2>/dev/null || echo "")
if [[ -n "$DB_PATH" && -f "$DB_PATH" ]]; then
  if [[ "$(uname)" == "Darwin" ]]; then
    DB_MTIME=$(stat -f %m "$DB_PATH" 2>/dev/null || echo 0)
  else
    DB_MTIME=$(stat -c %Y "$DB_PATH" 2>/dev/null || echo 0)
  fi
  NOW=$(date +%s)
  AGE=$(( NOW - DB_MTIME ))

  if (( AGE < 3600 )); then
    echo "ast-index: index is fresh (${AGE}s old), skipping rebuild"
    exit 0
  fi
fi

# Check if there are changes since last index
CHANGES=$(git status --porcelain -- '*.py' 2>/dev/null | head -5)
if [[ -z "$CHANGES" && -n "$DB_PATH" && -f "$DB_PATH" ]]; then
  echo "ast-index: no Python changes detected, running incremental update"
  ast-index update 2>/dev/null || ast-index rebuild 2>/dev/null
else
  echo "ast-index: rebuilding index..."
  ast-index rebuild 2>/dev/null
fi

STATS=$(ast-index stats 2>/dev/null | head -5)
echo "ast-index: $STATS"

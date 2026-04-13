---
phase: 40-moex-data-adapter-macro-features
plan: "01"
subsystem: scripts/ml-research
tags: [moex, tinkoff, ml-research, data-adapter, tdd]
dependency_graph:
  requires: []
  provides: [moex-data-loading-in-auto-ml-research]
  affects: [scripts/auto_ml_research.py]
tech_stack:
  added: []
  patterns: [TinkoffFetcher sync bridge, DEFAULT_SEGMENTS single source of truth]
key_files:
  created:
    - tests/unit/test_auto_ml_research_moex.py
  modified:
    - scripts/auto_ml_research.py
decisions:
  - "Dynamic _SEGMENT_SYMBOLS population from DEFAULT_SEGMENTS at module import (not hardcoded), so new segments automatically appear in CLI choices"
  - "Bond segments excluded by instrument_type=='stock' filter, not by name pattern"
  - "Lazy import of TinkoffFetcher/build_default_registry inside functions to avoid heavy gRPC init at module load"
  - "sandbox=False mandatory — sandbox endpoint has no historical candles"
  - "VIX explicitly set to None for MOEX segments (US-only index)"
metrics:
  duration_minutes: 12
  completed_date: "2026-04-13"
  tasks_completed: 1
  tasks_total: 1
  files_changed: 2
---

# Phase 40 Plan 01: MOEX Data Adapter for auto_ml_research Summary

**One-liner:** Wired TinkoffFetcher into auto_ml_research.py for all ru_* equity segments with dynamic symbol loading from DEFAULT_SEGMENTS, ATR uplift, and missing-token graceful skip.

## What Was Built

`scripts/auto_ml_research.py` now supports all MOEX equity segments as CLI choices. The script branches on `_is_moex_segment()` in `_prepare_data()` — MOEX segments fetch via `_fetch_moex_candles()` (TinkoffFetcher, `sandbox=False`), benchmark via `_fetch_moex_benchmark()` (IMOEX), and VIX is skipped. US segments are unchanged.

Symbols come from `config/segments.py DEFAULT_SEGMENTS` at import time, filtered to `instrument_type == "stock"` and `segment_id.startswith("ru_")`. Bond segments (`ru_ofz_pd`, `ru_ofz_pk`) are automatically excluded.

ATR barriers are widened 1.2x for MOEX in `build_full_dataset()` to account for higher volatility.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Add MOEX segment symbols and data loading to auto_ml_research.py | 1556e45 | scripts/auto_ml_research.py, tests/unit/test_auto_ml_research_moex.py |

## Decisions Made

1. **Dynamic symbol loading from DEFAULT_SEGMENTS** — Populating `_SEGMENT_SYMBOLS` at module import from `DEFAULT_SEGMENTS` means new MOEX segments added to `config/segments.py` automatically appear as CLI choices without touching `auto_ml_research.py`.

2. **Bond exclusion via `instrument_type` filter** — More robust than name-based filtering; correctly excludes all bond segments regardless of naming conventions.

3. **Lazy TinkoffFetcher import** — TinkoffFetcher is imported inside the fetch functions, not at module top-level. This avoids gRPC initialization when the script is running US segments or imported in tests.

4. **`sandbox=False`** — Required for historical candle data; sandbox endpoint returns no data.

## Deviations from Plan

### Auto-extended scope

**[Rule 2 - Enhancement] Additional MOEX stock segments included in CLI**
- **Found during:** Task 1, --help verification
- **Issue:** `DEFAULT_SEGMENTS` contains more than the 4 ru_* equity segments specified in the plan (also ru_metals, ru_consumer, ru_telecom, ru_utilities, ru_construction, ru_chemicals, ru_transport). The dynamic loading approach includes all of them.
- **Decision:** Kept all — the plan says "symbols sourced from config/segments.py DEFAULT_SEGMENTS, not hardcoded". Including all stock segments from the single source of truth is strictly correct behavior. Tests verify the 4 required segments are present; additional segments cause no harm.
- **Files modified:** None (behavior is inherent to the dynamic loading approach)

## Threat Surface Scan

No new network endpoints, auth paths, or schema changes introduced. The `FINALAYZE_TINKOFF_TOKEN` is read from environment only, never logged or printed (T-40-01 mitigation in place). TinkoffFetcher is lazily imported only when the token exists.

## Self-Check: PASSED

- scripts/auto_ml_research.py exists and contains all required symbols
- tests/unit/test_auto_ml_research_moex.py exists with 21 tests
- Commit 1556e45 exists in git log
- `uv run pytest tests/unit/test_auto_ml_research_moex.py --no-cov` exits 0 (21 passed)
- `uv run ruff check scripts/auto_ml_research.py` exits 0
- `uv run python scripts/auto_ml_research.py --help` shows ru_blue_chips, ru_energy, ru_tech, ru_finance in choices

---
phase: 38-presetapplicator-auto-apply-loop
plan: "02"
subsystem: orchestration, strategies
tags: [position-tracking, combiner-hook, forward-compatibility, tdd]
dependency_graph:
  requires: [38-01]
  provides: [_entry_strategy lifecycle, get_entry_strategies(), invalidate_segment_cache()]
  affects: [trading_loop.py, combiner.py]
tech_stack:
  added: []
  patterns: [parallel-dict-tracking, no-op-forward-compat-hook]
key_files:
  created:
    - tests/unit/test_combiner.py
  modified:
    - src/finalayze/orchestration/trading_loop.py
    - tests/unit/core/test_trading_loop.py
decisions:
  - "_entry_strategy is set unconditionally on BUY fill (not gated on candles), since strategy ownership tracking applies regardless of whether candles are provided for stop-loss computation"
  - "invalidate_segment_cache() placed after _load_config() as a natural extension point — same method scope"
  - "test_combiner.py created as a new focused file matching plan's acceptance criteria (not appended to test_strategy_combiner.py)"
metrics:
  duration_minutes: 15
  completed_date: "2026-04-12"
  tasks_completed: 2
  files_modified: 3
  files_created: 1
---

# Phase 38 Plan 02: _entry_strategy + invalidate_segment_cache Summary

**One-liner:** Position-ownership dict (`_entry_strategy`) wired into TradingLoop BUY/SELL/stop-loss paths, plus `invalidate_segment_cache()` forward-compat hook added to StrategyCombiner.

## Objective

Add `_entry_strategy: dict[str, str]` tracking to TradingLoop and `invalidate_segment_cache()` hook to StrategyCombiner, enabling Plan 38-01's PresetApplicator to check open positions before disabling a strategy, and to signal cache invalidation after YAML writes.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Add _entry_strategy dict to TradingLoop | 9622f11 | trading_loop.py, test_trading_loop.py |
| 2 | Add invalidate_segment_cache() to StrategyCombiner | 8ef1b34 | combiner.py, test_combiner.py |

## Decisions Made

1. `_entry_strategy[symbol] = strategy_name` is set immediately on BUY fill (not gated on `candles and fill_price` guard) — strategy ownership is unconditional on a filled BUY, regardless of whether stop-loss candles are available.

2. `invalidate_segment_cache()` is placed after `_load_config()` as the natural extension point. Currently a no-op with a debug log; any future caching addition only needs this method updated.

3. `tests/unit/test_combiner.py` created as a standalone file rather than appending to `test_strategy_combiner.py`, matching the plan's acceptance criteria grep targets and keeping concerns separated.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] _persist_to_db called in BUY fill path**
- **Found during:** Task 1 GREEN phase
- **Issue:** `_submit_order` calls `_persist_to_db(self._persist_order_async(...))` which tries a real asyncpg DB connection in tests, causing `InvalidPasswordError`
- **Fix:** Added `loop._persist_to_db = MagicMock()` in the BUY fill and SELL fill test cases to prevent DB connection attempts
- **Files modified:** tests/unit/core/test_trading_loop.py

**2. [Rule 1 - Bug] _entry_strategy wrongly placed inside candles guard**
- **Found during:** Task 1 GREEN phase
- **Issue:** Initial placement of `_entry_strategy[symbol] = strategy_name` inside `if order.side == "BUY" and candles and result.fill_price is not None:` meant no tracking when `candles=None`
- **Fix:** Moved to a separate `if order.side == "BUY":` guard (unconditional on candles)
- **Files modified:** src/finalayze/orchestration/trading_loop.py

## Known Stubs

None.

## Threat Flags

None — no new network endpoints, auth paths, or external trust boundaries introduced.

## Test Results

- `uv run pytest tests/unit/core/test_trading_loop.py -k entry_strategy -x -v` — 6 passed
- `uv run pytest tests/unit/test_combiner.py -k invalidate -x -v` — 3 passed
- `uv run pytest tests/unit/core/test_trading_loop.py tests/unit/test_combiner.py -x` — 39 passed

## Self-Check: PASSED

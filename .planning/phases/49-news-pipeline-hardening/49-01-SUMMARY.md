---
phase: 49-news-pipeline-hardening
plan: 01
subsystem: analysis/news-pipeline
tags: [bugfix, llm, timeout, budget-cap, pydantic]
dependency_graph:
  requires: []
  provides: [parse_structured, budget_cap, per_article_timeout]
  affects: [news_analyzer, event_classifier, llm_client, trading_loop, metrics]
tech_stack:
  added: []
  patterns: [parse_structured, asyncio.wait_for, pydantic-validated-llm-responses]
key_files:
  created:
    - tests/unit/test_news_pipeline.py
  modified:
    - src/finalayze/analysis/llm_client.py
    - src/finalayze/analysis/news_analyzer.py
    - src/finalayze/analysis/event_classifier.py
    - src/finalayze/core/trading_loop.py
    - src/finalayze/api/metrics.py
    - tests/unit/test_news_analyzer.py
    - tests/unit/test_event_classifier.py
decisions:
  - "Added parse_structured() as concrete method on LLMClient ABC (not abstract) with default complete()+model_validate_json() implementation"
  - "Used builtin TimeoutError instead of asyncio.TimeoutError per ruff UP041"
  - "Created EventClassifierResult Pydantic model for structured event classification responses"
  - "threading.Lock was already correctly used only in sync code paths -- no fix needed"
metrics:
  duration: 488s
  completed: "2026-04-14T20:21:04Z"
  tasks_completed: 2
  tasks_total: 2
  files_changed: 7
---

# Phase 49 Plan 01: News Pipeline Hardening Summary

Replaced json.loads parsing with Pydantic-validated parse_structured() in all LLM callers, added 5s per-article timeout via asyncio.wait_for, and enforced 20-article budget cap per news cycle with Prometheus metric.

## What Changed

### Task 1: Migrate json.loads to parse_structured + add per-article 5s timeout
- **Commit:** b2d0472
- Added `parse_structured()` method to `LLMClient` ABC base class (concrete, not abstract) that calls `complete()` + `model_validate_json()`
- Replaced `json.loads` parsing in `NewsAnalyzer.analyze()` with `parse_structured(SentimentResult)` wrapped in `asyncio.wait_for(timeout=5.0)`
- Replaced `json.loads` parsing in `EventClassifier.classify()` with `parse_structured(EventClassifierResult)` wrapped in `asyncio.wait_for(timeout=5.0)`
- Created `EventClassifierResult` Pydantic model with `event_types: list[str]` field
- Removed `_parse_response()` method from EventClassifier (logic moved to `_resolve_event_type()`)
- Both analyzers return neutral fallback on timeout or parse error
- Updated all tests to use `parse_structured` mocks instead of `complete` + JSON

### Task 2: Add article budget cap + verify threading safety
- **Commit:** 408d418
- Added `_MAX_ARTICLES_PER_CYCLE = 20` constant to `trading_loop.py`
- Added budget cap enforcement in `_news_cycle()` -- truncates articles and logs `news_budget_cap_hit`
- Added `news_budget_cap_total` Counter and `inc_news_budget_cap_hit()` to `MetricsCollector`
- Created `tests/unit/test_news_pipeline.py` with 4 tests: budget cap limit, metric increment, no-cap-under-limit, async lock safety
- AST-verified that `_sentiment_lock` is never referenced in any async method

## Deviations from Plan

### Adapted to Actual Codebase Structure

**1. [Rule 3 - Blocking] Files referenced in plan do not exist**
- **Found during:** Task 1
- **Issue:** Plan references `news_impact_analyzer.py` and `combined_analyzer.py` which do not exist. The actual codebase has `event_classifier.py` (LLM-based event classification) and `impact_estimator.py` (rule-based, no LLM).
- **Fix:** Applied the same parse_structured migration to `event_classifier.py` instead. Skipped `impact_estimator.py` (no LLM calls, no json.loads).
- **Files modified:** `src/finalayze/analysis/event_classifier.py`, `tests/unit/test_event_classifier.py`

**2. [Rule 3 - Blocking] Trading loop at different path**
- **Found during:** Task 2
- **Issue:** Plan references `src/finalayze/orchestration/trading_loop.py` but actual file is at `src/finalayze/core/trading_loop.py`.
- **Fix:** Applied changes to correct path.

**3. [Rule 1 - Bug] threading.Lock was already correct**
- **Found during:** Task 2
- **Issue:** Plan claims `_sentiment_lock` is acquired across async boundary. Actual code correctly uses the lock only in sync methods (`_process_news_article`, `_get_sentiment`). No `_analyze_impact_batch` async method exists.
- **Fix:** No code change needed. Added AST-based test to verify this invariant holds.

**4. [Rule 3 - Blocking] LLMClient missing parse_structured**
- **Found during:** Task 1
- **Issue:** Plan assumes `parse_structured` already exists on `LLMClient` ABC. It does not.
- **Fix:** Added `parse_structured()` as a concrete method on `LLMClient` base class with default implementation (complete + model_validate_json).
- **Files modified:** `src/finalayze/analysis/llm_client.py`

**5. [Rule 3 - Blocking] No _batch_timeout to reduce**
- **Found during:** Task 2
- **Issue:** Plan says to reduce `_batch_timeout` from 1800 to 120. No such variable exists in current code. `_run_async` has a 30s timeout which is appropriate for single-article processing.
- **Fix:** No change needed. Per-article timeout (5s) plus budget cap (20) ensures cycle completes in ~100s max.

## Decisions Made

1. `parse_structured()` added as concrete method (not abstract) on `LLMClient` -- subclasses can override for provider-specific structured output APIs
2. Used `TimeoutError` (builtin) instead of `asyncio.TimeoutError` per ruff UP041
3. `max_tokens` parameter on `parse_structured` suppressed with `noqa: ARG002` -- reserved for subclass implementations

## Self-Check: PASSED

All 8 files verified present. Both commits (b2d0472, 408d418) verified in git log. 20 tests pass.

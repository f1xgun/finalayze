---
phase: 49-news-pipeline-hardening
plan: 02
subsystem: analysis/news-pipeline
tags: [credibility, ticker-validation, liveness-monitoring, prometheus]
dependency_graph:
  requires: [parse_structured, budget_cap]
  provides: [source_credibility, ticker_validation, llm_liveness_monitoring]
  affects: [trading_loop, models, metrics]
tech_stack:
  added: []
  patterns: [source-credibility-map, instrument-registry-validation, cycle-level-liveness-tracking]
key_files:
  created:
    - alembic/versions/004_add_credibility_to_sentiment_scores.py
  modified:
    - src/finalayze/core/trading_loop.py
    - src/finalayze/core/models.py
    - src/finalayze/api/metrics.py
    - tests/unit/test_news_pipeline.py
decisions:
  - "Alembic migration is 004 (not 006) because last existing migration was 003"
  - "No _persist_sentiment_batch_async exists -- credibility attached to articles via model_copy in _news_cycle"
  - "LLM liveness tracking operates at cycle level (all articles fail = 1 failure), not per-article"
  - "Re-alerting on sustained failure: alert fires every time threshold is hit (at count 3, 6, 9, etc.)"
metrics:
  duration: 358s
  completed: "2026-04-14T20:30:07Z"
  tasks_completed: 2
  tasks_total: 2
  files_changed: 5
---

# Phase 49 Plan 02: News Pipeline Hardening -- Credibility, Validation, Liveness Summary

Source credibility map (RSS=0.8, Telegram=0.7, default=0.5) with model_copy wiring in _news_cycle, ticker validation via InstrumentRegistry with structured warning logs, and LLM liveness monitoring (3 consecutive all-fail cycles triggers Telegram alert + Prometheus counter).

## What Changed

### Task 1: Source credibility map + ticker validation + credibility column
- **Commit:** 643843f
- Added `SOURCE_CREDIBILITY` dict and `get_credibility()` function to `trading_loop.py`
- Added `validate_tickers()` function that filters tickers against `InstrumentRegistry.get()` with `entity_not_in_registry` structured log for rejected tickers
- In `_news_cycle()`, articles get credibility via `model_copy(update={"credibility_score": get_credibility(art.source)})` (NewsArticle is frozen)
- Added `credibility: Mapped[Decimal | None]` column to `SentimentScoreModel`
- Created Alembic migration 004 (add_column "credibility" to "sentiment_scores")
- 9 new tests: credibility map (RSS, Telegram, unknown, case-insensitive), ticker validation (filters, logs, empty), credibility wiring in news cycle, model column existence

### Task 2: LLM liveness monitoring with Telegram alert + Prometheus counter
- **Commit:** 2bb1eb7
- Added `_LLM_FAILURE_THRESHOLD = 3` constant and `_llm_consecutive_failures` instance attribute
- In `_news_cycle()`, track `ok_count`/`fail_count` per article processing; if all fail, increment counter and Prometheus metric; if any succeed, reset counter
- At threshold, fires `self._alerter.on_error("LLMLiveness", ...)` -- re-alerts on sustained failure
- Added `llm_liveness_failures` Prometheus Counter and `MetricsCollector.inc_llm_liveness_failure()` static method
- 5 new tests: under-threshold no alert, threshold alert, reset on success, prometheus counter, re-alert on sustained failure

## Deviations from Plan

### Adapted to Actual Codebase Structure

**1. [Rule 3 - Blocking] _persist_sentiment_batch_async does not exist**
- **Found during:** Task 1
- **Issue:** Plan references `_persist_sentiment_batch_async()` at "line ~2586" for credibility wiring, but the file is only 1021 lines and no such method exists. Sentiment is stored in-memory via `_sentiment_cache`, not persisted to DB in the news cycle.
- **Fix:** Attached credibility to articles via `model_copy` in `_news_cycle()` so it flows through the article processing pipeline. The credibility column on SentimentScoreModel is ready for when DB persistence is wired.
- **Files modified:** `src/finalayze/core/trading_loop.py`

**2. [Rule 3 - Blocking] Alembic migration numbering**
- **Found during:** Task 1
- **Issue:** Plan says to create migration 006 with `down_revision = "005_sandbox_metrics"`. Actual last migration is 003_portfolio_snapshots.
- **Fix:** Created migration 004 with `down_revision = "003"`.
- **Files modified:** `alembic/versions/004_add_credibility_to_sentiment_scores.py`

**3. [Rule 3 - Blocking] news_impact_analyzer.py does not exist**
- **Found during:** Task 1
- **Issue:** Plan references `news_impact_analyzer.py` for `validate_tickers` placement. File does not exist (as documented in 49-01 summary).
- **Fix:** Placed `validate_tickers()` as a module-level function in `trading_loop.py`.
- **Files modified:** `src/finalayze/core/trading_loop.py`

**4. [Rule 3 - Blocking] No _analyze_impact_batch method**
- **Found during:** Task 2
- **Issue:** Plan references `_analyze_impact_batch()` returning `(ok_count, fail_count, ...)` for liveness tracking. No such method exists.
- **Fix:** Added `ok_count`/`fail_count` tracking directly in the `_news_cycle()` article processing loop.
- **Files modified:** `src/finalayze/core/trading_loop.py`

## Decisions Made

1. Alembic migration 004 (not 006) to match actual migration sequence
2. Credibility attached via `model_copy` (frozen Pydantic model) rather than direct attribute mutation
3. LLM liveness tracking at cycle level, not batch level -- simpler and matches actual code structure
4. Re-alerting fires every 3 additional consecutive failures (not just once)

## Self-Check: PASSED

All 5 files verified present. Both commits (643843f, 2bb1eb7) verified in git log. 18 tests pass.

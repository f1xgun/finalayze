---
phase: 49-news-pipeline-hardening
plan: 03
subsystem: analysis/news-pipeline
tags: [gap-closure, ticker-validation, llm-liveness, is_fallback]
dependency_graph:
  requires: [parse_structured, budget_cap, source_credibility, ticker_validation, llm_liveness_monitoring]
  provides: [wired_ticker_validation, soft_failure_liveness, is_fallback_field]
  affects: [trading_loop, schemas, news_analyzer, sentiment_prompts]
tech_stack:
  added: []
  patterns: [soft-failure-detection, llm-ticker-extraction, return-value-liveness-signal]
key_files:
  created: []
  modified:
    - src/finalayze/core/schemas.py
    - src/finalayze/analysis/news_analyzer.py
    - src/finalayze/analysis/prompts/sentiment_en.txt
    - src/finalayze/analysis/prompts/sentiment_ru.txt
    - src/finalayze/core/trading_loop.py
    - tests/unit/test_news_analyzer.py
    - tests/unit/test_news_pipeline.py
decisions:
  - "SentimentResult.tickers and is_fallback fields use defaults ([] and False) for backward compatibility"
  - "_process_news_article returns bool (True=real LLM, False=fallback) as liveness signal to _news_cycle"
  - "llm_ok_count tracks real LLM successes separately from ok_count (which includes fallbacks)"
  - "validate_tickers() called only when sentiment.tickers is non-empty to avoid unnecessary registry lookups"
metrics:
  duration: 302s
  completed: "2026-04-15T06:56:25Z"
  tasks_completed: 2
  tasks_total: 2
  files_changed: 7
---

# Phase 49 Plan 03: Gap Closure -- Ticker Validation Wiring + LLM Liveness Blind Spot Summary

Wired validate_tickers() into _process_news_article() production path via SentimentResult.tickers field and LLM prompt instructions; fixed LLM liveness blind spot by propagating is_fallback signal from analyzers through _process_news_article() return value to _news_cycle() soft-failure tracking.

## What Changed

### Task 1: Add tickers + is_fallback to SentimentResult, wire validate_tickers into production path
- **Commit:** 6bb1e23
- Added `tickers: list[str] = []` and `is_fallback: bool = False` fields to SentimentResult in schemas.py
- Set `is_fallback=True` on `_FALLBACK` in news_analyzer.py so fallback results are identifiable
- Added "tickers" field instruction to both EN and RU sentiment prompts with examples (["SBER", "GAZP"])
- Wired `validate_tickers(sentiment.tickers, self._registry, market_id)` in `_process_news_article()` -- only when tickers is non-empty
- Added 6 new tests: SentimentResult field defaults, fallback is_fallback flag, validate_tickers wiring in production path, empty tickers skip validation

### Task 2: Fix LLM liveness blind spot -- count fallback results as soft failures
- **Commit:** 24819e8
- Changed `_process_news_article()` return type from `None` to `bool` (True = real LLM result, False = fallback used)
- Added `return not sentiment.is_fallback` at end of `_process_news_article()`
- Added `llm_ok_count` tracker in `_news_cycle()` that increments only when `_process_news_article()` returns True
- Changed liveness condition from `ok_count == 0` to `llm_ok_count == 0` -- now detects both hard failures (exceptions) and soft failures (LLM timeouts/parse errors returned as fallbacks)
- Updated existing test helpers to return `True` from mock `_process_news_article` for backward compatibility
- Added 5 new tests: soft failure increments counter, reset on real success, mixed soft+hard failures, mixed cycle resets counter

## Deviations from Plan

None -- plan executed exactly as written. Both gaps identified in 49-VERIFICATION.md are now closed.

## Decisions Made

1. SentimentResult fields use defaults (`tickers=[]`, `is_fallback=False`) for full backward compatibility with existing code and tests
2. `_process_news_article()` returns `bool` as liveness signal -- minimal API change, no new parameters needed
3. `llm_ok_count` tracked separately from `ok_count` to preserve existing exception-based tracking while adding soft-failure detection
4. `validate_tickers()` called conditionally (only when `sentiment.tickers` is non-empty) to avoid unnecessary InstrumentRegistry lookups

## Self-Check: PASSED

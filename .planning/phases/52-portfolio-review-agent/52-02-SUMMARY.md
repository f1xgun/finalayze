---
phase: 52-portfolio-review-agent
plan: 02
subsystem: core
tags: [portfolio-review, trading-loop, apscheduler, fire-and-forget, telegram, graceful-degradation]
dependency_graph:
  requires: [PortfolioReviewResult, build_review_prompt, format_review_telegram, PORTFOLIO_REVIEW_SYSTEM_PROMPT, REVIEW_LLM_TIMEOUT]
  provides: [_portfolio_review_cycle, _run_portfolio_review_async, _gather_portfolio_data]
  affects: [TradingLoop.start]
tech_stack:
  added: []
  patterns: [fire-and-forget-async, apscheduler-cron, graceful-degradation]
key_files:
  created:
    - tests/unit/test_portfolio_review_integration.py
  modified:
    - src/finalayze/core/trading_loop.py
key_decisions:
  - "Fire-and-forget via run_coroutine_threadsafe without .result() -- same pattern as anomaly enrichment"
  - "_gather_portfolio_data is sync and isolated from async handler, preventing _broker_router leaking into advisory path"
  - "Cron at hour=16 UTC (19:00 MSK) fires after MOEX close at 18:40 MSK"
metrics:
  duration_seconds: 225
  completed: "2026-04-15T09:05:58Z"
  tasks_completed: 2
  tasks_total: 2
  tests_added: 18
  files_created: 1
  files_modified: 1
---

# Phase 52 Plan 02: Portfolio Review TradingLoop Wiring Summary

APScheduler cron job at 16:00 UTC dispatches fire-and-forget async portfolio review via LLMClient.parse_structured with PortfolioReviewResult schema, delivering formatted advisory to Telegram with graceful degradation on LLM failure.

## What Was Built

### TradingLoop Methods (Layer 6: core/trading_loop.py)
- **_portfolio_review_cycle()**: Sync APScheduler callback; guards on _llm_client and _async_loop; dispatches via run_coroutine_threadsafe without .result()
- **_run_portfolio_review_async()**: Async fire-and-forget; gathers portfolio data, calls parse_structured with PortfolioReviewResult, formats via format_review_telegram, sends via _alerter._send; wraps in try/except logging portfolio_review_llm_failure
- **_gather_portfolio_data()**: Sync data collector; iterates circuit_breakers keys, calls BrokerRouter.route().get_portfolio()/get_positions() per market; per-market error isolation

### Cron Registration (in start())
- Registered after _daily_reset cron job: hour=16, minute=0 UTC (19:00 MSK, after MOEX close at 18:40 MSK)

### Safety Contract (PFRA-03)
- _run_portfolio_review_async writes ONLY to _alerter._send -- no _broker_router, place_order, generate_signal, or _submit_order references
- _broker_router used only in _gather_portfolio_data for read-only data collection
- Code-grep tests enforce this separation

### Integration Tests (18 tests)
- **TestCronDispatch** (5): dispatch coroutine, no-op when no LLM/None loop/closed loop, skip logging
- **TestLLMCallAndTelegramDelivery** (3): parse_structured with PortfolioReviewResult, Telegram format, system prompt
- **TestGracefulDegradation** (3): timeout/error logs failure, never raises
- **TestGatherPortfolioData** (3): multi-market data, broker error isolation, empty markets
- **TestCronRegistration** (1): hour=16 minute=0 in start() source
- **TestHandlerSafety** (2): no order pipeline references, no _broker_router in async handler
- **TestFireAndForget** (1): no .result() call in code lines

## Task Commits

| Task | Name | Commit | Key Changes |
|------|------|--------|-------------|
| 1 | Wire portfolio review into TradingLoop | fc13017 | 3 methods + cron registration, imports from portfolio_review_agent |
| 2 | Integration tests for portfolio review wiring | ee6a165 | 18 tests covering dispatch, LLM, Telegram, degradation, safety |

## Verification Results

- 47 tests pass (18 new + 29 from Plan 01): `uv run pytest tests/unit/test_portfolio_review_integration.py tests/unit/test_portfolio_review_agent.py -x -v`
- Ruff lint: clean (only pre-existing ARG002 on unrelated _get_event_type_code.ticker param)
- Ruff format: clean
- Handler safety: zero order-pipeline references in _run_portfolio_review_async (code-grep verified)

## Deviations from Plan

None -- plan executed exactly as written.

## Self-Check: PASSED

- [x] src/finalayze/core/trading_loop.py modified (3 new methods + cron + imports)
- [x] tests/unit/test_portfolio_review_integration.py created (419 lines, > 60 minimum)
- [x] Commit fc13017 exists in git log
- [x] Commit ee6a165 exists in git log
- [x] 18 integration tests pass
- [x] 47 total tests pass (Plan 01 + Plan 02)
- [x] ruff check clean (pre-existing ARG002 only)
- [x] ruff format clean

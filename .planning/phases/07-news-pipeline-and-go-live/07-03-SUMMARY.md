---
phase: 07-news-pipeline-and-go-live
plan: 03
subsystem: trading-loop, news, telegram, presets
tags: [rss, telegram, entity-extraction, event-driven, trading-loop, go-live, kill-switch]

# Dependency graph
requires:
  - phase: 07-01
    provides: RssNewsFetcher and EntityExtractor for news pipeline
  - phase: 07-02
    provides: TelegramChannelReader for Russian financial channel ingestion
provides:
  - TradingLoop news cycle wired with RSS + Telegram + entity extraction
  - event_driven strategy enabled on all 4 ru_* MOEX segments at 0.15 weight
  - /stop Telegram command for emergency trading halt
  - Go-live checklist for real money MOEX deployment
  - Backtest validation of preset weight redistribution
affects: [go-live, real-money-deployment]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Independent error handling per news source (RSS/Telegram/NewsAPI)
    - Entity extraction enriches articles before sentiment processing
    - Proportional weight redistribution when adding new strategy

key-files:
  created:
    - tests/unit/test_news_cycle_integration.py
    - tests/unit/test_event_driven_presets.py
    - tests/unit/test_telegram_stop_command.py
    - tests/unit/test_real_mode_guard.py
    - docs/operations/GO_LIVE_CHECKLIST.md
  modified:
    - src/finalayze/core/trading_loop.py
    - src/finalayze/main.py
    - src/finalayze/core/telegram_bot.py
    - src/finalayze/strategies/presets/ru_blue_chips.yaml
    - src/finalayze/strategies/presets/ru_energy.yaml
    - src/finalayze/strategies/presets/ru_finance.yaml
    - src/finalayze/strategies/presets/ru_tech.yaml

key-decisions:
  - "RSS and Telegram fetchers added as optional params to TradingLoop (backward compatible)"
  - "Legacy NewsAPI used as fallback only when RSS+Telegram return no articles"
  - "Entity extraction enriches articles with MOEX tickers before sentiment pipeline"
  - "event_driven weight 0.15 on all ru_* segments; other weights reduced proportionally"
  - "Backtest validates weight redistribution impact (event_driven shows 0 trades as expected -- news-only strategy)"

patterns-established:
  - "Independent error handling: each news source wrapped in try/except, one failure does not block others"
  - "Optional constructor injection: new TradingLoop dependencies are None by default for backward compat"

requirements-completed: [NWS-04, NWS-05, AUT-05]

# Metrics
duration: 15min
completed: 2026-03-15
---

# Phase 7 Plan 03: TradingLoop News Integration Summary

**RSS + Telegram + entity extraction wired into TradingLoop news cycle; event_driven enabled at 0.15 weight on all MOEX segments; /stop kill switch and go-live checklist for real money deployment**

## Performance

- **Duration:** ~15 min (across executor sessions with checkpoint)
- **Started:** 2026-03-15
- **Completed:** 2026-03-15
- **Tasks:** 4 (3 auto + 1 checkpoint:human-verify)
- **Files modified:** 12 source/test/doc files + backtest results

## Accomplishments

- TradingLoop._news_cycle() fetches from RSS and Telegram with independent error handling, entity extraction enriches articles with MOEX tickers
- OpenRouterClient wired to EntityExtractor in main.py via create_llm_client; RssNewsFetcher and TelegramChannelReader instantiated from settings
- event_driven strategy enabled on all 4 ru_* MOEX presets at 0.15 weight with proportional reduction of other strategy weights
- /stop Telegram command halts all trading cycles (emergency kill switch)
- real_confirmed preflight guard tested (Settings rejects REAL mode without explicit confirmation)
- Backtest-iteration on all ru_* segments validates weight redistribution does not degrade existing strategies
- Go-live checklist documented with prerequisites, environment config, safety verification, launch procedure, and emergency procedures

## Task Commits

Each task was committed atomically:

1. **Task 1: Wire RSS + Telegram + EntityExtractor into TradingLoop and main.py** - `4ab0881` (feat)
2. **Task 2: Enable event_driven on MOEX presets, add /stop command, real-mode guard test, go-live checklist** - `f7ab73d` (feat)
3. **Task 3: Run backtest-iteration on all ru_* segments** - `7f9d81b` (chore)
4. **Task 4: Verify news pipeline, backtest results, and go-live readiness** - checkpoint:human-verify (approved)

## Files Created/Modified

- `src/finalayze/core/trading_loop.py` - Added rss_fetcher, telegram_reader, entity_extractor params; rewrote _news_cycle with multi-source fetching
- `src/finalayze/main.py` - Wired EntityExtractor, RssNewsFetcher, TelegramChannelReader into TradingLoop constructor
- `src/finalayze/core/telegram_bot.py` - Added /stop command handler with trading_loop.stop() call
- `src/finalayze/strategies/presets/ru_blue_chips.yaml` - event_driven enabled at 0.15 weight
- `src/finalayze/strategies/presets/ru_energy.yaml` - event_driven enabled at 0.15 weight
- `src/finalayze/strategies/presets/ru_finance.yaml` - event_driven enabled at 0.15 weight
- `src/finalayze/strategies/presets/ru_tech.yaml` - event_driven enabled at 0.15 weight
- `tests/unit/test_news_cycle_integration.py` - Integration tests for multi-source news cycle
- `tests/unit/test_event_driven_presets.py` - Preset validation (weights sum to 1.00, event_driven enabled)
- `tests/unit/test_telegram_stop_command.py` - /stop command handler tests
- `tests/unit/test_real_mode_guard.py` - real_confirmed preflight guard validation
- `docs/operations/GO_LIVE_CHECKLIST.md` - Go-live procedure for real MOEX trading

## Decisions Made

- RSS and Telegram fetchers are optional constructor params (None by default) for backward compatibility
- Legacy NewsAPI kept as fallback when RSS+Telegram return no articles
- Entity extraction runs on every article before sentiment processing, populating article.symbols with MOEX tickers
- event_driven weight set to 0.15 across all ru_* segments; existing strategy weights reduced proportionally
- Backtest confirms weight redistribution has minimal impact (event_driven generates 0 trades in backtest since it requires live news)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

External services require manual configuration before go-live:
- `FINALAYZE_TELEGRAM_API_ID` and `FINALAYZE_TELEGRAM_API_HASH` for Telegram channel reading (from https://my.telegram.org/apps)
- `FINALAYZE_LLM_API_KEY` for OpenRouter entity extraction (from https://openrouter.ai/keys)
- `FINALAYZE_REAL_CONFIRMED=true` to enable real money mode
- See `docs/operations/GO_LIVE_CHECKLIST.md` for complete deployment procedure

## Next Phase Readiness

This is the final plan in Phase 7 (the last phase). The system is ready for real money MOEX deployment:
- All news pipeline components wired and tested
- event_driven strategy enabled on all MOEX segments
- Emergency kill switch (/stop) operational
- Go-live checklist documented
- Backtest validation passed

## Self-Check: PASSED

All 12 key files verified present. All 3 task commits (4ab0881, f7ab73d, 7f9d81b) verified in git log.

---
*Phase: 07-news-pipeline-and-go-live*
*Completed: 2026-03-15*

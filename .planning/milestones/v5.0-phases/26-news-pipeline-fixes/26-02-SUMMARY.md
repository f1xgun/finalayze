---
phase: 26-news-pipeline-fixes
plan: 02
subsystem: analysis, data
tags: [entity-extraction, telegram, dedup, moex-tickers]

requires:
  - phase: none
    provides: n/a
provides:
  - "Corrected _VALID_TICKERS frozenset with TCSG instead of bare T"
  - "URL-based message deduplication in TelegramChannelReader"
affects: [news-pipeline, entity-extraction, telegram-reader]

tech-stack:
  added: []
  patterns:
    - "OrderedDict LRU dedup (same pattern as RssNewsFetcher)"

key-files:
  created: []
  modified:
    - src/finalayze/analysis/entity_extractor.py
    - src/finalayze/data/fetchers/telegram_reader.py
    - tests/unit/test_entity_extractor.py
    - tests/unit/test_telegram_reader.py

key-decisions:
  - "Follow RssNewsFetcher OrderedDict dedup pattern for consistency"

patterns-established:
  - "OrderedDict with popitem(last=False) for bounded LRU dedup across all news fetchers"

requirements-completed: [NEWS-03, NEWS-04]

duration: 2min
completed: 2026-03-23
---

# Phase 26 Plan 02: TCSG Ticker Fix and Telegram Dedup Summary

**Fixed T-Bank ticker mismatch (T -> TCSG) in entity extractor and added URL-based message deduplication to Telegram reader with 5000-entry LRU eviction**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-23T20:06:53Z
- **Completed:** 2026-03-23T20:09:00Z
- **Tasks:** 1 (TDD: RED + GREEN)
- **Files modified:** 4

## Accomplishments

- TCSG is now in _VALID_TICKERS; bare "T" removed -- T-Bank news correctly maps to MOEX ticker
- TelegramChannelReader skips messages with previously-seen URLs across consecutive fetch calls
- Dedup seen set bounded at 5000 entries with oldest-first eviction (matching RssNewsFetcher pattern)
- 8 new tests covering ticker validity, extraction filtering, dedup behavior, and eviction

## Task Commits

Each task was committed atomically (TDD flow):

1. **Task 1 RED: Failing tests** - `fa1ab03` (test)
2. **Task 1 GREEN: Implementation** - `14daac4` (feat)

## Files Created/Modified

- `src/finalayze/analysis/entity_extractor.py` - Replaced "T" with "TCSG" in _VALID_TICKERS
- `src/finalayze/data/fetchers/telegram_reader.py` - Added OrderedDict _seen_urls dedup in _parse_message
- `tests/unit/test_entity_extractor.py` - Added TestValidTickers class (4 tests)
- `tests/unit/test_telegram_reader.py` - Added TestTelegramChannelReaderDedup class (4 tests)

## Decisions Made

- Followed exact same OrderedDict dedup pattern as RssNewsFetcher for consistency across news fetchers

## Deviations from Plan

None - plan executed exactly as written.

## Known Stubs

None.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Entity extractor now correctly resolves T-Bank to TCSG ticker
- Telegram reader will not waste LLM tokens on duplicate messages
- Both fixes are independent and do not affect other pipeline components

---
*Phase: 26-news-pipeline-fixes*
*Completed: 2026-03-23*

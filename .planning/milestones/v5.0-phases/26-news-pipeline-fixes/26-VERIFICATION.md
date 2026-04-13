---
phase: 26-news-pipeline-fixes
verified: 2026-03-24T00:00:00Z
status: passed
score: 4/4 must-haves verified
re_verification: false
---

# Phase 26: News Pipeline Fixes Verification Report

**Phase Goal:** No wasted LLM tokens when event_driven disabled, sentiment decays over time, ticker extraction correct, no duplicate Telegram processing
**Verified:** 2026-03-24
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| #  | Truth                                                                                                           | Status     | Evidence                                                                                                                            |
|----|----------------------------------------------------------------------------------------------------------------|------------|-------------------------------------------------------------------------------------------------------------------------------------|
| 1  | When no segment has event_driven enabled, _news_cycle() returns immediately without fetching articles or calling LLM | VERIFIED   | trading_loop.py:788 — early return guard `if not self._any_event_driven_enabled()` at top of `_news_cycle()`                       |
| 2  | Sentiment scores decay exponentially with 4-hour half-life (8h-old score reads as ~25% of original)            | VERIFIED   | `_read_decayed_sentiment` at line 1004 applies `score * math.exp(-_SENTIMENT_DECAY_LAMBDA * hours_elapsed)` where lambda = ln(2)/4  |
| 3  | Entity extractor _VALID_TICKERS contains TCSG and does not contain bare T                                       | VERIFIED   | entity_extractor.py:52 — frozenset contains "TCSG"; "T" is absent                                                                  |
| 4  | TelegramChannelReader skips messages whose URL was already processed within the dedup window                    | VERIFIED   | telegram_reader.py:165-169 — `_seen_urls` OrderedDict check in `_parse_message()` before appending article                         |

**Score:** 4/4 truths verified

---

### Required Artifacts

| Artifact                                                     | Expected                                      | Status     | Details                                                                                                       |
|--------------------------------------------------------------|-----------------------------------------------|------------|---------------------------------------------------------------------------------------------------------------|
| `src/finalayze/orchestration/trading_loop.py`                | News cycle skip guard and sentiment time-decay | VERIFIED   | `_any_event_driven_enabled`, `_read_decayed_sentiment`, `_SENTIMENT_DECAY_LAMBDA`, `_event_driven_active` all present and substantive |
| `src/finalayze/analysis/entity_extractor.py`                 | Corrected _VALID_TICKERS frozenset with TCSG   | VERIFIED   | "TCSG" at line 52, "T" absent from frozenset                                                                  |
| `src/finalayze/data/fetchers/telegram_reader.py`             | URL-based message deduplication                | VERIFIED   | `_seen_urls: OrderedDict[str, None]` in `__init__`, dedup block at lines 165-169 in `_parse_message()`        |
| `tests/unit/core/test_trading_loop.py`                       | Tests for news skip and sentiment decay        | VERIFIED   | 10 tests in `TestNewsCycleSkipGuard` and `TestSentimentTimeDecay` classes                                      |
| `tests/unit/test_news_cycle_integration.py`                  | Integration tests updated with skip-guard fix  | VERIFIED   | File exists; pre-existing tests fixed to set `_event_driven_active=True`                                       |
| `tests/unit/test_entity_extractor.py`                        | Tests TCSG valid, T invalid                    | VERIFIED   | `TestValidTickers` class at line 122 with `test_tcsg_in_valid_tickers` and `test_bare_t_not_in_valid_tickers`  |
| `tests/unit/test_telegram_reader.py`                         | Dedup tests                                    | VERIFIED   | `TestTelegramChannelReaderDedup` class at line 270 with 4 dedup scenario tests                                 |

---

### Key Link Verification

| From                           | To                         | Via                                        | Status   | Details                                                                                |
|-------------------------------|----------------------------|--------------------------------------------|----------|----------------------------------------------------------------------------------------|
| `_news_cycle`                 | `_any_event_driven_enabled` | early return guard                         | WIRED    | trading_loop.py:788 — `if not self._any_event_driven_enabled(): return`               |
| `_get_sentiment`              | `_sentiment_cache`          | exponential decay applied on read          | WIRED    | trading_loop.py:1027 — `return self._read_decayed_sentiment(seg_id)` in fallback path |
| `entity_extractor._VALID_TICKERS` | LLM ticker output       | frozenset membership check                 | WIRED    | "TCSG" confirmed in frozenset; used in `extract()` for filtering                       |
| `telegram_reader._seen_urls`  | `_parse_message`            | URL dedup check before appending           | WIRED    | Lines 165-169 — check `msg_url in self._seen_urls` before `return NewsArticle(...)`   |

---

### Data-Flow Trace (Level 4)

Not applicable — phase produces utility/guard code and configuration fixes, not components rendering dynamic UI data. The news pipeline is disabled (event_driven=false in all presets); data-flow trace would require a live integration test.

---

### Behavioral Spot-Checks

| Behavior                                | Command                                                                                                                      | Result        | Status |
|-----------------------------------------|------------------------------------------------------------------------------------------------------------------------------|---------------|--------|
| All phase 26 tests pass                  | `uv run pytest tests/unit/core/test_trading_loop.py tests/unit/test_news_cycle_integration.py tests/unit/test_entity_extractor.py tests/unit/test_telegram_reader.py -q` | 51 passed, 0 failed | PASS   |
| TCSG in frozenset, T absent             | grep confirms "TCSG" at entity_extractor.py:52, "T" not found                                                               | Confirmed     | PASS   |
| `_any_event_driven_enabled` caches result | `_event_driven_active: bool | None = None` in `__init__`; cache check at line 1035                                          | Confirmed     | PASS   |
| Decay constants correct (lambda=ln2/4) | `_SENTIMENT_DECAY_LAMBDA = math.log(2) / 4.0` at trading_loop.py:78                                                         | Confirmed     | PASS   |

---

### Requirements Coverage

| Requirement | Source Plan  | Description                                                                                  | Status     | Evidence                                                                                                    |
|-------------|-------------|----------------------------------------------------------------------------------------------|------------|-------------------------------------------------------------------------------------------------------------|
| NEWS-01     | 26-01-PLAN  | News cycle skipped entirely when no segment has event_driven enabled — no LLM calls wasted   | SATISFIED  | `_any_event_driven_enabled()` YAML scanning with caching; early return at _news_cycle():788                 |
| NEWS-02     | 26-01-PLAN  | Sentiment cache has time-based exponential decay (half-life 4 hours)                         | SATISFIED  | `_read_decayed_sentiment` + `_SENTIMENT_DECAY_LAMBDA`; all sentiment reads and writes updated               |
| NEWS-03     | 26-02-PLAN  | Entity extractor _VALID_TICKERS contains "TCSG" (not "T")                                   | SATISFIED  | entity_extractor.py:52 — "TCSG" in frozenset, "T" absent; TestValidTickers tests pass                      |
| NEWS-04     | 26-02-PLAN  | Telegram reader deduplicates messages by message link URL                                    | SATISFIED  | telegram_reader.py:39,165-169 — `_seen_urls` OrderedDict with LRU eviction; TestTelegramChannelReaderDedup passes |

**Note on REQUIREMENTS.md:** The traceability table (lines 80-81) still shows NEWS-03 and NEWS-04 as "Pending" and the checkboxes on lines 33-34 are unchecked. This is a stale documentation issue only — the code and tests fully implement both requirements. The document should be updated to mark them Complete/checked, but this does not affect goal achievement.

---

### Anti-Patterns Found

| File                                         | Line | Pattern                                                    | Severity | Impact                                  |
|----------------------------------------------|------|------------------------------------------------------------|----------|-----------------------------------------|
| `src/finalayze/orchestration/trading_loop.py` | 1704 | `TODO: Wire returns history for live correlation computation in future phase` | Info     | Pre-existing; unrelated to phase 26 work |

No phase-26-introduced anti-patterns found. All new code is substantive and fully connected.

---

### Human Verification Required

None. All observable truths for this phase are verifiable programmatically via code inspection and test execution.

---

### Gaps Summary

No gaps. All four must-have truths are fully verified:

1. NEWS-01 (skip guard) — implemented, tested (5 tests), wired at _news_cycle entry point.
2. NEWS-02 (sentiment decay) — implemented with correct lambda, tested (5 tests), wired in all sentiment read paths.
3. NEWS-03 (TCSG ticker) — one-line fix confirmed in frozenset, tested (4 tests).
4. NEWS-04 (Telegram dedup) — OrderedDict pattern correctly followed, tested (4 tests), dedup fires before `return NewsArticle(...)`.

The only minor housekeeping item is that REQUIREMENTS.md lines 33-34 and 80-81 still show NEWS-03/NEWS-04 as Pending — this is stale documentation and does not represent a code gap.

---

_Verified: 2026-03-24_
_Verifier: Claude (gsd-verifier)_

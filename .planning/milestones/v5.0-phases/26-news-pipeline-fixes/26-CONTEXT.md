# Phase 26: News Pipeline Fixes - Context

**Gathered:** 2026-03-23
**Status:** Ready for planning

<domain>
## Phase Boundary

Disable news cycle when event_driven is off (save LLM tokens), add time-based sentiment decay, fix TCSG ticker mismatch, add Telegram message deduplication.

</domain>

<decisions>
## Implementation Decisions

### Claude's Discretion
All implementation choices are at Claude's discretion — bug-fix phase.
Key constraints from audit:
- NEWS-01: Check if any active segment has event_driven enabled; if none, skip entire _news_cycle()
- NEWS-02: Apply time-based decay: score *= exp(-lambda * hours_since_last_update), half-life=4h → lambda=ln(2)/4
- NEWS-03: Replace "T" with "TCSG" in _VALID_TICKERS frozenset (entity_extractor.py:52)
- NEWS-04: TelegramChannelReader should track seen message URLs in OrderedDict (like RssNewsFetcher pattern)
- Must preserve ability to re-enable event_driven later without code changes

</decisions>

<code_context>
## Existing Code Insights

### Key Files
- `src/finalayze/orchestration/trading_loop.py:795` — _news_cycle() always runs
- `src/finalayze/orchestration/trading_loop.py:909` — EMA without time decay: new = old*0.7 + impact*0.3
- `src/finalayze/analysis/entity_extractor.py:52` — _VALID_TICKERS contains "T" not "TCSG"
- `src/finalayze/data/fetchers/telegram_reader.py` — no dedup mechanism
- `src/finalayze/data/fetchers/rss_fetcher.py` — OrderedDict URL dedup (reference pattern)
- `src/finalayze/strategies/presets/*.yaml` — event_driven.enabled: false everywhere

### Established Patterns
- RssNewsFetcher uses OrderedDict with maxlen=5000 for URL dedup
- Segment presets loaded per segment with strategy enable/disable flags
- _sentiment_cache protected by _sentiment_lock (threading.Lock)

</code_context>

<specifics>
## Specific Ideas

No specific requirements — bug-fix phase.

</specifics>

<deferred>
## Deferred Ideas

None.

</deferred>

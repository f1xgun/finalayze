# Phase 49: News Pipeline Hardening - Context

**Gathered:** 2026-04-14
**Status:** Ready for planning

<domain>
## Phase Boundary

Fix 3 confirmed latent bugs (json.loads parsing, 1800s no-op timeout, threading.Lock across await) and add production safeguards (article budget cap, source credibility tagging, ticker validation, LLM liveness monitoring) to the news ingestion pipeline before Phase 50 activates EventDrivenStrategy.

</domain>

<decisions>
## Implementation Decisions

### Claude's Discretion

All implementation choices are at Claude's discretion — pure infrastructure/bug-fix phase. Success criteria are fully prescriptive with specific values:
- Article budget: 20 per cycle, 5s per-article LLM timeout
- Source credibility: RSS=0.8, Telegram=0.7
- Ticker validation: reject against InstrumentRegistry
- LLM liveness: 3 consecutive failures → Telegram alert + Prometheus counter
- Structured parsing: replace json.loads with parse_structured() returning SentimentResult Pydantic model

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `NewsAnalyzer` in `src/finalayze/analysis/news_analyzer.py` — current LLM analysis with json.loads (bug target)
- `LLMClient` in `src/finalayze/analysis/llm_client.py` — three providers (OpenRouter, OpenAI, Anthropic), no explicit timeout
- `TelegramAlerter` in `src/finalayze/core/alerts.py` — existing alert dispatch
- `InstrumentRegistry` in `src/finalayze/markets/instruments.py` — ticker validation source
- `SentimentScoreModel` in `src/finalayze/core/models.py` — sentiment_scores table (needs `credibility` column)

### Established Patterns
- APScheduler `BackgroundScheduler` in `trading_loop.py` for scheduling news cycles
- `threading.Lock` used for `_sentiment_cache` and `_stop_loss_lock` — the sentiment lock crosses await boundary (bug target)
- `run_coroutine_threadsafe` bridges scheduler threads to async loop with 30s timeout
- Pydantic v2 schemas in `core/schemas.py` for all data contracts

### Integration Points
- `_news_cycle()` in `trading_loop.py` (line 284) — entry point for scheduled news processing
- `SentimentScoreModel` table — target for credibility column addition
- Prometheus metrics via existing counter/histogram patterns in api/metrics
- Telegram alerts for LLM liveness monitoring

</code_context>

<specifics>
## Specific Ideas

STATE.md confirms 3 latent bugs with HIGH research confidence:
1. `json.loads()` in news_analyzer.py — replace with `parse_structured()` returning Pydantic SentimentResult
2. 1800s no-op timeout — replace with 5s per-article + 20-article budget cap
3. `threading.Lock` across `await` — replace with `asyncio.Lock` or restructure async/sync boundary

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

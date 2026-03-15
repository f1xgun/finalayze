# Phase 7: News Pipeline and Go-Live - Context

**Gathered:** 2026-03-15
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase delivers the Russian news ingestion pipeline (RSS + Telegram), LLM-based sentiment analysis with ticker extraction, event_driven strategy enablement on all MOEX segments, and first real MOEX trades on a 500K RUB account after sandbox re-validation.

</domain>

<decisions>
## Implementation Decisions

### News Source Architecture
- Use feedparser library for RSS fetching (mature, handles RSS/Atom)
- Target 3 sources: RBC, Interfax, TASS (major Russian business wires)
- Use Telethon (async) for Telegram channel reading (NWS-03)
- Poll news sources every 5 minutes (balance freshness vs rate limits)

### LLM News Analysis Pipeline
- LLM entity extraction for mapping news articles to MOEX tickers (more accurate than keyword dict)
- Individual article analysis (reuse existing NewsAnalyzer.analyze() pattern)
- Weighted average in StrategyCombiner for combining news sentiment with technical signals
- Use OpenRouter client with free model for cost efficiency (user preference)

### Go-Live Safety & Real Trading
- Starting capital: 500K RUB (matches PROJECT.md constraint)
- Require passing sandbox validation report before go-live (re-confirm AUT-04)
- Same risk limits as sandbox (proven, no reason to change)
- Kill switch: existing circuit breaker + add Telegram /stop write command

### Event-Driven Strategy Enablement
- Enable event_driven on ALL ru_* segments (ru_blue_chips, ru_energy, ru_finance, ru_tech)
- Initial weight: 0.15 (15%) — meaningful but technical signals still dominate
- Event types: geopolitical, sanctions, cbr_rate, commodity_price, earnings (already in presets)
- Backtest validation required before enabling (mandatory per WORKFLOW.md)

### Claude's Discretion
- RSS URL selection within the 3 confirmed sources (validate at implementation)
- Telegram channel selection for financial sentiment
- LLM prompt design for entity extraction (extend existing sentiment_ru.txt or separate prompt)
- Deduplication strategy for news articles across sources

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/finalayze/analysis/news_analyzer.py` — NewsAnalyzer with RU prompt support
- `src/finalayze/analysis/llm_client.py` — LLMClient ABC with OpenRouter/Anthropic/OpenAI providers
- `src/finalayze/analysis/event_classifier.py` — EventClassifier for categorizing events
- `src/finalayze/analysis/impact_estimator.py` — ImpactEstimator for scoring event impact
- `src/finalayze/analysis/prompts/sentiment_ru.txt` — Russian sentiment prompt
- `src/finalayze/strategies/event_driven.py` — EventDrivenStrategy with sanctions proximity scoring
- `src/finalayze/core/schemas.py` — NewsArticle, SentimentResult schemas
- All ru_* strategy presets have event_driven config (currently disabled, weight 0.00)

### Established Patterns
- Async-first with httpx for HTTP, SQLAlchemy 2.0 async for DB
- Data fetchers in `src/finalayze/data/fetchers/` (base.py pattern)
- Strategy presets in YAML (`src/finalayze/strategies/presets/*.yaml`)
- TDD mandatory — failing test first, then implement
- Pydantic v2 for all schemas

### Integration Points
- StrategyCombiner — event_driven weight already wired, just needs enabling
- TradingLoop scheduler — news polling job needs scheduling (APScheduler pattern from Phase 6)
- Telegram bot — add /stop write command to existing read-only commands (/status, /breakers)
- Settings — add news-related config (RSS URLs, Telegram channels, polling interval)

</code_context>

<specifics>
## Specific Ideas

- User wants OpenRouter with free model (not Claude Sonnet directly) for LLM analysis cost
- All pruned MOEX symbols restored in Phase 2 specifically for news/sentiment integration
- Russian news RSS URLs flagged as MEDIUM confidence in STATE.md — validate during implementation
- Sanctions proximity scoring already built into EventDrivenStrategy for Russian equities

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

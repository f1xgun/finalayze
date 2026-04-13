# Phase 27: Intelligent News Impact Analysis - Context

**Gathered:** 2026-03-24
**Status:** Ready for planning

<domain>
## Phase Boundary

Replace EntityExtractor + CombinedAnalyzer (2 LLM calls, naive ticker-in-text extraction) with unified NewsImpactAnalyzer (1 LLM call, sector-aware impact prediction). Add SectorTickerMapper (static registry, no LLM). Update _sentiment_cache to store per-ticker sentiment. Update event_driven strategy to consume per-ticker scores.

</domain>

<decisions>
## Implementation Decisions

### NewsImpactAnalyzer Design
- Single LLM call per article replaces both EntityExtractor and CombinedAnalyzer
- Prompt returns JSON: event_type, sentiment, confidence, reasoning, affected_sectors[], direct_tickers[]
- Each affected_sector has: sector name, direction (-1/+1), magnitude (0.0-1.0), reasoning
- Russian and English prompt variants (analyze_impact_ru.txt, analyze_impact_en.txt)
- Reuse existing circuit breaker pattern from EntityExtractor
- Reuse _CachingLLMClient SHA-256 cache for dedup

### Sector Taxonomy (MOEX-focused)
- oil_gas: ROSN, LKOH, TATN, TATNP, NVTK, SNGS, SNGSP
- banking: SBER, VTBR, TCSG
- metals_mining: GMKN, NLMK, CHMF, MAGN, RUAL, ALRS, PLZL
- telecom: MTSS
- tech: YDEX, OZON, HHRU
- utilities: IRAO, MSNG, HYDR
- real_estate: PIKK
- retail: MGNT
- transport: AFLT, TRNFP
- fertilizers: PHOR
- conglomerate: AFKS
- exchange: MOEX
- bonds_fixed: OFZ-PD segment
- bonds_floating: OFZ-PK segment
- Stored as dict[str, list[str]] constant in sector_ticker_mapper.py

### Sentiment Cache Structure Change
- Current: _sentiment_cache[segment_id] = (score, timestamp)
- New: _sentiment_cache[(segment_id, ticker)] = (score, timestamp)
- Fallback: if no per-ticker score, use segment-level average
- event_driven strategy receives per-ticker sentiment via generate_signal(sentiment_score=...)
- _read_decayed_sentiment(segment_id, ticker) reads per-ticker first, falls back to segment average

### Integration with TradingLoop
- Replace _extract_entities_batch + _analyze_sentiment_batch with single _analyze_impact_batch
- _process_articles_batch calls NewsImpactAnalyzer.analyze() per article
- Results routed to _sentiment_cache via SectorTickerMapper
- direct_tickers from LLM response used as-is (validated against _VALID_TICKERS)
- Sector-mapped tickers receive sector.magnitude * sector.direction * article.sentiment

### Claude's Discretion
- Exact prompt wording and chain-of-thought structure
- Error handling for malformed sector names from LLM
- Whether to keep EntityExtractor as fallback or remove entirely
- Batch size for concurrent LLM calls (current: semaphore of 5)

</decisions>

<code_context>
## Existing Code Insights

### Files to Replace/Modify
- `src/finalayze/analysis/entity_extractor.py` — REPLACE with NewsImpactAnalyzer
- `src/finalayze/analysis/combined_analyzer.py` — REPLACE (functionality merged into NewsImpactAnalyzer)
- `src/finalayze/analysis/impact_estimator.py` — SIMPLIFY (SectorTickerMapper replaces rule-based routing)
- `src/finalayze/analysis/event_classifier.py` — KEEP EventType enum, remove classifier class
- `src/finalayze/orchestration/trading_loop.py` — Update _news_cycle, _process_articles_batch, _sentiment_cache structure
- `src/finalayze/strategies/event_driven.py` — Update to use per-ticker sentiment

### Files to Create
- `src/finalayze/analysis/news_impact_analyzer.py` — Unified LLM analyzer
- `src/finalayze/analysis/sector_ticker_mapper.py` — Static sector→ticker registry
- `src/finalayze/analysis/prompts/analyze_impact_ru.txt` — Russian prompt
- `src/finalayze/analysis/prompts/analyze_impact_en.txt` — English prompt

### Established Patterns
- LLMClient.complete(user_prompt, system_prompt) → str
- _CachingLLMClient with SHA-256 LRU cache (1000 entries)
- Circuit breaker: _CIRCUIT_BREAKER_THRESHOLD=5, _CIRCUIT_BREAKER_RESET_SECONDS=300
- JSON response parsing with _CODE_FENCE_RE for markdown fence stripping
- Pydantic models for structured outputs (SentimentResult, SegmentImpact)

### Integration Points
- TradingLoop._news_cycle() orchestrates the pipeline
- _process_articles_batch() runs concurrent LLM calls with semaphore
- _sentiment_cache protected by _sentiment_lock (threading.Lock)
- event_driven strategy receives sentiment_score via generate_signal()

</code_context>

<specifics>
## Specific Ideas

Example LLM prompt structure for sector impact:
```
Analyze this Russian financial news. Determine:
1. Event type (cbr_rate, oil_price, sanctions, geopolitical, macro, earnings, sector_specific, other)
2. Sentiment [-1.0 to +1.0] with confidence [0.0 to 1.0]
3. Which MOEX sectors are affected, direction, and magnitude

Sectors: oil_gas, banking, metals_mining, telecom, tech, utilities,
real_estate, retail, transport, fertilizers, conglomerate, exchange,
bonds_fixed, bonds_floating

Return JSON:
{"event_type": "...", "sentiment": 0.0, "confidence": 0.0,
 "reasoning": "...",
 "affected_sectors": [{"sector": "...", "direction": -1, "magnitude": 0.8, "reasoning": "..."}],
 "direct_tickers": ["SBER"]}
```

</specifics>

<deferred>
## Deferred Ideas

- Article persistence to database (NEWS-F01) — separate phase
- Prompt injection sanitization (NEWS-F02) — separate phase
- Dynamic sector registry from InstrumentRegistry (load from DB instead of static dict)
- Multi-hop reasoning (article about Turkey → MOEX via tourism/currency channel)

</deferred>

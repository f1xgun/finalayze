# News Analysis Pipeline

## Overview

News articles are ingested from multiple sources, analyzed by Claude LLM for
sentiment and classification, then routed to affected market segments.

## Pipeline Steps

1. **Ingestion:** Fetch from NewsAPI (EN), Finnhub (EN), Russian news sources (RU)
2. **Deduplication:** Hash-based duplicate detection
3. **LLM Analysis:** Claude Sonnet performs:
   - Sentiment scoring (-1.0 to +1.0)
   - Fact-checking (cross-reference across sources)
   - Event classification (earnings, regulatory, macro, M&A, geopolitical)
   - Entity extraction (tickers, companies, markets)
   - Geographic scope determination (global, us, russia, sector)
4. **Impact Routing:** Route to affected segments based on scope
5. **Aggregation:** Composite sentiment score per symbol per segment

## Scope Routing

| Scope | Affects | Impact Weight |
|-------|---------|---------------|
| Global | All segments | 1.0 primary, 0.3-0.5 secondary |
| US | `us_*` segments | 1.0 primary |
| Russia | `ru_*` segments | 1.0 primary |
| Sector | Matching sector across markets | 1.0 primary |

## LLM Prompts

Stored in `src/finalayze/analysis/prompts/`:
- `sentiment_en.txt` -- English news sentiment
- `sentiment_ru.txt` -- Russian news sentiment
- `fact_check.txt` -- Cross-source fact checking
- `classify_event.txt` -- Event type classification

## Status

**Phase 2:** Full pipeline implemented with EN + RU support.

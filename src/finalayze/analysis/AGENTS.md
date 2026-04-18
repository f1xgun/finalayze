# Analysis

## Purpose
News sentiment analysis using LLMs (Claude/OpenAI/OpenRouter), entity extraction, event classification, fact checking, and impact estimation.

## Layer
Layer 3 -- Analysis / ML. Can import from layers 0-2. Never import from layers 4-6.

## Key Files
- `news_analyzer.py` -- NewsAnalyzer: LLM-based sentiment scoring with EN/RU prompt selection
- `llm_client.py` -- LLMClient ABC and implementations: OpenRouterClient, OpenAIClient, AnthropicClient
- `entity_extractor.py` -- Extract company/ticker entities from news text
- `event_classifier.py` -- Classify news into event types (earnings, M&A, regulatory, etc.)
- `fact_checker.py` -- Cross-reference claims against known data
- `impact_estimator.py` -- Estimate price impact magnitude from classified events
- `prompts/` -- LLM prompt templates (sentiment_en.txt, sentiment_ru.txt, classify_event.txt, etc.)

## Public API
- `NewsAnalyzer` -- `async analyze(article: NewsArticle) -> SentimentResult`
- `LLMClient` -- `async complete(prompt, system) -> str` (abstract)
- `EntityExtractor` -- extract ticker mentions from text
- `EventClassifier` -- classify news events

## Contracts
- Input: `NewsArticle` (with language field for prompt selection)
- Output: `SentimentResult` (sentiment in [-1.0, 1.0], confidence in [0.0, 1.0])
- Invariants: Parse errors return neutral fallback (sentiment=0.0, confidence=0.0). LLM clients use SHA-256 LRU cache (max 1000 entries) and exponential backoff retry (3 attempts). `event_driven` strategy is currently DISABLED (no real-time news feed).

## Testing
- Test location: `tests/unit/test_news_analyzer.py`, `tests/unit/test_llm_client.py`
- Run: `uv run pytest tests/unit/test_news_analyzer.py tests/unit/test_llm_client.py -v`

## Common Patterns
- All LLM clients extend `_CachingLLMClient` for deduplication and retry
- Prompt files are loaded lazily and cached in-memory
- LLMRateLimitError vs LLMError distinction for retry logic
- JSON response parsing with fallback to neutral on any parse failure

---

## Graph

- **Parent:** [`src/finalayze/AGENTS.md`](../AGENTS.md)
- **Agent owner:** `analysis-agent` (pipeline orchestration: `news-pipeline-agent`)
- **Layer:** 3
- **Imports from:** `core/`, `config/`, `data/`
- **Imported by:** `strategies/` (event-driven, disabled), `backtest/`, `orchestration/`, `api/`
- **Keywords:** `LLM`, `Anthropic`, `OpenAI`, `OpenRouter`, `NewsAnalyzer`, `sentiment`, `entity_extractor`, `event_classifier`, `fact_checker`, `impact_estimator`, `prompts`, `SentimentResult`

# Phase 49: News Pipeline Hardening - Research

**Researched:** 2026-04-14
**Domain:** News ingestion pipeline bug fixes and production safeguards
**Confidence:** HIGH

## Summary

Phase 49 addresses 3 confirmed latent bugs and adds 4 production safeguards to the news pipeline before Phase 50 activates EventDrivenStrategy. The codebase is well-structured with clear separation between `NewsAnalyzer` (Layer 3), `LLMClient` (Layer 3), and `TradingLoop` (Layer 6). All changes are internal to existing modules with no new external dependencies.

The `json.loads()` bug in `NewsAnalyzer.analyze()` (line 53) and `EventClassifier._parse_response()` must be replaced with structured output via the OpenAI SDK's `beta.chat.completions.parse()` method, which is available in the installed openai 2.21.0. The `threading.Lock` crossing `await` boundary is in `_sentiment_lock` (line 135 of trading_loop.py) -- it guards `_sentiment_cache` which is only accessed from `_process_news_article()` running on the persistent async loop via `_run_async()`. The 1800s timeout is actually a 30s timeout in `_run_async()` but with no per-article LLM timeout and no article count cap.

**Primary recommendation:** Fix bugs first (parse_structured, lock, timeout), then layer safeguards (budget cap, credibility, ticker validation, liveness) on top of the corrected code.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
All implementation choices are at Claude's discretion -- pure infrastructure/bug-fix phase. Success criteria are fully prescriptive with specific values:
- Article budget: 20 per cycle, 5s per-article LLM timeout
- Source credibility: RSS=0.8, Telegram=0.7
- Ticker validation: reject against InstrumentRegistry
- LLM liveness: 3 consecutive failures -> Telegram alert + Prometheus counter
- Structured parsing: replace json.loads with parse_structured() returning SentimentResult Pydantic model

### Claude's Discretion
All implementation choices.

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| NEWS-01 | News pipeline processes MOEX RSS feeds with 5s per-article LLM timeout | LLMClient timeout parameter, asyncio.wait_for wrapping |
| NEWS-02 | NewsAnalyzer migrated from json.loads() to parse_structured() | OpenAI SDK beta.chat.completions.parse() with Pydantic model |
| NEWS-03 | Source credibility map wired (RSS: 0.8, Telegram: 0.7) | NewsArticle.credibility_score field + SentimentScoreModel.credibility column |
| NEWS-04 | Ticker whitelist validation filters against InstrumentRegistry | InstrumentRegistry.get() raises InstrumentNotFoundError |
| NEWS-05 | LLM liveness check with Telegram alert on sustained failure | TelegramAlerter.on_error() + new Prometheus Counter |
| NEWS-06 | Article budget cap (max 20 articles/cycle) prevents cost explosion | Slice articles list in _news_cycle before processing loop |
</phase_requirements>

## Standard Stack

### Core (already installed)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| openai | 2.21.0 | LLM API client (OpenRouter, OpenAI) | `beta.chat.completions.parse()` for structured output [VERIFIED: uv pip show] |
| anthropic | 0.83.0 | Anthropic LLM API client | Already used in AnthropicClient [VERIFIED: uv pip show] |
| pydantic | v2 | Schema validation for SentimentResult | Already used throughout codebase [VERIFIED: codebase] |
| prometheus_client | (installed) | Metrics counters/gauges | Already in api/metrics.py [VERIFIED: codebase] |
| structlog | (installed) | Structured logging | Already used throughout codebase [VERIFIED: codebase] |
| apscheduler | (installed) | BackgroundScheduler for news cycles | Already in trading_loop.py [VERIFIED: codebase] |

### No New Dependencies Required
This phase modifies existing code only. No new packages needed.

## Architecture Patterns

### Affected Files Map
```
src/finalayze/
├── analysis/
│   ├── news_analyzer.py       # BUG FIX: json.loads -> parse_structured (NEWS-02)
│   ├── llm_client.py          # ADD: timeout param, parse_structured method (NEWS-01, NEWS-02)
│   └── prompts/
│       ├── sentiment_en.txt   # UPDATE: add tickers extraction field (NEWS-04)
│       └── sentiment_ru.txt   # UPDATE: add tickers extraction field (NEWS-04)
├── core/
│   ├── trading_loop.py        # BUG FIX: threading.Lock -> asyncio.Lock (threading bug)
│   │                          # ADD: article budget cap (NEWS-06)
│   │                          # ADD: ticker validation (NEWS-04)
│   │                          # ADD: LLM liveness tracking (NEWS-05)
│   ├── schemas.py             # UPDATE: SentimentResult add tickers field (NEWS-04)
│   └── models.py              # ADD: credibility column to SentimentScoreModel (NEWS-03)
├── api/
│   └── metrics.py             # ADD: llm_liveness_failures Counter, news_budget_cap_hit Counter (NEWS-05, NEWS-06)
alembic/
└── versions/
    └── 004_add_credibility.py # NEW: add credibility column to sentiment_scores (NEWS-03)
```

### Pattern 1: Structured LLM Output via parse_structured()
**What:** Replace `json.loads(raw)` with a method that uses the LLM provider's native structured output
**When to use:** All LLM calls that return structured data

The OpenAI SDK (used for both OpenAI and OpenRouter) supports `beta.chat.completions.parse()` which takes a Pydantic model as `response_format` and returns a parsed object. [CITED: platform.openai.com/docs/guides/structured-outputs]

For Anthropic, structured output is achieved via `tool_use` with a JSON schema, or by parsing the text response with Pydantic. Since the Anthropic SDK v0.83.0 does not have an equivalent `parse()` method, the implementation should:
1. Add a `parse_structured()` method to `LLMClient` ABC that accepts a Pydantic model class
2. In `OpenRouterClient` and `OpenAIClient`: use `beta.chat.completions.parse(response_format=Model)`
3. In `AnthropicClient`: keep text completion + validate with `SentimentResult.model_validate_json(raw)` (Pydantic v2 built-in) [ASSUMED]

```python
# Source: OpenAI structured outputs docs
from pydantic import BaseModel

class SentimentResult(BaseModel):
    sentiment: float
    confidence: float
    reasoning: str
    tickers: list[str] = []  # NEW: LLM-extracted ticker symbols

# In OpenAIClient / OpenRouterClient:
completion = await self._client.beta.chat.completions.parse(
    model=self._model,
    messages=[...],
    response_format=SentimentResult,
)
result = completion.choices[0].message.parsed
```

### Pattern 2: Per-Article LLM Timeout
**What:** Wrap each LLM call with `asyncio.wait_for(coro, timeout=5.0)` 
**When to use:** Every `_analyze_article()` call

```python
# In _process_news_article or _analyze_article:
try:
    sentiment, event = await asyncio.wait_for(
        self._analyze_article(article), timeout=5.0
    )
except asyncio.TimeoutError:
    _log.warning("llm_timeout", article_id=str(article.id))
    return  # skip article
```

Note: The existing `_run_async()` uses `run_coroutine_threadsafe` with a 30s timeout at the thread boundary. The 5s per-article timeout must be inside the async coroutine, not at the thread boundary. [VERIFIED: codebase trading_loop.py line 184]

### Pattern 3: threading.Lock -> asyncio.Lock Fix
**What:** Replace `self._sentiment_lock = threading.Lock()` with proper async-safe access
**When to use:** `_sentiment_cache` access in `_process_news_article()`

The current code path is: APScheduler thread -> `_run_async()` -> async event loop -> `_process_news_article()`. The `threading.Lock` is used inside `_process_news_article()` which runs on the async event loop via `run_coroutine_threadsafe`. Using `threading.Lock` in async code can block the event loop.

**Fix approach:** Since `_sentiment_cache` is accessed from both the async loop (write in `_process_news_article`) and from sync `get_sentiment()` (read), two approaches:
1. Use `asyncio.Lock` for writes + keep `threading.Lock` for the sync reader, OR
2. Restructure so writes happen outside the async boundary (collect results async, apply under threading.Lock in the sync caller)

Approach 2 is simpler: `_process_news_article()` returns the impacts, and `_news_cycle()` applies them under `threading.Lock` after `_run_async()` returns. [ASSUMED]

### Pattern 4: Article Budget Cap
**What:** Limit articles processed per news cycle to 20
**When to use:** In `_news_cycle()` after `fetch_news()` returns

```python
_MAX_ARTICLES_PER_CYCLE = 20

articles = self._news_fetcher.fetch_news(...)
if len(articles) > _MAX_ARTICLES_PER_CYCLE:
    _log.warning("news_budget_cap_hit", total=len(articles), cap=_MAX_ARTICLES_PER_CYCLE)
    MetricsCollector.inc_news_budget_cap_hit()  # new metric
    articles = articles[:_MAX_ARTICLES_PER_CYCLE]
```

### Anti-Patterns to Avoid
- **json.loads for LLM output:** Fragile -- LLMs add markdown fences, extra text. Use SDK structured output instead.
- **threading.Lock in async code:** Blocks the event loop. Use asyncio.Lock or restructure the sync/async boundary.
- **Unbounded article processing:** Without a budget cap, a busy news day can cause runaway LLM costs and cycle timeouts.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| JSON parsing from LLM | Regex/json.loads with fallbacks | OpenAI `beta.chat.completions.parse()` | Handles markdown fences, partial JSON, validation automatically |
| Pydantic JSON parsing | Custom parser | `SentimentResult.model_validate_json()` | Pydantic v2 built-in, handles validation + type coercion |
| Prometheus metrics | Custom counters | `prometheus_client.Counter` | Thread-safe, already used in api/metrics.py |
| Async timeout | Custom timer threads | `asyncio.wait_for()` | Built into asyncio, handles cancellation properly |

## Common Pitfalls

### Pitfall 1: OpenRouter May Not Support Structured Outputs
**What goes wrong:** `beta.chat.completions.parse()` with `response_format` requires model-level support for structured outputs. OpenRouter proxies many models, not all support JSON Schema mode.
**Why it happens:** OpenRouter is an aggregator -- structured output support depends on the underlying model.
**How to avoid:** Fall back to text completion + `SentimentResult.model_validate_json(raw)` for OpenRouter. Or verify the model supports structured output before using parse().
**Warning signs:** 400/422 errors from OpenRouter when using response_format parameter.
[ASSUMED -- needs validation with OpenRouter docs]

### Pitfall 2: Timeout Must Be Per-Article, Not Per-Cycle
**What goes wrong:** If timeout is on the whole `_news_cycle()`, a single slow article wastes the entire cycle's budget.
**Why it happens:** Confusing per-article timeout (5s) with cycle timeout (2 minutes).
**How to avoid:** Apply `asyncio.wait_for(timeout=5.0)` around each individual `_analyze_article()` call, not the entire loop.
**Warning signs:** Cycle completing with 0 articles processed after one slow LLM call.

### Pitfall 3: LLM Liveness Counter Must Persist Across Cycles
**What goes wrong:** If the failure counter resets each cycle, consecutive failures are never detected.
**Why it happens:** Counter stored as local variable instead of instance attribute.
**How to avoid:** Store `_llm_consecutive_failures: int` as instance attribute on TradingLoop, reset to 0 on success.
**Warning signs:** Telegram alert never fires even though LLM is down.

### Pitfall 4: Credibility Column Migration
**What goes wrong:** Missing Alembic migration means the credibility column doesn't exist in production DB.
**Why it happens:** Adding column to ORM model without corresponding migration.
**How to avoid:** Create alembic migration 004 that adds `credibility` column (nullable Numeric) to `sentiment_scores`.
**Warning signs:** SQLAlchemy IntegrityError or "column does not exist" at runtime.

### Pitfall 5: SentimentResult Tickers vs NewsArticle Symbols
**What goes wrong:** Confusion between `NewsArticle.symbols` (pre-populated by fetcher) and `SentimentResult.tickers` (LLM-extracted).
**Why it happens:** Two different sources of ticker symbols with different provenance.
**How to avoid:** Ticker validation (NEWS-04) should validate LLM-extracted tickers from the sentiment analysis output, not the fetcher's pre-populated symbols.
**Warning signs:** Ghost tickers appearing in sentiment_scores from LLM hallucination.

## Code Examples

### Structured Parse Method on LLMClient
```python
# Source: OpenAI docs + codebase pattern
from typing import TypeVar
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

class LLMClient(ABC):
    @abstractmethod
    async def complete(self, prompt: str, system: str) -> str: ...

    async def parse_structured(
        self, prompt: str, system: str, response_model: type[T]
    ) -> T:
        """Parse LLM response into a Pydantic model.
        
        Default implementation: complete() + model_validate_json().
        OpenAI/OpenRouter subclasses override with beta.chat.completions.parse().
        """
        raw = await self.complete(prompt, system)
        return response_model.model_validate_json(raw)
```

### Source Credibility Map
```python
# Source: CONTEXT.md prescriptive values
SOURCE_CREDIBILITY: dict[str, float] = {
    # RSS feeds
    "rbc": 0.8,
    "interfax": 0.8,
    "tass": 0.8,
    "moex_iss": 0.8,
    "reuters": 0.8,
    # Telegram channels
    "telegram": 0.7,
}

def get_credibility(source: str) -> float:
    """Return credibility score for a news source. Default 0.5 for unknown."""
    return SOURCE_CREDIBILITY.get(source.lower(), 0.5)
```

### Ticker Validation Against InstrumentRegistry
```python
# Source: codebase instruments.py pattern
from finalayze.core.exceptions import InstrumentNotFoundError

def validate_tickers(
    tickers: list[str],
    registry: InstrumentRegistry,
    market_id: str,
) -> list[str]:
    """Filter tickers to only those in the instrument registry."""
    valid = []
    for ticker in tickers:
        try:
            registry.get(ticker, market_id)
            valid.append(ticker)
        except InstrumentNotFoundError:
            _log.warning(
                "entity_not_in_registry",
                ticker=ticker,
                market_id=market_id,
            )
    return valid
```

### LLM Liveness Monitor
```python
# Source: codebase alerts.py + metrics.py pattern
_LLM_FAILURE_THRESHOLD = 3

class LLMLivenessTracker:
    def __init__(self, alerter: TelegramAlerter) -> None:
        self._consecutive_failures = 0
        self._alerter = alerter

    def record_success(self) -> None:
        self._consecutive_failures = 0

    def record_failure(self) -> None:
        self._consecutive_failures += 1
        llm_liveness_failures.inc()  # Prometheus counter
        if self._consecutive_failures >= _LLM_FAILURE_THRESHOLD:
            self._alerter.on_error(
                "LLMLiveness",
                f"LLM failed {self._consecutive_failures} consecutive cycles",
            )
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `json.loads()` for LLM output | `beta.chat.completions.parse()` with Pydantic | OpenAI SDK 1.40+ (Aug 2024) | Eliminates parse errors from markdown fences, extra text |
| `threading.Lock` in mixed async/sync | `asyncio.Lock` or restructured boundary | Python 3.10+ best practice | Prevents event loop blocking |

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Anthropic SDK 0.83.0 lacks an equivalent to OpenAI's `beta.chat.completions.parse()` | Pattern 1 | Would need different structured output approach for Anthropic provider |
| A2 | OpenRouter supports `response_format` for structured outputs on supported models | Pitfall 1 | May need fallback to text + model_validate_json for OpenRouter |
| A3 | Restructuring _process_news_article to return impacts (approach 2) is simpler than dual locks | Pattern 3 | May need asyncio.Lock approach instead |

## Open Questions (RESOLVED)

1. RESOLVED: **OpenRouter structured output support** -- The `LLMClient.parse_structured()` ABC already exists in the codebase with a default implementation that does `complete() + model_validate_json()`. OpenAI/OpenRouter subclasses can override with `beta.chat.completions.parse()` when supported, falling back to the base implementation automatically. Plan 49-01 uses `parse_structured()` which handles this transparently.

2. RESOLVED: **Ticker extraction scope** -- Tickers are extracted by `NewsImpactAnalyzer` via `NewsImpactResult.direct_tickers`, not by `NewsAnalyzer`/`SentimentResult`. Plan 49-02 Task 1 validates `result.direct_tickers` against `InstrumentRegistry` after impact analysis, keeping ticker extraction separate from sentiment analysis.

3. RESOLVED: **Credibility column location** -- Add nullable `credibility` column to `SentimentScoreModel` via Alembic migration 006 (not 004 as originally noted; 005 already exists for sandbox_metrics). Credibility flows from `get_credibility(article.source)` through `_persist_sentiment_batch_async()` (line ~2586 of trading_loop.py) into DB rows.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest + pytest-asyncio |
| Config file | pyproject.toml `[tool.pytest.ini_options]` |
| Quick run command | `uv run pytest tests/unit/test_news_analyzer.py -x` |
| Full suite command | `uv run pytest tests/ -x --timeout=30` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| NEWS-01 | 5s per-article LLM timeout | unit | `uv run pytest tests/unit/test_news_analyzer.py::test_llm_timeout -x` | Wave 0 |
| NEWS-02 | parse_structured returns SentimentResult | unit | `uv run pytest tests/unit/test_news_analyzer.py::test_parse_structured -x` | Wave 0 |
| NEWS-03 | Credibility from source map in DB rows | unit | `uv run pytest tests/unit/test_news_pipeline.py::test_credibility_map -x` | Wave 0 |
| NEWS-04 | Invalid ticker rejected with structured log | unit | `uv run pytest tests/unit/test_news_pipeline.py::test_ticker_validation -x` | Wave 0 |
| NEWS-05 | 3 consecutive LLM failures -> alert + counter | unit | `uv run pytest tests/unit/test_news_pipeline.py::test_llm_liveness -x` | Wave 0 |
| NEWS-06 | Budget cap 20 articles + metric logged | unit | `uv run pytest tests/unit/test_news_pipeline.py::test_budget_cap -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/unit/test_news_analyzer.py tests/unit/test_news_pipeline.py -x`
- **Per wave merge:** `uv run pytest tests/ -x`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/test_news_pipeline.py` -- covers NEWS-03, NEWS-04, NEWS-05, NEWS-06 (new file)
- [ ] Update `tests/unit/test_news_analyzer.py` -- covers NEWS-01, NEWS-02 (update existing)

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | -- |
| V3 Session Management | no | -- |
| V4 Access Control | no | -- |
| V5 Input Validation | yes | Pydantic v2 validation on SentimentResult; ticker whitelist via InstrumentRegistry |
| V6 Cryptography | no | -- |

### Known Threat Patterns for News Pipeline

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| LLM prompt injection via news content | Tampering | Output validated via Pydantic schema; tickers whitelisted against InstrumentRegistry |
| LLM hallucinated tickers | Information Disclosure | InstrumentRegistry validation rejects unknown symbols with structured log |
| Cost explosion from unbounded LLM calls | Denial of Service | 20 article budget cap + 5s per-article timeout |

## Sources

### Primary (HIGH confidence)
- Codebase files: news_analyzer.py, llm_client.py, trading_loop.py, models.py, schemas.py, metrics.py, alerts.py, instruments.py -- all verified by direct reading
- OpenAI SDK version 2.21.0 [VERIFIED: uv pip show openai]
- Anthropic SDK version 0.83.0 [VERIFIED: uv pip show anthropic]
- pyproject.toml: openai>=1.50.0, anthropic>=0.42.0 [VERIFIED: codebase]

### Secondary (MEDIUM confidence)
- [OpenAI Structured Outputs docs](https://platform.openai.com/docs/guides/structured-outputs) -- beta.chat.completions.parse() API

### Tertiary (LOW confidence)
- OpenRouter structured output compatibility -- not verified, flagged in Assumptions Log

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - all packages already installed and verified
- Architecture: HIGH - all affected files read and understood
- Pitfalls: MEDIUM - OpenRouter structured output support unverified

**Research date:** 2026-04-14
**Valid until:** 2026-05-14 (stable domain, no fast-moving dependencies)

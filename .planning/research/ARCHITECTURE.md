# Architecture Research

**Domain:** Runtime LLM agent integration — MOEX trading system
**Researched:** 2026-04-14
**Confidence:** HIGH (all findings from direct codebase inspection)

## Existing Architecture Summary

The system uses a layered dependency model (L0→L6) enforced at import time. The
`TradingLoop` (L5, `orchestration/trading_loop.py`) is the central orchestrator. It owns
an `APScheduler` (BackgroundScheduler) with three named executor pools: `default`
(max_workers=4), `news` (max_workers=1), and `retrain` (max_workers=1).

Currently scheduled jobs and their executors:

| Job | Trigger | Executor | Notes |
|-----|---------|----------|-------|
| `_news_cycle` | interval (news_poll_interval_minutes) | `news` | RSS sync + Telegram async via `_run_async` |
| `_strategy_cycle` | interval (strategy_cycle_minutes) | `default` | reads `_sentiment_cache` |
| `_daily_reset` | cron (daily_reset_hour_utc) | `default` | resets CBs, sends P&L summary |
| `_weekly_digest` | cron (Sunday, weekly_digest_hour_utc) | `default` | weekly P&L Telegram message |
| `_retrain_cycle` | interval (168h default) | `retrain` | optional, ML only |
| `_fx_update_cycle` | interval (60min) | `default` | optional, FX service only |
| `_bond_cycle` | cron (07:30 UTC) | `default` | optional, bond processor only |
| `_macro_refresh` | cron (07:00 UTC) | `default` | optional, bond processor only |

### Async bridge pattern

`_run_async(coro, timeout=30)` bridges sync APScheduler threads → the persistent
`_async_loop` (HTTP/DB/Telegram). `_run_grpc(coro, timeout=30)` targets the separate
`_grpc_loop` (Tinkoff gRPC). Using `_run_async` from inside `_async_loop` deadlocks —
async paths call async methods directly (see `_apply_impact_result` pattern).

### Sentiment pipeline (current)

```
RssNewsFetcher.fetch_news()          [L2, sync]
TelegramChannelReader.fetch_*()      [L2, async via _run_async]
    ↓ list[NewsArticle]
_analyze_impact_batch()              [async, on _async_loop]
    ↓ per article
NewsImpactAnalyzer.analyze()         [L3, async LLM call]
    ↓ NewsImpactResult
_apply_impact_result()               [async, on _async_loop]
    ↓ writes
_sentiment_cache: dict[(seg, ticker), (score, monotonic_ts)]   [threading.Lock]
Redis cache (RedisCache.set_sentiment)
DB: sentiment_scores (SentimentScoreModel) via fire-and-forget
```

### Anomaly detection (current)

```
_strategy_cycle_impl() builds CycleMetrics
    ↓
SandboxMonitorService.on_cycle_complete(metrics)
    ↓
AnomalyDetector.check(metrics)  → triggered: list[str]
    ↓ if triggered
AnomalyDetector._fire_alert()
    ↓
TelegramAlerter.send_alert(msg, priority=CRITICAL)
```

`AnomalyDetector` outputs raw metric strings ("drawdown", "fill_rate", "slippage") with
numeric values and z-scores. No LLM explanation exists today.

---

## Integration Points for the 4 New Agents

### Agent 1: News Pipeline (RSS/Telegram → EventDrivenStrategy)

**Current state:** Pipeline structure is complete (RssNewsFetcher, TelegramChannelReader,
NewsImpactAnalyzer, _sentiment_cache). `event_driven` strategy has a `credibility`
parameter accepted in `generate_signal()`. The pipeline already calls it with
`credibility=1.0` unconditionally — no source credibility is computed per-source.
T-Pulse (T-Invest's in-app news) is **not available** in the gRPC SDK (no news
service exists in `services.py`).

**Integration required:**

1. **Source credibility cap (new logic, no new files):** `_news_cycle` currently passes
   `credibility=1.0` when dispatching to `_apply_impact_result`. Add a per-source
   credibility map (e.g. `{"rss": 0.8, "telegram": 0.7}`) and propagate it through
   `_apply_impact_result` → `EventDrivenStrategy.generate_signal()`. This is a
   TradingLoop modification only; EventDrivenStrategy already accepts the parameter.

2. **Latency SLA gate (new constant, TradingLoop):** Add `_NEWS_LATENCY_SLA_SECONDS`
   constant (e.g. 300s = 5min). In `_news_cycle`, after fetching articles, skip any
   article where `now - article.published_at > SLA`. This prevents stale news from
   reaching the strategy cycle with sentiment decay not yet applied.

3. **event_driven preset activation:** The `enabled: false` flag in all `ru_*`
   preset YAMLs must be flipped to `true`. No code change — YAML edit only.
   Clears `_event_driven_active` cache on restart.

**Data flow change:**
```
NewsArticle.source → credibility_map[source]
    ↓
_apply_impact_result(result, source_credibility)
    ↓
_sentiment_cache[(seg, ticker)] = weighted score
    ↓ (already wired)
_strategy_cycle → _get_sentiment() → EventDrivenStrategy.generate_signal(credibility=X)
```

**New components:** None. Modifications only in `trading_loop.py` and preset YAMLs.

---

### Agent 2: Portfolio Review Agent

**Current state:** `_daily_reset` already fetches portfolio state for every market via
`broker.get_portfolio()` and sends a structured Telegram message (`on_daily_summary`).
No LLM analysis exists. `LLMClient` is available (L3) and `parse_structured()` is
already implemented for typed Pydantic output.

**Integration required:**

1. **New file: `src/finalayze/analysis/portfolio_review_agent.py` (L3)**
   - Class `PortfolioReviewAgent(llm_client: LLMClient)`
   - `async def review(portfolio: PortfolioState, market_context: dict) -> PortfolioReviewResult`
   - `PortfolioReviewResult(BaseModel)`: Pydantic model with fields like
     `summary: str`, `risk_flags: list[str]`, `recommendations: list[str]`,
     `overall_sentiment: str` — advisory only, no trade directives
   - Uses `parse_structured()` with a new `prompts/portfolio_review_ru.txt` system prompt
   - Must be **advisory only**: the prompt must explicitly forbid trade directives

2. **TradingLoop modification:**
   - Add `portfolio_review_agent: PortfolioReviewAgent | None = None` constructor param
   - Add `_portfolio_review_cycle()` method (sync, calls `_run_async`)
   - Register as daily cron job in `start()`, separate from `_daily_reset` to avoid
     blocking it: `scheduler.add_job(_portfolio_review_cycle, "cron", hour=daily_reset_hour_utc+1)`
   - Result delivered via `self._alerter.send_alert(formatted_review, priority=AlertPriority.INFO)`

**Data flow:**
```
_portfolio_review_cycle() [cron, "default" executor]
    ↓ _run_grpc → broker.get_portfolio() per market
    ↓ builds portfolio summary dict
    ↓ _run_async → PortfolioReviewAgent.review(portfolio, context)
    ↓ LLMClient.parse_structured() → PortfolioReviewResult
    ↓ TelegramAlerter.send_alert(formatted, INFO)
    ↓ (optional) _persist_to_db(portfolio_review_record)
```

**New components:**
- `src/finalayze/analysis/portfolio_review_agent.py` (L3, new file)
- `src/finalayze/analysis/prompts/portfolio_review_ru.txt` (new prompt)
- `PortfolioReviewResult` Pydantic schema (can live in `portfolio_review_agent.py` or `core/schemas.py`)

**Layer boundary:** L3 (analysis) is the correct layer — LLMClient already lives there,
PortfolioReviewAgent imports LLMClient (L3) and PortfolioState (L0 schema). TradingLoop
(L5) imports it as a constructor injection (TYPE_CHECKING guard for the forward reference).

---

### Agent 3: Anomaly Interpreter Agent

**Current state:** `AnomalyDetector._fire_alert()` calls `TelegramAlerter.send_alert(msg)`
with a raw string like `"Sandbox anomaly: drawdown = 0.0523 (z-score: 2.41, threshold: 2.0σ)"`.
No context or explanation is provided.

**Integration required:**

1. **New file: `src/finalayze/analysis/anomaly_interpreter.py` (L3)**
   - Class `AnomalyInterpreter(llm_client: LLMClient)`
   - `async def interpret(metric: str, value: float, z_score: float | None, context: dict) -> str`
   - Returns a plain string explanation (1-2 sentences) — not a structured Pydantic model
     since this is fire-and-forget, not ML-consumable data
   - Uses `LLMClient.complete()` with a short `prompts/anomaly_interpret_ru.txt` system prompt
   - Must have its own LLM call timeout (short, 15s) and silent fallback: if LLM fails,
     the original raw alert still goes out

2. **`AnomalyDetector` modification:**
   - Add `anomaly_interpreter: AnomalyInterpreter | None = None` constructor param
   - In `_fire_alert()`: if interpreter is set, schedule fire-and-forget async interpretation
   - **Layer check:** `AnomalyDetector` is in `monitoring/` (L6). `AnomalyInterpreter` would be
     in `analysis/` (L3). L6 importing L3 is allowed per the layer rules (monitoring/ CLAUDE.md:
     "may import from L0-L5 and L6 (api/)"). Use `TYPE_CHECKING` guard for the import.
   - The fire-and-forget call must not block `_fire_alert()`. Use
     `asyncio.run_coroutine_threadsafe(coro, _async_loop)` where `_async_loop` is passed
     as a constructor param (the TradingLoop's `_async_loop`).

3. **`SandboxMonitorService` modification:**
   - Pass the `_async_loop` from TradingLoop into `AnomalyDetector` via `SandboxMonitorService`
   - Current: `SandboxMonitorService.__init__` creates `AnomalyDetector` internally with
     no loop reference. Add `async_loop: asyncio.AbstractEventLoop | None = None` param.

**Data flow:**
```
AnomalyDetector._fire_alert(metric, value, threshold, z_score)
    ↓ (existing path, unchanged)
TelegramAlerter.send_alert(raw_msg, CRITICAL)
    ↓ (new async path, fire-and-forget)
asyncio.run_coroutine_threadsafe(
    AnomalyInterpreter.interpret(metric, value, z_score, context),
    _async_loop
)
    ↓ LLMClient.complete() with short timeout
    ↓ on success
TelegramAlerter.send_alert(explanation, INFO)   ← follow-up message
```

**New components:**
- `src/finalayze/analysis/anomaly_interpreter.py` (L3, new file)
- `src/finalayze/analysis/prompts/anomaly_interpret_ru.txt` (new prompt)

---

### Agent 4: Sentiment DB Persistence and Rolling Aggregation

**Current state:** `_persist_sentiment_batch_async` already writes per-ticker rows to
`sentiment_scores` (symbol, market_id, timestamp, news_sentiment, composite_sentiment,
confidence). The table is a TimescaleDB hypertable. No rolling aggregation exists.

**Integration required:**

1. **Rolling aggregation — TimescaleDB continuous aggregate (migration only):**
   Create a Continuous Aggregate View:
   ```sql
   CREATE MATERIALIZED VIEW sentiment_7d_avg
   WITH (timescaledb.continuous) AS
   SELECT
     time_bucket('1 day', timestamp) AS day,
     symbol,
     market_id,
     AVG(composite_sentiment) AS avg_sentiment_7d,
     AVG(confidence) AS avg_confidence_7d,
     COUNT(*) AS sample_count
   FROM sentiment_scores
   GROUP BY 1, 2, 3;
   ```
   Add `REFRESH POLICY` for automatic refresh. This lives in an Alembic migration.
   No Python code change required for the write path.

2. **ML feature accessor — new method in L2 data layer:**
   New file: `src/finalayze/data/sentiment_store.py` (L2)
   - Class `SentimentStore(session_factory: async_sessionmaker)`
   - `async def get_rolling_sentiment(symbol, market_id, window_days=7) -> float | None`
   - Queries `sentiment_7d_avg` materialized view or falls back to raw `sentiment_scores`
     with a date-range filter
   - This is a read-only accessor for future ML feature extraction in `features/technical.py`

3. **No change to write path:** The `_persist_sentiment_batch_async` path in TradingLoop
   already writes the data correctly. The rolling aggregation is purely a DB-side concern.

**Data flow (read path for future ML):**
```
auto_ml_research.py / features/technical.py
    ↓ calls
SentimentStore.get_rolling_sentiment(symbol, market_id)
    ↓ queries
TimescaleDB: sentiment_7d_avg (continuous aggregate)
    ↓ returns
float: rolling 7-day average sentiment score
```

**New components:**
- `src/finalayze/data/sentiment_store.py` (L2, new file)
- Alembic migration: `sentiment_7d_avg` continuous aggregate + refresh policy

---

## Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  L6: API / Monitoring                                                        │
│  ┌──────────────────┐   ┌──────────────────────────────────────────────┐   │
│  │  TelegramAlerter  │   │  SandboxMonitorService (MODIFIED)             │   │
│  │  (existing)       │   │  + async_loop param                          │   │
│  └────────┬──────────┘   │  ┌──────────────────────────────────────┐   │   │
│           │               │  │ AnomalyDetector (MODIFIED)            │   │   │
│           │               │  │ + anomaly_interpreter param          │   │   │
│           │               │  │ + async_loop param                   │   │   │
│           │               │  └──────────────────────────────────────┘   │   │
│           │               └──────────────────────────────────────────────┘   │
└───────────┼──────────────────────────────────────────────────────────────────┘
            │
┌───────────┼──────────────────────────────────────────────────────────────────┐
│  L5: Orchestration                                                            │
│  ┌────────▼──────────────────────────────────────────────────────────────┐  │
│  │  TradingLoop (MODIFIED)                                                │  │
│  │  + portfolio_review_agent: PortfolioReviewAgent | None                │  │
│  │  + _portfolio_review_cycle() [new daily cron job]                     │  │
│  │  + source credibility map in _news_cycle                              │  │
│  │  + latency SLA gate in _news_cycle                                    │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
            │
┌───────────┼──────────────────────────────────────────────────────────────────┐
│  L3: Analysis                                                                  │
│  ┌────────▼──────────────────┐  ┌──────────────────────────────────────┐    │
│  │  NewsImpactAnalyzer       │  │  PortfolioReviewAgent  (NEW)          │    │
│  │  (existing)               │  │  + review() -> PortfolioReviewResult │    │
│  └───────────────────────────┘  └──────────────────────────────────────┘    │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │  AnomalyInterpreter  (NEW)                                            │    │
│  │  + interpret(metric, value, z_score) -> str                          │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │  LLMClient (existing, shared by all three new agents)                │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘
            │
┌───────────┼──────────────────────────────────────────────────────────────────┐
│  L2: Data                                                                      │
│  ┌────────▼───────────────────┐  ┌────────────────────────────────────┐     │
│  │  RssNewsFetcher (existing)  │  │  SentimentStore  (NEW)             │     │
│  │  TelegramChannelReader      │  │  + get_rolling_sentiment()         │     │
│  │  (existing)                 │  └────────────────────────────────────┘     │
│  └─────────────────────────────┘                                              │
└──────────────────────────────────────────────────────────────────────────────┘
            │
┌───────────┼──────────────────────────────────────────────────────────────────┐
│  L0: Core                                                                      │
│  ┌────────▼──────────────────────────────────────────────────────────────┐   │
│  │  PortfolioState, NewsArticle, SentimentScoreModel (existing)          │   │
│  │  PortfolioReviewResult  (NEW Pydantic model, frozen)                  │   │
│  └───────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
            │
┌───────────┼──────────────────────────────────────────────────────────────────┐
│  DB / Infra                                                                    │
│  ┌────────▼──────────────────────────────────────────────────────────────┐   │
│  │  sentiment_scores hypertable (existing)                               │   │
│  │  sentiment_7d_avg continuous aggregate  (NEW — Alembic migration)     │   │
│  └───────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## New vs Modified Components

### New files

| File | Layer | Purpose |
|------|-------|---------|
| `src/finalayze/analysis/portfolio_review_agent.py` | L3 | PortfolioReviewAgent class + PortfolioReviewResult schema |
| `src/finalayze/analysis/anomaly_interpreter.py` | L3 | AnomalyInterpreter class |
| `src/finalayze/analysis/prompts/portfolio_review_ru.txt` | L3 | LLM system prompt for portfolio review |
| `src/finalayze/analysis/prompts/anomaly_interpret_ru.txt` | L3 | LLM system prompt for anomaly explanation |
| `src/finalayze/data/sentiment_store.py` | L2 | SentimentStore rolling aggregation reader |
| `alembic/versions/XXX_sentiment_7d_avg.py` | Infra | TimescaleDB continuous aggregate migration |

### Modified files

| File | Change type | Details |
|------|-------------|---------|
| `src/finalayze/orchestration/trading_loop.py` | Additive | New constructor param, new cron method, credibility map, SLA gate |
| `src/finalayze/monitoring/anomaly_detector.py` | Additive | New optional params, fire-and-forget in `_fire_alert` |
| `src/finalayze/monitoring/sandbox_monitor.py` | Additive | Thread `async_loop` param through to `AnomalyDetector` |
| `src/finalayze/core/schemas.py` | Additive | Add `PortfolioReviewResult` Pydantic model |
| `src/finalayze/strategies/presets/ru_*.yaml` | Config | Set `event_driven.enabled: true` |

---

## Suggested Build Order

Build in dependency order, not feature order. Each step is independently testable.

### Step 1: Anomaly Interpreter (isolated, no TradingLoop changes)

- Create `anomaly_interpreter.py` and prompt file
- Modify `AnomalyDetector` to accept optional `anomaly_interpreter` and `async_loop`
- Modify `SandboxMonitorService` to accept and thread `async_loop`
- Tests: mock LLMClient, verify fallback when LLM fails, verify raw alert fires even if
  interpreter raises, verify `run_coroutine_threadsafe` scheduling pattern

**Why first:** Self-contained L3→L6 addition. Does not touch the critical-path trading
loop. Validates the fire-and-forget async pattern before applying it elsewhere.

### Step 2: Portfolio Review Agent (new cron, isolated from news pipeline)

- Add `PortfolioReviewResult` to `core/schemas.py`
- Create `portfolio_review_agent.py` and prompt file
- Add `_portfolio_review_cycle()` to TradingLoop with new constructor param
- Tests: mock `broker.get_portfolio()`, mock LLMClient, verify advisory-only output,
  verify it schedules after `_daily_reset`, not concurrent with it

**Why second:** No dependency on news pipeline. All dependencies (LLMClient, PortfolioState,
TelegramAlerter) already exist. The cron job pattern is well-established in TradingLoop.

### Step 3: News Pipeline activation (credibility cap + latency SLA + preset flip)

- Add `_SOURCE_CREDIBILITY` constant dict to TradingLoop
- Add `_NEWS_LATENCY_SLA_SECONDS` constant and filter in `_news_cycle`
- Propagate source credibility through `_apply_impact_result` → `EventDrivenStrategy`
- Flip `enabled: true` in `ru_*` preset YAMLs
- Tests: credibility scaling math, SLA gate (articles older than SLA are skipped),
  integration test that `event_driven` strategy receives `credibility < 1.0` for
  telegram-sourced articles

**Why third:** Requires deep familiarity with `_news_cycle` internals. Steps 1 and 2
already force reading this code. The changes are additive guards, not structural changes.

### Step 4: Sentiment DB persistence + rolling aggregation

- Write Alembic migration for `sentiment_7d_avg` continuous aggregate + refresh policy
- Create `SentimentStore` class
- Tests: integration test against TimescaleDB (Docker Compose)

**Why last:** The write path already works. Only the read accessor and DB migration are
new. No trading-critical path is changed. Can be deferred without blocking any other step.

---

## Critical Design Constraints

### Do not call `_run_async` from `_async_loop` (deadlock risk)

`_apply_impact_result` already demonstrates the correct pattern: it is called from within
`_async_loop` and uses `await` directly, not `_run_async`. The same applies to any new
async path that runs on `_async_loop`. The Portfolio Review Agent must use `_run_async`
from the APScheduler thread, not from inside an async function already on `_async_loop`.

### Portfolio Review Agent: advisory only, no trade directives

Expert debate unanimously rejected the Pre-Trade Reasoning Agent because LLM modifiers
in the sizing pipeline break determinism, calibration, and backtestability. The
`PortfolioReviewResult` schema must not contain fields that TradingLoop can interpret as
signals. The prompt must explicitly state "do not recommend specific buy/sell actions."
The result is delivered only via Telegram, not wired into any strategy or sizing pipeline.

### AnomalyInterpreter must not delay the raw alert

`AnomalyDetector._fire_alert()` is called from an APScheduler thread synchronously.
The raw alert must fire synchronously (existing path, unchanged). LLM interpretation
fires asynchronously via `run_coroutine_threadsafe`. If `_async_loop` is None (e.g.,
during TradingLoop startup before `_run_async` has been called), the interpreter silently
skips — the raw alert has already been sent.

### T-Pulse is not available in the current T-Invest gRPC SDK

The `t_tech.invest` SDK (`services.py`) exposes exactly 10 gRPC services: Instruments,
MarketData, Operations, Orders, Sandbox, StopOrders, Users, MarketDataStream,
OperationsStream, OrdersStream, and Signals. No news or pulse service exists. T-Pulse is
a T-Bank mobile app feature not exposed via the investment API. The news pipeline must
continue using RSS (feedparser) and Telegram (Telethon) as the only available sources.
No new data source adapter is needed for this milestone.

### Sentiment rolling aggregation: TimescaleDB continuous aggregate, not Python-side

Computing rolling averages in Python from raw rows would be slow and unscalable as data
grows. The `sentiment_scores` table is already a TimescaleDB hypertable. The correct
approach is a continuous aggregate (materialized, auto-refreshed by TimescaleDB). The
`SentimentStore` reader queries the materialized view, not the raw table, keeping the
query latency at O(1) regardless of historical data volume.

---

## Sources

- Direct inspection: `src/finalayze/orchestration/trading_loop.py` (full file, 2062+ LOC)
- Direct inspection: `src/finalayze/monitoring/anomaly_detector.py`
- Direct inspection: `src/finalayze/monitoring/sandbox_monitor.py`
- Direct inspection: `src/finalayze/monitoring/CLAUDE.md`
- Direct inspection: `src/finalayze/analysis/news_impact_analyzer.py`
- Direct inspection: `src/finalayze/analysis/llm_client.py`
- Direct inspection: `src/finalayze/analysis/CLAUDE.md`
- Direct inspection: `src/finalayze/strategies/event_driven.py`
- Direct inspection: `src/finalayze/core/models.py` (SentimentScoreModel, NewsArticleModel)
- Direct inspection: `src/finalayze/core/schemas.py` (PortfolioState)
- Direct inspection: `src/finalayze/core/CLAUDE.md`
- Direct inspection: `src/finalayze/data/fetchers/tinkoff_data.py`
- Direct inspection: `src/finalayze/data/CLAUDE.md`
- Direct inspection: `.venv/lib/.../t_tech/invest/services.py` (SDK services — no news service)
- T-Bank Dev Portal: https://developer.tbank.ru/invest/api (confirmed 10 gRPC services, no news/pulse)

---

*Architecture research for: v10.0 Runtime LLM Trading Agents — MOEX trading system*
*Researched: 2026-04-14*

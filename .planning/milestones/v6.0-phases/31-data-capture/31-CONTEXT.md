# Phase 31: Data Capture - Context

**Gathered:** 2026-03-31
**Status:** Ready for planning

<domain>
## Phase Boundary

Wire DB persistence for orders, signals, news articles, and sentiment scores. All 4 tables exist with migrations — the gap is purely wiring in the trading loop. Writes must be fire-and-forget.

</domain>

<decisions>
## Implementation Decisions

### DB Persistence Pattern

- All DB writes are fire-and-forget — wrap in try/except, log warning on failure, never crash trading loop
- DB write failures must NOT increment _consecutive_errors counter (from v4.0 decision)
- Use existing async_sessionmaker from TradingLoop._session_factory
- Write happens via _run_async() on _async_loop (not _grpc_loop)
- Each persist call creates its own async session, commits, and closes

### Order Persistence (PERSIST-01)

- After order fill in _strategy_cycle, persist to `orders` table
- Fields: symbol, side, quantity, fill_price, order_id, timestamp, strategy_name, market_id
- Use existing OrderModel from core/models.py

### Signal Persistence (PERSIST-02)

- After signal generation in _strategy_cycle, persist to `signals` table
- Fields: symbol, direction, confidence, strategy, reasoning, timestamp, segment_id
- Use existing SignalModel from core/models.py

### News Article Persistence (PERSIST-03)

- After article processing in _analyze_impact_batch, persist to `news_articles` table
- Fields: title, source, url, published_at, content_hash, processed_at
- Use existing NewsArticleModel from core/models.py

### Sentiment Score Persistence (PERSIST-04)

- After sentiment analysis in _analyze_impact_batch, persist to `sentiment_scores` table
- Fields: ticker, score, source, timestamp, article_id (if available)
- Use existing SentimentScoreModel from core/models.py

### Fire-and-Forget Safety (PERSIST-05)

- Helper method _persist_to_db(coro) that wraps any async persist call
- Catches all exceptions, logs "db_persist_failed" with table name and error
- Never re-raises, never affects trading loop flow

### Claude's Discretion

- Whether to batch inserts (bulk) or insert one-by-one
- Exact field mapping from runtime objects to ORM models
- Whether to add a flush/commit after each insert or batch per cycle

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/finalayze/core/models.py` — OrderModel, SignalModel, NewsArticleModel, SentimentScoreModel
- `src/finalayze/orchestration/trading_loop.py` — _session_factory, _run_async, _async_loop
- `alembic/versions/` — migrations already created these tables

### Established Patterns
- _persist_equity_snapshots() in trading_loop.py — existing fire-and-forget DB write pattern
- async_sessionmaker usage in _load_baseline_async()

### Integration Points
- _strategy_cycle() — after order execution, persist order + signals
- _analyze_impact_batch() — after article analysis, persist articles + sentiment

</code_context>

<specifics>
## Specific Ideas

From sandbox analysis: all 4 tables had 0 rows after 5 days of operation.
The ORM models and DB tables exist — only the write calls are missing.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

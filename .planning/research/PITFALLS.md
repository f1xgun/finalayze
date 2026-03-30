# Domain Pitfalls: Sandbox Stability & Observability (v6.0)

**Domain:** Adding stability fixes (gRPC isolation, DB persistence, log pipeline, ticker updates) to a live MOEX trading system
**Researched:** 2026-03-30
**Confidence:** HIGH (codebase analysis + production sandbox validation logs)

---

## Critical Pitfalls

Mistakes that crash the trading loop, cause duplicate orders, or lose trade data.

### Pitfall 1: gRPC Event Loop Isolation Breaks Existing Broker Calls

**What goes wrong:**
The fix for BlockingIOError involves isolating gRPC calls to a dedicated event loop thread (already partially done via `TinkoffBroker._run_async()` and `TradingLoop._run_async()`). The system currently has TWO independent background event loops: one in `TinkoffBroker` (line 97-98, `self._loop`) and one in `TradingLoop` (line 219, `self._async_loop`). If the gRPC isolation fix creates a third loop, or if it modifies the existing loops to use `asyncio.to_thread()` or `run_in_executor()` instead, the `_grpc_lock` (asyncio.Lock on TradingLoop's loop, line 223) stops protecting concurrent gRPC calls because the lock lives on a different loop than the calls.

**Why it happens:**
The `asyncio.Lock` in TradingLoop (`self._grpc_lock`, line 223) is bound to TradingLoop's `_async_loop`. If gRPC calls move to TinkoffBroker's separate `_loop`, the lock cannot serialize them -- `asyncio.Lock` only works within a single event loop. The equity cycle and bond cycle currently rely on this lock to prevent overlapping gRPC calls (the T-Bank SDK reuses a single gRPC channel that is not thread-safe for concurrent requests).

**Consequences:**
- Race condition: equity cycle and bond cycle send overlapping gRPC requests on the same channel, causing `StreamStreamCall` corruption or `StatusCode.INTERNAL` errors.
- Potential duplicate orders: if `post_order` is called twice because the first call's response was corrupted by a concurrent `get_portfolio` call.
- Silent data corruption: portfolio equity reads interleaved with order submissions return stale data, causing circuit breaker miscalculation.

**Prevention:**
- Use a single background event loop for ALL async operations (merge TinkoffBroker's `_loop` into TradingLoop's `_async_loop`, or vice versa). The `_grpc_lock` must live on the same loop that executes gRPC calls.
- If isolation requires a separate thread for gRPC, use a `threading.Lock` (not `asyncio.Lock`) to serialize access from APScheduler threads.
- Test: submit_order and get_portfolio concurrently from two threads; verify no `StatusCode.INTERNAL` or `StreamStreamCall` errors.

**Detection:**
- Log `grpc_call_start` / `grpc_call_end` with thread ID and loop ID.
- Monitor for `StatusCode.INTERNAL` or `StatusCode.UNAVAILABLE` errors that appear only under concurrent load.
- Watch for strategy cycles that take >5 minutes (sign of lock contention or deadlock).

---

### Pitfall 2: Channel Reconnection Causes Duplicate Orders

**What goes wrong:**
`TinkoffBroker.reconnect_client()` (line 445) calls `self.close()` which nulls out `_client`, `_services`, `_loop`, and `_loop_thread`. During the window between `close()` and successful reconnection, any concurrent call to `submit_order()` or `get_portfolio()` will fail. If the strategy cycle is running when reconnection triggers (from `_attempt_grpc_reconnect`), in-flight orders may appear to fail (timeout in `_run_async` at line 173, timeout=30s), causing the calling code to believe the order was not placed. But the gRPC server may have received and executed the order before the channel dropped.

**Why it happens:**
The `reconnect_client()` method holds `_client_lock` (threading.Lock) during the reconnect, but `_run_async()` does not acquire `_client_lock` before scheduling coroutines. So a reconnect can happen while `_run_async()` has already scheduled a coroutine on the old loop that is about to be stopped. The future.result(timeout=30) call will raise `TimeoutError` or `concurrent.futures.CancelledError`, but the gRPC server may have already processed the request.

**Consequences:**
- Order is executed on the exchange but the system believes it failed. No position tracking, no stop-loss, no risk management for that position.
- On next cycle, the system may generate the same BUY signal and submit a duplicate order (doubling the position).
- The `_reconcile_inflight_orders()` method (line 335) only runs at startup, not after reconnection, so the orphaned order is not detected until the next restart.

**Prevention:**
- After every successful reconnect, call `_reconcile_inflight_orders()` to detect any orders that were filled during the reconnection window.
- Add a "reconnecting" flag that `_strategy_cycle_impl` checks before submitting orders. If reconnecting, skip the cycle entirely.
- Store every `post_order` call's parameters in a local journal (in-memory dict keyed by idempotency key) BEFORE calling the gRPC API. After reconnect, compare journal against `get_orders()` response.
- The T-Bank API supports idempotency via `order_id` parameter -- use a UUID generated client-side and pass it to `post_order()`.

**Detection:**
- Compare `get_positions()` after reconnect against internal position tracking (`_entry_prices`, `_stop_states`).
- Alert if positions exist in broker that are not tracked internally.

---

### Pitfall 3: DB Persistence Crashes the Trading Loop on Connection Failure

**What goes wrong:**
Adding DB writes for orders, signals, and news articles introduces a new failure mode in `_strategy_cycle_impl()` and `_news_cycle()`. If the DB write is done synchronously in the cycle (e.g., `await session.execute(insert(...))` bridged via `_run_async()`), a PostgreSQL connection timeout or pool exhaustion will raise an exception that propagates up and kills the current cycle. Worse: if the DB write happens AFTER the order is submitted to the broker but BEFORE the cycle completes, the order is executed but not persisted -- creating an invisible position.

**Why it happens:**
The current code uses broad `except Exception` in `_strategy_cycle` (line 1102) which catches DB errors, but the `_consecutive_equity_errors` counter triggers a CRITICAL alert after 3 failures. If the DB is temporarily unreachable (common with TimescaleDB vacuum, WAL replay, or container restart), the system will alert and potentially halt trading after 3 cycles (3 hours at 60-min intervals) -- even though the trading logic itself is fine.

**Consequences:**
- Trading halts because the DB persistence layer (a non-critical path) fails 3 times in a row.
- Orders executed but not persisted create "ghost positions" -- the system's internal state diverges from the audit trail.
- If the DB write is placed inside a transaction that also reads portfolio state, a DB lock timeout could delay the entire strategy cycle beyond the APScheduler misfire grace time (300s), causing the next cycle to be skipped.

**Prevention:**
- DB writes MUST be fire-and-forget with exception swallowing: wrap every DB write in its own try/except that logs the failure but never propagates it to the cycle.
- Use a separate async task (not blocking the cycle) for DB persistence. Queue writes to a bounded in-memory buffer; a background coroutine drains the buffer to DB.
- NEVER put DB writes in the critical path between order submission and position tracking.
- Separate the error counter: `_consecutive_equity_errors` should only count errors from trading logic (broker, strategy), not from persistence.

**Detection:**
- Add a `db_write_failures` Prometheus counter. Alert if it exceeds 10 in 1 hour.
- Log `db_persist_failed` at WARNING level (not ERROR, not exception) to avoid noise in Loki.
- Periodic reconciliation job: compare DB `orders` table against broker `get_orders()` response.

---

### Pitfall 4: Ticker Rename Breaks Position Tracking and Stop-Loss State

**What goes wrong:**
Renaming tickers (FIVE->X5, YNDX->YDEX, HHRU->HH) in `DEFAULT_MOEX_INSTRUMENTS` changes the `(symbol, market_id)` key in `InstrumentRegistry`. But the runtime state in `TradingLoop` uses ticker strings as keys: `_stop_states` (line 179), `_entry_prices` (line 205), `_last_prices` (line 235), `_sentiment_cache` (line 170). If the system holds a position in YNDX (tracked by FIGI in broker), but the registry now only knows YDEX, the `has_position("YDEX")` call works (FIGI lookup), but the stop-loss state stored under "YNDX" is orphaned -- the trailing stop disappears.

**Why it happens:**
The `InstrumentRegistry` is keyed by `(symbol, market_id)`. Portfolio positions from `TinkoffBroker.get_portfolio()` are FIGI-keyed (line 396). The mapping between FIGI and symbol goes through `get_by_figi()` which does a linear scan. If the old ticker YNDX is removed from the registry and replaced with YDEX (same FIGI), the `get_by_figi()` call returns the new Instrument with symbol="YDEX". But `_stop_states["YNDX"]` still exists from before the rename -- the trailing stop is never applied to YDEX, and YDEX starts with no stop-loss.

**Consequences:**
- Position held under old ticker loses its trailing stop protection.
- `_entry_prices["YNDX"]` is orphaned -- Kelly P&L computation fails silently (returns 0 P&L for the position).
- If the old ticker is in `_cycle_exited_symbols`, the new ticker is not protected by the re-entry guard.
- Worst case: a position with no stop-loss rides a -15% drawdown before the circuit breaker trips.

**Prevention:**
- When updating tickers, add a migration mapping: `_TICKER_RENAMES = {"YNDX": "YDEX", "FIVE": "X5", "HHRU": "HH"}`. On startup, migrate `_stop_states`, `_entry_prices`, `_sentiment_cache` keys.
- Keep old tickers in the registry as inactive (`is_active=False`) with the same FIGI, so `get_by_figi()` still returns a result during the transition.
- For delisted tickers (FIXP, POLY): if the broker reports a position in a delisted FIGI, log a CRITICAL alert but do NOT auto-sell (delisted instruments cannot be traded).
- Test: hold a position in YNDX, rename to YDEX, verify stop-loss state is migrated.

**Detection:**
- On startup, compare `broker.get_positions()` FIGIs against registry FIGIs. Any FIGI in broker but not in registry triggers a WARNING.
- Log `orphaned_stop_state` when `_stop_states` contains keys not matching any active instrument.

---

## Moderate Pitfalls

### Pitfall 5: Promtail Log Volume Overwhelms Loki

**What goes wrong:**
The current Promtail config (promtail-config.yml) uses `docker_sd_configs` to scrape JSON logs from the `finalayze-sandbox-app` container. It extracts `event` and `level` as labels. Loki uses labels for indexing, and each unique label combination creates a new stream. The trading system logs 50+ unique event types (`order_filled`, `strategy_cycle_summary`, `portfolio_fetched`, `grpc_reconnect_attempt`, etc.). With the `event` label, Loki creates 50+ streams per container. Under load (e.g., news cycle processing 20 articles with `news_article_analyzed` events), the stream count can spike, causing Loki to reject ingestion with `429 Too Many Requests` or `stream limit exceeded`.

**Why it happens:**
The `event` label has HIGH cardinality (50+ values). Loki's default `max_streams_per_user` is 10000, but more critically, `max_label_names_per_series` defaults to 15 and `ingestion_rate_mb` defaults to 4MB. The real issue is that during strategy cycles, the app logs 30-40 entries per cycle (one per instrument), creating burst ingestion that exceeds Loki's default rate limits.

**Prevention:**
- Remove `event` from Promtail labels. Use it only for filtering in LogQL queries (parsed at query time via `| json | event = "order_filled"`).
- Keep only `level` and `container` as labels (low cardinality).
- Add `limits_config` to Loki: `ingestion_rate_mb: 10`, `ingestion_burst_size_mb: 20`, `max_streams_per_user: 5000`.
- Add rate limiting in Promtail: `rate` stage with `rate: 100` (lines per second) to cap burst ingestion.
- The `drop` stage for health/metrics endpoints is correct but uses regex on raw lines -- JSON logs will not match `^INFO:.*` patterns because structlog outputs JSON, not `INFO:` prefixed lines. Fix the drop regex to match JSON format.

**Detection:**
- Check `loki_distributor_lines_received_total` Prometheus metric. If 0, Promtail is not shipping logs.
- Check `promtail_targets_active_total`. If 0, the docker_sd_config is not discovering the container.

---

### Pitfall 6: CBR XML API Blocks the Main Trading Loop

**What goes wrong:**
The `CBRFetcher` (cbr.py) uses synchronous `httpx.Client` with a 30-second timeout and 3 retries with exponential backoff (`1s, 2s, 4s`). In the worst case, a CBR API call takes `3 * 30s + 1s + 2s + 4s = 97 seconds`. If this is called from `_strategy_cycle_impl()` (to get FX rates as a fallback when gRPC fails), it blocks the APScheduler thread for ~97 seconds. With `max_instances=1` on the strategy_cycle job and `misfire_grace_time=300`, the next cycle will be delayed but not skipped. However, during those 97 seconds, no stop-loss checks are running.

**Why it happens:**
The CBR fetcher docstring explicitly warns "Sync only -- do NOT call from async code without asyncio.to_thread()". But the natural temptation when adding FX fallback is to call `cbr_fetcher.fetch_fx_rates()` directly in the strategy cycle (which runs in an APScheduler ThreadPoolExecutor thread). This is technically "sync" code, but it blocks the ONE thread allocated to the strategy_cycle job.

**Consequences:**
- Stop-loss trailing updates are delayed by up to 97 seconds. In volatile MOEX sessions, a 2% move can happen in 2 minutes.
- If the CBR API is down (cbr.ru has occasional outages), the fallback itself becomes a failure point, defeating the purpose of having a fallback.
- The `_macro_refresh` job already calls CBR at 07:00 UTC. If the FX fallback also calls CBR during the strategy cycle, rate limiting may cause the backoff to kick in more often.

**Prevention:**
- Cache the last-known FX rate in memory. The FX fallback should read from cache, not make a live HTTP call during the strategy cycle.
- Run CBR fetch in a separate background task (dedicated APScheduler job, e.g., every 30 minutes). Store result in `_fx_cache: dict[str, Decimal]`.
- The strategy cycle reads `_fx_cache.get("USDRUB")`. If the cache is empty AND gRPC failed, use the hardcoded fallback rate from `_MARKET_CURRENCY` or the last portfolio-reported rate.
- Set CBR timeout to 10s (not 30s) and reduce retries to 2. FX rates change slowly; a stale rate from 1 hour ago is acceptable.

**Detection:**
- Log `cbr_fx_fallback_used` when the cache is read instead of a fresh gRPC rate.
- Monitor `strategy_cycle_summary.duration_ms` -- if it exceeds 120000ms (2 minutes), investigate.

---

### Pitfall 7: Market-Hours Gate Misses Signals at Boundary Minutes

**What goes wrong:**
The MOEX market hours are defined as `_MOEX_OPEN_UTC = (7, 0)` and `_MOEX_CLOSE_UTC = (15, 45)` in trading_loop.py (lines 90-91). A market-hours gate that checks `now < open or now >= close` will skip cycles that start at exactly 07:00 UTC (market open). But MOEX has a pre-open auction from 06:50 to 07:00 UTC and a closing auction from 15:40 to 15:50 UTC. If the strategy cycle is scheduled at 60-minute intervals and the last cycle before close is at 15:00 UTC, the gate will let it through. But by the time the cycle finishes processing 40 instruments (2-3 minutes) and submits a market order at 15:03, the MOEX main session is still open. However, if the gate uses `now.hour >= 15 and now.minute >= 45`, it will block cycles starting at 15:00 even though they should run.

**Why it happens:**
The gate comparison is on cycle START time, but orders are submitted MINUTES later. The cycle could start at 15:30 UTC (within hours), process for 5 minutes, and attempt to submit an order at 15:35 -- still within main session. But a gate that blocks cycles starting after 15:15 (to be safe) will miss the 15:00-15:15 window unnecessarily.

**Consequences:**
- Too aggressive: skipping the 15:00 cycle means no trading in the last 45 minutes of the session, losing potential exit signals.
- Too lenient: submitting orders at 15:43 UTC (during closing auction) gets different execution prices than expected (auction matching, not continuous trading).
- Weekend/holiday edge: if the system starts on a MOEX holiday, the gate blocks all cycles, but the news cycle should still run (news happens on weekends).

**Prevention:**
- Gate should block ORDER SUBMISSION, not cycle execution. Let the cycle generate signals and check hours only at the `submit_order()` call point.
- Use the existing `is_moex_trading_day()` from `moex_calendar.py` for date-level gating. Use hour/minute check only for intraday gating.
- Close the gate 15 minutes before exchange close (15:30 UTC) to avoid auction-period submissions.
- Do NOT gate the news cycle on market hours -- news analysis should run 24/7 and cache sentiment for the next trading session.

**Detection:**
- Log `market_hours_gate_blocked` with the current time and gate thresholds.
- Monitor for order rejections with T-Bank error codes related to "trading session closed" -- these indicate the gate is too lenient.

---

### Pitfall 8: Article Deduplication Hash Collisions or Unbounded Memory

**What goes wrong:**
Adding article deduplication to reduce LLM rate limit fallbacks requires storing previously seen article identifiers. If using `hash(article.title + article.url)` (Python's built-in hash), two risks emerge: (1) Python's `hash()` is not collision-resistant -- different articles can produce the same hash, causing legitimate articles to be skipped. (2) The set of seen hashes grows unboundedly if articles are never evicted, consuming memory proportional to the total number of articles ever processed.

**Why it happens:**
The news cycle runs every `news_poll_interval_minutes` (configurable, likely 15-30 minutes). Each cycle fetches articles from 3+ RSS feeds and Telegram channels. Over a week, this is 1000-3000 articles. A set of hashes is small (8 bytes each), but the real risk is if the dedup key includes the full article text or URL (for collision resistance), growing to 500 bytes per entry. At 3000 articles/week, this is ~1.5MB/week -- manageable, but over months without restart, it grows without bound.

**Consequences:**
- Hash collision: a genuinely new article is skipped because its hash matches a previous article. A critical GAZP dividend announcement is missed, and the event_driven strategy generates no signal.
- Memory leak: if the dedup set stores full article content for comparison (to avoid collisions), memory grows linearly with time. After 3 months, the set contains 40K entries.
- False dedup across feeds: the same news story from RBC, Interfax, and TASS has different URLs and slightly different titles. A URL-based dedup won't catch these (which is actually correct -- the LLM should analyze each source's unique framing). A title-based fuzzy dedup might incorrectly merge them.

**Prevention:**
- Use `hashlib.sha256(article.url.encode()).hexdigest()[:16]` for a collision-resistant 16-char key. URL-based dedup is correct for "same article re-fetched" (the actual rate limit problem).
- Use a TTL-based eviction: store `{hash: timestamp}` in a dict, and evict entries older than 24 hours at the start of each news cycle. This caps memory at ~200 entries (24h / 15min intervals * 3 sources * ~1 article per source per cycle).
- Do NOT dedup across sources (RBC vs Interfax). Only dedup within the same source, or use the full URL as the dedup key (different sources have different URLs).
- Consider using Redis SET with TTL if the system already has Redis running (it does -- `finalayze-sandbox-redis`). This survives restarts and has built-in TTL.

**Detection:**
- Log `article_deduped` with the hash and source when an article is skipped.
- Track `dedup_cache_size` as a Prometheus gauge. Alert if it exceeds 1000 (indicates eviction is not working).

---

## Minor Pitfalls

### Pitfall 9: Loki Positions File on tmpfs Loses State on Container Restart

**What goes wrong:**
The Promtail config stores the positions file at `/tmp/positions.yaml` (promtail-config.yml line 5). This file tracks which log positions Promtail has shipped to Loki. On container restart, `/tmp` is cleared, and Promtail re-ships all available logs from the Docker JSON log file. This causes duplicate log entries in Loki.

**Prevention:**
- Mount a persistent volume for the positions file: add `- promtail_positions:/positions` to the promtail service in docker-compose, and set `filename: /positions/positions.yaml`.

---

### Pitfall 10: Bond Broker Shares gRPC Client State After Reconnect

**What goes wrong:**
`make_bond_broker()` (tinkoff_broker.py line 529) creates a bond broker that shares `equity_broker._client` and `_account_id`. After `reconnect_client()` on the equity broker, the bond broker's `_client` reference is stale (pointing to the old, closed client). The bond broker will fail on its next call with "Channel closed" or similar gRPC error.

**Prevention:**
- After equity broker reconnect, update the bond broker's `_client` and `_services` references.
- Or: make bond broker use the same `reconnect_client()` method, which re-creates its own client.
- Better: both brokers should reference a shared `_client` holder object that is updated atomically during reconnect.

---

### Pitfall 11: asyncpg Session Leak in Fire-and-Forget DB Writes

**What goes wrong:**
If DB persistence uses `async with async_session() as session:` inside a fire-and-forget task, and the task is cancelled (e.g., during shutdown), the session may not be properly closed. asyncpg connection pool has a limited number of connections (default 10 for asyncpg). Leaked sessions exhaust the pool, causing subsequent DB operations (including health checks) to hang.

**Prevention:**
- Always use `async with` for sessions, and wrap the entire task in a try/finally.
- Set `pool_size=5` and `max_overflow=5` in SQLAlchemy async engine config. This caps total connections at 10.
- Add a `pool_timeout=5` so DB operations fail fast rather than blocking indefinitely.
- During shutdown, cancel all pending DB tasks and wait for them to complete (with timeout).

---

### Pitfall 12: Promtail Drop Stage Regex Mismatch with JSON Logs

**What goes wrong:**
The current Promtail drop stage (promtail-config.yml lines 37-41) uses regex `^INFO:.*"GET /metrics.*` to drop health check logs. But the app uses structlog with JSON output. Health check logs look like `{"event": "http_request", "method": "GET", "path": "/metrics", "level": "info"}`, not `INFO: "GET /metrics"`. The drop stage never matches anything, so noisy health check logs fill Loki.

**Prevention:**
- After the `json` parsing stage, use a `match` stage with a LogQL-style selector: `selector: '{level="info"}'` and a `drop` stage with `source: event` matching `http_request` where path is `/metrics` or `/api/v1/health`.
- Alternatively, filter in the application: configure uvicorn access log to exclude `/metrics` and `/api/v1/health` paths.

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation | Severity |
|-------------|---------------|------------|----------|
| gRPC event loop isolation | Event loop mismatch breaks asyncio.Lock serialization (Pitfall 1) | Merge to single event loop OR use threading.Lock | CRITICAL |
| Channel reconnection | Orphaned orders during reconnect window (Pitfall 2) | Post-reconnect reconciliation + idempotency keys | CRITICAL |
| DB persistence wiring | DB errors crash trading loop (Pitfall 3) | Fire-and-forget with separate error counter | CRITICAL |
| Ticker registry updates | Orphaned stop-loss and entry price state (Pitfall 4) | Migration mapping on startup + inactive old entries | CRITICAL |
| Promtail/Loki pipeline | High cardinality labels + wrong drop regex (Pitfalls 5, 12) | Remove `event` label, fix JSON-aware drop stage | MODERATE |
| CBR XML fallback | Sync HTTP call blocks strategy cycle thread (Pitfall 6) | Background fetch + in-memory cache | MODERATE |
| Market-hours gate | Gate cycle start vs order submission timing (Pitfall 7) | Gate at order submission, not cycle start | MODERATE |
| Article dedup | Hash collisions or unbounded memory (Pitfall 8) | SHA256 URL hash + 24h TTL eviction | MODERATE |
| Bond broker reconnect | Stale client reference after equity reconnect (Pitfall 10) | Shared client holder or coordinated reconnect | MODERATE |
| DB connection pool | Session leak from cancelled fire-and-forget tasks (Pitfall 11) | Pool sizing + timeout + try/finally | MINOR |
| Promtail positions | Duplicate logs after container restart (Pitfall 9) | Persistent volume for positions file | MINOR |

## Integration Risk Matrix

These pitfalls interact with each other. The most dangerous combination:

1. **gRPC isolation + reconnection + bond broker** (Pitfalls 1, 2, 10): Changing the event loop architecture affects all three. If the fix for Pitfall 1 (merging loops) is done incorrectly, it can worsen Pitfall 2 (reconnection) and break Pitfall 10 (bond broker sharing). These three MUST be addressed in the same phase, tested together.

2. **DB persistence + ticker rename** (Pitfalls 3, 4): If orders are persisted with the old ticker symbol and then the ticker is renamed, the audit trail shows YNDX but the system tracks YDEX. DB queries by symbol will miss historical orders. Use FIGI as the primary key in `OrderModel`, not symbol.

3. **Market-hours gate + CBR fallback** (Pitfalls 6, 7): If the market-hours gate runs at cycle start and the CBR fallback adds 97 seconds to the cycle, a cycle that starts at 15:30 UTC (within gate) may submit orders at 15:32 -- after the gate would have blocked it. The gate must check at submission time, not cycle start.

4. **Log pipeline + DB persistence** (Pitfalls 5, 12, 3): If DB write failures are logged at ERROR level (high volume during DB outage), and Promtail ships all logs to Loki without rate limiting, a DB outage creates a log storm that overwhelms Loki, destroying the ability to diagnose the DB issue.

## Recommended Implementation Order

Based on dependency analysis and risk:

1. **Phase 1: gRPC + reconnection + bond broker** (Pitfalls 1, 2, 10) -- highest risk, tightest coupling
2. **Phase 2: DB persistence** (Pitfalls 3, 11) -- depends on understanding the event loop architecture from Phase 1
3. **Phase 3: Ticker updates** (Pitfall 4) -- can be done independently but benefits from DB persistence being wired
4. **Phase 4: Log pipeline** (Pitfalls 5, 9, 12) -- independent, low risk, high observability value
5. **Phase 5: CBR fallback + market-hours gate + article dedup** (Pitfalls 6, 7, 8) -- independent features, moderate risk

## Sources

- Codebase analysis: `src/finalayze/execution/tinkoff_broker.py` (gRPC client, reconnection)
- Codebase analysis: `src/finalayze/orchestration/trading_loop.py` (event loops, locks, cycles)
- Codebase analysis: `src/finalayze/data/fetchers/cbr.py` (sync HTTP, timeouts)
- Codebase analysis: `src/finalayze/markets/instruments.py` (registry, FIGI mapping)
- Codebase analysis: `src/finalayze/core/models.py` (DB models for persistence)
- Codebase analysis: `monitoring/promtail/promtail-config.yml` (log pipeline)
- Codebase analysis: `monitoring/loki/loki-config.yml` (Loki limits)
- Codebase analysis: `docker/docker-compose.sandbox.yml` (service topology)
- PROJECT.md: v6.0 requirements and known issues from sandbox validation

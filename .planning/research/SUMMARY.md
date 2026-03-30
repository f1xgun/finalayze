# Project Research Summary

**Project:** Finalayze v6.0 — Sandbox Stability & Observability
**Domain:** Production MOEX autonomous trading system hardening
**Researched:** 2026-03-30
**Confidence:** HIGH

## Executive Summary

This milestone is not greenfield feature work — it is hardening an existing, functional trading system that showed 10 concrete failure modes during a week-long sandbox run (March 20-30, 2026). The system already trades, has strategies, ML models, a broker integration, and a monitoring stack. What it lacks is the operational reliability required for production. Key failures observed: 127 missed scheduler jobs due to gRPC event loop contention, 62 portfolio fetch failures from T-Bank error 70001, zero DB rows written across 5 days, zero log entries reaching Loki, and FX rates stuck at 0.0. These are not design flaws — they are integration gaps and configuration omissions. Every fix has a specific, measured failure behind it.

The recommended approach is a four-phase fix plan ordered strictly by severity and dependency. gRPC event loop isolation is the top blocker: the PollerCompletionQueue from grpcio >= 1.75 saturates asyncio's self-pipe when gRPC channels share an event loop with non-gRPC work, causing strategy cycles to drift from 5 minutes to 60+ minutes. Until this is fixed, the scheduler cannot fire reliably and nothing else matters. After gRPC stability is restored, DB persistence and log observability unlock the ability to diagnose all remaining issues. Finally, lower-priority fixes (FX fallback, market-hours gate, article dedup, stale tickers) complete the hardening. Zero new external dependencies are needed: every fix uses existing stdlib or already-installed libraries.

The critical risk across all fixes is incorrect event loop architecture. The system currently has three independent asyncio event loops (TradingLoop, TinkoffBroker, TinkoffFetcher), which is architecturally wrong and the root cause of the PollerCompletionQueue problem. The consolidation to a two-loop design (dedicated gRPC loop + general async loop) must be done atomically for all three components in the same phase. The `asyncio.Lock` used for gRPC serialization must live on the same loop as the gRPC calls it protects — if this invariant is violated during the refactor, race conditions can produce duplicate orders during concurrent equity and bond cycle execution.

## Key Findings

### Recommended Stack

No new libraries are required for any of the six core fixes. The system's existing stack (grpcio 1.78.1, SQLAlchemy 2.0.46, asyncpg, httpx, structlog, APScheduler, Promtail 3.4.2, Loki 3.4.2) is correct and sufficient. Explicitly avoid adding uvloop (worsens PollerCompletionQueue contention), grpclib (incompatible with t-tech-investments SDK), celery, tenacity, or custom log shipping. The only new code uses stdlib: `asyncio`, `threading`, `hashlib`, `collections.OrderedDict`.

**Core technologies (existing, no changes needed):**
- `grpcio` 1.78.1 + `t-tech-investments` 0.3.3: T-Bank gRPC transport — must be isolated to a dedicated event loop to eliminate PollerCompletionQueue contention
- `SQLAlchemy` 2.0.46 async + `asyncpg`: DB persistence — `async_sessionmaker` pattern is correct and thread-safe; just not wired into trading loop call sites
- `APScheduler` BackgroundScheduler: job scheduling — already works; recovers to 5-min cycle interval once gRPC isolation removes the 60-min drift
- `Promtail` 3.4.2 + `Loki` 3.4.2: log pipeline — correct stack; single missing volume mount (`/var/lib/docker/containers`) is the only fix needed
- `httpx` + `xml.etree` + `FXRateService`: CBR XML FX fallback — fully implemented in `markets/fx_service.py`; only wiring into `TradingLoop._fx_update_cycle()` is missing

**What NOT to add:**
- `uvloop`: worsens the PollerCompletionQueue EAGAIN issue; standard asyncio is safer
- `grpclib`: t-tech-investments SDK requires grpcio; cannot swap
- `redis` for article dedup: overkill; in-memory OrderedDict with TTL is sufficient for single-process
- `tenacity`: `RetryPolicy` already exists in `execution/retry.py`

### Expected Features

All 9 features are small enough to ship in one milestone. Nothing should be deferred.

**Must have (table stakes — system is broken without these):**
- TS-1: gRPC event loop isolation — 127 missed scheduler jobs/week, 60-min cycle drift; no trading system tolerates this
- TS-2: T-Bank API error 70001 resilience — multi-hour portfolio blind windows, no position sizing, no risk checks
- TS-3: DB persistence for orders/signals/news/sentiment — 0 rows after 5 days, complete data loss, no audit trail
- TS-4: Loki log pipeline operational — 0 log entries, Grafana log dashboards useless, cannot search for failures
- TS-5: FX rate fallback via CBR XML — FX = 0.0 on gRPC failure, all MOEX position sizing is wrong
- TS-6: Market-hours gate at cycle level — strategy cycles fire 24/7, MOEX open 07:00-15:45 UTC only

**Should have (operational quality — significant reduction in burden):**
- D-1: LLM article deduplication — 35 Groq fallback activations/day, wastes LLM quota, slower analysis
- D-2: Stale ticker cleanup — HHRU->HH rename missing in `config/segments.py`; causes failed instrument lookups
- D-3: Telegram alerter startup resilience — startup crash if Telegram token invalid; alerter should be best-effort

**Defer:** Nothing. All 9 items fit in a single milestone (estimated 2-4 days, 4 phases).

**All features are fully independent** — no blocking dependency chains between them. Can be parallelized freely within phases.

### Architecture Approach

The post-fix architecture introduces a strict two-loop separation within TradingLoop: a dedicated gRPC event loop thread (`_grpc_loop`) for all TinkoffBroker and TinkoffFetcher calls, and the existing general async loop (`_async_loop`) for HTTP, DB, and Telegram work. TinkoffBroker and TinkoffFetcher no longer manage their own event loops — they accept the shared gRPC loop via constructor injection from TradingLoop. All gRPC serialization (`asyncio.Lock`) moves to live on the gRPC loop.

DB persistence is extracted to a new `orchestration/persistence.py` module (fire-and-forget helpers: `persist_signal()`, `persist_order()`, `persist_news_article()`, `persist_sentiment()`) to avoid further bloating `trading_loop.py` (already 2400+ lines). Article deduplication gets its own `analysis/dedup.py` module with title-hash + TTL eviction logic. Infrastructure changes are limited to two YAML lines in docker-compose and Promtail config.

**Major components and their changes:**
1. `TradingLoop` (`orchestration/trading_loop.py`) — gains `_grpc_loop` + `_run_grpc()` dispatcher; adds persist call sites at signal/order/news points; adds market-hours early exit; ~200 new lines
2. `TinkoffBroker` (`execution/tinkoff_broker.py`) — removes self-managed `_loop`/`_loop_thread`; gains last-known-portfolio cache; channel reset on 70001/INTERNAL; ~80 new lines
3. `TinkoffFetcher` (`data/fetchers/tinkoff_data.py`) — removes self-managed loop; accepts external gRPC loop via constructor; ~30 changed lines
4. `persistence.py` (NEW `orchestration/`) — fire-and-forget DB write helpers; ~120 lines
5. `dedup.py` (NEW `analysis/`) — `deduplicate_articles()` with SHA256 URL hash + 24h TTL; ~40 lines
6. `docker/docker-compose.sandbox.yml` + `monitoring/promtail/promtail-config.yml` — 1-2 line additions for Docker log volume mount and `__path__` relabeling

**Total scope:** 9 modified files, 2 new files, ~540 new/changed lines.

### Critical Pitfalls

1. **gRPC asyncio.Lock mismatch during event loop consolidation** — The existing `_grpc_lock` (`asyncio.Lock`) is bound to `TradingLoop._async_loop`. If gRPC calls move to `_grpc_loop` but the lock stays on `_async_loop`, it stops serializing concurrent equity and bond cycle broker calls. Race condition produces duplicate orders. Fix: re-create `_grpc_lock` on the gRPC loop; or use a `threading.Lock` for cross-thread serialization. This is the highest-risk element of the entire milestone.

2. **Channel reconnection orphans in-flight orders** — During the reconnect window, a `post_order` call may time out client-side while the gRPC server already executed it. On the next cycle, the system generates the same BUY signal and submits a duplicate (doubling the position). Fix: call `_reconcile_inflight_orders()` after every reconnect; add a "reconnecting" guard flag that blocks order submission during the window; use T-Bank's `order_id` idempotency key.

3. **DB errors crash the trading loop** — If `_consecutive_equity_errors` counts DB write failures the same as trading failures, a 3-cycle DB outage triggers a CRITICAL alert and halts trading even though the trading logic is fine. DB persistence is non-critical path. Fix: fire-and-forget with a separate `db_write_failures` Prometheus counter; never propagate DB exceptions to the cycle; never place DB writes between order submission and position tracking update.

4. **Ticker rename orphans stop-loss state** — Runtime state (`_stop_states`, `_entry_prices`) is keyed by ticker symbol. After renaming HHRU->HH, any open position under the old key loses its trailing stop. Fix: add `_TICKER_RENAMES` migration mapping applied at startup; keep old tickers as inactive in registry with same FIGI during transition.

5. **CBR XML sync call blocks strategy cycle** — `CBRFetcher` uses sync `httpx.Client` (30s timeout, 3 retries = worst-case 97s). Calling it inline as an FX fallback during `_strategy_cycle` blocks the APScheduler thread for 97s with no stop-loss checks running. Fix: pre-fetch CBR rate in a background APScheduler job every 30 minutes into `_fx_cache`; strategy cycle reads from cache only, never makes a live CBR HTTP call.

## Implications for Roadmap

Based on the combined dependency graph from ARCHITECTURE.md and risk ordering from PITFALLS.md, the recommended phase structure is four phases:

### Phase 1: Quick Wins — Config & Guard Fixes
**Rationale:** Zero-risk changes (config-only, try/except wrappers). Ship same day to reduce noise before tackling the architectural changes in Phase 2. Resolves the ticker-rename pitfall before it can interact with Phase 3 (DB persistence keys orders by symbol — must be correct before any orders are persisted).
**Delivers:** Correct MOEX ticker universe (HHRU->HH), no startup crash from invalid Telegram token, off-hours cycles skipped at cycle entry
**Addresses:** D-2 (stale tickers), D-3 (alerter resilience), TS-6 (market-hours gate)
**Avoids:** Pitfall 4 (ticker rename orphaning stop-loss state), Pitfall 7 (gate timing — gate at cycle entry AND at order submission for defense-in-depth)
**Research flag:** None — standard config patterns, no research needed

### Phase 2: gRPC Isolation & Log Visibility
**Rationale:** gRPC event loop isolation (TS-1) is the root cause of 127 missed scheduler jobs. Nothing else can be properly debugged until cycles are stable. Loki pipeline fix (TS-4) enables log-based debugging for all subsequent phases. These two can be worked in parallel (infrastructure track: Loki config; code track: gRPC consolidation) but must both land before Phase 3.
**Delivers:** Stable 5-minute strategy cycles, Grafana/Loki logs visible, gRPC calls isolated from HTTP/DB event loop
**Addresses:** TS-1 (gRPC isolation), TS-4 (Loki pipeline)
**Avoids:** Pitfall 1 (asyncio.Lock mismatch — MUST be solved in this phase, tested with concurrent equity+bond cycles), Pitfall 5 (Loki high-cardinality `event` label — remove from Promtail labels, parse at query time), Pitfall 12 (JSON drop stage regex mismatch — fix for structlog JSON format)
**Research flag:** The three-way refactoring (TradingLoop + TinkoffBroker + TinkoffFetcher) must be designed and implemented atomically. Consider `/gsd:research-phase` if the asyncio.Lock migration approach is not fully clear before implementation starts.

### Phase 3: Resilience — Portfolio Cache, FX Fallback & Bond Broker
**Rationale:** Builds on the broker refactoring from Phase 2. T-Bank 70001 resilience (TS-2) requires the broker event loop to already be sorted out, then adds the last-known-portfolio cache layer. FX fallback (TS-5) is independent but low-priority relative to gRPC stability. Bond broker reconnect coordination (Pitfall 10) must be addressed in the same phase as channel reconnection logic.
**Delivers:** Multi-hour portfolio blind windows eliminated, FX rate always available via CBR fallback, bond broker stays in sync after equity broker reconnect
**Addresses:** TS-2 (70001 resilience), TS-5 (FX rate fallback + staleness tracking)
**Avoids:** Pitfall 2 (duplicate orders during reconnect — post-reconnect reconciliation + idempotency keys), Pitfall 6 (CBR sync call blocking cycle — background fetch job, not inline call), Pitfall 10 (bond broker stale client reference after reconnect)
**Research flag:** None — last-known-good cache and channel reset are textbook patterns. ARCHITECTURE.md provides code sketches for all three.

### Phase 4: Data Capture & Noise Reduction
**Rationale:** DB persistence (TS-3) requires stable cycles from Phase 2 to produce meaningful data (60-min drift would generate misleading timestamps). Article deduplication (D-1) is independent but lower urgency than core stability. These are the highest-LOC changes but lowest operational risk — all writes are additive and fire-and-forget.
**Delivers:** Full audit trail for orders, signals, news, and sentiment; ~50% reduction in LLM API calls from deduplication
**Addresses:** TS-3 (DB persistence for all four tables), D-1 (article deduplication across RSS and Telegram sources)
**Avoids:** Pitfall 3 (DB errors crash trading loop — strict fire-and-forget, separate error counter, never in critical path between order submission and position tracking), Pitfall 8 (dedup hash collisions — SHA256 URL hash, not Python built-in hash; 24h TTL eviction), Pitfall 11 (asyncpg session leak — `async with` + try/finally + pool_timeout=5)
**Research flag:** None — SQLAlchemy `async_sessionmaker` fire-and-forget pattern and SHA256 dedup with TTL are fully documented in STACK.md and ARCHITECTURE.md.

### Phase Ordering Rationale

- Phase 1 before Phase 2: config fixes take minutes and eliminate confounding factors for the architectural changes; ticker rename must be correct before DB persistence keys orders by symbol
- Phase 2 before Phase 3: broker event loop refactoring must be complete before adding cache/fallback logic on top of the refactored broker
- Phase 2 before Phase 4: DB persistence needs stable 5-min cycles to produce useful data; 127 missed-job events being persisted would create misleading audit data
- Phases 3 and 4 are independent of each other — can be parallelized if two implementation tracks are available
- gRPC isolation, reconnect, and bond broker (Pitfalls 1, 2, 10) must all be addressed in Phases 2-3 — they interact tightly and partial fixes create dangerous race conditions with duplicate orders

### Research Flags

Phases needing deeper research during planning:
- **Phase 2 (gRPC loop consolidation):** The three-way refactoring touches three files simultaneously with asyncio.Lock semantics that must be preserved. If the exact implementation approach for gRPC loop sharing and lock migration is not clear before coding starts, run `/gsd:research-phase` scoped to the asyncio.Lock + gRPC shared loop pattern.

Phases with standard patterns (skip additional research):
- **Phase 1:** Config-only changes. No research needed.
- **Phase 3:** Last-known-good cache pattern. Code sketches already in ARCHITECTURE.md. No research needed.
- **Phase 4:** SQLAlchemy async fire-and-forget and SHA256 dedup with TTL are fully specified in STACK.md and ARCHITECTURE.md. No research needed.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | No new dependencies. All fixes verified against existing installed library docs. The gRPC PollerCompletionQueue EAGAIN behavior confirmed by official gRPC docs + LangGraph community reports. |
| Features | HIGH | Based on direct sandbox observation with specific measured metrics (127 missed jobs, 62 portfolio failures, 0 DB rows, 0 Loki entries, FX=0.0). No speculation — each feature addresses a measured failure. |
| Architecture | HIGH | Based on direct codebase inspection of all 11 files to be modified, with line numbers. Code sketches for all fixes are in ARCHITECTURE.md and validated against existing patterns in the codebase. |
| Pitfalls | HIGH | Based on codebase analysis + production sandbox validation logs. All pitfalls are grounded in specific code constructs (e.g., `asyncio.Lock` on wrong loop, `_client` shared reference in bond broker). Prevention strategies are concrete and testable. |

**Overall confidence:** HIGH

### Gaps to Address

- **Bond broker reconnect coordination (Pitfall 10):** `make_bond_broker()` in `tinkoff_broker.py` (line 529) shares the equity broker's `_client` reference. After equity reconnect, bond broker gets a stale client. The exact fix (shared client holder vs. coordinated reconnect) depends on the event loop consolidation design chosen in Phase 2 — resolve during Phase 2 planning before implementation.
- **Promtail drop stage regex (Pitfall 12):** The current drop regex (`^INFO:.*"GET /metrics.*`) does not match structlog JSON output. The exact structlog JSON key structure and Promtail `json` pipeline stage config need to be verified against a live log sample during Phase 2. Low risk to delay to Phase 2 implementation rather than planning.
- **Market-hours gate placement (Pitfall 7):** PITFALLS.md recommends gating at order submission rather than cycle start to avoid missing the last 45 minutes of the session. FEATURES.md (TS-6) recommends gating at cycle entry for simplicity. Resolve during Phase 1: implement both (cycle-entry gate as early exit + submission-time gate as defense-in-depth). This is not a research question — it is an implementation decision.

## Sources

### Primary (HIGH confidence)
- Direct codebase inspection: `src/finalayze/orchestration/trading_loop.py` (2400+ lines, all cycle methods), `src/finalayze/execution/tinkoff_broker.py` (519 lines), `src/finalayze/data/fetchers/tinkoff_data.py`, `src/finalayze/core/models.py`, `src/finalayze/markets/fx_service.py`, `src/finalayze/data/fetchers/cbr.py`, `config/segments.py`, `monitoring/promtail/promtail-config.yml`, `monitoring/loki/loki-config.yml`, `docker/docker-compose.sandbox.yml`
- Sandbox validation logs (March 20-30, 2026): 127 missed scheduler jobs, 62 portfolio fetch failures, 0 DB rows, 0 Loki log entries, FX=0.0
- [gRPC Python AsyncIO API docs](https://grpc.github.io/grpc/python/grpc_asyncio.html) — PollerCompletionQueue behavior, channel lifecycle
- [grpc/grpc#25364](https://github.com/grpc/grpc/issues/25364) — multi-thread async client thread safety discussion
- [SQLAlchemy 2.0 AsyncIO docs](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html) — `async_sessionmaker` thread safety, `expire_on_commit=False` requirement

### Secondary (MEDIUM confidence)
- [LangGraph PollerCompletionQueue thread](https://forum.langchain.com/t/pollercompletionqueue-handle-events-blockingioerror-spam-in-langgraph-cloud-logs/3232) — EAGAIN suppression pattern validated by community at grpcio >= 1.75
- [Promtail Docker SD troubleshooting](https://community.grafana.com/t/promtail-does-not-collect-logs-from-other-containers/87000) — `/var/lib/docker/containers` volume mount requirement confirmed by multiple independent reports
- [Grafana Loki issue #5955](https://github.com/grafana/loki/issues/5955) — "Unable to find any logs to tail" resolution
- [gRPC environment variables reference](https://grpc.github.io/grpc/core/md_doc_environment_variables.html) — GRPC_POLL_STRATEGY analysis

### Tertiary (LOW confidence — validate during implementation)
- CBR XML_daily.asp API stability claim ("unchanged for 10+ years") — credible based on API age and widespread use, but not independently benchmarked; will be confirmed empirically when FX fallback is wired in Phase 3

---
*Research completed: 2026-03-30*
*Ready for roadmap: yes*

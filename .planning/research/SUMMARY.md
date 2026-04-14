# Project Research Summary

**Project:** Finalayze v10.0 — Runtime LLM Trading Agents
**Domain:** Live MOEX equities trading system with LLM-powered news analysis agents
**Researched:** 2026-04-14
**Confidence:** HIGH

## Executive Summary

Finalayze v10.0 is an incremental milestone that activates and extends a largely-built LLM news analysis subsystem. The core infrastructure — `LLMClient` with 5 providers, `NewsImpactAnalyzer`, `EventDrivenStrategy`, `RssNewsFetcher`, `TelegramChannelReader`, and the `sentiment_scores` TimescaleDB hypertable — already exists and is verified. The milestone's central challenge is not building new infrastructure but closing specific activation gaps: the `EventDrivenStrategy` has never fired a live signal because `_news_cycle()` is gated behind `_any_event_driven_enabled()` which always returns false, and the news pipeline calls LLM analysis with a hardcoded 1800-second timeout that stalls on free-tier OpenRouter latency. The recommended approach is surgical, additive-only work: config flips, parameter injection, two new L3 agent classes, one new L2 accessor, and one DB migration — with zero new package dependencies.

The two new agents follow a clear pattern from research. The `PortfolioReviewAgent` must be advisory-only (Telegram output, no trade directives) because the expert panel unanimously rejected LLM modifiers in the sizing pipeline as they break backtestability and determinism. The `AnomalyInterpreterAgent` must use fire-and-forget dispatch so the primary anomaly alert is never delayed by LLM latency. Academic literature (TrustTrade 2025, LLM Multi-Agent Anomaly Detection 2024) confirms both patterns are production-validated approaches. T-Pulse integration was investigated and must be deferred: the T-Invest gRPC SDK exposes no news or pulse service, and the undocumented REST endpoint changed authentication in June 2024 with unclear current status.

The critical risks cluster in Phase 1 (news pipeline activation). Three latent bugs exist in the current codebase that will manifest the moment the live pipeline is enabled: `NewsAnalyzer.analyze()` uses `json.loads()` instead of `parse_structured()` causing silent signal loss on any LLM format variation; the `threading.Lock` over `_sentiment_cache` is acquired inside an async coroutine that can deadlock if an `await` is ever inserted inside the lock block; and the 1800-second batch timeout causes APScheduler thread starvation. All three must be fixed before activating `event_driven` on any live segment. The sentiment decay clock also requires a market-hours gate to prevent the first article of the trading day from producing outsized signals after the overnight quiet period.

## Key Findings

### Recommended Stack

v10.0 requires zero new package dependencies. Every technology needed is already installed and verified in `pyproject.toml`. The implementation surface is entirely new Python modules and configuration changes on top of the existing stack.

**Core technologies:**
- `anthropic` / `openai` via `LLMClient` ABC: LLM backbone for new agents — `parse_structured()` already handles typed Pydantic output across all 5 providers; new agents call it directly
- `httpx` (>=0.28.0): T-Pulse REST calls if implemented — `TelegramChannelReader` is the reference pattern; implement as `TPulseFetcher` default-disabled
- `feedparser` (>=6.0.12): MOEX ISS official news feeds — add `moex.com/export/news.aspx?cat=200` and `cat=202` as config-only change to `news_rss_urls`
- `APScheduler` (>=3.10.4): Portfolio review daily cron registration — extend `TradingLoop.start()` with one new `add_job()` call
- `SQLAlchemy` 2.0 async + `asyncpg`: Rolling sentiment aggregation — `session.execute(text(...))` pattern already used for raw SQL; new `SentimentStore` queries continuous aggregate view
- `asyncio.run_coroutine_threadsafe` + strong reference set: Fire-and-forget pattern for `AnomalyInterpreterAgent` — documented Python 3.12+ fix for task GC; already used in `TradingLoop._persist_sentiment_scores()`
- TimescaleDB continuous aggregates: `sentiment_7d_avg` materialized view via Alembic migration 004 — pure SQL, incremental O(new data only) refresh, no Python library change

### Expected Features

**Must have (table stakes):**
- RSS + Telegram sources enabled on `ru_blue_chips` / `ru_diversified` with `event_driven.enabled: true` — closes the "EventDrivenStrategy never fires" gap; all code exists, YAML flip only
- Per-source credibility weights (RBC=0.8, Interfax=0.8, TASS=0.7, Telegram=0.5) with hard cap at 0.7 — `EventDrivenStrategy.generate_signal(credibility=)` already accepts float; inject per-source map in `_news_cycle`
- Per-article LLM timeout (5s) + per-cycle timeout (30s) replacing the existing 1800s no-op — `asyncio.wait_for` wrapper; prevents APScheduler thread starvation
- Deduplication gate moved before `NewsImpactAnalyzer` — eliminates wasted LLM tokens on repeated articles; current check runs after the LLM call
- `AnomalyInterpreterAgent`: fire-and-forget async task after `AnomalyDetector.check()` fires; Haiku-tier LLM; appends explanation to Telegram alert; never blocks raw alert
- `PortfolioReviewAgent`: daily Pydantic-structured report; Sonnet tier; runs in `daily_reset` job outside market hours; Telegram digest only

**Should have (competitive):**
- Sentiment rolling aggregation: TimescaleDB `time_bucket()` continuous aggregate on `sentiment_scores`; `SentimentStore` L2 class for future ML feature extraction — wire now so data accumulates; ML use is v11+
- FIGI-resolved sentiment routing: map `EntityExtractor` output through `InstrumentRegistry` before `sentiment_scores` write — improves ML feature quality; low implementation cost after Phase 1

**Defer (v2+):**
- XGBoost sentiment features — requires 30+ days of accumulated `sentiment_scores` data; full ML experiment cycle; v11+
- T-Pulse integration — T-Invest gRPC SDK has no news service; undocumented REST endpoint with authentication changes in June 2024; defer until programmatic API confirmed
- Multi-source consensus scoring (TrustTrade-style cross-agent agreement) — meaningful complexity; reserve until baseline sentiment features show ML lift
- Streaming news (WebSocket/Kafka) — RSS polling every 5 min sufficient for MOEX daily-bar strategies; streaming adds operational complexity with no measurable edge at daily timeframes

### Architecture Approach

The architecture is strictly additive: two new L3 agent files, one new L2 accessor, one Alembic migration, modifications to `TradingLoop` (L5), `AnomalyDetector` and `SandboxMonitorService` (L6). The layered dependency model (L0→L6) is maintained throughout. The key structural constraint is the two async loop separation: `_async_loop` (HTTP/DB/Telegram) and `_grpc_loop` (Tinkoff gRPC). New agents must use `_run_async()` from APScheduler threads, never call it from within `_async_loop`, and use `asyncio.run_coroutine_threadsafe()` for the fire-and-forget cross-thread pattern.

**Major components:**
1. `PortfolioReviewAgent` (L3, new) — accepts `PortfolioState`, calls `parse_structured()` with `PortfolioReviewResult` Pydantic schema; advisory only; scheduled as daily cron in `TradingLoop._portfolio_review_cycle()`
2. `AnomalyInterpreterAgent` (L3, new) — accepts triggered metric names + z-scores from `AnomalyDetector._fire_alert()`; dispatched via `asyncio.run_coroutine_threadsafe`; follow-up Telegram message only; raw alert fires first unconditionally
3. `SentimentStore` (L2, new) — read-only accessor for `sentiment_7d_avg` continuous aggregate view; used by future ML feature pipeline; write path unchanged
4. `TradingLoop` modifications (L5, additive) — source credibility map, latency SLA gate in `_news_cycle`, new `_portfolio_review_cycle()` cron job, Portfolio Review Agent constructor injection
5. `AnomalyDetector` / `SandboxMonitorService` modifications (L6, additive) — optional `anomaly_interpreter` and `async_loop` constructor params; fire-and-forget dispatch in `_fire_alert()`

### Critical Pitfalls

1. **`threading.Lock` held across async LLM call boundary (deadlock)** — never place any `await` expression inside `with self._sentiment_lock:`; compute values async outside the lock, acquire lock only for the dict write; this is a latent bug if new `await` is added to `_apply_impact_result`

2. **LLM timeout blocking news cycle, starving strategy execution** — replace `_run_async(coro, timeout=1800)` with per-article 5s timeout via `asyncio.wait_for`; cap batch at 20 articles maximum; log `news_cycle_duration` as metric; fix in Phase 1 before any live activation

3. **`NewsAnalyzer.analyze()` uses `json.loads()` instead of `parse_structured()`** — silent signal loss on any LLM format variation (code fences, trailing whitespace, explanatory text); migrate to `parse_structured(SentimentResult)` before activating live pipeline; this is the single highest-priority code fix

4. **Portfolio Review Agent suggestions interpreted as executable orders** — `PortfolioReviewResult` schema must contain no `direction`, `confidence`, or `symbol`+`market_id` fields matching `Signal`/`OrderRequest`; named distinctly; handler must write to Telegram only, never to `BrokerRouter`; add type assertion at handler entry

5. **Sentiment cache decay distorting first-article-of-day signal** — `_SENTIMENT_HALF_LIFE_HOURS = 4.0` was tuned for continuous intraday flow; apply decay only during MOEX market hours (10:00-18:45 MSK); freeze the decay clock overnight to prevent first morning article from spiking from near-zero baseline

6. **LLM hallucinated ticker extraction creating ghost signals** — all `EntityExtractor` output must be validated against `InstrumentRegistry` whitelist before touching `_sentiment_cache`; log rejected tickers with `entity_not_in_registry` reason

## Implications for Roadmap

Based on research, the build order is dictated by two principles: fix latent bugs before activation, and validate fire-and-forget async pattern on the least risky component first.

### Phase 1: News Pipeline Activation and Hardening

**Rationale:** Three latent bugs (`json.loads`, 1800s timeout, lock/async mixing risk) will cause immediate silent failure or deadlock when the news pipeline goes live. These must be fixed before any live segment activation. This phase also adds the safeguards (ticker whitelist, LLM liveness alerting, article budget cap, dedup gate reordering) that prevent cost explosions and ghost signals.

**Delivers:** A production-safe news ingestion pipeline that activates `EventDrivenStrategy` on `ru_*` segments for the first time; credibility-weighted sentiment scores written to DB; HealthMonitor tracking LLM liveness

**Addresses features:** Real MOEX news sources wired to EventDrivenStrategy; credibility cap; LLM timeout; deduplication gate; MOEX ISS official feeds (config-only); LLM liveness alerting

**Avoids pitfalls:** `json.loads` silent signal loss; LLM timeout starvation; hallucinated ticker ghost signals; `threading.Lock` deadlock; LLM API downtime silent failure; per-article cost explosion

**Research flag:** Standard patterns — newsparser + APScheduler wiring is well-documented. No phase research needed. Codebase inspection provides complete implementation context.

### Phase 2: EventDrivenStrategy Activation and Signal Quality

**Rationale:** After Phase 1 hardens the pipeline, the event_driven strategy can be safely enabled on ru_* segment presets. Phase 2 validates the sentiment decay behavior, adds the CBR/dividend duplicate signal guard in the combiner, and runs a backtest on CBR announcement dates to verify combined confidence does not double-count correlated catalysts.

**Delivers:** `event_driven` strategy enabled at 15% weight on `ru_blue_chips` and `ru_diversified`; combiner hook preventing CBR/dividend signal amplification; sentiment decay gated on market hours; sandbox validation week with live news data

**Addresses features:** EventDrivenStrategy live on MOEX segments; sentiment decay market-hours gate; per-source credibility tuning after observing real signal distribution

**Avoids pitfalls:** CBR/dividend duplicate signal (EventDriven + cbr_calendar amplification); sentiment decay first-article-of-day distortion; Telegram low-credibility source overweighting

**Research flag:** Needs shallow validation research during planning — specifically, review the `StrategyCombiner` hook system to confirm `_on_strategy_signal` hook can suppress correlated catalyst signals without architectural changes.

### Phase 3: Portfolio Review Agent

**Rationale:** No dependency on news pipeline activation. All dependencies (LLMClient, PortfolioState, TelegramAlerter, daily cron pattern) already exist. Can be built in parallel with Phase 2 but scheduled after Phase 1 completes to reuse async patterns validated there.

**Delivers:** Daily advisory Pydantic-structured LLM analysis of portfolio; `PortfolioReviewAgent` L3 class; `PortfolioReviewResult` Pydantic schema in `core/schemas.py`; daily Telegram digest

**Addresses features:** Portfolio Review Agent; structured daily advisory report

**Avoids pitfalls:** Agent output interpreted as executable orders (type schema design enforced at this phase); Portfolio Review running during market hours (off-hours scheduling gate)

**Research flag:** Standard patterns — LLM structured output with `parse_structured()` is well-established in codebase. No phase research needed.

### Phase 4: Anomaly Interpreter Agent

**Rationale:** Builds on the fire-and-forget async pattern from Phase 1 and the async loop integration validated in Phase 3. Self-contained L3 addition that threads through `AnomalyDetector` and `SandboxMonitorService` via optional constructor params. Lowest-risk phase since the raw alert path is unchanged.

**Delivers:** `AnomalyInterpreterAgent` L3 class; follow-up LLM explanation appended to anomaly Telegram alerts; two-step alert pattern (immediate raw alert + async enrichment)

**Addresses features:** Anomaly Interpreter Agent; human-readable anomaly diagnosis

**Avoids pitfalls:** LLM interpretation blocking raw alert delivery (two-step pattern enforced); `asyncio.run_coroutine_threadsafe` correctness (validated in Phase 1)

**Research flag:** Standard patterns — fire-and-forget async is established from Phase 1. No phase research needed.

### Phase 5: Sentiment Rolling Aggregation Infrastructure

**Rationale:** The write path is already working. This phase adds only the read accessor and DB migration — no trading-critical path changes. Must be done before v11 ML work but can be deferred until 2+ weeks of `sentiment_scores` data confirms the table is populating correctly after Phase 1 activation.

**Delivers:** `SentimentStore` L2 accessor; `sentiment_7d_avg` TimescaleDB continuous aggregate with auto-refresh policy; `SentimentFeatureProvider` stub in Layer 3 for v11 ML integration

**Addresses features:** Sentiment rolling aggregation infrastructure; FIGI-resolved sentiment routing

**Avoids pitfalls:** Python-side rolling aggregation on full raw table (TimescaleDB incremental refresh is O(new data only)); ML pipeline crashing on empty/minimal data (query handles sparse data)

**Research flag:** Needs confirmation of TimescaleDB continuous aggregate window function syntax (`timescaledb.enable_cagg_window_functions` setting) during planning. One config flag, low risk.

### Phase Ordering Rationale

- Phase 1 must be first: latent bugs (`json.loads`, 1800s timeout) cause immediate failure on live activation; fixing before enabling prevents silent degradation
- Phase 2 depends on Phase 1: EventDrivenStrategy activation only makes sense after the pipeline is production-safe
- Phase 3 is independent of news pipeline but benefits from Phase 1's async pattern validation; schedule after Phase 1 completes
- Phase 4 is independent but applies fire-and-forget pattern from Phase 1; schedule after Phase 1 and 3 for pattern reuse
- Phase 5 is least urgent: write path works; read accessor and DB migration needed only when ML feature work begins in v11

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 2:** Review `StrategyCombiner._on_strategy_signal` hook capability for correlated catalyst suppression — confirm hook has access to other active strategy signals in the same cycle
- **Phase 5:** Verify `timescaledb.enable_cagg_window_functions` is enabled in Docker Compose PostgreSQL config; confirm continuous aggregate refresh policy syntax for current TimescaleDB version

Phases with standard patterns (skip research-phase):
- **Phase 1:** All patterns are codebase-native; `RssNewsFetcher`, `TelegramChannelReader`, `asyncio.wait_for` are well-understood
- **Phase 3:** `parse_structured()` + Pydantic + APScheduler cron — established patterns throughout codebase
- **Phase 4:** Fire-and-forget with `asyncio.run_coroutine_threadsafe` — documented pattern, validated in Phase 1

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All findings from direct `pyproject.toml` and source file inspection; zero new packages needed; all version constraints verified |
| Features | HIGH | Cross-validated against codebase, `.planning/PROJECT.md` expert debate outcomes, and 4 academic papers; anti-features (Pre-Trade Reasoning) explicitly rejected by expert panel |
| Architecture | HIGH | All integration points verified by direct inspection of `trading_loop.py` (2062+ LOC), `anomaly_detector.py`, `sandbox_monitor.py`, `llm_client.py`, T-Invest SDK `services.py`; no inference required |
| Pitfalls | HIGH | Three pitfalls (`json.loads`, 1800s timeout, `threading.Lock` across await) confirmed as latent bugs by direct code inspection; academic sources validate hallucination and advisory-only patterns |

**Overall confidence:** HIGH

### Gaps to Address

- **T-Pulse API availability**: REST endpoint `https://www.tinkoff.ru/api/invest-gw/social/v1/post/instrument/{ticker}` has uncertain authentication status post-June 2024. If T-Pulse is desired in v10.x, validate in sandbox environment before implementing. For v10.0, T-Pulse is excluded.

- **OpenRouter free tier RPD cap**: Exact daily request limit for free-tier models varies by model and is not publicly documented. The article-budget cap (N=10 per cycle) and `FallbackLLMClient` provide mitigation, but actual production cost/limit behavior needs one week of live observation after Phase 1 activation to tune appropriately.

- **Sentiment decay clock freeze implementation**: `is_market_open_now()` helper exists in `market_schedule`; verify it handles MOEX holiday calendar correctly before the decay gate depends on it in Phase 2.

- **MOEX ISS RSS feed stability**: `moex.com/export/news.aspx?cat=200` and `cat=202` URLs verified via web research (MEDIUM confidence); availability should be confirmed with a live HTTP check during Phase 1 implementation.

## Sources

### Primary (HIGH confidence)
- `src/finalayze/orchestration/trading_loop.py` — full async architecture, APScheduler jobs, sentiment pipeline, fire-and-forget pattern
- `src/finalayze/analysis/llm_client.py` — `LLMClient` ABC, `parse_structured()`, `_CachingLLMClient` confirmed
- `src/finalayze/analysis/news_analyzer.py` — `json.loads()` latent bug confirmed
- `src/finalayze/monitoring/anomaly_detector.py` — `check()` returns `list[str]`, `_fire_alert()` synchronous path confirmed
- `src/finalayze/strategies/event_driven.py` — `generate_signal(credibility=)` parameter confirmed
- `src/finalayze/core/models.py` — `SentimentScoreModel` hypertable schema confirmed
- `.venv/.../t_tech/invest/services.py` — T-Invest gRPC SDK exposes 10 services; no news/pulse service confirmed
- `pyproject.toml` — all installed package versions confirmed
- Python asyncio docs + Python 3.12 task GC documentation — `create_task()` + strong reference set pattern

### Secondary (MEDIUM confidence)
- [TrustTrade: Selective Consensus in LLM Trading Agents (2025)](https://arxiv.org/html/2603.22567) — per-source credibility weighting rationale
- [LLM Multi-Agent Framework for Anomaly Detection in Finance (2024)](https://arxiv.org/html/2403.19735v1) — advisory-only anomaly explanation pattern validated
- [LLM Architectures for Financial Document Processing (2025)](https://arxiv.org/html/2603.22651) — latency/hierarchy tradeoffs; hierarchical at 97.7% of reflexive accuracy at 60.9% cost
- [Large Language Models in equity markets (Frontiers, 2025)](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1608365/full) — advisory-vs-execution boundary in production
- TimescaleDB continuous aggregates official docs — `time_bucket()` + `add_continuous_aggregate_policy()` verified
- `https://www.moex.com/s355` — MOEX RSS feed catalog; cat=200 and cat=202 URLs

### Tertiary (LOW confidence)
- artydev.ru/posts/pulse-parser/ — T-Pulse REST endpoint structure (community blog, unofficial, authentication changes noted June 2024)
- github.com/meanother/tpulse-py — T-Pulse library last commit Dec 2021, confirms unmaintained status

---
*Research completed: 2026-04-14*
*Ready for roadmap: yes*

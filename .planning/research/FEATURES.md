# Feature Research

**Domain:** Runtime LLM trading agents for live MOEX equities system (v10.0)
**Researched:** 2026-04-14
**Confidence:** HIGH (primary findings verified against codebase + production literature)

---

## Context: What Already Exists

This is a subsequent milestone. The following are shipped and must NOT be re-built:

| Existing Component | State | Gap for v10.0 |
|--------------------|-------|---------------|
| `NewsAnalyzer` + `ImpactEstimator` (Layer 3) | COMPLETE | No live segments have event_driven enabled |
| `EventDrivenStrategy` (Layer 4) | COMPLETE, DISABLED | Never receives live sentiment_score input |
| `RssNewsFetcher` (RBC, Interfax, TASS RSS) | COMPLETE | Wired in TradingLoop but segments not configured |
| `TelegramChannelReader` (t.me/s/ HTTP scraping) | COMPLETE | Wired in TradingLoop |
| `AnomalyDetector` (z-score + threshold rules) | COMPLETE | Fires raw alerts, no LLM explanation |
| `LLMClient` (5 providers, fallback chain, parse_structured) | COMPLETE | Available for new agents |
| `TradingLoop._news_cycle()` | COMPLETE | Skips when no event_driven enabled |
| DB tables: `news_articles`, `sentiment_scores` | COMPLETE (migration 002) | Fire-and-forget writes already wired |
| Telegram alerter with priority queue | COMPLETE | Available for agent output routing |
| `PortfolioState` schema | COMPLETE | Available as Portfolio Review Agent input |

New features must build **on top of** these components, not replace them.

---

## Feature Landscape

### Table Stakes (Users Expect These)

Features a live trading system with "LLM news agents" must have. Missing these = the agent subsystem is dead or dangerous.

| Feature | Why Expected | Complexity | Depends On |
|---------|--------------|------------|------------|
| Real MOEX news sources wired to EventDrivenStrategy on live segments | EventDrivenStrategy has 0 live trades because it never receives a non-zero sentiment_score. The entire news analysis pipeline (NewsAnalyzer, ImpactEstimator, EntityExtractor) runs but its output never reaches strategy signals | MEDIUM | Existing RssNewsFetcher + TelegramChannelReader; `ru_*` segment presets need `event_driven.enabled: true`; `_news_cycle()` must pass sentiment to strategy cycle |
| Credibility cap of 0.7 on news-derived signals | Expert panel mandated this. LLM sentiment from unverified Telegram channels is noisy; uncapped credibility creates overconfident signals. EventDrivenStrategy already accepts `credibility=` param — the cap needs enforcement at the source weighting layer | LOW | `EventDrivenStrategy.generate_signal(credibility=)` already implemented; needs per-source weight config and cap enforcement |
| Per-article LLM timeout (5s) and per-cycle timeout (30s) | Current `_batch_timeout = 1800` in `_news_cycle` means a single slow OpenRouter call can hold an APScheduler thread for 30 minutes, stalling strategy cycles. A 5-second timeout per article forces graceful degradation on rate-limited LLM | LOW | `asyncio.wait_for` wrapper; replaces the existing 1800s timeout |
| Anomaly Interpreter Agent: LLM explanation for triggered anomalies | `AnomalyDetector` fires "drawdown z-score 2.3σ" alerts. Operators need human-readable diagnosis: "Unusual SBER drawdown — likely related to CBR rate surprise or sector rotation". Advisory only, no writes to circuit breakers | MEDIUM | `AnomalyDetector.check()` already returns triggered metric names; new fire-and-forget async task; Haiku tier; appends explanation to existing Telegram alert |
| Portfolio Review Agent: daily structured LLM analysis | After market close, advisory-only Pydantic-structured summary of portfolio PnL attribution, risk concentration, and next-session context. No trade execution. Sonnet tier. | MEDIUM | `PortfolioState` schema; must run in `daily_reset` job gated by `not market_schedule.is_market_open()`; output as structured Pydantic model + Telegram digest |

### Differentiators (Competitive Advantage)

Features beyond baseline that provide edge specific to this system (MOEX, Russian news, autonomous loop).

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Per-source credibility weights for Russian news | RBC/Interfax carry higher factual weight than anonymous Telegram channels. TrustTrade (2025) shows dynamic source weighting improves risk-adjusted returns vs. uniform trust. Prevents pump-and-dump Telegram posts from triggering real trades | MEDIUM | `NewsArticle.source` field available; credibility map in settings.yaml; injected into `EventDrivenStrategy.generate_signal(credibility=)` which already exists |
| Sentiment rolling aggregation for future XGBoost features | Collect article-level sentiment now, expose 1h/4h/24h rolling averages as ML features later. `sentiment_scores` table already populated; adding TimescaleDB time_bucket views and a query hook in the ML feature pipeline costs little now but enables a whole ML feature category in v11+ | MEDIUM | `sentiment_scores` table exists; needs a `time_bucket()` view + `SentimentFeatureProvider` class in Layer 3; XGBoost integration is future milestone but data must accumulate now |
| Article-level deduplication before LLM analysis | Same article reposted on RBC RSS and Telegram should not double-count sentiment. `_is_article_duplicate()` in TradingLoop already exists but applies AFTER `NewsImpactAnalyzer` — it needs to gate BEFORE the LLM call | LOW | SHA-256 dedup already implemented; re-ordering the gate eliminates wasted LLM tokens and inflated sentiment_score writes |

### Anti-Features (Commonly Requested, Often Problematic)

Features that appear valuable but create systemic risk or were explicitly rejected.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Pre-trade LLM reasoning (LLM modifier in sizing pipeline) | Seems like smarter position sizing | Expert panel unanimous REJECT. Breaks backtestability (non-deterministic), is uncalibrated (no historical accuracy), adds 1-3s latency per trade in hot path. A LLM hallucinating "sanctions risk" could zero out a well-priced position | Keep LLM advisory-only. Use XGBoost with sentiment features (deterministic, backtestable) for sizing influence |
| Sentiment-based circuit breaker auto-halt on negative news | News-triggered halt seems prudent | Triggers on unverified LLM output; a model hallucination could halt a profitable position. Circuit breakers must remain rules-based (drawdown %, fill rate) | Log high-negative-sentiment events; alert via Telegram; let operator decide; existing circuit breaker remains untouched |
| Real-time streaming news (WebSocket/Kafka) | Lower latency than RSS polling | RSS polling every 5 min is sufficient for MOEX daily-bar strategies. Streaming adds operational complexity with no measurable signal edge at daily timeframes. MOEX news moves prices over hours, not seconds | Stick with APScheduler polling at configurable interval |
| Autonomous portfolio rebalancing based on LLM advice | Natural extension of Portfolio Review | Advisory-to-action path bypasses strategy/risk/backtest validation loop. "Reduce LKOH due to geopolitical risk" has never been backtested and cannot be | Portfolio Review output is structured Pydantic; human reads it, decides manually |
| LLM-powered news source discovery (auto-find new RSS feeds) | More coverage | LLM autonomously adding feeds to production config creates unaudited sources with unknown credibility and potential injection vectors | Manually curate feed list; add T-Pulse RSS manually once confirmed available |
| T-Pulse (Tinkoff social network) as news source in v10.0 | Tinkoff's own network for MOEX investors should have high-quality signal | No public REST/RSS API documented. Tinkoff launched Pulse as a mobile app social network in 2020; no evidence of programmatic data access in research | Defer to v11+; investigate T-Invest gRPC API for any news/pulse endpoint |

---

## Feature Dependencies

```
[RSS + Telegram sources]
    └──feeds──> [TradingLoop._news_cycle()]
                    └──analyzes via──> [NewsImpactAnalyzer]
                                           └──writes──> [sentiment_scores table]
                                           └──updates──> [_sentiment_cache]
                                                             └──read by──> [EventDrivenStrategy]
                                                                               └──gated by──> [credibility cap 0.7]

[AnomalyDetector.check() returns triggered names]
    └──triggers──> [Anomaly Interpreter Agent (fire-and-forget)]
                       └──appends LLM explanation to──> [Telegram alert]

[PortfolioState schema]
    └──input to──> [Portfolio Review Agent]
                       └──outputs──> [PortfolioReviewReport (Pydantic)]
                                         └──sends to──> [Telegram digest]

[sentiment_scores table (accumulates over time)]
    └──enables (future)──> [Rolling aggregation views]
                               └──enables (future v11+)──> [XGBoost sentiment features]

[Per-source credibility weights config]
    └──injects into──> [EventDrivenStrategy credibility= param]
                           └──hard cap at──> [0.7 maximum]
```

### Dependency Notes

- **EventDrivenStrategy requires live sentiment_score**: Strategy code is complete. The gap is that `_news_cycle()` updates `_sentiment_cache` but `_strategy_cycle()` does not read from it when calling `EventDrivenStrategy.generate_signal()`. The wiring between the two cycles is the critical path.
- **Anomaly Interpreter must not block AnomalyDetector**: The interpreter must be dispatched as an `asyncio.create_task()` or `ThreadPoolExecutor` task — not awaited inline — so `AnomalyDetector.check()` returns synchronously as today.
- **Portfolio Review requires off-hours gate**: Sonnet tier during market hours competes with news_cycle LLM quota. Enforce `not market_schedule.is_market_open()` before every Portfolio Review invocation.
- **Sentiment rolling aggregation depends on data accumulation**: The feature is useless for ML until 30+ days of data exist. Wire the infrastructure now; do not attempt ML training on it until v11.
- **Deduplication gate order matters**: Move `_is_article_duplicate()` check before `_analyze_impact_batch()`, not after. Current code wastes LLM tokens on duplicates.

---

## MVP Definition

### Launch With (v10.0)

Minimum to make the news/agent subsystem functional end-to-end for the first time.

- [ ] RSS + Telegram sources enabled on `ru_blue_chips` and `ru_diversified` segments with `event_driven.enabled: true` in preset YAML — closes the "EventDrivenStrategy never fires" gap
- [ ] Per-source credibility weights (RBC=0.8, Interfax=0.8, TASS=0.7, Telegram=0.5) injected into `EventDrivenStrategy`; hard cap enforced at 0.7
- [ ] Article-level LLM timeout (`asyncio.wait_for`, 5s) + cycle-level timeout (30s) replacing the existing 1800s no-op
- [ ] Deduplication gate moved before `NewsImpactAnalyzer` batch call
- [ ] Anomaly Interpreter Agent: `asyncio.create_task()` dispatch after `AnomalyDetector.check()` fires; calls Haiku-tier LLM; appends explanation text to Telegram alert; no circuit-breaker writes
- [ ] Portfolio Review Agent: daily Pydantic-structured report; Sonnet tier; runs in `daily_reset` job outside market hours; sends Telegram digest

### Add After Validation (v10.x)

Features to add once the core pipeline is confirmed working in sandbox for 1+ weeks.

- [ ] Sentiment rolling aggregation: TimescaleDB `time_bucket()` views (1h/4h/24h) on `sentiment_scores`; `SentimentFeatureProvider` class in Layer 3 — add after 2+ weeks of data accumulation confirms the table is populating correctly
- [ ] FIGI-resolved sentiment routing: map `EntityExtractor` output through `InstrumentRegistry` before `sentiment_scores` write, enabling symbol-level ML features

### Future Consideration (v11+)

Defer until v10.0 validated in live trading and sentiment data accumulated.

- [ ] XGBoost sentiment features using rolling aggregation — requires 30+ days of data
- [ ] T-Pulse integration — requires confirming a programmatic API exists
- [ ] Multi-source consensus scoring (TrustTrade-style cross-agent agreement on news interpretation) — adds meaningful complexity; reserve for after baseline sentiment features show ML lift

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| News sources enabled + EventDrivenStrategy live | HIGH — activates dead subsystem | LOW (config + glue) | P1 |
| Credibility cap + per-source weights | HIGH — prevents signal noise | LOW (param injection + config) | P1 |
| LLM timeout (5s article / 30s cycle) | HIGH — prevents APScheduler stall cascade | LOW (asyncio.wait_for) | P1 |
| Deduplication gate moved before LLM batch | MEDIUM — eliminates wasted tokens | LOW (code reorder) | P1 |
| Anomaly Interpreter Agent | MEDIUM — advisory quality-of-life | MEDIUM (new async agent) | P1 |
| Portfolio Review Agent | MEDIUM — operator insight | MEDIUM (new agent + prompt + Pydantic schema) | P1 |
| Sentiment rolling aggregation infrastructure | HIGH (future ML) / LOW (immediate) | MEDIUM (DB view + Layer 3 class) | P2 |
| FIGI-resolved sentiment routing | MEDIUM — improves ML feature quality | LOW (InstrumentRegistry lookup) | P2 |
| T-Pulse integration | LOW (unconfirmed availability) | UNKNOWN | P3 |
| XGBoost sentiment features | HIGH (future) | HIGH (full ML experiment cycle) | P3 |

---

## Complexity Assessment

| Component | Complexity | Key Risk |
|-----------|------------|----------|
| News sources + EventDrivenStrategy wiring | LOW — all code exists; config + glue | `_sentiment_cache` must be read in strategy cycle with correct symbol-level mapping |
| Credibility weights config | LOW — `EventDrivenStrategy.generate_signal(credibility=)` already accepts float | Telegram channel credibility is subjective; 0.5 default may still be too high for some channels |
| LLM timeouts | LOW — `asyncio.wait_for` wrapper | The existing `_batch_timeout = 1800` must be REPLACED, not layered alongside |
| Anomaly Interpreter Agent | MEDIUM — new class, async dispatch, Haiku tier | Must not block `AnomalyDetector.check()` synchronous return path |
| Portfolio Review Agent | MEDIUM — new class, Pydantic output schema design, off-hours scheduling | Pydantic output schema and prompt design are the main work; Sonnet response parsing |
| Sentiment rolling aggregation | MEDIUM — TimescaleDB `time_bucket` views + query in ML pipeline | Data is sparse for 30+ days; queries must handle empty/minimal data without crashing ML pipeline |

---

## Sources

- Codebase: `src/finalayze/analysis/news_analyzer.py`, `src/finalayze/strategies/event_driven.py`, `src/finalayze/monitoring/anomaly_detector.py`, `src/finalayze/orchestration/trading_loop.py`, `src/finalayze/data/fetchers/rss_fetcher.py`, `src/finalayze/data/fetchers/telegram_reader.py`, `docs/database/SCHEMA.md`
- `.planning/PROJECT.md` — v10.0 requirements and expert debate outcomes (Pre-Trade Reasoning REJECT, others APPROVE)
- [TrustTrade: Human-Inspired Selective Consensus in LLM Trading Agents (2025)](https://arxiv.org/html/2603.22567) — source credibility weighting rationale
- [Enhancing Anomaly Detection in Financial Markets with LLM Multi-Agent Framework (2024)](https://arxiv.org/html/2403.19735v1) — LLM explanation for anomalies pattern; advisory-only approach validated
- [Benchmarking Multi-Agent LLM Architectures for Financial Document Processing (2025)](https://arxiv.org/html/2603.22651) — latency, hierarchy tradeoffs; hierarchical provides 97.7% of reflexive accuracy at 60.9% cost
- [Large Language Models in equity markets (Frontiers, 2025)](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1608365/full) — production deployment patterns, advisory-vs-execution boundary
- [Tinkoff Pulse social network for investors](https://www.fintechfutures.com/venture-capital-funding/tinkoff-launches-pulse-social-network-for-investors) — T-Pulse is mobile-app only, no public API documented

---
*Feature research for: Runtime LLM trading agents (v10.0 milestone)*
*Researched: 2026-04-14*

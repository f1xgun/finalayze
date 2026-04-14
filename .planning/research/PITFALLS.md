# Pitfalls Research

**Domain:** Runtime LLM agents in a live Python trading system (MOEX focus)
**Researched:** 2026-04-14
**Confidence:** HIGH — based on direct codebase inspection + verified external sources

---

## Critical Pitfalls

### Pitfall 1: threading.Lock Held Across Async LLM Call Boundary

**What goes wrong:**
`_apply_impact_result` acquires `_sentiment_lock` (a `threading.Lock`) while running
on `_async_loop`. If any code path that holds the lock then awaits an LLM call,
the lock is held across the await yield point. A second coroutine on the same event loop
cannot acquire it — it tries `_lock.acquire()`, which is a blocking call that deadlocks
the entire `_async_loop`.

**Why it happens:**
The TradingLoop uses `threading.Lock` for `_sentiment_cache` because the outer
`_news_cycle` / `_strategy_cycle` methods run on APScheduler `ThreadPoolExecutor` threads.
When batch analysis runs inside `_analyze_impact_batch` (which is an async method on
`_async_loop`), the mixing of `threading.Lock` with `await` expressions is not safe.
The existing code in `_apply_impact_result` actually acquires `_sentiment_lock` with
`with self._sentiment_lock:` and then does Redis/DB awaits. This is a latent deadlock:
if any LLM call were inserted inside that `with` block, it would freeze the loop.

**How to avoid:**
Never hold `threading.Lock` across an `await`. Extract all LLM calls and async
operations to happen either before or after the `with self._sentiment_lock:` block.
The pattern is: compute new scores async (outside lock), then acquire lock, write, release.
For new agents that run exclusively on `_async_loop`, use `asyncio.Lock` for any shared
state accessed only from that loop.

**Warning signs:**
- News cycle hangs indefinitely and `_run_async` timeout fires (30s default)
- `concurrent.futures.TimeoutError` in `_news_cycle` logs
- `_strategy_cycle` shows no sentiment updates despite articles being processed

**Phase to address:**
Phase 1 (News Pipeline) — verify lock acquisition pattern before adding any new
`await` inside `_apply_impact_result` or downstream handlers.

---

### Pitfall 2: LLM Timeout Blocking the News Cycle, Starving Strategy Execution

**What goes wrong:**
`_news_cycle` is scheduled on the APScheduler `"news"` executor (separate
`ThreadPoolExecutor`). But `_run_async(coro, timeout=1800)` is a blocking call from
that thread. If the LLM batch takes longer than the strategy cycle interval, the news
cycle job occupies its executor slot. With `max_workers=1` on the `"news"` executor,
the next news cycle job queues and grows unbounded. Worse: if the batch timeout is
extended and an article's LLM call hangs, the entire batch waits.

**Why it happens:**
OpenRouter free-tier models have unpredictable latency (sometimes 30-120 seconds per
call). The existing semaphore caps at 5 concurrent articles, but with 30+ articles per
cycle and a per-call timeout not set at the HTTP client level, the batch can exceed
the scheduler interval.

**How to avoid:**
Set a hard per-call timeout on the LLM HTTP client (httpx `timeout=` parameter or
OpenAI `timeout=`). The recommended value for news analysis is 15 seconds per article.
Add an article-level circuit breaker (already exists at 5 consecutive failures) and
cap total batch size at 20 articles to bound worst-case duration. Log batch duration
as a metric so alerts fire when news_cycle_duration > 80% of the cycle interval.

**Warning signs:**
- Log line `news_processing_circuit_opened` firing every cycle
- APScheduler job queue depth > 1 for `"news"` executor
- Strategy cycle logs showing `sentiment_score=0.0` for all tickers consistently

**Phase to address:**
Phase 1 (News Pipeline) — add per-request timeout to LLM client constructor.

---

### Pitfall 3: LLM Hallucinated Ticker Extraction Creates Ghost Signals

**What goes wrong:**
The entity extractor maps free-text Russian news to MOEX tickers. LLMs regularly
hallucinate plausible but non-existent tickers (e.g., "GAZPM" instead of "GAZP",
or entirely invented symbols for companies not on MOEX). When
`EventDrivenStrategy.generate_signal()` receives a hallucinated ticker, it produces
a `Signal` with that symbol. If the symbol is not in `InstrumentRegistry`, the
pre-trade check rejects it — but only if the registry lookup is strictly enforced.
Any gap in the pre-trade gate lets a hallucinated ticker reach the broker.

**Why it happens:**
LLMs have no authoritative MOEX symbol list in context. The model knows Russian
companies by legal name, not their exchange ticker. Mapping ambiguous company names
to exact MOEX tickers is a known LLM failure mode, especially for subsidiaries,
preferred share classes (SBERP vs SBER), and companies with recent renames
(YNDX -> YDEX, FIVE removed, HHRU -> HEAD).

**How to avoid:**
Post-process entity extraction output with a whitelist filter: only tickers present
in `InstrumentRegistry` are passed to `EventDrivenStrategy`. Reject anything not in
the registry before touching the sentiment cache. Log rejected tickers with reason
`entity_not_in_registry` to detect extraction drift over time.

**Warning signs:**
- Log entries for symbols that never appear in strategy cycle instrument lists
- `entity_extracted` log events with symbols absent from `InstrumentRegistry.list_all()`
- Increasing rate of `instrument_not_found` errors in execution layer

**Phase to address:**
Phase 1 (News Pipeline) — add registry whitelist validation in the entity extraction
post-processing step before updating `_sentiment_cache`.

---

### Pitfall 4: EventDriven Signal Duplicating cbr_calendar / Dividend Gap Signals

**What goes wrong:**
CBR rate decision news is covered by both the `EventDrivenStrategy` (via sentiment
analysis of news articles) and the `cbr_calendar` strategy (deterministic rule using
the yield curve slope). On CBR announcement days, both strategies fire in the same
direction. The `StrategyCombiner` aggregates weighted signals, so the combined
confidence exceeds threshold twice as easily, leading to oversized entries on the exact
day when everyone in the market is already positioned.

Similarly, dividend announcement news triggers `EventDrivenStrategy` BUY signals
at the same time as `dividend_gap` strategy's pre-event positioning. Double-weight
on the same underlying cause is not alpha — it is leverage disguised as diversification.

**Why it happens:**
Each strategy was designed independently without a deduplication layer for correlated
catalysts. The combiner sums weighted signals without checking whether multiple
strategies fired on the same fundamental cause.

**How to avoid:**
Implement a catalyst deduplication tag: when `EventDrivenStrategy` classifies an
event as `"cbr_rate"` or `"dividend"`, suppress or halve its weight in the combiner
if `cbr_calendar` or `dividend_gap` has already fired in the same direction for the
same symbol within the last 2 bars. This is achievable via a combiner hook
(`_on_strategy_signal`) that checks active strategy signals before adding weight.

**Warning signs:**
- Abnormally large combined confidence on CBR decision dates
- `event_driven` and `cbr_calendar` both appearing in `reasoning` field of same Signal
- Position sizes spiking 2x on SBER/MOEX financials on CBR meeting dates

**Phase to address:**
Phase 2 (EventDrivenStrategy activation) — add signal correlation guard in combiner
before enabling event_driven at 15% weight on ru_* segments.

---

### Pitfall 5: Portfolio Review Agent Suggestions Interpreted as Executable Orders

**What goes wrong:**
The Portfolio Review Agent produces a structured Pydantic output containing
recommendations such as `{"action": "REDUCE", "symbol": "LKOH", "pct": 20}`. If
any downstream code iterates these recommendations and passes them to `BrokerRouter`
or `PreTradeChecker`, it bypasses the strategy combiner entirely. The recommendations
have no circuit-breaker gate, no ADX regime check, no stop-loss state check. A single
misconfigured handler can cause the agent to autonomously liquidate positions.

**Why it happens:**
Structured output from an LLM agent looks identical to a trade instruction from a
strategy — both are Python objects with action fields. Developers adding a handler
for agent output often copy the pattern from `_process_signal()` without realizing
the agent output is advisory.

**How to avoid:**
The Portfolio Review Agent output type must be named `PortfolioReviewSuggestion` (not
`Signal`, `Order`, or `Recommendation`). It must contain no `direction`, `confidence`,
or `symbol`+`market_id` fields that match `Signal` or `OrderRequest` schemas.
The handler that processes it must write to a separate Telegram alert channel, not to
`BrokerRouter`. Add a type assertion at the handler entry point:
`assert not isinstance(output, Signal), "PortfolioReviewSuggestion must not reach broker"`

**Warning signs:**
- `order_submitted` log entries with `strategy=portfolio_review`
- Portfolio review output objects appearing in `_cycle_exited_symbols` or signal logs
- Any code path from `_portfolio_review_cycle` touching `self._broker_router`

**Phase to address:**
Phase 3 (Portfolio Review Agent) — enforce at schema design time, not just code review.

---

### Pitfall 6: Anomaly Interpreter LLM Call Blocking Alert Delivery

**What goes wrong:**
`AnomalyDetector.check()` fires synchronously from the strategy cycle. If the anomaly
interpretation requires an LLM call and that call is blocking (or times out in 30s),
the anomaly alert is delayed by the LLM latency. During that window, the drawdown may
deepen further without operator awareness. The primary purpose of the anomaly alert
is speed; adding LLM interpretation before the alert inverts the priority.

**Why it happens:**
The intuitive design is: detect anomaly → get LLM explanation → send alert with
explanation. But this makes the most time-sensitive path (the alert) dependent on
the slowest path (LLM API call).

**How to avoid:**
Fire the raw anomaly alert immediately, then schedule the LLM interpretation as a
fire-and-forget task on `_async_loop`. The operator sees the alert within seconds;
the enriched explanation arrives as a follow-up message 10-30 seconds later. Use
`asyncio.create_task` or `asyncio.run_coroutine_threadsafe` with no blocking wait.
Never `await` the LLM call from inside `AnomalyDetector.check()`.

**Warning signs:**
- Alert latency exceeds 10 seconds from anomaly detection to Telegram message
- `_strategy_cycle` duration increases on anomaly cycles
- `consecutive_equity_errors` counter incrementing without operator awareness

**Phase to address:**
Phase 4 (Anomaly Interpreter Agent) — architecture diagram must show the two-step
pattern (immediate alert + async enrichment) before implementation begins.

---

### Pitfall 7: NewsAnalyzer Using json.loads() Instead of parse_structured()

**What goes wrong:**
`NewsAnalyzer.analyze()` calls `self._llm.complete()` and then parses the response
with `json.loads(raw)`. When the LLM returns a response with a code fence
(triple-backtick json blocks), trailing whitespace, or explanatory text before the JSON,
`json.loads` raises `JSONDecodeError`. The fallback returns `SentimentResult(0.0, 0.0)`,
silently converting a potentially high-confidence article into neutral sentiment.
This is already present in the codebase and will affect all real news once the
pipeline is activated.

**Why it happens:**
`NewsAnalyzer` predates the `parse_structured()` method on `LLMClient`. It was written
before v8.0 when structured output was standardized. The fallback to neutral is safe
from an execution standpoint but represents silent signal loss — the system appears
to work while generating no sentiment signal.

**How to avoid:**
Migrate `NewsAnalyzer.analyze()` to use `self._llm.parse_structured(prompt, system, SentimentResult)`.
This routes through the provider-specific structured output implementation (OpenAI
`beta.chat.completions.parse`, Anthropic `messages.parse`) which guarantees valid
schema adherence. Remove the manual `json.loads` block. Keep the fallback only as
a last resort for `LLMError`.

**Warning signs:**
- `sentiment_score=0.0` for >50% of articles in logs despite clear market-moving news
- `json_decode_error` log events in news_analyzer
- `processed_fail` counter in `news_cycle_complete` log consistently non-zero

**Phase to address:**
Phase 1 (News Pipeline) — fix NewsAnalyzer before activating the live feed, not after.

---

### Pitfall 8: Sentiment Cache Decay Causing Stale Signals During Low-News Periods

**What goes wrong:**
`_SENTIMENT_HALF_LIFE_HOURS = 4.0` applies exponential decay to sentiment scores.
On weekends, holidays, and MOEX low-news windows, the sentiment cache decays to zero
over 12-16 hours. When a new article arrives after this quiet period, the EWM
combination formula `existing * 0.7 + new_score * 0.3` starts from near-zero,
causing the first article to have outsized impact relative to its actual significance.
A single article from a low-credibility source after a weekend can spike sentiment
to 0.3 * credibility on tickers that had no prior signal.

**Why it happens:**
The decay lambda was tuned for continuous intraday news flow. MOEX operates on
10:00-18:45 MSK; on weekdays after close and during weekends there is no data to
sustain the cache. The first article of the trading day starts from an artificially
deflated baseline.

**How to avoid:**
Apply sentiment decay only during market hours. Check `is_market_open_now()` before
decaying scores; freeze the decay clock outside market hours. Alternatively, increase
the half-life to 24h to cover overnight gaps. The strategy-level `min_sentiment`
threshold (0.5) provides a secondary guard, but the first article of the day can
still produce a sub-threshold spike that warps the EWM.

**Warning signs:**
- `event_driven` signals firing exclusively on the first news article each morning
- Sentiment score jumping from 0.01 to 0.28 on single article
- Strategy firing on Monday morning news that pre-dates Friday market close

**Phase to address:**
Phase 2 (EventDrivenStrategy activation) — validate decay behavior on a week of
simulated intraday data including overnight gaps before enabling event_driven on
live segments.

---

### Pitfall 9: LLM API Downtime During Trading Hours Silently Disables News Signal

**What goes wrong:**
OpenRouter (the current provider) has had outages. During an outage, `_news_cycle`
catches `LLMError` and logs a warning, but `_sentiment_cache` is not updated.
The strategy cycle continues running with stale (decayed to zero) sentiment. The
system does not alert operators that the LLM pipeline is down; it silently degrades
to technical-signal-only mode. This is acceptable behavior but must be observable.

**Why it happens:**
The `FallbackLLMClient` handles rate limits but falls through both primary and
fallback on a full outage. The `_analyze_impact_batch` inline circuit breaker
suppresses retries after 5 consecutive failures, which is correct, but does not
surface the degradation to the health monitor or Telegram alerter.

**How to avoid:**
Wire `HealthMonitor` to track `last_successful_llm_call` timestamp. When the LLM
pipeline has been down for more than one strategy cycle interval, `HealthMonitor`
should fire a Telegram alert: "News sentiment pipeline degraded — LLM unavailable
for Xm. Running on technical signals only." This converts silent failure into
observable degradation. The trading system continues; operators are aware.

**Warning signs:**
- `news_processing_circuit_opened` in logs without corresponding Telegram alert
- `_sentiment_cache` entries all at zero despite strategy cycle running
- `HealthMonitor.feed_timestamps["llm"]` not updated for > 30 minutes

**Phase to address:**
Phase 1 (News Pipeline) — add LLM liveness tracking to HealthMonitor at the same
time as the news pipeline goes live.

---

### Pitfall 10: Per-Article LLM Cost Explosion on High-Volume News Days

**What goes wrong:**
On MOEX-relevant macro events (CBR rate decisions, geopolitical escalations, sanctions
announcements), RSS feeds and Telegram channels produce 50-200 articles per cycle.
With OpenRouter free tier having daily request caps (~50 RPD for some models), hitting
the cap mid-day silences the sentiment pipeline for the remainder of the trading day.
Upgrading to paid tier at $0.03/1k tokens means 200 articles * ~500 tokens = 100k
tokens per cycle * 12 cycles/day = $36/day, exceeding operational cost budgets.

**Why it happens:**
Per-article LLM calls scale linearly with news volume. High-volatility days produce
the most news and the highest LLM costs, exactly when accurate sentiment matters most.

**How to avoid:**
Implement article prioritization: score articles by title keywords before sending to
LLM (regex match for CBR, sanctions, earnings tickers). Only send top-N articles
(N=10 recommended) to the LLM per cycle; deprioritize routine earnings and
boilerplate agency reports. Use the `_ARTICLE_DEDUP_MAX_SIZE` cap aggressively.
Log `llm_calls_skipped_budget` as a metric for cost tracking.

**Warning signs:**
- Daily LLM request counter reaching provider limit before market close
- `LLMRateLimitError` appearing after previously-successful batches
- Processed article count per cycle unexpectedly high on volatility events

**Phase to address:**
Phase 1 (News Pipeline) — add article scoring/prioritization layer before LLM batch.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| `json.loads()` in NewsAnalyzer (existing) | Simple, works for compliant LLMs | Silent signal loss on any format variation | Never — migrate to `parse_structured()` in Phase 1 |
| No hit/miss rate tracking on `_CachingLLMClient` | Less code | Cannot detect cache thrashing or cache-miss spikes causing cost explosions | Never for production — add cache metrics in Phase 1 |
| `_event_driven_active` cached once at startup | Avoids YAML re-reads | Does not reflect runtime preset changes via `PresetApplicator` | Acceptable for MVP; fix when dynamic preset updates are added |
| Hardcoded `_SENTIMENT_HALF_LIFE_HOURS = 4.0` | Simple | Wrong for 18h overnight gaps; first-article-of-day distortion | Acceptable until Phase 2 validation confirms or refutes |
| No LLM liveness in HealthMonitor | Simpler health monitor | Silent degradation during LLM outages goes unnoticed | Never for production — add in Phase 1 alongside news pipeline |
| Article credibility hardcoded to 1.0 on RSS | Simpler ingestion | All sources treated equal; Interfax press releases weighted same as TASS emergency alerts | Acceptable for MVP; tune per-source credibility in Phase 2 |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| OpenRouter free tier | Treating free tier as production-grade SLA | Set `llm_fallback_provider` to Groq (also free, different infrastructure) for resiliency |
| APScheduler + async LLM | Running `await llm.complete()` directly inside APScheduler job | Bridge via `_run_async()` — APScheduler jobs are sync; async work must go through the background loop |
| `_sentiment_lock` (threading.Lock) in async context | Acquiring the lock inside an async coroutine, then awaiting | Compute values outside lock, then acquire lock only for the dict write (no `await` inside `with _sentiment_lock`) |
| `_analyze_impact_batch` on `_async_loop` | Calling `_run_async()` from inside a coroutine running on `_async_loop` | Use `await` directly or `asyncio.create_task` — `_run_async` submits to the same loop, causing deadlock |
| Telegram channel reader (Telethon) | Sharing Telethon session across multiple event loops | Telethon client must be created and used on the same event loop; use `_async_loop` consistently |
| HealthMonitor feed freshness | Not wiring LLM pipeline to HealthMonitor timestamp tracking | Register `"llm"` as a feed in `HealthMonitor.update_feed_timestamp()` after each successful news batch |
| `InstrumentRegistry` symbol validation | Skipping registry check for LLM-extracted tickers | All entity extractor outputs must be validated against `InstrumentRegistry` before touching sentiment cache |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Per-article LLM call with no prioritization | LLM cost spike on high-news days, rate limit hit | Pre-score articles, cap batch at N=10 | At 30+ articles per cycle (~50+ on macro events) |
| `parse_structured()` bypass in NewsAnalyzer | 40-60% of articles return 0.0 sentiment silently | Migrate to `parse_structured()` | Immediately on live deployment with any LLM non-compliance |
| `asyncio.Semaphore(5)` for batch concurrency | Queue depth grows if per-call latency > 3s; 30 articles * 6s = 3min batch | Add per-call timeout at HTTP level; reduce semaphore for free-tier models | At LLM latency > 5s per call (free tier: common) |
| `_CachingLLMClient` LRU cache with no TTL | Stale cached responses for articles that recur (e.g. recurring agency wire) | Cache key includes full content hash — body changes break cache naturally; acceptable | Cache hit rate misleadingly high if headlines repeat |
| Portfolio Review Agent running during market hours | Competes with strategy cycle for `_async_loop` execution | Schedule portfolio review only after market close (18:45 MSK) | If scheduled during trading hours on busy news days |

---

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| Prompt injection via RSS article content | Malicious article content overrides system prompt, causes agent to emit invalid signals or dump credentials | Sanitize article content: strip HTML, cap at 2000 chars, never inject raw content into system prompt |
| LLM output containing executable Python or eval() targets | Agent output used as dynamic config or eval'd | All agent output typed as Pydantic models; never `eval()` or `exec()` LLM responses |
| API key leakage in structured LLM prompts | System prompt logging exposes provider keys | Never include `settings.llm_api_key` in logged prompts; log prompt hash only |
| Portfolio Review Agent with write access to PresetApplicator | Agent suggestions trigger automatic strategy parameter changes | `PortfolioReviewSuggestion` must not contain fields that match `PresetApplicator` input schema |

---

## "Looks Done But Isn't" Checklist

- [ ] **NewsAnalyzer**: Uses `parse_structured()`, not `json.loads()` — verify `news_analyzer.py` does not call `json.loads(raw)` on LLM response
- [ ] **Ticker validation**: Entity extractor output filtered against `InstrumentRegistry` before sentiment cache update — verify `_apply_impact_result` rejects unknown tickers
- [ ] **LLM liveness**: `HealthMonitor` tracks `last_llm_success` and fires Telegram alert on silence > 30min — verify feed registered in HealthMonitor
- [ ] **Portfolio Review advisory-only**: No code path from portfolio review output reaches `BrokerRouter` — verify with type assertions + integration test
- [ ] **Anomaly Interpreter non-blocking**: Alert fired before LLM interpretation, not after — verify Telegram message timestamp vs LLM call start timestamp in tests
- [ ] **Sentiment lock safety**: No `await` expression inside any `with self._sentiment_lock:` block — verify with grep `"with self._sentiment_lock"` in trading_loop.py
- [ ] **Per-call LLM timeout**: HTTP client has request-level timeout <= 15s — verify in LLM client constructor or httpx settings
- [ ] **Duplicate signal guard**: CBR/dividend event_driven signals suppressed when cbr_calendar/dividend_gap active in same direction — verify combiner hook logic
- [ ] **Decay clock frozen off-hours**: Sentiment decay only runs during MOEX market hours — verify `_SENTIMENT_DECAY_LAMBDA` application is gated on market hours check
- [ ] **Article budget cap**: LLM batch capped at N=10 articles per cycle — verify `_analyze_impact_batch` has explicit article count limit

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| threading.Lock deadlock in async | HIGH | Requires trading loop restart; if in production, trigger kill switch, fix lock pattern, redeploy |
| LLM timeout blocking news cycle | LOW | Reduce per-call timeout; reduce semaphore concurrency; deploy config change without restart |
| Hallucinated ticker in signal | LOW | Existing `InstrumentRegistry` check prevents execution; add registry validation in entity extractor post-processing |
| Portfolio review executing orders | HIGH | Requires immediate kill switch, position audit, manual unwinding of any orphan positions |
| NewsAnalyzer json.loads failure | LOW | Fix: migrate to `parse_structured()` in single file; deploy; no data migration needed |
| LLM API downtime | LOW | `FallbackLLMClient` handles rate limits; for full outage, system degrades gracefully to technical signals only |
| Cost explosion on news spike | LOW | Reduce `_analyze_impact_batch` article cap to 5; monitor `processed_ok` count in logs |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| threading.Lock across await | Phase 1: News Pipeline | No `with _sentiment_lock:` block contains any `await`; static analysis check |
| LLM timeout blocking cycle | Phase 1: News Pipeline | Per-call timeout test: mock slow LLM returning after 20s; verify cycle completes in < 35s |
| Hallucinated ticker extraction | Phase 1: News Pipeline | Integration test: entity extractor output for unknown tickers rejected at sentiment cache boundary |
| cbr_calendar duplicate signal | Phase 2: EventDrivenStrategy activation | Backtest CBR announcement dates; confirm combined confidence does not exceed 2x individual strategy |
| Portfolio review autonomous execution | Phase 3: Portfolio Review Agent | Architecture review + type assertion test proving no `Signal`/`OrderRequest` produced |
| Anomaly interpreter blocking alert | Phase 4: Anomaly Interpreter Agent | Latency test: anomaly detected to Telegram message < 3s; LLM enrichment arrives asynchronously |
| NewsAnalyzer json.loads | Phase 1: News Pipeline | Unit test: mock LLM returning code-fence-wrapped response; verify `SentimentResult` parsed correctly |
| Sentiment decay first-article distortion | Phase 2: EventDrivenStrategy activation | Simulation: run news cycle at 09:00 MSK after 16h gap; verify first article confidence <= min_sentiment threshold |
| LLM API downtime silent failure | Phase 1: News Pipeline | HealthMonitor test: simulate LLM failure for 35min; verify Telegram alert fired |
| Per-article cost explosion | Phase 1: News Pipeline | Stress test: inject 100 articles; verify only N=10 reach LLM; verify LLM call counter |

---

## Sources

- Direct codebase inspection: `src/finalayze/orchestration/trading_loop.py`, `src/finalayze/analysis/news_analyzer.py`, `src/finalayze/analysis/llm_client.py`, `src/finalayze/strategies/event_driven.py`, `src/finalayze/risk/position_sizing_pipeline.py`, `src/finalayze/monitoring/anomaly_detector.py`
- Expert debate findings (v10.0 milestone, 2 rounds, 5 agents): documented in `.planning/PROJECT.md` milestone context
- [TradeTrap: Are LLM-based Trading Agents Truly Reliable and Faithful?](https://arxiv.org/html/2512.02261v1) — LLM trading agent robustness under adversarial conditions
- [Auditing LLM Agents in Finance Must Prioritize Risk](https://arxiv.org/pdf/2502.15865) — hallucination, systemic bias, error propagation in multi-step agent chains
- [LLM Hallucinations: Implications for Financial Institutions](https://biztechmagazine.com/article/2025/08/llm-hallucinations-what-are-implications-financial-institutions) — $250M+ annual losses from hallucination-related incidents
- [Using a Threading Lock in Asyncio Results in a Deadlock](https://superfastpython.com/asyncio-use-threading-lock/) — technical explanation of threading.Lock + asyncio deadlock pattern
- [Limitations of News Sentiment Analysis in Stock Return Prediction](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5086825) — signal quality, stale news overreaction, decay dynamics
- [Sentiment trading with large language models](https://arxiv.org/abs/2412.19245) — LLM sentiment signals: accuracy, latency sensitivity, signal degradation

---
*Pitfalls research for: Runtime LLM agents in live MOEX trading system (v10.0)*
*Researched: 2026-04-14*

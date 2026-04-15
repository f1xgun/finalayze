# Phase 51: Anomaly Interpreter Agent - Context

**Gathered:** 2026-04-15
**Status:** Ready for planning

<domain>
## Phase Boundary

Build statistical anomaly detection (>3σ price moves, volume spikes vs 20-day rolling) that fires a raw Telegram alert immediately, then dispatches a fire-and-forget LLM enrichment as a follow-up message labeled "AI interpretation (unverified)".

</domain>

<decisions>
## Implementation Decisions

### Anomaly Detection Approach
- No AnomalyDetector class exists yet — this is greenfield
- Detect >3σ price moves and volume spikes vs 20-day rolling mean/std in `_process_instrument()` after candle fetch
- Raw alert = `TelegramAlerter.send_alert()` with ticker, magnitude, direction — fires IMMEDIATELY before any LLM call
- Keep detection as a helper function or lightweight class within the trading loop module — no complex architecture needed
- Existing alert patterns: `on_trade_filled()`, `on_error()`, `on_circuit_breaker_trip()` all use `send_alert()` which is already fire-and-forget via `loop.create_task()`

### LLM Enrichment Pattern
- Fire-and-forget via `loop.create_task()` — matches existing TelegramAlerter pattern
- LLM prompt includes: ticker, price move %, volume ratio, recent news headlines if available from sentiment cache
- Follow-up message format: "AI interpretation (unverified): {explanation}"
- 30s timeout on LLM call via `asyncio.wait_for()`
- On failure: log `anomaly_llm_failure` via structlog, do NOT send follow-up message
- Raw alert is NEVER delayed — send_alert() happens synchronously before create_task(llm_enrichment)

### Claude's Discretion
- Whether to create a new `AnomalyDetector` class in a separate file or keep as functions in trading_loop.py
- Exact σ threshold (3σ suggested, adjustable)
- Exact rolling window (20-day suggested)
- LLM prompt wording and context included
- Whether to add `on_anomaly()` method to TelegramAlerter or use generic `send_alert()`

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `TelegramAlerter` in `core/alerts.py` — fire-and-forget `send_alert()` via `loop.create_task()`
- `LLMClient` ABC in `analysis/llm_client.py` — `async complete()`, `async parse_structured()`
- `create_llm_client(settings)` factory for instantiation
- Existing fire-and-forget pattern: `loop.create_task(coro)` with `# noqa: RUF006`
- `asyncio.run_coroutine_threadsafe()` for bridging sync→async in trading loop

### Established Patterns
- All alerts go through TelegramAlerter.send_alert() → _send() → httpx POST
- Errors are always suppressed in alert path — never crash the caller
- structlog for all logging with bound context
- Prometheus counters in api/metrics.py for observability

### Integration Points
- `TradingLoop._process_instrument()` — after candle fetch, check for anomalies
- `TelegramAlerter` — send raw alert + follow-up LLM message
- `LLMClient` — async LLM call for interpretation
- `MetricsCollector` — optional Prometheus counter for anomaly events

</code_context>

<specifics>
## Specific Ideas

- Success criterion explicitly requires: "a unit test asserting that TelegramAlerter.send() is called before any LLM await"
- The "AI interpretation (unverified)" label is mandatory per success criteria
- `anomaly_llm_failure` structlog entry is the specific key required
- "Suppressing the raw alert on LLM failure is impossible by design" — architectural guarantee, not just error handling

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

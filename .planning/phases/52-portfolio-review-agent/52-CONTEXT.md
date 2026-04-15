# Phase 52: Portfolio Review Agent - Context

**Gathered:** 2026-04-15
**Status:** Ready for planning

<domain>
## Phase Boundary

Build a daily advisory-only LLM portfolio review that runs at 19:00 MSK (after MOEX close), sends a structured Telegram report, and has zero write path to the order pipeline. Safety enforced by schema design and code-level verification.

</domain>

<decisions>
## Implementation Decisions

### Schema Safety
- `PortfolioReviewResult` Pydantic schema with position summaries, concentration warnings, catalyst list
- Schema MUST NOT have `direction`, `confidence`, or `symbol`+`market_id` combination that matches `Signal` or `OrderRequest`
- Type-checker assertion at handler entry prevents trade-like fields from being added
- Code-grep verification: `BrokerRouter`, `place_order`, `generate_signal` must return zero results inside handler

### Scheduling & Output
- Scheduled via APScheduler at 19:00 MSK daily (after MOEX close at 18:40 MSK)
- LLM receives: current positions, daily P&L, sector/ticker concentration, upcoming events/catalysts
- Output: structured Telegram message with clear sections (not free-form prose)
- Handler writes ONLY to `TelegramAlerter` — no other output path

### Claude's Discretion
- Exact Telegram message format and section layout
- LLM prompt design and context selection
- Whether to create a separate `PortfolioReviewAgent` class or use a scheduled function
- How to gather portfolio state (direct broker query vs cached positions)

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `TelegramAlerter` in `core/alerts.py` — fire-and-forget `send_alert()`
- `LLMClient` in `analysis/llm_client.py` — `async complete()`, `parse_structured()`
- `APScheduler BackgroundScheduler` in `core/trading_loop.py` — existing job scheduling
- Pattern from Phase 51: fire-and-forget LLM enrichment, graceful degradation

### Established Patterns
- Pydantic v2 frozen schemas for all data contracts
- Advisory agents: fire-and-forget async, no blocking the main loop
- structlog for all logging

### Integration Points
- APScheduler in TradingLoop for 19:00 MSK daily job
- TelegramAlerter for output
- LLMClient for portfolio analysis
- Broker/position state for portfolio data

</code_context>

<specifics>
## Specific Ideas

- Success criterion explicitly requires: code search for BrokerRouter/place_order/generate_signal returns zero results
- PortfolioReviewResult must be verifiably distinct from Signal and OrderRequest schemas
- "Structured, not free-form prose" — sections with specific data points, not paragraphs
- STATE.md notes: "PortfolioReviewAgent: advisory-only schema enforced — no direction/confidence/symbol+market_id fields"

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

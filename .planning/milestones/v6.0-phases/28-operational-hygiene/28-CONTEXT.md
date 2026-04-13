# Phase 28: Operational Hygiene - Context

**Gathered:** 2026-03-30
**Status:** Ready for planning

<domain>
## Phase Boundary

Eliminate confounding factors before architectural changes: fix stale tickers, add market-hours gate to strategy cycles, deduplicate LLM articles, make Telegram alerter startup-safe.

</domain>

<decisions>
## Implementation Decisions

### Claude's Discretion

All implementation choices are at Claude's discretion — pure infrastructure phase.

Key constraints from sandbox analysis:
- MOEX market hours: 07:00-15:45 UTC (already defined in codebase as constants)
- Stale tickers: FIVE, FIXP, POLY to remove; YNDX→YDEX; HHRU→HH (verify FIGI via T-Bank API)
- Article dedup: SHA-256 hash on URL or content, OrderedDict with 24h TTL, in-memory (single process)
- Telegram alerter: wrap send_alert in try/except at startup, log warning, continue

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/finalayze/markets/schedule.py` — MOEX market hours constants
- `config/segments.py` — ticker definitions for all ru_* segments
- `src/finalayze/analysis/news_impact_analyzer.py` — news processing pipeline
- `src/finalayze/api/alerts.py` — TelegramAlerter class

### Established Patterns
- Market hours check already exists for health monitor (`feed: off-hours`)
- Telegram alerter already has try/except for individual send failures
- News pipeline has `_seen_urls` set in Telegram reader (URL-based dedup)

### Integration Points
- `src/finalayze/orchestration/trading_loop.py` — strategy_cycle method needs market-hours guard
- `config/segments.py` — ticker lists for ru_blue_chips, ru_energy, etc.
- News cycle in TradingLoop — article dedup before LLM call
- `scripts/run_sandbox.py` — Telegram alerter startup

</code_context>

<specifics>
## Specific Ideas

No specific requirements — infrastructure phase.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

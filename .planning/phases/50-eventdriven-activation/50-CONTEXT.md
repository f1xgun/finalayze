# Phase 50: EventDriven Activation - Context

**Gathered:** 2026-04-15
**Status:** Ready for planning

<domain>
## Phase Boundary

Enable EventDrivenStrategy on all ru_* segments with weight 0.15, add CBR/dividend duplicate-signal protection in the combiner, and freeze sentiment decay during MOEX closed hours.

</domain>

<decisions>
## Implementation Decisions

### CBR Duplicate-Signal Protection
- No `cbr_calendar` strategy exists in the codebase — the success criterion's reference to it is aspirational
- Duplicate suppression happens in StrategyCombiner: if event_driven + another strategy both fire on the same ticker with the same `cbr_rate` event type in one cycle, zero the lower-weight signal
- Detection via `event_types` field in `Signal.features` dict (already populated by EventDrivenStrategy)
- Suppression scope: same ticker + same cycle only — cross-ticker and cross-cycle signals are independent

### Sentiment Decay & Market Hours
- Current decay is Redis TTL (30min binary expiry in `data/cache.py`), not a mathematical half-life curve
- "Freezing" = extend TTL when market closes so the last sentiment survives until next open; resume normal 30min TTL at open
- Use `MOEX_MARKET_SCHEDULE.is_market_open()` from `markets/schedule.py` for gating
- Weight change in all ru_* presets: `event_driven.weight: 0.10` → `0.15` per success criteria
- Last sentiment before close is preserved via extended TTL until next trading session open

### Claude's Discretion
- Exact TTL extension duration during closed hours (recommend: time until next MOEX open + 30min buffer)
- Whether to add dividend-specific duplicate suppression or only CBR (recommend: both, since mechanism is the same)
- How to structure the combiner deduplication (hook vs inline check in generate_signal loop)

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `EventDrivenStrategy` in `strategies/event_driven.py` — accepts `credibility` param, already functional
- `StrategyCombiner` in `strategies/combiner.py` — synchronous signal collection loop, no existing dedup
- `MOEX_MARKET_SCHEDULE` in `markets/schedule.py` — `is_market_open()` for gating
- Preset YAMLs in `strategies/presets/ru_*.yaml` — all have `event_driven` section (disabled, weight=0.10)
- `SentimentCache` in `data/cache.py` — Redis TTL-based, `_SENTIMENT_TTL_SECONDS = 1800`

### Established Patterns
- Strategy presets are YAML-configured with `enabled`, `weight`, `min_sentiment` fields
- Combiner iterates strategies synchronously, collects Signal objects, normalizes by weight
- Signal.features dict carries metadata (event_types, strategy-specific data)

### Integration Points
- Preset YAMLs: flip `enabled: true`, set `weight: 0.15` in all 4 ru_* files
- StrategyCombiner.generate_signal(): add dedup check after collecting all signals
- SentimentCache: modify TTL logic based on market hours
- trading_loop.py: pass credibility from NewsArticle through to EventDrivenStrategy

</code_context>

<specifics>
## Specific Ideas

- STATE.md research flag: "Verify StrategyCombiner._on_strategy_signal hook has access to other active signals" — RESOLVED: no such hook exists. Combiner collects signals synchronously in a loop. Dedup must be post-collection.
- EventDrivenStrategy already has price-move guard (>5% suppresses signal as "already priced in")
- The ±10% tolerance on criterion 3 suggests approximate preservation is fine, not exact value retention

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

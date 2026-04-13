# Phase 38: PresetApplicator + Auto-Apply Loop - Context

**Gathered:** 2026-04-12
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase delivers `PresetApplicator` for atomic YAML write-back with safety gates (circuit breaker, position ownership, sandbox validation), INCONCLUSIVE Telegram routing, `_entry_strategy` tracking in TradingLoop, `combiner.invalidate_segment_cache()`, and the `POST /experiments/{id}/apply` REST endpoint.

</domain>

<decisions>
## Implementation Decisions

### PresetApplicator Design
- Lives in `orchestration/preset_applicator.py` (Layer 5) — reads ExperimentManager, writes YAML presets
- Backup naming: `{segment}.yaml.bak.{ISO-timestamp}` in same directory as original
- Key validation: check keys exist in current YAML + value types match — reject unknown keys
- Deep merge: only overrides specified keys, preserves all other settings
- Atomic write: write to `{segment}.yaml.pending`, then `os.replace()` to final path

### Safety Gates
- Circuit breaker check is FIRST line of `apply_verdict()` — import CircuitBreaker from `risk/`, check level. Raise if `CircuitLevel != NORMAL`
- `_entry_strategy`: Dict `{symbol: strategy_name}` in TradingLoop, set on fill, cleared on close
- `_entry_strategy` lives in `orchestration/trading_loop.py` in the order execution path
- Block strategy-disable auto-apply if `_entry_strategy` has positions for that strategy
- INCONCLUSIVE → use existing `TelegramAlerter.send_alert()` with priority=HIGH, includes experiment_id and key metrics

### Sandbox Gate + Cache Invalidation
- `SandboxGate` class with `check(experiment_id) -> bool` — reads sandbox monitor metrics
- Pass criteria: 3+ trading days with fill_rate > 0 AND no circuit breaker trips during the period
- `combiner.invalidate_segment_cache()`: clear the `_presets` dict for the affected segment
- Call invalidate immediately after `os.replace()` in PresetApplicator

### Claude's Discretion
- Internal error handling in PresetApplicator (rollback on failure vs leave pending file)
- SandboxGate metric collection mechanism (read from DB or sandbox monitor API)
- Telegram alert message formatting for INCONCLUSIVE verdicts
- REST endpoint response model structure for apply results

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `core/experiment_manager.py` — ExperimentManager with verdict computation, `preset_overrides` field on ExperimentState
- `risk/circuit_breaker.py` — CircuitBreaker with CircuitLevel enum (NORMAL, CAUTION, CRITICAL)
- `orchestration/trading_loop.py` — TradingLoop with `_stop_states`, `_entry_prices` dicts (pattern for `_entry_strategy`)
- `strategies/combiner.py` — StrategyCombiner with per-segment preset loading
- `monitoring/sandbox_monitor.py` — SandboxMonitor with metric collection
- `api/telegram_bot.py` — TelegramAlerter with `send_alert()` method
- `api/v1/experiments.py` — existing read-only experiments router (from Phase 37)

### Established Patterns
- Atomic file write: `os.replace()` used in DebateManager for safe file operations
- YAML read/write: `yaml.safe_load()` + `yaml.dump()` in existing preset loading
- Strategy presets: `src/finalayze/strategies/presets/*.yaml` per segment
- Dict-based position tracking: `_stop_states: dict[str, StopState]`, `_entry_prices: dict[str, float]`

### Integration Points
- `orchestration/preset_applicator.py` imports from Layer 0 (schemas, experiment_manager) and Layer 4 (risk/circuit_breaker)
- `orchestration/trading_loop.py` gets `_entry_strategy` dict
- `strategies/combiner.py` gets `invalidate_segment_cache()` method
- `api/v1/experiments.py` gets `POST /{id}/apply` endpoint
- `api/telegram_bot.py` used for INCONCLUSIVE alerts

</code_context>

<specifics>
## Specific Ideas

- The staging file pattern (`*.yaml.pending` → `os.replace()`) prevents partial writes from being visible to the strategy cycle
- Backup files should never be cleaned up automatically — operator reviews and removes manually
- SandboxGate should use the same metrics that go-live-scorecard skill checks

</specifics>

<deferred>
## Deferred Ideas

- Auto-revert on post-apply metric degradation (rollback gate) — deferred to v8.x
- Scheduled orchestrator + auto-apply loop (cron-based) — deferred to v8.x
- `ou_mean_reversion` and `pairs` strategies blocked from auto-apply by schema flag — deferred to v8.x

</deferred>

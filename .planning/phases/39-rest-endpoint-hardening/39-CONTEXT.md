# Phase 39: REST Endpoint Hardening - Context

**Gathered:** 2026-04-12
**Status:** Ready for planning
**Source:** Gap closure from v8.0-MILESTONE-AUDIT.md

<domain>
## Phase Boundary

Close REST API gaps identified by v8.0 audit: wire real Telegram alerter into /apply endpoint, inject circuit breaker state, fix multi-debate response truncation, and add POST /debates/{id}/finalize endpoint.

</domain>

<decisions>
## Implementation Decisions

### Audit Gap Fixes (all locked)
- POST /experiments/{id}/apply must use a real or injectable TelegramAlerter — not no-op. Use FastAPI dependency injection with a factory that creates a real alerter from settings.
- POST /experiments/{id}/apply must check circuit breaker state from a shared source. Use a singleton CircuitBreaker registry or read from DB/Redis. If no live breakers available (test/debug mode), log warning and proceed.
- POST /debates response must include `debate_ids: list[str]` containing ALL created debate IDs — not just the first one.
- POST /debates/{id}/finalize endpoint must accept a FactCheckReport JSON body and call AgentOrchestrator.finalize_debate() — makes the arbiter-to-experiment loop REST-triggerable.

### Claude's Discretion
- All implementation choices are at Claude's discretion — gap closure phase with clear audit-defined scope
- Whether to use FastAPI dependency injection or module-level factory for alerter/circuit breaker wiring
- Response model structure for finalize endpoint

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `api/v1/experiments.py` — existing experiments router with POST /apply (needs modification)
- `api/v1/debates.py` — existing debates router (needs finalize endpoint + response fix)
- `orchestration/agent_orchestrator.py` — AgentOrchestrator.finalize_debate()
- `orchestration/preset_applicator.py` — PresetApplicator with injectable alerter/circuit_breakers
- `api/telegram_bot.py` — TelegramAlerter.send_alert()
- `risk/circuit_breaker.py` — CircuitBreaker, CircuitLevel

### Integration Points
- `api/v1/experiments.py` — replace no-op alerter with real one, inject circuit breakers
- `api/v1/debates.py` — fix CreateDebateResponse, add finalize endpoint
- `api/v1/router.py` — already includes both routers

</code_context>

<specifics>
## Specific Ideas

No specific requirements beyond audit gap closure.

</specifics>

<deferred>
## Deferred Ideas

None — gap closure phase covers all audit items.

</deferred>

---
phase: 15
slug: schemas-config-and-rollout-foundation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-21
---

# Phase 15 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | pyproject.toml |
| **Quick run command** | `uv run pytest tests/unit/test_rollout.py tests/unit/test_pre_trade_check.py tests/unit/test_circuit_breaker.py -x -q` |
| **Full suite command** | `uv run pytest tests/ -x -q` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/unit/test_rollout.py tests/unit/test_pre_trade_check.py tests/unit/test_circuit_breaker.py -x -q`
- **After every plan wave:** Run `uv run pytest tests/ -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 15-01-01 | 01 | 1 | ROLL-01 | unit | `uv run pytest tests/unit/test_rollout.py -x -q` | ❌ W0 | ⬜ pending |
| 15-01-02 | 01 | 1 | ROLL-02 | unit | `uv run pytest tests/unit/test_pre_trade_check.py tests/unit/test_circuit_breaker.py -x -q` | ✅ | ⬜ pending |
| 15-02-01 | 02 | 2 | ROLL-03 | integration | `uv run python scripts/validate_capital_ladder.py` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/test_rollout.py` — stubs for ROLL-01 (RolloutPhase enum, RolloutLimits, Settings integration)
- [ ] `tests/unit/test_capital_ladder.py` — stubs for ROLL-03 (lot size validation at capital tiers)

*Existing test infrastructure covers PreTradeChecker and CircuitBreaker (ROLL-02).*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| End-to-end rollout phase switch | ROLL-02 | Requires running system with env var | Set FINALAYZE_ROLLOUT_PHASE=MINIMAL, start system, verify log output shows tightened limits |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending

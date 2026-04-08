---
phase: 34
slug: experiment-registry-runner
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-08
---

# Phase 34 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (already configured) |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `uv run pytest tests/unit/core/test_experiment_schemas.py tests/unit/core/test_experiment_manager.py -x` |
| **Full suite command** | `uv run pytest tests/unit/core/ -x` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/unit/core/test_experiment_schemas.py tests/unit/core/test_experiment_manager.py -x`
- **After every plan wave:** Run `uv run pytest tests/unit/core/ -x`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 5 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 34-01-01 | 01 | 1 | EXP-01 | unit | `uv run pytest tests/unit/core/test_experiment_schemas.py -x` | ❌ W0 | ⬜ pending |
| 34-01-02 | 01 | 1 | EXP-01, EXP-04 | unit | `uv run pytest tests/unit/core/test_experiment_manager.py -x` | ❌ W0 | ⬜ pending |
| 34-02-01 | 02 | 2 | EXP-02 | unit | `uv run pytest tests/unit/core/test_experiment_runner.py -x` | ❌ W0 | ⬜ pending |
| 34-02-02 | 02 | 2 | EXP-03 | unit | `uv run pytest tests/unit/core/test_experiment_runner.py::test_interaction_test -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/core/test_experiment_schemas.py` — stubs for EXP-01 schema tests
- [ ] `tests/unit/core/test_experiment_manager.py` — stubs for EXP-01 CRUD + EXP-04 verdict tests
- [ ] `tests/unit/core/test_experiment_runner.py` — stubs for EXP-02, EXP-03 runner tests

*Existing pytest infrastructure covers framework needs.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Full end-to-end experiment run with live backtest | EXP-02, EXP-03 | Requires live backtest engine with market data | Run `python scripts/run_iteration.py --hypothesis test-001 --segments us_tech` |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 5s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending

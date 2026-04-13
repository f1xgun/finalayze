---
phase: 38
slug: presetapplicator-auto-apply-loop
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-12
---

# Phase 38 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | `pyproject.toml` ([tool.pytest.ini_options]) |
| **Quick run command** | `uv run pytest tests/unit/core/test_preset_applicator.py tests/unit/test_api_experiments.py -v --tb=short` |
| **Full suite command** | `uv run pytest tests/unit/ -v --tb=short` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run quick command
- **After every plan wave:** Run full suite
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

All plans use embedded TDD (tdd=true) — tests created within each task, not in separate Wave 0.

---

## Wave 0 Requirements

Covered by in-task TDD protocol (behavior → failing test → implementation).

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Sandbox gate with live MOEX data | APPLY-06 | Requires live sandbox metrics | Run sandbox for 3 days, then test apply endpoint |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies (embedded TDD)
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references (embedded TDD)
- [x] No watch-mode flags
- [x] Feedback latency < 30s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** pending

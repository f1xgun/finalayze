---
phase: 10
slug: macro-regime
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-20
---

# Phase 10 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (existing) |
| **Config file** | pyproject.toml [tool.pytest.ini_options] |
| **Quick run command** | `uv run pytest tests/unit/test_position_sizing_pipeline.py tests/unit/test_bond_cycle.py -x -q` |
| **Full suite command** | `uv run pytest tests/unit/ -x --timeout=30` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/unit/test_position_sizing_pipeline.py tests/unit/test_bond_cycle.py -x -q`
- **After every plan wave:** Run `uv run pytest tests/unit/ -x --timeout=30`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 10-01-01 | 01 | 1 | MACRO-01 | unit | `uv run pytest tests/unit/test_position_sizing_pipeline.py -k "cbr_regime" -x` | No | ⬜ pending |
| 10-01-02 | 01 | 1 | MACRO-03 | unit | `uv run pytest tests/unit/test_position_sizing_pipeline.py -k "sector" -x` | No | ⬜ pending |
| 10-02-01 | 02 | 1 | MACRO-02 | unit | `uv run pytest tests/unit/test_bond_cycle.py -k "ofz_rotation" -x` | No | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/test_position_sizing_pipeline.py` — add CBRRegimeStep and SectorAllocationStep tests
- [ ] `tests/unit/test_bond_cycle.py` — add OFZ rotation trigger tests

*Existing infrastructure covers framework needs. Only new test cases required.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Backtest equity curve improves with macro regime | MACRO-01..03 | Requires full iteration run | Run `scripts/run_iteration.py` on ru_blue_chips, compare Sharpe before/after |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending

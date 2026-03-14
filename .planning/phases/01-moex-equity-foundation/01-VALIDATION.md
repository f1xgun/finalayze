---
phase: 1
slug: moex-equity-foundation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-14
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.3+ |
| **Config file** | `pyproject.toml` ([tool.pytest.ini_options]) |
| **Quick run command** | `uv run pytest tests/unit/ -x -q --no-header` |
| **Full suite command** | `uv run pytest --cov -q` |
| **Estimated runtime** | ~45 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/unit/ -x -q --no-header`
- **After every plan wave:** Run `uv run pytest --cov -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 45 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 01-01-01 | 01 | 1 | EQF-01 | unit | `uv run pytest tests/unit/test_position_sizing.py -x` | ✅ | ⬜ pending |
| 01-01-02 | 01 | 1 | EQF-05 | unit | `uv run pytest tests/unit/test_costs.py -x` | ✅ | ⬜ pending |
| 01-02-01 | 02 | 1 | EQF-04 | unit | `uv run pytest tests/unit/test_moex_calendar.py -x` | ✅ | ⬜ pending |
| 01-02-02 | 02 | 1 | EQF-01,04,05 | integration | `uv run python scripts/run_iteration.py --name phase1-validation --segments ru_blue_chips` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- Existing test infrastructure covers unit tests for all phase requirements
- Integration test (backtest iteration) uses existing `scripts/run_iteration.py`

*Existing infrastructure covers all phase requirements.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Position size visually 10-20% of equity in backtest output | EQF-01 | Requires reading backtest CSV output | Check `results/iterations/phase1-*/` CSV for position sizes |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 45s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending

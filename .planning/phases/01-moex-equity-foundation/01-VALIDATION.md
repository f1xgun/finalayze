---
phase: 1
slug: moex-equity-foundation
status: draft
nyquist_compliant: true
wave_0_complete: true
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
| 01-01-01 | 01 | 1 | EQF-04 | unit | `uv run pytest tests/unit/test_moex_calendar.py -x` | YES | pending |
| 01-01-02 | 01 | 1 | EQF-05 | unit | `uv run pytest tests/unit/test_costs.py -x` | YES | pending |
| 01-01-03 | 01 | 1 | EQF-04 | unit | `uv run pytest tests/unit/test_trading_loop_holidays.py -x` | NO (created by 01-01 Task 2) | pending |
| 01-02-01 | 02 | 2 | EQF-01 | unit | `uv run pytest tests/unit/test_moex_sizing.py -x` | NO (created by 01-02 Task 1, TDD) | pending |
| 01-02-02 | 02 | 2 | EQF-01 | integration | `uv run python scripts/run_iteration.py --name phase1-validation --segments ru_blue_chips --start-date 2020-01-01 --end-date 2025-12-31` | YES (script exists) | pending |

*Status: pending / green / red / flaky*

---

## Wave 0 Requirements

- `tests/unit/test_moex_calendar.py` — already exists (extended by Plan 01-01 Task 1)
- `tests/unit/test_costs.py` — already exists (extended by Plan 01-01 Task 1)
- `tests/unit/test_trading_loop_holidays.py` — created by Plan 01-01 Task 2
- `tests/unit/test_moex_sizing.py` — created by Plan 01-02 Task 1 (TDD: test file written first in RED phase, before source changes)

*All test files are created by their respective tasks before source changes. No separate Wave 0 needed.*

---

## Requirement Coverage

| Requirement | Plan | Tasks | Test Files |
|-------------|------|-------|------------|
| EQF-04 (MOEX calendar) | 01 | Task 1 (calendar), Task 2 (TradingLoop wiring) | `test_moex_calendar.py`, `test_trading_loop_holidays.py` |
| EQF-05 (commission) | 01 | Task 1 (costs fix) | `test_costs.py` |
| EQF-01 (RUB sizing) | 02 | Task 1 (sizing fix), Task 2 (backtest validation) | `test_moex_sizing.py` |

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Position size visually 10-20% of equity in backtest output | EQF-01 | Requires reading backtest CSV output | Check `results/iterations/phase1-*/` CSV: position_value / equity between 0.10 and 0.20 |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references (TDD tasks create test files in RED phase)
- [x] No watch-mode flags
- [x] Feedback latency < 45s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** pending

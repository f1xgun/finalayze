---
phase: 9
slug: strategy-wiring
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-20
---

# Phase 9 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (via uv run) |
| **Config file** | pyproject.toml [tool.pytest.ini_options] |
| **Quick run command** | `uv run pytest tests/unit/ -x -q` |
| **Full suite command** | `uv run pytest tests/ --cov -q` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/unit/ -x -q`
- **After every plan wave:** Run `uv run pytest tests/ --cov -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 09-01-01 | 01 | 1 | STRAT-01 | unit | `uv run pytest tests/unit/test_dividend_gap.py -x` | Partial | ⬜ pending |
| 09-01-02 | 01 | 1 | STRAT-01 | unit | `uv run pytest tests/unit/test_strategy_combiner.py -x` | No | ⬜ pending |
| 09-02-01 | 02 | 1 | STRAT-02 | unit | `uv run pytest tests/unit/test_strategy_combiner.py -x` | No | ⬜ pending |
| 09-03-01 | 03 | 2 | STRAT-03 | unit | `uv run pytest tests/unit/test_position_sizing_pipeline.py -x` | No | ⬜ pending |
| 09-03-02 | 03 | 2 | STRAT-04 | unit | `uv run pytest tests/unit/test_position_sizing_pipeline.py -x` | No | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/test_position_sizing_pipeline.py` — add BrentGateStep and RubOilRegimeStep tests
- [ ] `tests/unit/test_strategy_combiner.py` — add _EVENT_STRATEGIES bypass and confidence floor tests
- [ ] `tests/unit/test_dividend_gap.py` — add yield-based hold bar tier tests

*Existing infrastructure covers framework needs. Only new test stubs required.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Backtest equity curve positive | STRAT-01..04 | Requires full iteration run | Run `scripts/run_iteration.py` on ru_blue_chips, check PnL > 0 |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending

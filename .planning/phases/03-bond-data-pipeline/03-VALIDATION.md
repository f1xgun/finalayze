---
phase: 03
slug: bond-data-pipeline
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-03-14
---

# Phase 03 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | pyproject.toml `[tool.pytest.ini_options]` |
| **Quick run command** | `uv run pytest tests/unit/test_bond*.py tests/unit/test_macro*.py tests/unit/test_cbr*.py tests/unit/test_instruments*.py tests/unit/test_ofzin*.py -x -q` |
| **Full suite command** | `uv run pytest tests/ -x --timeout=120` |
| **Estimated runtime** | ~15 seconds (quick), ~60 seconds (full) |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/unit/test_bond*.py tests/unit/test_macro*.py tests/unit/test_cbr*.py tests/unit/test_instruments*.py tests/unit/test_ofzin*.py -x -q`
- **After every plan wave:** Run `uv run pytest tests/ -x --timeout=120`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 03-01-01 | 01 | 1 | BDP-02, BDP-04 | unit | `uv run pytest tests/unit/test_bond_math.py -x` | ✅ | ⬜ pending |
| 03-01-02 | 01 | 1 | BDP-02, BDP-04 | unit | `uv run pytest tests/unit/test_bond_math_quantlib.py -x` | ❌ created by task | ⬜ pending |
| 03-02-01 | 02 | 1 | BDP-03 | unit | `uv run pytest tests/unit/test_cbr_yield_curve.py -x` | ❌ created by task | ⬜ pending |
| 03-02-02 | 02 | 1 | BDP-03 | unit | `uv run pytest tests/unit/test_macro_persistence.py tests/unit/test_macro_cache.py -x` | ❌ created by task | ⬜ pending |
| 03-02-03 | 02 | 1 | BDP-03 | unit | `uv run pytest tests/unit/test_ofzin_indexation.py -x` | ❌ created by task | ⬜ pending |
| 03-03-01 | 03 | 2 | BDP-01, BDP-05 | unit | `uv run pytest tests/unit/test_tinkoff_data.py tests/unit/test_instruments.py -x -q` | ✅ | ⬜ pending |
| 03-03-02 | 03 | 2 | BDP-01, BDP-05 | unit | `uv run pytest tests/unit/test_bond_discovery.py tests/unit/test_instruments.py -x -v` | ❌ created by task | ⬜ pending |
| 03-03-03 | 03 | 2 | BDP-01 | unit | `uv run pytest tests/unit/test_bond_candle_fetch.py -x -v` | ❌ created by task | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

All test files are either pre-existing or created by their respective tasks (TDD approach: test file is written as part of the task's RED phase). No separate Wave 0 stub creation needed.

Pre-existing test files:
- `tests/unit/test_bond_math.py` — covers existing bond math (extended in 03-01-01)
- `tests/unit/test_tinkoff_data.py` — covers existing TinkoffFetcher (extended in 03-03-01)
- `tests/unit/test_instruments.py` — covers InstrumentRegistry (extended in 03-03-02)
- `tests/unit/test_macro_cache.py` — covers MacroCacheService (extended in 03-02-02)

Test files created by tasks (TDD — test written first, then implementation):
- `tests/unit/test_bond_math_quantlib.py` — created by 03-01-02
- `tests/unit/test_cbr_yield_curve.py` — created by 03-02-01
- `tests/unit/test_macro_persistence.py` — created by 03-02-02
- `tests/unit/test_ofzin_indexation.py` — created by 03-02-03
- `tests/unit/test_bond_discovery.py` — created by 03-03-02
- `tests/unit/test_bond_candle_fetch.py` — created by 03-03-03

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| T-Invest bond data freshness | BDP-01 | Requires live API with token | Set FINALAYZE_TINKOFF_TOKEN, run `uv run python -c "from finalayze.data.fetchers.tinkoff_data import TinkoffFetcher; f=TinkoffFetcher(); print(f.fetch_bond_info('SU26244RMFS2'))"` |
| CBR yield curve live fetch | BDP-03 | Requires internet access to cbr.ru | Run `uv run python -c "from finalayze.data.fetchers.cbr import CBRFetcher; f=CBRFetcher(); print(f.fetch_yield_curve(date.today()))"` |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify commands
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all test dependencies (TDD tasks create their own test files)
- [x] No watch-mode flags
- [x] Feedback latency < 15s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** pending execution

---
phase: 03
slug: bond-data-pipeline
status: draft
nyquist_compliant: false
wave_0_complete: false
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
| **Quick run command** | `uv run pytest tests/unit/test_bond*.py tests/unit/test_macro*.py tests/unit/test_instruments*.py -x -q` |
| **Full suite command** | `uv run pytest tests/ -x --timeout=120` |
| **Estimated runtime** | ~15 seconds (quick), ~60 seconds (full) |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/unit/test_bond*.py tests/unit/test_macro*.py tests/unit/test_instruments*.py -x -q`
- **After every plan wave:** Run `uv run pytest tests/ -x --timeout=120`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 03-01-01 | 01 | 1 | BDP-05 | unit | `uv run pytest tests/unit/test_bond_registry.py -x` | ❌ W0 | ⬜ pending |
| 03-01-02 | 01 | 1 | BDP-01 | unit | `uv run pytest tests/unit/test_tinkoff_data.py -x -k bond` | ✅ | ⬜ pending |
| 03-02-01 | 02 | 1 | BDP-02 | unit | `uv run pytest tests/unit/test_bond_math.py -x` | ✅ | ⬜ pending |
| 03-02-02 | 02 | 1 | BDP-04 | unit | `uv run pytest tests/unit/test_quantlib_bond.py -x` | ❌ W0 | ⬜ pending |
| 03-03-01 | 03 | 2 | BDP-03 | unit | `uv run pytest tests/unit/test_macro_cache.py -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/test_bond_registry.py` — stubs for BDP-05 (bond auto-discovery, FIGI mapping)
- [ ] `tests/unit/test_quantlib_bond.py` — stubs for BDP-04 (QuantLib integration)
- [ ] `tests/unit/test_macro_cache.py` — stubs for BDP-03 (MacroCacheService persistence)

*Existing tests cover bond_math.py and tinkoff_data.py bond methods.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| T-Invest bond data freshness | BDP-01 | Requires live API with token | Set FINALAYZE_TINKOFF_TOKEN, run `uv run python -c "from finalayze.data.fetchers.tinkoff_data import TinkoffFetcher; f=TinkoffFetcher(); print(f.fetch_bond_info('SU26244RMFS2'))"` |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending

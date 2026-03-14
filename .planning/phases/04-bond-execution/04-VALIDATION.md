---
phase: 4
slug: bond-execution
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-14
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (latest via uv) |
| **Config file** | pyproject.toml [tool.pytest.ini_options] |
| **Quick run command** | `uv run pytest tests/unit/test_bond_cycle.py tests/unit/test_dv01_sizing.py tests/unit/test_yield_stop.py tests/unit/test_layer_ledger.py tests/unit/test_broker_router.py -x` |
| **Full suite command** | `uv run pytest tests/ -x --timeout=300` |
| **Estimated runtime** | ~30 seconds (quick), ~120 seconds (full) |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/unit/test_bond_cycle.py tests/unit/test_dv01_sizing.py tests/unit/test_layer_ledger.py tests/unit/test_broker_router.py -x`
- **After every plan wave:** Run `uv run pytest tests/ -x --timeout=300`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 04-01-01 | 01 | 1 | BEX-04 | unit | `uv run pytest tests/unit/test_dv01_sizing.py -k "dirty" -x` | ❌ W0 | ⬜ pending |
| 04-01-02 | 01 | 1 | BEX-01 | unit | `uv run pytest tests/unit/test_bond_cycle.py -k "size_and_execute" -x` | ❌ W0 | ⬜ pending |
| 04-01-03 | 01 | 1 | BEX-02 | unit | `uv run pytest tests/unit/test_bond_cycle.py -k "yield_stop" -x` | ❌ W0 | ⬜ pending |
| 04-02-01 | 02 | 1 | BEX-03 | unit | `uv run pytest tests/unit/test_broker_router.py -k "moex_bonds" -x` | ❌ W0 | ⬜ pending |
| 04-02-02 | 02 | 1 | BEX-06 | unit | `uv run pytest tests/unit/test_layer_ledger.py -k "reconcil" -x` | ❌ W0 | ⬜ pending |
| 04-03-01 | 03 | 2 | BEX-05 | integration | `uv run pytest tests/integration/test_bond_walk_forward.py -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/test_bond_cycle.py` — new tests for _size_and_execute (mocked broker, fill confirmation, timeout, partial fills) and _process_yield_stops (mocked GetLastPrices, regime-adaptive)
- [ ] `tests/unit/test_dv01_sizing.py` — new tests for dirty price parameter instead of face_value
- [ ] `tests/unit/test_layer_ledger.py` — new tests for BondPositionRecord, DB persistence, startup reconciliation
- [ ] `tests/unit/test_broker_router.py` — new test for "moex_bonds" key registration
- [ ] `tests/integration/test_bond_walk_forward.py` — walk-forward validation test with OFZ data

*Existing infrastructure covers framework and fixtures.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Sandbox order submission | BEX-01 | Requires live T-Invest sandbox connection | Run `scripts/run_sandbox.py` with FINALAYZE_TINKOFF_TOKEN, verify order appears in T-Invest sandbox UI |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending

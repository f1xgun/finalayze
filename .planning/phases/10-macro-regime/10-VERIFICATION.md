---
phase: 10-macro-regime
verified: 2026-03-20T10:00:00Z
status: gaps_found
score: 11/12 must-haves verified
gaps:
  - truth: "Both steps are wired in the engine pipeline after BrentGate, before Copula/EVT/MetaLabel/HardCaps"
    status: partial
    reason: "Pipeline code placement is correct but _compute_moex_sizing_data returns a 2-tuple (0.0, None) on the early-exit path (moex_data is None) while the call site unpacks 4 values — runtime ValueError if MOEX market data load fails. Confirmed by mypy: scripts/run_iteration.py:672 error: Incompatible return value type (got 'tuple[float, None]', expected 'tuple[float, RubOilRegimeSignal | None, float, str]')"
    artifacts:
      - path: "scripts/run_iteration.py"
        issue: "Early return at line 672 returns 2-tuple (0.0, None) instead of 4-tuple (0.0, None, 0.0, \"\") — mismatches updated return type annotation and call-site unpack"
    missing:
      - "Fix early return: change `return 0.0, None` to `return 0.0, None, 0.0, \"\"` at scripts/run_iteration.py line 672"
---

# Phase 10: Macro Regime Verification Report

**Phase Goal:** MOEX equity positions are sized according to CBR rate regime, sector allocation rotates based on macro conditions, and OFZ allocation shifts when CBR cutting cycle begins
**Verified:** 2026-03-20T10:00:00Z
**Status:** gaps_found (1 gap — latent runtime bug in early-exit path)
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|---------|
| 1 | CBRRegimeStep scales ru_* equity positions by yield curve slope tier: >100bps -> 1.2x, 0-100bps -> 1.0x, <0bps -> 0.6x | VERIFIED | `CBRRegimeStep.adjust()` at position_sizing_pipeline.py:185-194 implements exact thresholds; 5 tests in TestCBRRegimeStep |
| 2 | CBRRegimeStep passes through non-ru_* segments unchanged | VERIFIED | `if not self._segment_id.startswith("ru_"): return size` at line 186; test_cbr_regime_passthrough passes |
| 3 | CBRRegimeStep gracefully returns 1.0x when yield_slope_bps is 0.0 (missing data) | VERIFIED | yield_slope_bps=0.0 falls into else branch -> Decimal("1.0") at line 193; test_cbr_regime_missing_data passes |
| 4 | SectorAllocationStep scales ru_energy by Brent-in-RUB thresholds: >6000 -> 1.3x, <4000 -> 0.7x, between -> 1.0x | VERIFIED | SectorAllocationStep.adjust() lines 211-217; 3 energy tests pass |
| 5 | SectorAllocationStep scales ru_finance by CBR direction: cutting -> 1.2x, hiking -> 0.8x, hold -> 1.0x | VERIFIED | Lines 218-224; 3 finance tests pass |
| 6 | SectorAllocationStep passes through non-sector segments unchanged | VERIFIED | `else: return size` at line 226; tests for ru_blue_chips and us_tech passthrough pass |
| 7 | Both steps are wired in the engine pipeline after BrentGate, before Copula/EVT/MetaLabel/HardCaps | PARTIAL | Code placement correct (engine.py:191-194), but _compute_moex_sizing_data early-exit returns 2-tuple (0.0, None) at line 672 instead of 4-tuple — latent ValueError if moex_data is None; mypy error confirmed |
| 8 | OFZ rotation shifts CORE capital_pct from 0.45 to 0.30 and STRATEGIC from 0.275 to 0.425 when CBR cutting cycle is detected (2+ consecutive cuts) | VERIFIED | apply_ofz_rotation() at bond_cycle.py:70-100 uses relative -0.15/+0.15 shift; test_ofz_rotation_cutting_cycle passes |
| 9 | OFZ rotation does NOT trigger when only 1 cut has occurred | VERIFIED | `if not all(d == "cut" for d in last_two)` guard; test_ofz_rotation_single_cut_not_cycle passes |
| 10 | OFZ rotation reverts to default allocations on the first hike after a cutting cycle | VERIFIED | Last-two-decisions check returns unchanged configs when latest is hike; test_ofz_rotation_revert_on_hike passes |
| 11 | OFZ rotation preserves TACTICAL and SHORT allocations unchanged | VERIFIED | Only CORE and STRATEGIC keys are replaced; test_ofz_rotation_preserves_tactical_short passes |
| 12 | BondCycleProcessor uses rotated configs in its layer loop when cutting cycle active | VERIFIED | `effective_configs = apply_ofz_rotation(...)` at bond_cycle.py:199; loop at line 208 uses `effective_configs.items()` |

**Score:** 11/12 truths verified (1 partial — latent bug in early-exit path)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/finalayze/data/fetchers/cbr.py` | Static yield curve slope data + helpers | VERIFIED | `_YIELD_CURVE_SLOPE_BPS` dict at line 505 (17 entries 2022-03 to 2025-12); `get_recent_cbr_decisions`, `is_cutting_cycle`, `get_yield_slope_bps` at lines 526-551 |
| `src/finalayze/risk/position_sizing_pipeline.py` | CBRRegimeStep and SectorAllocationStep classes | VERIFIED | Both classes at lines 174-227 with correct scaling logic and Protocol compliance |
| `src/finalayze/backtest/config.py` | yield_slope_bps and cbr_direction fields on BacktestConfig | VERIFIED | Lines 174-177: `yield_slope_bps: float = 0.0` and `cbr_direction: str = ""` |
| `src/finalayze/backtest/engine.py` | Pipeline wiring for CBRRegimeStep and SectorAllocationStep | VERIFIED | Lines 191-194: both steps inserted after BrentGate, before CopulaStep |
| `scripts/run_iteration.py` | Yield slope + CBR direction computation and passing to BacktestConfig | PARTIAL | Computation correct (lines 719-733), passing correct (lines 1201-1202), but early-exit 2-tuple at line 672 mismatches 4-tuple call-site |
| `src/finalayze/core/bond_cycle.py` | apply_ofz_rotation function and BondCycleProcessor integration | VERIFIED | Function at lines 70-100, wired in run_cycle() at lines 199-208 |
| `tests/unit/test_moex_sizing.py` | CBRRegimeStep and SectorAllocationStep tests | VERIFIED | 8 CBRRegimeStep tests (TestCBRRegimeStep) + 8 SectorAllocationStep tests (TestSectorAllocationStep) |
| `tests/unit/test_cbr_meeting_calendar.py` | CBR helpers and yield slope tests | VERIFIED | TestGetRecentCBRDecisions, TestIsCuttingCycle, TestGetYieldSlopeBps test classes confirmed |
| `tests/unit/test_bond_cycle.py` | OFZ rotation tests | VERIFIED | 6 tests in TestOFZRotation covering all required scenarios |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `src/finalayze/backtest/engine.py` | `src/finalayze/risk/position_sizing_pipeline.py` | import and instantiation in _build_sizing_pipeline | WIRED | Lines 45, 54: `CBRRegimeStep`, `SectorAllocationStep` imported; lines 192, 194: instantiated with cfg values |
| `scripts/run_iteration.py` | `src/finalayze/backtest/config.py` | BacktestConfig constructor kwargs | WIRED | Lines 783-784: `yield_slope_bps=yield_slope_bps, cbr_direction=cbr_direction` in BacktestConfig call |
| `src/finalayze/core/bond_cycle.py` | `src/finalayze/data/fetchers/cbr.py` | deferred import CBR_MEETINGS | WIRED | Line 80: `from finalayze.data.fetchers.cbr import CBR_MEETINGS` inside apply_ofz_rotation |
| `src/finalayze/core/bond_cycle.py` | `src/finalayze/core/schemas.py` | dataclasses.replace on frozen LayerConfig | WIRED | Lines 92-98: `replace(configs[PortfolioLayer.CORE], capital_pct=...)` and `replace(configs[PortfolioLayer.STRATEGIC], capital_pct=...)` |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| MACRO-01 | 10-01-PLAN.md | CBRRegimeStep in sizing pipeline — CBR rate level + direction affects equity allocation sizing | SATISFIED | CBRRegimeStep implemented and wired; 5 tests; yield curve slope data in cbr.py |
| MACRO-02 | 10-02-PLAN.md | OFZ PK-to-PD rotation trigger — detects CBR cutting cycle start for bond allocation shift | SATISFIED | apply_ofz_rotation() in bond_cycle.py; BondCycleProcessor.run_cycle() uses rotated configs; 6 tests |
| MACRO-03 | 10-01-PLAN.md | SectorAllocationStep in sizing pipeline for sector rotation using MOEX sector data | SATISFIED | SectorAllocationStep implemented and wired; ru_energy Brent thresholds + ru_finance CBR direction; 8 tests |

All 3 requirement IDs (MACRO-01, MACRO-02, MACRO-03) are claimed by plans and satisfied by codebase. REQUIREMENTS.md confirms all 3 marked Complete for Phase 10. No orphaned requirements.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `scripts/run_iteration.py` | 672 | Return 2-tuple `(0.0, None)` from function typed as 4-tuple; call site unpacks 4 values | Blocker | Runtime `ValueError: not enough values to unpack` if MOEX market data load fails (moex_data is None). Confirmed by mypy type error. |

The engine docstring comment at line 180-181 still reads `[BrentGate] -> [Copula] -> [EVT]` without mentioning the Phase 10 steps, but this is informational only (no functional impact).

### Human Verification Required

None — all checks are automatable for this phase.

### Gaps Summary

Phase 10 implemented all three MACRO requirements with correct logic, tests, and wiring. The single gap is a latent runtime bug introduced during Phase 10's wiring commit (`eb36da9`):

`_compute_moex_sizing_data()` in `scripts/run_iteration.py` was upgraded from returning a 2-tuple to a 4-tuple, but the early-exit guard at line 672 (`if moex_data is None: return 0.0, None`) was not updated to return the full 4-tuple `(0.0, None, 0.0, "")`. The call site at line 1155 unconditionally unpacks 4 values. This works at runtime today only because MOEX segments always produce a `moex_data` object — but if the market data loader fails to populate it, the backtest will crash instead of gracefully degrading.

**Fix:** Change line 672 from `return 0.0, None` to `return 0.0, None, 0.0, ""`.

---

_Verified: 2026-03-20T10:00:00Z_
_Verifier: Claude (gsd-verifier)_

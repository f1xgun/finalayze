---
phase: 47-cross-asset-features-asymmetric-barriers
verified: 2026-04-14T13:00:00Z
status: passed
score: 7/7
overrides_applied: 0
---

# Phase 47: Cross-Asset Features & Asymmetric Barriers — Verification Report

**Phase Goal:** ru_energy models have access to Brent crude return features, and energy stocks use asymmetric triple barriers that account for commodity-linked downside volatility
**Verified:** 2026-04-14T13:00:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | compute_features() output for MOEX segments contains brent_ret_5d and brent_ret_21d keys | VERIFIED | `_compute_brent_return_features` returns 3-key dict; `**brent_return_features` merged into `all_features` in `compute_features()` at line 845 |
| 2 | Each multi-period Brent feature falls back to 0.0 independently when insufficient data | VERIFIED | Per-feature `if len(brent) >= lag+N` guards at lines 703, 710, 717; `result = dict(_default)` ensures independent fallback |
| 3 | Existing brent_return (1-bar) is preserved unchanged for backward compatibility | VERIFIED | 1-bar logic block unchanged at lines 702-707; 18 Brent tests pass including legacy TestBrentReturnFeatures |
| 4 | ru_energy gets asymmetric barriers: lower_atr_mult > upper_atr_mult after MOEX uplift | VERIFIED | `_get_barrier_params("ru_energy")` → upper=1.8, lower=2.4; lower > upper confirmed by spot-check |
| 5 | Other MOEX segments keep symmetric defaults | VERIFIED | `_get_barrier_params("ru_finance")` → (2.4, 2.4); us_tech → (2.0, 2.0) confirmed by spot-check |
| 6 | Barrier config is a dict lookup, not hardcoded per-segment if/else | VERIFIED | `_SEGMENT_BARRIER_CONFIG` dict in both scripts; old `is_moex = _segment_id.startswith("ru_")` block absent; old inline `_TB_UPPER_ATR_MULT * _MOEX_ATR_UPLIFT` if/else absent from `_get_triple_barrier_params` |
| 7 | auto_ml_research.py and train_models.py produce identical barrier values for the same segment | VERIFIED | Parity spot-check: both return (1.7999..., 2.4) for ru_energy; functions are identical implementations |

**Score:** 7/7 truths verified

### Roadmap Success Criteria

| # | Success Criterion | Status | Evidence |
|---|------------------|--------|----------|
| 1 | Feature matrix for ru_energy contains brent_ret_5d and brent_ret_21d with non-zero values | VERIFIED | Keys present in `_default` dict; computed when `len(brent) >= lag+6` (8) or `lag+22` (24); merged via `**brent_return_features` in `compute_features()` |
| 2 | Brent features derived from existing `_fetch_moex_macro_data()` — no new data fetch | VERIFIED | `_fetch_moex_macro_data()` populates `commodity_candles={"BZ=F": ...}`; `_compute_brent_return_features` accesses via `moex_data.commodity_candles.get("BZ=F")` at line 695 |
| 3 | When `--segment ru_energy` passed, lower multiplier wider than upper | VERIFIED | upper=1.8, lower=2.4 confirmed; `_get_triple_barrier_params("ru_energy")` returns `{"upper_atr_mult": 1.8, "lower_atr_mult": 2.4, ...}` |
| 4 | Barrier asymmetry configurable per segment via config dict | VERIFIED | `_SEGMENT_BARRIER_CONFIG: dict[str, tuple[float, float]]` in both scripts; test_config_driven confirms dict mutation reflects in output |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/finalayze/ml/features/technical.py` | `_compute_brent_return_features` returns 3-key dict | VERIFIED | Lines 686-690: `{"brent_return": 0.0, "brent_ret_5d": 0.0, "brent_ret_21d": 0.0}`; independent per-feature computation blocks present |
| `tests/unit/test_features_moex.py` | TestBrentMultiPeriodReturnFeatures test class | VERIFIED | Class at line 482 with 9 test methods; all 18 Brent tests pass |
| `scripts/auto_ml_research.py` | `_SEGMENT_BARRIER_CONFIG` dict and `_get_barrier_params` helper | VERIFIED | Dict at line 101 with `"ru_energy": (1.5, 2.0)`; helper at line 243; `build_full_dataset` uses `upper_mult, lower_mult = _get_barrier_params(_segment_id)` at line 504 |
| `scripts/train_models.py` | `_SEGMENT_BARRIER_CONFIG` dict and `_get_barrier_params` helper | VERIFIED | Dict at line 69 with `"ru_energy": (1.5, 2.0)`; helper at line 557; `_get_triple_barrier_params` uses `upper, lower = _get_barrier_params(segment_id)` at line 572 |
| `tests/unit/test_auto_ml_research_moex.py` | TestBarrierConfig test class | VERIFIED | Class at line 41 with 5 test methods; all pass |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `src/finalayze/ml/features/technical.py` | `MoexMarketData.commodity_candles` | `_compute_brent_return_features` accesses `BZ=F` candles | VERIFIED | `moex_data.commodity_candles.get("BZ=F")` at line 695 |
| `scripts/auto_ml_research.py` | `_SEGMENT_BARRIER_CONFIG` | `_get_barrier_params()` called in `build_full_dataset` | VERIFIED | `upper_mult, lower_mult = _get_barrier_params(_segment_id)` at line 504 |
| `scripts/train_models.py` | `_SEGMENT_BARRIER_CONFIG` | `_get_barrier_params()` called in `_get_triple_barrier_params` | VERIFIED | `upper, lower = _get_barrier_params(segment_id)` at line 572 |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|-------------------|--------|
| `_compute_brent_return_features` | `brent` candles | `moex_data.commodity_candles.get("BZ=F")` populated by `_fetch_moex_macro_data()` | Yes — fetched from T-Bank gRPC API via TinkoffFetcher | FLOWING |
| `_get_barrier_params` | `base_upper, base_lower` | `_SEGMENT_BARRIER_CONFIG` module constant | Yes — config-driven, not hardcoded inline | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| ru_energy barrier asymmetry (lower > upper) | `uv run python -c "from scripts.train_models import _get_barrier_params as f; u,l=f('ru_energy'); print(l>u)"` | True | PASS |
| Parity auto_ml_research == train_models | `uv run python -c "from scripts.train_models import _get_barrier_params as tm; from scripts.auto_ml_research import _get_barrier_params as amr; print(tm('ru_energy')==amr('ru_energy'))"` | True | PASS |
| ru_finance symmetric after uplift | `uv run python -c "from scripts.train_models import _get_barrier_params as f; u,l=f('ru_finance'); print(u==l)"` | True | PASS |
| All 18 Brent feature tests pass | `uv run pytest tests/unit/test_features_moex.py -k Brent -q` | 18 passed | PASS |
| All 5 barrier config tests pass | `uv run pytest tests/unit/test_auto_ml_research_moex.py -k TestBarrierConfig -q` | 5 passed | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| FEAT-01 | 47-01-PLAN.md | Brent crude return features (ret_5d, ret_21d) available in technical feature set for MOEX segments | SATISFIED | `brent_ret_5d`, `brent_ret_21d` present in `_compute_brent_return_features` output, merged into `compute_features()` |
| FEAT-02 | 47-01-PLAN.md | Brent features wired from existing `_fetch_moex_macro_data()` into feature engineering pipeline | SATISFIED | `_fetch_moex_macro_data()` populates `commodity_candles`; no new data fetch introduced |
| BARR-01 | 47-02-PLAN.md | Energy stocks use asymmetric triple barrier (wider lower ATR multiplier for commodity-linked volatility) | SATISFIED | `ru_energy` lower=2.4 > upper=1.8; `_get_triple_barrier_params("ru_energy")` confirms asymmetry |
| BARR-02 | 47-02-PLAN.md | Barrier asymmetry configurable per segment in autoresearch | SATISFIED | `_SEGMENT_BARRIER_CONFIG` dict in both scripts; test_config_driven validates dict-driven behavior |

### Anti-Patterns Found

No anti-patterns found in modified files. No TODO/FIXME/PLACEHOLDER markers in the new code paths. No stub returns. The pre-existing `import numpy as np` inside a function body in `scripts/train_models.py` (line 1073) predates this phase and is not introduced by these changes.

### Human Verification Required

None. All must-haves are verifiable programmatically, tests pass, and spot-checks confirm correct barrier values and feature data flow.

### Gaps Summary

No gaps. All 7 must-have truths verified, all 4 roadmap success criteria satisfied, all 4 requirement IDs (FEAT-01, FEAT-02, BARR-01, BARR-02) satisfied. All commits exist (71c3357, 8ff9640, ae9412e, ed20775).

---

_Verified: 2026-04-14T13:00:00Z_
_Verifier: Claude (gsd-verifier)_

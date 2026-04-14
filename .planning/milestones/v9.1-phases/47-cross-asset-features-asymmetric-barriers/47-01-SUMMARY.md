---
phase: 47-cross-asset-features-asymmetric-barriers
plan: "01"
subsystem: ml
tags: [features, brent, cross-asset, moex, energy, log-return]

requires:
  - phase: 46-feature-selection-stability
    provides: stable feature selection pipeline (once-before-folds) that makes new features meaningful

provides:
  - _compute_brent_return_features returning 3-key dict (brent_return, brent_ret_5d, brent_ret_21d)
  - Independent per-feature fallback logic (each feature computes or falls back independently)
  - TestBrentMultiPeriodReturnFeatures with 9 test methods covering all edge cases

affects:
  - phase 48 (segment config changes) — new Brent features will appear in feature vectors for ru_energy
  - ml-experiment skill — re-training ru_energy models will now have multi-horizon oil momentum signals

tech-stack:
  added: []
  patterns:
    - "Independent per-feature fallback: compute each feature in separate if-block, copy default dict, mutate in place"
    - "Clip bounds scale with horizon: 1d=0.15, 5d=0.30, 21d=0.50"

key-files:
  created:
    - tests/unit/test_features_moex.py (TestBrentMultiPeriodReturnFeatures class added)
  modified:
    - src/finalayze/ml/features/technical.py (_compute_brent_return_features extended to 3 keys)

key-decisions:
  - "Independent per-feature fallback over single min_required check — each feature computes if it has enough data, falls back to 0.0 independently"
  - "Clip bounds scale with horizon: 1d=[-0.15,0.15], 5d=[-0.30,0.30], 21d=[-0.50,0.50]"

patterns-established:
  - "Multi-period log return features: use lag+N candle check, brent[-lag-N].close as start, brent[-lag-1].close as end"

requirements-completed: [FEAT-01, FEAT-02]

duration: 8min
completed: "2026-04-14"
---

# Phase 47 Plan 01: Cross-Asset Features — Multi-Period Brent Returns Summary

**Extended _compute_brent_return_features to return brent_ret_5d and brent_ret_21d alongside existing brent_return, with independent per-feature fallback logic and horizon-scaled clip bounds**

## Performance

- **Duration:** ~8 min
- **Started:** 2026-04-14T12:00:00Z
- **Completed:** 2026-04-14T12:08:00Z
- **Tasks:** 2 (TDD: RED + GREEN)
- **Files modified:** 2

## Accomplishments

- Added `brent_ret_5d` (5-bar log return, clipped [-0.30, 0.30], requires lag+6=8 candles minimum)
- Added `brent_ret_21d` (21-bar log return, clipped [-0.50, 0.50], requires lag+22=24 candles minimum)
- Each feature falls back to 0.0 independently — partial data yields partial features
- `TestBrentMultiPeriodReturnFeatures` class with 9 tests; all 18 Brent tests pass (including legacy `TestBrentReturnFeatures`)
- Ruff clean, format clean

## Task Commits

1. **Task 1: Add tests for multi-period Brent return features** - `71c3357` (test — TDD RED)
2. **Task 2: Implement multi-period Brent return features** - `8ff9640` (feat — TDD GREEN)

## Files Created/Modified

- `src/finalayze/ml/features/technical.py` - _compute_brent_return_features extended: 3-key default dict, per-feature independent computation blocks
- `tests/unit/test_features_moex.py` - TestBrentMultiPeriodReturnFeatures added; TestBrentReturnFeatures default assertions updated to expect 3 keys

## Decisions Made

- Independent per-feature fallback (separate `if len(brent) >= lag+N` blocks) over single `min_required` check — gives models partial Brent signal when 5d history exists but 21d does not
- Clip bounds scale with horizon to prevent multi-period returns from dominating feature space

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- ruff E501 in docstring (line 683, 106 chars > 100) — auto-fixed inline by splitting docstring line. Not tracked as a deviation (trivial formatting).

## Known Stubs

None.

## Threat Flags

None — feature computation is internal, reads from in-memory MoexMarketData, no new trust boundaries.

## Next Phase Readiness

- brent_ret_5d and brent_ret_21d now appear in compute_features() output for MOEX segments automatically
- Ready for Phase 47 Plan 02 (asymmetric triple barrier for energy stocks)
- ru_energy models need retraining after Phase 47 completes to pick up new features

---
*Phase: 47-cross-asset-features-asymmetric-barriers*
*Completed: 2026-04-14*

---
phase: 47
plan: "02"
subsystem: ml-training
tags: [asymmetric-barriers, triple-barrier, moex, ru_energy, feature-labeling]
dependency_graph:
  requires: []
  provides: [asymmetric-triple-barrier-config]
  affects: [scripts/auto_ml_research.py, scripts/train_models.py]
tech_stack:
  added: []
  patterns: [dict-lookup-barrier-config, per-segment-uplift-helper]
key_files:
  created:
    - tests/unit/test_auto_ml_research_moex.py (TestBarrierConfig class added)
  modified:
    - scripts/auto_ml_research.py
    - scripts/train_models.py
decisions:
  - "_SEGMENT_BARRIER_CONFIG is a module-level dict; new segments only need a new entry, no if/else logic"
  - "MOEX uplift applied in _get_barrier_params(), not in the config dict — keeps pre-uplift values readable"
  - "train_models.py mirrors auto_ml_research.py config exactly; comment notes must-sync requirement"
metrics:
  duration: "~10 minutes"
  completed: "2026-04-14"
  tasks_completed: 2
  tasks_total: 2
  files_changed: 3
---

# Phase 47 Plan 02: Asymmetric Barrier Config Summary

**One-liner:** Per-segment asymmetric triple barriers via `_SEGMENT_BARRIER_CONFIG` dict; `ru_energy` gets wider downside (upper=1.5, lower=2.0 pre-uplift → 1.8/2.4 after MOEX 1.2x) in both training scripts.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Add tests and implement in auto_ml_research.py | ae9412e | tests/unit/test_auto_ml_research_moex.py, scripts/auto_ml_research.py |
| 2 | Add matching barrier config to train_models.py | ed20775 | scripts/train_models.py |

## What Was Built

### _SEGMENT_BARRIER_CONFIG dict (both scripts)

```python
_SEGMENT_BARRIER_CONFIG: dict[str, tuple[float, float]] = {
    "ru_energy": (1.5, 2.0),  # (upper, lower) — wider downside for commodity-linked volatility
}
```

Segments not listed fall back to symmetric `(_TB_UPPER_ATR_MULT, _TB_LOWER_ATR_MULT)` = `(2.0, 2.0)`.

### _get_barrier_params() helper (both scripts)

```python
def _get_barrier_params(segment_id: str) -> tuple[float, float]:
    """Return (upper_atr_mult, lower_atr_mult) with MOEX uplift applied."""
    base_upper, base_lower = _SEGMENT_BARRIER_CONFIG.get(
        segment_id, (_TB_UPPER_ATR_MULT, _TB_LOWER_ATR_MULT)
    )
    if _is_moex_segment(segment_id):
        return base_upper * _MOEX_ATR_UPLIFT, base_lower * _MOEX_ATR_UPLIFT
    return base_upper, base_lower
```

### Barrier values by segment

| Segment | upper_atr_mult | lower_atr_mult | Asymmetric? |
|---------|---------------|----------------|-------------|
| ru_energy | 1.8 (1.5×1.2) | 2.4 (2.0×1.2) | Yes (lower > upper) |
| ru_finance | 2.4 (2.0×1.2) | 2.4 (2.0×1.2) | No |
| ru_blue_chips | 2.4 | 2.4 | No |
| us_tech | 2.0 | 2.0 | No |

### Usage in scripts

- `auto_ml_research.py`: `build_full_dataset()` replaced 3-line `is_moex/upper_mult/lower_mult` block with `upper_mult, lower_mult = _get_barrier_params(_segment_id)`
- `train_models.py`: `_get_triple_barrier_params()` replaced `if _is_moex_segment(...):` block with `upper, lower = _get_barrier_params(segment_id)`

## Tests Added

`TestBarrierConfig` in `tests/unit/test_auto_ml_research_moex.py`:
- `test_ru_energy_asymmetric` — upper≈1.8, lower≈2.4
- `test_ru_energy_lower_wider_than_upper` — lower > upper
- `test_ru_finance_symmetric` — upper=lower=2.4
- `test_us_tech_no_uplift` — upper=lower=2.0
- `test_config_driven` — dict mutation reflected in output (5 passed)

## Deviations from Plan

### Out-of-scope pre-existing issue found

**[Deferred - Pre-existing]** `scripts/train_models.py` line 1073 has `import numpy as np` inside a function body (PLC0415). This predates our changes (not in our diff). Deferred to `deferred-items.md`.

Otherwise — plan executed exactly as written.

## Known Stubs

None.

## Threat Flags

None. `_SEGMENT_BARRIER_CONFIG` is a module-level constant with no external input path (as noted in plan threat model T-47-02).

## Self-Check: PASSED

- [x] `scripts/auto_ml_research.py` contains `_SEGMENT_BARRIER_CONFIG` and `_get_barrier_params`
- [x] `scripts/train_models.py` contains `_SEGMENT_BARRIER_CONFIG` and `_get_barrier_params`
- [x] `tests/unit/test_auto_ml_research_moex.py` contains `TestBarrierConfig`
- [x] Commits ae9412e and ed20775 exist
- [x] Parity verified: both scripts produce (1.8, 2.4) for ru_energy
- [x] All 5 TestBarrierConfig tests pass

# Phase 47: Cross-Asset Features & Asymmetric Barriers - Research

**Researched:** 2026-04-14
**Domain:** ML feature engineering (Brent return features) + triple barrier labeling (asymmetric barriers)
**Confidence:** HIGH

## Summary

Phase 47 adds two orthogonal enhancements to the MOEX ML training pipeline. First, it extends `_compute_brent_return_features()` in `technical.py` to emit multi-period Brent log returns (`brent_ret_5d`, `brent_ret_21d`) alongside the existing 1-bar `brent_return`. Second, it makes triple barrier ATR multipliers per-segment configurable via a `_SEGMENT_BARRIER_CONFIG` dict in both `auto_ml_research.py` and `train_models.py`, with `ru_energy` getting asymmetric barriers (`lower=2.0, upper=1.5 ATR` after MOEX uplift).

Both files that build the triple-barrier dataset (`scripts/auto_ml_research.py` and `scripts/train_models.py`) contain their own inline barrier constant logic. They must be updated in lockstep — the CONTEXT.md decision explicitly calls this out as a parity requirement. The approach is the same pattern already used in both files for `_is_moex_segment()` routing.

**Primary recommendation:** Extend `_compute_brent_return_features()` to return 3 keys, add `_SEGMENT_BARRIER_CONFIG` dict to both scripts, and route barrier lookups through a `_get_barrier_params(segment_id)` helper mirroring the existing `_get_hparams()` pattern.

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Add `brent_ret_5d` and `brent_ret_21d` (5-day and 21-day log returns) to feature pipeline
- Compute inside existing `_compute_brent_return_features()` in `technical.py:679` — already has Brent data access and lag logic
- Keep existing `brent_return` (1-bar) for backward compatibility — models may already use it
- Return 0.0 for any individual feature with insufficient history (same fallback pattern as `_compute_commodity_features`)
- ru_energy asymmetry: `lower_atr_mult=2.0, upper_atr_mult=1.5` (wider downside for commodity-linked volatility)
- Config via `_SEGMENT_BARRIER_CONFIG` dict in `auto_ml_research.py` mapping segment_id to `(upper_mult, lower_mult)`, fallback to current symmetric defaults
- `train_models.py` also gets asymmetric barriers — same `_SEGMENT_BARRIER_CONFIG` pattern for parity
- Only ru_energy gets custom barriers; other ru_* segments keep symmetric defaults
- Success criteria requires columns named exactly `brent_ret_5d` and `brent_ret_21d`
- Barrier asymmetry logged at run start: barrier parameters already printed, just need to show different upper/lower
- `_SEGMENT_BARRIER_CONFIG` should use segment_id keys (e.g., "ru_energy") not patterns

### Claude's Discretion
- Exact implementation of multi-period return computation (rolling vs point-to-point)
- Whether to add clipping to multi-period returns (existing 1-bar clips to [-0.15, 0.15])
- Test structure and naming

### Deferred Ideas (OUT OF SCOPE)
- None — discussion stayed within phase scope
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| FEAT-01 | Brent crude return features (ret_5d, ret_21d) available in technical feature set for MOEX segments | Extend `_compute_brent_return_features()` to return 3-key dict; integrate at `compute_features()` merge point (line 830) |
| FEAT-02 | Brent features wired from existing `_fetch_moex_macro_data()` into feature engineering pipeline | No new data fetch needed — `commodity_candles["BZ=F"]` already flows into `_compute_brent_return_features()` via `MoexMarketData`; only the transformation needs extending |
| BARR-01 | Energy stocks use asymmetric triple barrier (wider lower ATR multiplier for commodity-linked volatility) | Replace inline `upper_mult/lower_mult` computation in `build_full_dataset()` (auto_ml_research.py:490-491) and `_get_triple_barrier_params()` (train_models.py:557-562) with `_SEGMENT_BARRIER_CONFIG` lookup |
| BARR-02 | Barrier asymmetry configurable per segment in autoresearch | Add `_SEGMENT_BARRIER_CONFIG: dict[str, tuple[float, float]]` at module level; `_get_barrier_params(segment_id)` helper applies MOEX uplift then config override |
</phase_requirements>

---

## Standard Stack

### Core (no new dependencies)
| Library | Purpose | Note |
|---------|---------|------|
| numpy | Log return computation (`np.log`, `np.clip`) | Already imported in `technical.py` [VERIFIED: codebase grep] |
| pandas | Series rolling for multi-period returns | Already imported [VERIFIED: codebase grep] |

No new packages required. [VERIFIED: both `technical.py` and `auto_ml_research.py` already import numpy and pandas]

**Installation:** None needed.

---

## Architecture Patterns

### Existing Pattern: Feature Function Returns `dict[str, float]` with 0.0 Defaults

`_compute_brent_return_features()` currently returns `{"brent_return": 0.0}` as the default dict and `{"brent_return": <float>}` on success. The extension follows the same pattern: return 3 keys, default all to `0.0`, compute each independently so insufficient history for one feature does not suppress others. [VERIFIED: reading `technical.py:679-708` and `_compute_commodity_features` at `technical.py:425-463`]

```python
# Source: src/finalayze/ml/features/technical.py (current pattern, lines 679-708)
def _compute_brent_return_features(moex_data: MoexMarketData | None) -> dict[str, float]:
    _default: dict[str, float] = {"brent_return": 0.0}
    ...
    return {"brent_return": brent_return}
```

Extended pattern (multi-period):
```python
# New signature — three keys, individual fallbacks
_default: dict[str, float] = {
    "brent_return": 0.0,
    "brent_ret_5d": 0.0,
    "brent_ret_21d": 0.0,
}
```

Multi-period returns use **point-to-point log returns**, not rolling mean — consistent with how `brent_return` (1-bar) works: `log(close[−lag−1] / close[−lag−period−1])`. The existing `_EXTERNAL_DATA_LAG_BARS = 2` applies to all windows (the `close` series is indexed from the lagged end). [VERIFIED: `technical.py:695-706`]

**Minimum candle requirements by window:**
- 1-bar return: `lag + 2` = 4 candles (existing)
- 5d return: `lag + 6` = 8 candles
- 21d return: `lag + 22` = 24 candles

Each window falls back to `0.0` independently if insufficient data.

**Clipping decision (Claude's discretion):** Apply `np.clip(_, -0.3, 0.3)` for 5d and `np.clip(_, -0.5, 0.5)` for 21d. Wider clips than 1-bar because multi-period returns have larger natural range (cumulative log return over 5/21 days). The existing 1-bar uses `[-0.15, 0.15]`. [ASSUMED — no project standard for multi-period clip bounds; analyst judgment]

### Existing Pattern: `_get_hparams()` Routing Helper for Per-Segment Config

`auto_ml_research.py` already uses the "segment router" pattern:

```python
# Source: scripts/auto_ml_research.py:233-235
def _get_hparams(segment_id: str) -> dict[str, float | int]:
    """Return hyperparameters: reduced complexity for MOEX, standard for US."""
    return dict(_MOEX_HPARAMS) if _is_moex_segment(segment_id) else dict(_DEFAULT_HPARAMS)
```

`_get_barrier_params(segment_id)` follows the same shape:

```python
# New helper — mirrors _get_hparams pattern
_SEGMENT_BARRIER_CONFIG: dict[str, tuple[float, float]] = {
    # segment_id -> (upper_atr_mult, lower_atr_mult) BEFORE MOEX uplift
    "ru_energy": (1.5, 2.0),
}

def _get_barrier_params(segment_id: str) -> tuple[float, float]:
    """Return (upper_atr_mult, lower_atr_mult) for the segment, with MOEX uplift applied."""
    base_upper, base_lower = _SEGMENT_BARRIER_CONFIG.get(
        segment_id, (_TB_UPPER_ATR_MULT, _TB_LOWER_ATR_MULT)
    )
    if _is_moex_segment(segment_id):
        return base_upper * _MOEX_ATR_UPLIFT, base_lower * _MOEX_ATR_UPLIFT
    return base_upper, base_lower
```

This replaces the inline computation at `auto_ml_research.py:489-491`:
```python
# Before (symmetric, MOEX-only branching)
is_moex = _segment_id.startswith("ru_")
upper_mult = _TB_UPPER_ATR_MULT * (_MOEX_ATR_UPLIFT if is_moex else 1.0)
lower_mult = _TB_LOWER_ATR_MULT * (_MOEX_ATR_UPLIFT if is_moex else 1.0)

# After
upper_mult, lower_mult = _get_barrier_params(_segment_id)
```

`train_models.py` has the equivalent logic in `_get_triple_barrier_params()` at lines 552-569. That function gets the same `_SEGMENT_BARRIER_CONFIG` dict and replaces its symmetric-only branching. [VERIFIED: `train_models.py:552-569`]

### Anti-Patterns to Avoid

- **Duplicating the barrier config dict in both scripts:** Both scripts are standalone (they sys.path.insert their own root), so they cannot share a common module constant directly. Accept the minor duplication — same config values in two places — and keep them in sync. This mirrors how `_TB_UPPER_ATR_MULT` and `_MOEX_ATR_UPLIFT` are already duplicated between the two scripts. [VERIFIED: both scripts define same constants independently]
- **Mutating `ExperimentConfig` in place for barrier params:** Phase 46 established the pattern of creating a *new* `ExperimentConfig` rather than mutating the caller's config. Barrier params are passed through `build_full_dataset()`, not stored in `ExperimentConfig`, so this is not applicable — but do not add barrier params to `ExperimentConfig`. The barrier routing belongs in `build_full_dataset()`.
- **Suppressing all 3 Brent features if any single window is insufficient:** Each window must fall back independently. A stock with 10 candles of Brent data should get `brent_return` (4 required) and `brent_ret_5d` (8 required) but `brent_ret_21d=0.0` (24 required). [VERIFIED: pattern from `_compute_commodity_features`]

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead |
|---------|-------------|-------------|
| Multi-period log returns | Custom loop | `np.log(close_curr / close_prev)` with index arithmetic — same as existing 1-bar |
| Config dispatch | Strategy pattern or class hierarchy | Simple dict + helper function — established project pattern |

---

## Common Pitfalls

### Pitfall 1: Brent Feature Not Surviving Feature Selection
**What goes wrong:** `brent_ret_5d` and `brent_ret_21d` appear in the raw feature matrix but are dropped by `select_features_efficient()` before any fold runs. Success criterion 1 requires "not dropped by feature selection."
**Why it happens:** Feature selection on `ru_energy` with ~850 samples and `max_features=10` may not select the new Brent features if they correlate weakly with the target in the training window.
**How to avoid:** Success criterion validation must check the *raw* feature matrix (before selection), not the selected features. The test for FEAT-01 should verify the feature exists in the output of `compute_features()` and `build_full_dataset()`, not in `selected_features.json`.
**Warning signs:** Test passes on `compute_features()` but fails on the final experiment's `features_used` list.

### Pitfall 2: Barrier Config Applied Before or After MOEX Uplift Inconsistently
**What goes wrong:** `_SEGMENT_BARRIER_CONFIG` stores raw (pre-uplift) multipliers in one script but post-uplift in another, resulting in different effective barriers for the same segment.
**Why it happens:** `auto_ml_research.py` applies uplift inline at lines 490-491; `train_models.py` applies uplift in `_get_triple_barrier_params()`.
**How to avoid:** `_SEGMENT_BARRIER_CONFIG` in BOTH scripts stores pre-uplift multipliers. `_get_barrier_params()` in both scripts applies uplift as the final step. Success criterion 3 can be verified by inspecting the logged `upper_mult`/`lower_mult` values at run start.
**Warning signs:** ru_energy `lower_mult` does not equal `2.0 * 1.2 = 2.4` in the logs.

### Pitfall 3: Index Arithmetic for Multi-Period Brent Returns
**What goes wrong:** Using `brent[-lag-period-1]` as the start close but `brent` may not have enough elements, causing an `IndexError` or silently reading from the beginning of the list.
**Why it happens:** Python negative indexing wraps around silently if `abs(index) <= len(list)`.
**How to avoid:** Check `len(brent) >= lag + period + 2` before computing each window. The `lag` variable is `_EXTERNAL_DATA_LAG_BARS = 2`. For 5d: check `len >= 8`; for 21d: check `len >= 24`.

### Pitfall 4: `brent_return` Key Absent from Return Dict
**What goes wrong:** Refactoring the default dict while adding new keys accidentally drops `"brent_return"`, breaking backward compatibility with models that already use it as a selected feature.
**Why it happens:** Editing the `_default` dict and forgetting the original key.
**How to avoid:** The CONTEXT.md decision explicitly says keep `brent_return` for backward compatibility. Tests should assert all 3 keys present.

---

## Code Examples

### Extended `_compute_brent_return_features()` skeleton

```python
# Source: derived from technical.py:679-708 [VERIFIED: existing pattern]
def _compute_brent_return_features(moex_data: MoexMarketData | None) -> dict[str, float]:
    """Compute Brent crude log return features.

    Returns 3 features:
    - brent_return: 1-bar log return, lagged by _EXTERNAL_DATA_LAG_BARS. Clipped [-0.15, 0.15].
    - brent_ret_5d: 5-bar log return, lagged. Clipped [-0.30, 0.30].
    - brent_ret_21d: 21-bar log return, lagged. Clipped [-0.50, 0.50].
    Each feature falls back to 0.0 independently if insufficient history.
    """
    _default: dict[str, float] = {
        "brent_return": 0.0,
        "brent_ret_5d": 0.0,
        "brent_ret_21d": 0.0,
    }

    if moex_data is None or not moex_data.commodity_candles:
        return _default

    brent = moex_data.commodity_candles.get("BZ=F")
    if not brent:
        return _default

    lag = _EXTERNAL_DATA_LAG_BARS
    result = dict(_default)

    # 1-bar return (existing logic, unchanged)
    if len(brent) >= lag + 2:
        close_prev = float(brent[-lag - 2].close)
        close_curr = float(brent[-lag - 1].close)
        if close_prev > 0 and close_curr > 0:
            result["brent_return"] = float(np.clip(np.log(close_curr / close_prev), -0.15, 0.15))

    # 5-bar return
    if len(brent) >= lag + 6:
        c0 = float(brent[-lag - 6].close)
        c1 = float(brent[-lag - 1].close)
        if c0 > 0 and c1 > 0:
            result["brent_ret_5d"] = float(np.clip(np.log(c1 / c0), -0.30, 0.30))

    # 21-bar return
    if len(brent) >= lag + 22:
        c0 = float(brent[-lag - 22].close)
        c1 = float(brent[-lag - 1].close)
        if c0 > 0 and c1 > 0:
            result["brent_ret_21d"] = float(np.clip(np.log(c1 / c0), -0.50, 0.50))

    return result
```

### `_SEGMENT_BARRIER_CONFIG` + `_get_barrier_params()` (both scripts)

```python
# Source: derived from auto_ml_research.py:94-98, 233-235, 489-491 [VERIFIED: existing patterns]

# Pre-uplift multipliers. MOEX uplift applied in _get_barrier_params().
# Default fallback: symmetric _TB_UPPER_ATR_MULT / _TB_LOWER_ATR_MULT.
_SEGMENT_BARRIER_CONFIG: dict[str, tuple[float, float]] = {
    "ru_energy": (1.5, 2.0),  # (upper, lower) — wider downside for commodity-linked volatility
}


def _get_barrier_params(segment_id: str) -> tuple[float, float]:
    """Return (upper_atr_mult, lower_atr_mult) with MOEX uplift applied."""
    base_upper, base_lower = _SEGMENT_BARRIER_CONFIG.get(
        segment_id, (_TB_UPPER_ATR_MULT, _TB_LOWER_ATR_MULT)
    )
    if _is_moex_segment(segment_id):
        return base_upper * _MOEX_ATR_UPLIFT, base_lower * _MOEX_ATR_UPLIFT
    return base_upper, base_lower
```

Usage in `build_full_dataset()` (auto_ml_research.py):
```python
# Replace lines 489-491:
upper_mult, lower_mult = _get_barrier_params(_segment_id)
```

Usage in `_get_triple_barrier_params()` (train_models.py):
```python
# Replace lines 557-562:
upper_atr, lower_atr = _get_barrier_params(segment_id)
return {
    "upper_atr_mult": upper_atr,
    "lower_atr_mult": lower_atr,
    ...
}
```

---

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| Symmetric barriers for all MOEX segments | Per-segment asymmetric barriers via `_SEGMENT_BARRIER_CONFIG` | ru_energy downside barrier wider, matching commodity-linked volatility asymmetry |
| Single 1-bar Brent return | 3 Brent return features (1d, 5d, 21d) | Model gets multi-horizon commodity momentum signal |

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Multi-period return clip bounds: `[-0.30, 0.30]` for 5d and `[-0.50, 0.50]` for 21d are appropriate | Code Examples | Over-clipping suppresses real signal; under-clipping lets outliers through. Bounded risk — feature still exists, just with different range. |

---

## Open Questions

1. **Clip bounds for multi-period returns**
   - What we know: 1-bar uses `[-0.15, 0.15]` (roughly 3 sigma for daily oil moves)
   - What's unclear: No project standard for multi-period clip bounds
   - Recommendation: Use proportional scaling (`5 * 0.15 = 0.75` is too wide; `[-0.30, 0.30]` for 5d and `[-0.50, 0.50]` for 21d are reasonable; Claude's discretion per CONTEXT.md)

---

## Environment Availability

Step 2.6: SKIPPED — phase is purely code changes within existing scripts. No external tools, services, or CLIs beyond what the project already uses.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (uv run pytest) |
| Config file | pyproject.toml [tool.pytest.ini_options] |
| Quick run command | `uv run pytest tests/unit/test_features_moex.py tests/unit/test_auto_ml_research_moex.py -x` |
| Full suite command | `uv run pytest tests/unit/ -x` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| FEAT-01 | `brent_ret_5d` and `brent_ret_21d` present in `compute_features()` output with non-zero values | unit | `uv run pytest tests/unit/test_features_moex.py -k brent -x` | Partial (file exists, new tests needed) |
| FEAT-01 | Both features return `0.0` when Brent data insufficient | unit | `uv run pytest tests/unit/test_features_moex.py -k brent -x` | Partial |
| FEAT-02 | Features computed from existing `commodity_candles["BZ=F"]` — no new fetch | unit | same as FEAT-01 | Partial |
| BARR-01 | `ru_energy` `lower_mult > upper_mult` after `_get_barrier_params()` | unit | `uv run pytest tests/unit/test_auto_ml_research_moex.py -k barrier -x` | ❌ Wave 0 |
| BARR-02 | Other MOEX segments use symmetric defaults via `_get_barrier_params()` | unit | `uv run pytest tests/unit/test_auto_ml_research_moex.py -k barrier -x` | ❌ Wave 0 |
| BARR-02 | `_SEGMENT_BARRIER_CONFIG` key change reflects in `_get_barrier_params()` (config-driven) | unit | same | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/unit/test_features_moex.py tests/unit/test_auto_ml_research_moex.py -x`
- **Per wave merge:** `uv run pytest tests/unit/ -x`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/test_features_moex.py` — add `TestBrentMultiPeriodReturnFeatures` class covering FEAT-01/FEAT-02 (file exists, new test class needed)
- [ ] `tests/unit/test_auto_ml_research_moex.py` — add `TestBarrierConfig` class covering BARR-01/BARR-02 (file exists, new test class needed)

---

## Security Domain

Phase is pure ML feature engineering + config dict changes in internal scripts. No network calls, auth, user input, or secrets handling introduced.

ASVS: Not applicable. `security_enforcement` is not disabled in config but this phase has no security-relevant surface.

---

## Sources

### Primary (HIGH confidence)
- `src/finalayze/ml/features/technical.py` — `_compute_brent_return_features()` at line 679, `_compute_commodity_features()` at line 425, `compute_features()` at line 711, all MOEX feature constants lines 78-88 [VERIFIED: direct read]
- `scripts/auto_ml_research.py` — `_TB_UPPER_ATR_MULT`, `_TB_LOWER_ATR_MULT`, `_MOEX_ATR_UPLIFT` at lines 94-98, `build_full_dataset()` at lines 477-534, `_get_hparams()` at line 233, `_SEGMENT_BARRIER_CONFIG` not yet present [VERIFIED: direct read]
- `scripts/train_models.py` — `_get_triple_barrier_params()` at lines 552-569, same barrier constants at lines 62-66 [VERIFIED: direct read]
- `tests/unit/test_features_moex.py` — existing `TestBrentReturnFeatures` class, test patterns [VERIFIED: direct read]
- `tests/unit/test_auto_ml_research_moex.py` — existing MOEX test patterns [VERIFIED: direct read]
- `.planning/phases/47-cross-asset-features-asymmetric-barriers/47-CONTEXT.md` — locked decisions [VERIFIED: direct read]

### Secondary (MEDIUM confidence)
- `.planning/REQUIREMENTS.md` — FEAT-01, FEAT-02, BARR-01, BARR-02 requirement definitions [VERIFIED: direct read]

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new dependencies, all library usage verified in codebase
- Architecture: HIGH — patterns verified by direct code reading; proposed patterns mirror existing ones
- Pitfalls: HIGH — pitfalls derived from direct reading of existing code logic and success criteria

**Research date:** 2026-04-14
**Valid until:** 2026-05-14 (stable internal codebase, no external dependencies)

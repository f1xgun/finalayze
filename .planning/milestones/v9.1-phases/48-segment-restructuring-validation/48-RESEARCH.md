# Phase 48: Segment Restructuring & Validation - Research

**Researched:** 2026-04-14
**Domain:** ML training pipeline — segment configuration, symbol history gating, MOEX experiment validation
**Confidence:** HIGH

## Summary

Phase 48 has three tightly scoped changes: remove SBERP from `config/segments.py` (one line), add a 500-trading-day minimum history gate inside `build_full_dataset()` in `auto_ml_research.py` and the per-symbol loop in `_build_dataset_triple_barrier()` in `train_models.py`, then document validation runs as a manual gate.

All three changes are purely mechanical edits in well-understood code. The `_SEGMENT_SYMBOLS` dict in `auto_ml_research.py` is populated dynamically from `DEFAULT_SEGMENTS` at import time (lines 180-182), so removing SBERP from `config/segments.py` line 114 automatically propagates to the script — no second edit needed. The existing per-symbol skip pattern (`if not candles: print(…); continue`) is the direct model for the history gate.

The validation runs (SEGM-03) cannot run in CI; they require `FINALAYZE_TINKOFF_TOKEN` and a live T-Bank gRPC connection. The code changes make them possible; a manual verification step confirms the verdicts.

**Primary recommendation:** Make three targeted edits — one config line removal, one constant + guard in `auto_ml_research.py::build_full_dataset`, one matching guard in `train_models.py::_build_dataset_triple_barrier` — then write unit tests before touching production code (TDD).

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**SBERP Removal**
- Remove SBERP from ru_finance symbol list in `config/segments.py:114` — single source of truth
- auto_ml_research.py picks up ru_* symbols via `DEFAULT_SEGMENTS` loop (line 178-182) — no manual change needed there

**Minimum History Gate**
- Threshold: 500 trading days (~2 years) — matches success criterion
- Location: In `build_full_dataset()` in `auto_ml_research.py` after candle fetch — skip symbol if `len(candles) < 500`
- `train_models.py` also gets the history gate — same parity pattern as barrier config (Phase 47)
- Log message: `"Skipping {symbol}: {len(candles)} trading days < 500 minimum"` at WARNING level (use print, consistent with existing script logging)
- Continue processing remaining symbols after skipping one

**Validation Runs**
- Success criterion #3 (ACCEPT/INCONCLUSIVE verdict on ru_energy, ru_finance, ru_tech) is a human-verification item
- Requires FINALAYZE_TINKOFF_TOKEN and live T-Bank API — cannot run in CI
- Code changes make the validation possible; actual experiment runs are documented as manual verification steps

### Claude's Discretion
- Exact placement within `build_full_dataset()` for the history gate
- Whether to add a `_MIN_HISTORY_DAYS = 500` constant or inline the value
- Test structure and naming for the history gate tests

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SEGM-01 | SBERP removed from ru_finance segment (rho > 0.95 with SBER adds noise without signal) | Single edit to `config/segments.py:114`; propagates automatically to `auto_ml_research.py` via `DEFAULT_SEGMENTS` loop |
| SEGM-02 | Minimum history check (500 trading days) gates ML eligibility per symbol in autoresearch | Add `_MIN_HISTORY_DAYS = 500` constant + guard inside `build_full_dataset()` symbol loop; mirror in `train_models.py::_build_dataset_triple_barrier` |
| SEGM-03 | ru_tech segment has defined ML policy (disabled, merged, or min-history filtered) | Min-history gate naturally handles ru_tech: HEAD (370d), YDEX (450d) are skipped; remaining symbols (OZON, VKCO, POSI) must still meet 500-day floor — segment itself is not disabled, just filtered |
</phase_requirements>

---

## Standard Stack

No new libraries required. All work is in existing files. [VERIFIED: codebase grep]

| File | Current Version | Purpose |
|------|----------------|---------|
| `config/segments.py` | in-tree | Segment definitions — single source of truth for ru_* symbols |
| `scripts/auto_ml_research.py` | in-tree | Autoresearch loop — builds datasets, runs experiments |
| `scripts/train_models.py` | in-tree | Walk-forward model training script |
| `tests/unit/test_segments.py` | in-tree | Config-level segment assertions |
| `tests/unit/test_auto_ml_research_moex.py` | in-tree | Auto-ML MOEX behavior tests |
| `tests/unit/test_train_models_script.py` | in-tree | train_models.py unit tests |

## Architecture Patterns

### Pattern 1: Single-Source Segment Config
`config/segments.py::DEFAULT_SEGMENTS` is the authoritative symbol list. `auto_ml_research.py` populates `_SEGMENT_SYMBOLS` by iterating `DEFAULT_SEGMENTS` at module import (lines 178-182). Removing SBERP from `config/segments.py` is sufficient — `auto_ml_research.py` requires no change. [VERIFIED: Read of auto_ml_research.py lines 178-182]

```python
# auto_ml_research.py lines 178-182 (existing, no change needed)
for _seg in DEFAULT_SEGMENTS:
    if _seg.segment_id.startswith("ru_") and _seg.instrument_type == "stock":
        _SEGMENT_SYMBOLS[_seg.segment_id] = list(_seg.symbols)
```

### Pattern 2: Per-Symbol Skip in build_full_dataset
`build_full_dataset()` iterates over `candles_by_sym.values()`. The existing skip is `if len(candles) < min_candles: continue` (line 507) where `min_candles = _WINDOW_SIZE + _TB_MAX_HOLD + 1 = 101`. The 500-day gate is a new named constant + guard at the symbol level, applied per symbol BEFORE `build_triple_barrier_dataset`. [VERIFIED: Read of auto_ml_research.py lines 492-547]

The `candles_by_sym` dict is keyed by symbol name (str), so the guard needs access to the key to print the warning. The existing code only iterates `.values()` — the planner will need to iterate `.items()` instead for the warning message. [VERIFIED: Read of auto_ml_research.py line 506]

```python
# Proposed change in build_full_dataset (Claude's discretion on exact placement)
_MIN_HISTORY_DAYS = 500  # module-level constant

for sym, candles in candles_by_sym.items():  # change .values() -> .items()
    if len(candles) < _MIN_HISTORY_DAYS:
        print(f"Skipping {sym}: {len(candles)} trading days < {_MIN_HISTORY_DAYS} minimum")
        continue
    if len(candles) < min_candles:  # existing guard stays
        continue
    ...
```

### Pattern 3: Parity Between auto_ml_research and train_models
Established in Phases 45-47: any behavioral gate added to `auto_ml_research.py` is mirrored in `train_models.py`. In `train_models.py`, the per-symbol loop lives in `_build_dataset_triple_barrier()` (line 851: `for symbol in symbols:`). The skip pattern there is already `if len(candles) < min_candles_tb: print(…); continue` (lines 853-858). The 500-day gate is inserted before the existing triple-barrier minimum check. [VERIFIED: Read of train_models.py lines 851-858]

```python
# Proposed change in _build_dataset_triple_barrier in train_models.py
_MIN_HISTORY_DAYS = 500  # module-level constant (same name for parity)

for symbol in symbols:
    candles = _fetch_symbol_candles(symbol, market_id, settings, segment_id=segment_id)
    if len(candles) < _MIN_HISTORY_DAYS:
        print(
            f"  [{segment_id}] Skipping {symbol}: {len(candles)} trading days "
            f"< {_MIN_HISTORY_DAYS} minimum"
        )
        continue
    if len(candles) < min_candles_tb:  # existing guard stays
        ...
```

### Pattern 4: Test for Segment Symbol Membership
`tests/unit/test_segments.py` already has exclusion-style tests (e.g., `assert "SU26238RMFS4" not in seg.symbols`). SEGM-01 gets a direct membership assertion in that file. [VERIFIED: Read of test_segments.py lines 46-49]

### Pattern 5: Test for History Gate in auto_ml_research
`tests/unit/test_auto_ml_research_moex.py` is the established home for autoresearch behavior tests. The history gate test follows the `build_full_dataset` integration pattern already in that file (lines 330-468): construct synthetic candles, call `build_full_dataset`, assert behavior. A symbol with 499 candles must be skipped (empty features returned if it's the only symbol). [VERIFIED: Read of test_auto_ml_research_moex.py]

### Anti-Patterns to Avoid
- **Iterating `.values()` and losing the key:** `build_full_dataset` currently uses `.values()`. Must change to `.items()` only for the lines that need `sym` for the log message — or use a nested approach. Keep changes minimal.
- **Adding the gate after min_candles check:** The 500-day gate is a semantic quality gate (independent of technical window requirements). It must come BEFORE the existing `min_candles` check, not after.
- **Two constants with different names:** `_MIN_HISTORY_DAYS` in both scripts for parity. Do not use `_MIN_HISTORY_BARS` in one and `_MIN_HISTORY_DAYS` in the other.
- **Touching `_SEGMENT_SYMBOLS` directly for SBERP:** The single source of truth is `config/segments.py`. Never patch the script constant — only the config.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead |
|---------|-------------|-------------|
| Log warning | Custom logging infra | `print(…)` consistent with existing script style |
| Symbol history validation | New validator class | Inline guard in existing symbol loop |

## Common Pitfalls

### Pitfall 1: build_full_dataset Iterates .values() Losing Symbol Name
**What goes wrong:** The warning message requires `{symbol}`, but `build_full_dataset` currently iterates `candles_by_sym.values()` — the symbol name is not available.
**Why it happens:** The existing code only cares about candle lists, not keys.
**How to avoid:** Change the loop from `.values()` to `.items()` and unpack `(sym, candles)`. [VERIFIED: Read of auto_ml_research.py line 506]
**Warning signs:** If the loop variable is `candles` with no paired `sym`, the print statement will fail.

### Pitfall 2: Test for SEGM-01 Fails After SBERP Removal If Hardcoded Symbol Count Expected
**What goes wrong:** `test_segments.py` has no test asserting SBERP is absent, but if any downstream test counts ru_finance symbols, it will break.
**Why it happens:** ru_finance has 6 symbols; after removal it has 5. Any `len(seg.symbols) == 6` assertion will fail.
**How to avoid:** Search test suite for hardcoded ru_finance symbol count assertions before committing.
**Warning signs:** CI failure on `test_segments.py` or `test_auto_ml_research_moex.py` with assertion on symbol count.

### Pitfall 3: train_models.py MIN_HISTORY_DAYS Gate in Wrong Function
**What goes wrong:** `train_models.py` has multiple dataset-building paths (`_build_dataset_triple_barrier`, `_build_dataset_direction`, `_build_dataset_trend_scanning`). Adding the gate only to `_build_dataset_triple_barrier` misses the others.
**Why it happens:** MOEX segments use triple barrier exclusively in practice, but the gate should be consistent.
**How to avoid:** Check all three `_build_dataset_*` functions and confirm which ones are invoked for MOEX segments. For Phase 48, the decisive path is triple-barrier — but the planner should note whether the other paths also need the gate for completeness.
**Warning signs:** `--label-mode direction` invocation bypasses the gate entirely.

### Pitfall 4: SEGM-03 Conflation with Code Gate
**What goes wrong:** Treating SEGM-03 (at least one ACCEPT/INCONCLUSIVE verdict) as a unit-testable requirement rather than a manual verification step.
**Why it happens:** The success criterion reads like a functional requirement.
**How to avoid:** SEGM-03 is satisfied by the combination of SEGM-01 + SEGM-02 code changes plus a live experiment run documented in the verification step. No unit test can assert an experiment produces ACCEPT — that depends on real MOEX data.

## Code Examples

### Existing symbol-skip pattern in auto_ml_research.py
```python
# Source: scripts/auto_ml_research.py lines 506-508 [VERIFIED]
for candles in candles_by_sym.values():
    if len(candles) < min_candles:
        continue
```

### Existing symbol-skip pattern in train_models.py
```python
# Source: scripts/train_models.py lines 851-858 [VERIFIED]
for symbol in symbols:
    candles = _fetch_symbol_candles(symbol, market_id, settings, segment_id=segment_id)
    if len(candles) < min_candles_tb:
        print(
            f"  [{segment_id}] {symbol}: only {len(candles)} candles, "
            f"need {min_candles_tb}+ for triple barrier -- skipping."
        )
        continue
```

### Existing segment exclusion test pattern
```python
# Source: tests/unit/test_segments.py line 48 [VERIFIED]
def test_26238_excluded(self) -> None:
    seg = _get("ru_ofz_pd")
    assert "SU26238RMFS4" not in seg.symbols
```

## State of the Art

| Aspect | Current State | After Phase 48 |
|--------|--------------|----------------|
| ru_finance symbols | SBER, SBERP, T, CBOM, BSPB, MOEX (6 symbols) | SBER, T, CBOM, BSPB, MOEX (5 symbols) |
| History gate in autoresearch | None (only technical `min_candles=101` gate) | `_MIN_HISTORY_DAYS=500` skips HEAD, YDEX, borderline T |
| ru_tech ML training | Degenerate — HEAD+YDEX with ~370-450d dominate and produce zero labels | OZON, VKCO, POSI training with full history |
| SEGM-01/02/03 | Pending | All complete |

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | T (Tinkoff/T-Bank) has ~500 trading days history since relisting — borderline, may be skipped | Standard Stack, Code Examples | If T has >500d, it will train; if <500d, it will be skipped — either outcome is valid per decision | 
| A2 | OZON, VKCO, POSI in ru_tech have >500 trading days history | Phase Requirements (SEGM-03) | If any of these also have <500d, ru_tech may have too few symbols for meaningful training — needs checking in validation run |

## Open Questions

1. **Does T (Tinkoff/T-Bank) clear the 500-day floor?**
   - What we know: Context says ~500 trading days since relisting — borderline
   - What's unclear: Exact candle count from T-Bank API
   - Recommendation: The gate handles this automatically; no code decision required. Document in manual verification step whether T was included or skipped.

2. **Should `_build_dataset_direction` and `_build_dataset_trend_scanning` also get the 500-day gate?**
   - What we know: MOEX segments exclusively use triple barrier (`LABEL_MODE_TRIPLE_BARRIER`) in practice
   - What's unclear: Whether parity principle requires gating all three paths
   - Recommendation: Gate only `_build_dataset_triple_barrier` for Phase 48 (matches MOEX usage). Leave others ungated unless explicitly requested.

## Environment Availability

Step 2.6: SKIPPED (phase is purely code/config changes with no new external dependencies; existing T-Bank API dependency is already handled by validation gate).

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest |
| Config file | `pyproject.toml` (pytest section) |
| Quick run command | `uv run pytest tests/unit/test_segments.py tests/unit/test_auto_ml_research_moex.py tests/unit/test_train_models_script.py -x` |
| Full suite command | `uv run pytest` |

### Phase Requirements -> Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SEGM-01 | SBERP not in ru_finance symbols | unit | `uv run pytest tests/unit/test_segments.py -x -k sberp` | Wave 0 — add test to existing file |
| SEGM-02 | Symbol with <500 candles is skipped with warning in auto_ml_research | unit | `uv run pytest tests/unit/test_auto_ml_research_moex.py -x -k history` | Wave 0 — add test to existing file |
| SEGM-02 | Symbol with <500 candles is skipped with warning in train_models | unit | `uv run pytest tests/unit/test_train_models_script.py -x -k history` | Wave 0 — add test to existing file |
| SEGM-03 | ru_energy/ru_finance/ru_tech produce non-REJECT verdict | manual | N/A — requires live T-Bank API | N/A — manual verification |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/unit/test_segments.py tests/unit/test_auto_ml_research_moex.py tests/unit/test_train_models_script.py -x`
- **Per wave merge:** `uv run pytest`
- **Phase gate:** Full suite green + manual verification of SEGM-03 before `/gsd:verify-work`

### Wave 0 Gaps

The test files exist. New test methods must be added:

- [ ] `tests/unit/test_segments.py` — add `TestRuFinance::test_sberp_not_in_ru_finance` (SEGM-01)
- [ ] `tests/unit/test_auto_ml_research_moex.py` — add `TestMinHistoryGate` class (SEGM-02, autoresearch)
- [ ] `tests/unit/test_train_models_script.py` — add test for history gate in `_build_dataset_triple_barrier` (SEGM-02, train_models)

## Security Domain

Not applicable. This phase makes no changes to authentication, session management, access control, or cryptography. No network-facing code is modified.

## Sources

### Primary (HIGH confidence)
- `config/segments.py` — read directly, ru_finance symbols confirmed at line 114 [VERIFIED: Read]
- `scripts/auto_ml_research.py` — read `_SEGMENT_SYMBOLS` population (lines 178-182), `build_full_dataset` (lines 492-547), constant values [VERIFIED: Read]
- `scripts/train_models.py` — read `_build_dataset_triple_barrier` (lines 818-897), per-symbol loop pattern [VERIFIED: Read]
- `tests/unit/test_segments.py` — read existing test structure and exclusion pattern [VERIFIED: Read]
- `tests/unit/test_auto_ml_research_moex.py` — read existing MOEX test class and `build_full_dataset` integration tests [VERIFIED: Read]
- `tests/unit/test_train_models_script.py` — read existing train_models test patterns [VERIFIED: Read]
- `48-CONTEXT.md` — locked decisions, specific constants and file locations [VERIFIED: Read]

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all code read directly from codebase
- Architecture: HIGH — patterns verified in both scripts
- Pitfalls: HIGH — identified from direct code inspection (`.values()` loop, symbol count tests)

**Research date:** 2026-04-14
**Valid until:** Stable — changes are config/script only, no external dependencies

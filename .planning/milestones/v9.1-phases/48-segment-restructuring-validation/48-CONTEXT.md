# Phase 48: Segment Restructuring & Validation - Context

**Gathered:** 2026-04-14
**Status:** Ready for planning

<domain>
## Phase Boundary

Remove SBERP from ru_finance segment, add minimum history gate (500 trading days) to ML training pipeline, and validate that ru_energy/ru_finance/ru_tech segments produce non-REJECT experiment verdicts. Changes span `config/segments.py`, `auto_ml_research.py`, `train_models.py`, and tests.

</domain>

<decisions>
## Implementation Decisions

### SBERP Removal
- Remove SBERP from ru_finance symbol list in `config/segments.py:114` — single source of truth
- auto_ml_research.py picks up ru_* symbols via `DEFAULT_SEGMENTS` loop (line 178-182) — no manual change needed there

### Minimum History Gate
- Threshold: 500 trading days (~2 years) — matches success criterion
- Location: In `build_full_dataset()` in `auto_ml_research.py` after candle fetch — skip symbol if `len(candles) < 500`
- `train_models.py` also gets the history gate — same parity pattern as barrier config (Phase 47)
- Log message: `"Skipping {symbol}: {len(candles)} trading days < 500 minimum"` at WARNING level (use print, consistent with existing script logging)
- Continue processing remaining symbols after skipping one

### Validation Runs
- Success criterion #3 (ACCEPT/INCONCLUSIVE verdict on ru_energy, ru_finance, ru_tech) is a human-verification item
- Requires FINALAYZE_TINKOFF_TOKEN and live T-Bank API — cannot run in CI
- Code changes make the validation possible; actual experiment runs are documented as manual verification steps

### Claude's Discretion
- Exact placement within `build_full_dataset()` for the history gate
- Whether to add a `_MIN_HISTORY_DAYS = 500` constant or inline the value
- Test structure and naming for the history gate tests

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `DEFAULT_SEGMENTS` in `config/segments.py` — ru_* segment definitions with symbol lists
- `_SEGMENT_SYMBOLS` dict in `auto_ml_research.py:129` populated from `DEFAULT_SEGMENTS` at import time
- `build_full_dataset()` — iterates symbols, fetches candles, builds features + labels
- `_fetch_moex_candles()` — fetches candles per symbol via TinkoffFetcher

### Established Patterns
- Script-level constants: `_TB_UPPER_ATR_MULT`, `_MOEX_LOOKBACK_DAYS`, `_MIN_SIGNALS_*` etc.
- Symbol skip pattern: `if not candles: print(f"  {sym}: no candles"); continue` already exists in both scripts
- Parity between `auto_ml_research.py` and `train_models.py` for config/behavior (established in phases 45-47)

### Integration Points
- `config/segments.py:114` — ru_finance symbols list (contains SBERP)
- `auto_ml_research.py` `build_full_dataset()` — symbol loop with candle fetch
- `train_models.py` symbol processing loop — same pattern

</code_context>

<specifics>
## Specific Ideas

- SBERP and SBER have rho>0.95 correlation — removing SBERP eliminates near-zero-independent-signal redundancy
- T (Tinkoff/T-Bank) has ~500 trading days history since relisting — borderline, may be skipped by the 500-day gate
- HEAD (HeadHunter) has ~370 trading days — will be skipped by the gate
- YDEX (Yandex) has ~450 trading days — will be skipped by the gate

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 48-segment-restructuring-validation*
*Context gathered: 2026-04-14 via autonomous smart discuss*

# Phase 11: Advanced Strategies and ML - Research

**Researched:** 2026-03-20
**Domain:** Preferred share arbitrage (MOEX), Russian macro ML features, ML ensemble enablement for ru_* segments
**Confidence:** HIGH

## Summary

Phase 11 wires three related capabilities: (1) preferred share arbitrage using the existing `PairsStrategy` with Kalman filter for SBER/SBERP and TATN/TATNP pairs, (2) extension of the MOEX ML feature set from 4 to ~11 features including CBR rate level/delta/direction, USDRUB return/vol, Brent return, and IMOEX relative strength, and (3) ML ensemble enablement for `ru_blue_chips` segment in reinforcer-only mode.

All three workstreams build on established patterns. The PairsStrategy already supports cointegration testing, Kalman hedge ratios, and z-score entry/exit. The feature pipeline in `technical.py` already has 4 MOEX features with the `_EXTERNAL_DATA_LAG_BARS = 2` convention. ML enablement follows the exact `us_tech` pattern (preset YAML change + model training).

**Primary recommendation:** Execute as three sequential plans: (1) pairs arbitrage wiring + tests, (2) macro feature extension + schema bump, (3) ML training and enablement. Each plan is independently testable.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Cointegration validation period: post-2022 data only (2023-2025) -- avoids sanction-era structural break
- Z-score thresholds: z_entry=2.0, z_exit=0.5 -- matches success criteria (z>2.0 entry), 0.5 for profit capture
- Long-only constraint: set `allow_short=False` on PairsStrategy for MOEX pairs -- success criteria says "long-only trades"
- Target preset: ru_blue_chips (both SBER and TATN are blue chips)
- Uses existing PairsStrategy with Kalman filter support already implemented
- Feature list: CBR rate level, CBR rate delta, CBR direction (one-hot: cut/hike binary, hold=baseline), USDRUB return, USDRUB zscore, USDRUB vol, Brent return, IMOEX relative strength, turnover zscore, realized_vol_ratio
- Computation: extend existing `_compute_moex_features()` in technical.py -- already has 7 MOEX features, add remaining 3 (CBR rate level, CBR delta, CBR direction one-hot)
- CBR encoding: one-hot binary columns (`cbr_direction_cut`, `cbr_direction_hike`), "hold" is baseline (0, 0)
- Look-ahead safety: all macro features lagged by 2 bars (matching existing MOEX features)
- Training data period: 2023-01 to 2025-12 (post-sanctions calm period)
- Target segment for initial enablement: ru_blue_chips (most liquid, highest data quality)
- Feature selection: 10 features (fewer than US's 15 -- smaller MOEX dataset, avoid overfitting)
- Quality gates: same dynamic gates as US with 2024-2025 validation window
- Mode: reinforcer-only (weight=0.10, same as us_tech enablement pattern)
- Training command pattern: `scripts/train_models.py --segment ru_blue_chips --walk-forward --excess-returns --sequential-bootstrap --force-save`

### Claude's Discretion
- TATNP FIGI registration (needs lookup or placeholder)
- Feature schema version bump (v2 -> v3 for new features)
- Model hyperparameter tuning for MOEX (tree depth, learning rate)
- Training script MOEX-specific flags

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| ADV-01 | Preferred share arbitrage (SBER/SBERP, TATN/TATNP) via adapted PairsStrategy with Kalman filter | PairsStrategy exists with full cointegration + Kalman support. Needs: preset YAML config, `allow_short` parameter, TATNP FIGI, long-only gating |
| ADV-02 | 10 Russian macro ML features (CBR rate/delta/direction, USDRUB return/zscore/vol, Brent return, IMOEX relative, turnover zscore) | 4 existing MOEX features in technical.py. Need 7 new features in extended `_compute_macro_features_v2()` or split into new function. All data sources available in MoexMarketData |
| ADV-03 | ML ensemble enabled for ru_* segments with macro features, reinforcer-only mode | us_tech enablement pattern established. Preset YAML change + model training + quality gate validation. Schema version bump v2->v3 required |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| statsmodels | (existing) | Cointegration test (Engle-Granger) | Already used in PairsStrategy |
| numpy | (existing) | Kalman filter, z-score computation | Already used throughout |
| pandas | (existing) | Feature engineering, rolling windows | Already used throughout |
| xgboost/lightgbm/catboost | (existing) | ML ensemble models | Already used in EnsembleModel |

### Supporting
No new dependencies required. All functionality builds on existing libraries.

### Alternatives Considered
None -- CONTEXT.md locks all implementation choices to existing patterns.

## Architecture Patterns

### Relevant Project Structure
```
src/finalayze/
  strategies/
    pairs.py              # PairsStrategy -- add allow_short parameter
    presets/
      ru_blue_chips.yaml  # Enable pairs + ml_ensemble
  ml/
    features/
      technical.py        # Extend with 7 new MOEX features
    loader.py             # FEATURE_SCHEMA_VERSION bump v2 -> v3
    training/
      quality_gates.py    # Same gates, 2024-2025 validation window
  markets/
    instruments.py        # TATNP FIGI registration
  core/
    schemas.py            # MoexMarketData (already has all needed data sources)
```

### Pattern 1: Long-Only Pairs Gating
**What:** PairsStrategy currently generates both BUY and SELL signals based on z-score direction. For MOEX long-only constraint, SELL signals from pairs must be suppressed.
**When to use:** When `allow_short=False` in preset params.
**Implementation approach:**
```python
# In PairsStrategy._compute_signal():
allow_short = bool(params.get("allow_short", True))
if z > z_entry and not allow_short:
    return None  # Suppress short signal for long-only MOEX
```
The `allow_short` parameter defaults to `True` for backward compatibility with existing US pairs configs.

### Pattern 2: MOEX Feature Extension
**What:** Add 7 new features to the MOEX feature pipeline. All use the existing `_EXTERNAL_DATA_LAG_BARS = 2` convention.
**Current state:**
- `usdrub_zscore_60d` -- exists, maps to "USDRUB zscore" requirement
- `brent_zscore_60d` -- exists, partially covers "Brent return" (z-score, not raw return)
- `real_rate_zscore` -- exists, related to but distinct from "CBR rate level"
- `market_turnover_zscore` -- exists, maps to "turnover zscore" requirement

**New features to add:**
1. `cbr_rate_level` -- Raw CBR key rate (from `moex_data.key_rates`), lagged 2 bars
2. `cbr_rate_delta` -- Change in CBR rate vs previous value, lagged 2 bars
3. `cbr_direction_cut` -- 1.0 if last rate change was negative (rate cut), else 0.0
4. `cbr_direction_hike` -- 1.0 if last rate change was positive (rate hike), else 0.0
5. `usdrub_return` -- Log return of USD/RUB rate over 1 bar, lagged 2 bars
6. `usdrub_vol` -- 20-day rolling std of USD/RUB returns, lagged 2 bars
7. `brent_return` -- Log return of Brent crude over 1 bar, lagged 2 bars
8. `imoex_relative` -- Relative strength of stock vs IMOEX benchmark, 21-day window

**Data sources:** All available in `MoexMarketData`:
- CBR: `moex_data.key_rates` (KeyRateRecord with `rate` and `timestamp`)
- FX: `moex_data.fx_rates` (FXRate with `rate` and `timestamp`)
- Brent: `moex_data.commodity_candles["BZ=F"]` (list of Candle)
- IMOEX: Available via `market_context.benchmark_candles` for MOEX segments

### Pattern 3: ML Enablement (Established from us_tech)
**What:** Enable ML ensemble for a segment.
**Steps:**
1. Bump `FEATURE_SCHEMA_VERSION` in `loader.py` (v2 -> v3)
2. Update `ru_blue_chips.yaml`: `ml_ensemble.enabled: true`, `ml_ensemble.weight: 0.10`
3. Train models: `scripts/train_models.py --segment ru_blue_chips --walk-forward --excess-returns --sequential-bootstrap --force-save`
4. Validate quality gates pass on 2024-2025 data
5. Run backtest iteration to compare metrics

### Anti-Patterns to Avoid
- **Adding IMOEX relative as a MOEX-specific feature when cross-asset features already exist:** The existing `_compute_cross_asset_features()` already computes `relative_strength_21d` using `benchmark_candles`. For MOEX segments, benchmark IS IMOEX. The "IMOEX relative" feature may already be covered -- verify before duplicating.
- **Breaking backward compatibility on existing feature names:** The 4 existing MOEX features (`usdrub_zscore_60d`, `brent_zscore_60d`, `real_rate_zscore`, `market_turnover_zscore`) must keep their current names. New features get new names.
- **Forgetting to update `selected_features.json` after training:** The training script handles this automatically, but verify the new features appear.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Cointegration test | Custom ADF on spread | `statsmodels.tsa.stattools.coint` | Already used in PairsStrategy |
| Kalman hedge ratio | Custom state-space | `compute_kalman_hedge_ratio()` in pairs.py | Already implemented and tested |
| Feature z-scoring | Manual z-score | `_rolling_zscore_clipped()` helper | Handles edge cases, clipping, min observations |
| Feature selection | Manual pruning | `select_features()` in training/feature_selection.py | Automated importance-based selection |
| Quality gates | Ad-hoc checks | `quality_gates.py` (6 AFML gates) | Established Brier + accuracy + calibration gates |

## Common Pitfalls

### Pitfall 1: TATNP Missing FIGI
**What goes wrong:** TATNP is registered in `instruments.py` (line 521) but has `figi=None` (no FIGI field set). The TinkoffFetcher needs FIGI to fetch candle data.
**Why it happens:** TATNP was added later as an additional instrument without FIGI lookup.
**How to avoid:** Look up TATNP FIGI from T-Invest API or use known value. TATN FIGI is `BBG004RVFFC0`, TATNP should be `BBG004RVFCY3` (already registered at line 244 for a different entry -- verify this is actually SNGSP, not TATNP).
**Warning signs:** Backtest crashes with `InstrumentNotFoundError` or empty candle data for TATNP.

**Correction on TATNP FIGI:** Looking at instruments.py line 239-244, FIGI `BBG004RVFCY3` is registered for SNGSP (Surgutneftegaz Preferred), not TATNP. TATNP at line 521-527 truly has no FIGI. Must look up the correct FIGI.

### Pitfall 2: Feature Schema Version Mismatch
**What goes wrong:** Adding new features without bumping `FEATURE_SCHEMA_VERSION` means old models (trained on v2 features) load successfully but receive features they weren't trained on.
**Why it happens:** Schema version is checked at load time in `loader.py`.
**How to avoid:** Bump `FEATURE_SCHEMA_VERSION` from 2 to 3 in `loader.py`. This will reject old us_tech models until they're retrained. Since us_tech models are already trained and working, this means us_tech must also be retrained after the schema bump.
**Warning signs:** Silent model performance degradation due to feature mismatch.

### Pitfall 3: Look-Ahead Bias in CBR Direction Features
**What goes wrong:** CBR rate decisions are announced at known dates. If the feature uses the current bar's rate to compute direction, it leaks information.
**Why it happens:** CBR rates change infrequently (8 meetings/year). The "direction" is between consecutive rate records.
**How to avoid:** Apply the standard 2-bar lag (`_EXTERNAL_DATA_LAG_BARS`). Forward-fill the sparse CBR series to daily, then lag. Direction is computed from the last two rate changes BEFORE the lag window.
**Warning signs:** Suspiciously high feature importance for `cbr_direction_*` features.

### Pitfall 4: PairsStrategy SELL Signal in Long-Only Mode
**What goes wrong:** When spread z > 2.0, PairsStrategy generates a SELL signal. In long-only mode for MOEX, this should be suppressed.
**Why it happens:** Current code (line 262) returns `SignalDirection.SELL` when z > z_entry.
**How to avoid:** Add `allow_short` parameter check. When False, return None for positive z (SELL direction).
**Warning signs:** Backtest shows short positions on MOEX preferred shares.

### Pitfall 5: IMOEX Relative Feature Duplication
**What goes wrong:** Creating a separate `imoex_relative` feature when `relative_strength_21d` from `_compute_cross_asset_features()` already does the same thing using `benchmark_candles`.
**Why it happens:** For MOEX segments, the benchmark IS IMOEX. The cross-asset features already compute relative strength.
**How to avoid:** Check whether `benchmark_candles` is populated for MOEX segments in backtest runs. If yes, `relative_strength_21d` already serves as the IMOEX relative feature. If not, either wire IMOEX as benchmark or create a dedicated MOEX feature.
**Warning signs:** Two highly correlated features in the feature set.

### Pitfall 6: Small Training Dataset for MOEX
**What goes wrong:** 2023-2025 is only ~500 trading days. With walk-forward splitting (12mo train + 6mo test), there's very limited training data.
**Why it happens:** Post-2022 constraint is correct (structural break), but it limits data.
**How to avoid:** Use fewer features (10 vs US 15 -- already decided). Consider reducing tree depth and increasing regularization. Feature selection will automatically drop low-importance features.
**Warning signs:** Overfitting, quality gates failing, high variance across walk-forward folds.

## Code Examples

### Pairs Preset Configuration (ru_blue_chips.yaml addition)
```yaml
# In strategies section of ru_blue_chips.yaml
pairs:
  enabled: true
  weight: 0.12
  params:
    pairs: [[SBER, SBERP], [TATN, TATNP]]
    z_entry: 2.0
    z_exit: 0.5
    use_kalman: true
    min_confidence: 0.40
    allow_short: false
```

### CBR Feature Computation Pattern
```python
# Follows established pattern from _compute_macro_features
def _compute_cbr_features(
    moex_data: MoexMarketData | None,
    candle_timestamps: list[datetime] | None = None,
) -> dict[str, float]:
    """Compute CBR rate level, delta, and direction features."""
    _default = {
        "cbr_rate_level": 0.0,
        "cbr_rate_delta": 0.0,
        "cbr_direction_cut": 0.0,
        "cbr_direction_hike": 0.0,
    }
    if moex_data is None or not moex_data.key_rates:
        return _default

    rates = moex_data.key_rates
    # Forward-fill to daily, apply 2-bar lag
    # ... (follows same pattern as _compute_macro_features)

    # Direction: compare last two rate changes
    # hold = (0, 0), cut = (1, 0), hike = (0, 1)
```

### ML Enablement YAML Pattern (from us_tech)
```yaml
ml_ensemble:
  enabled: true
  weight: 0.10
  params:
    threshold: 0.10
    min_confidence: 0.30
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| 4 MOEX features | 11 MOEX features (with CBR, FX return/vol, Brent return) | Phase 11 | Richer signal for MOEX ML |
| FEATURE_SCHEMA_VERSION=2 | v3 (includes new MOEX features) | Phase 11 | Requires retraining all segment models |
| No MOEX pairs trading | Pref share arbitrage on SBER/SBERP, TATN/TATNP | Phase 11 | New alpha source |
| ML only on us_tech | ML on us_tech + ru_blue_chips | Phase 11 | First MOEX ML enablement |

## Open Questions

1. **TATNP FIGI value**
   - What we know: TATNP is registered in instruments.py (line 521) without a FIGI. TATN FIGI is `BBG004RVFFC0`.
   - What's unclear: The actual FIGI for TATNP. Needs T-Invest API lookup.
   - Recommendation: Use T-Invest instruments search API during implementation (Claude's Discretion per CONTEXT.md). Likely candidate: search for "Tatneft pref" via `services.instruments.shares()`.

2. **IMOEX benchmark availability in backtest**
   - What we know: `market_context.benchmark_candles` is populated for segments in `run_iteration.py`. For MOEX segments, this should be IMOEX data.
   - What's unclear: Whether the iteration runner currently sets IMOEX as benchmark for ru_* segments.
   - Recommendation: Verify during implementation. If already wired, `relative_strength_21d` covers "IMOEX relative". If not, add as explicit MOEX feature.

3. **Impact of schema version bump on us_tech**
   - What we know: Bumping FEATURE_SCHEMA_VERSION from 2 to 3 will reject existing us_tech models at load time.
   - What's unclear: Whether us_tech should be retrained as part of this phase or separately.
   - Recommendation: Retrain us_tech after schema bump. The new features default to 0.0 for US segments (no MoexMarketData), so model behavior should be unchanged.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.x |
| Config file | pyproject.toml `[tool.pytest.ini_options]` |
| Quick run command | `uv run pytest tests/unit/test_pairs_strategy.py tests/unit/test_features_moex.py -x` |
| Full suite command | `uv run pytest tests/ -x --timeout=120` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ADV-01a | Pairs preset enables SBER/SBERP, TATN/TATNP with z_entry=2.0 | unit | `uv run pytest tests/unit/test_pairs_strategy.py -x -k "ru_blue_chips"` | Needs new tests |
| ADV-01b | allow_short=False suppresses SELL signals | unit | `uv run pytest tests/unit/test_pairs_strategy.py -x -k "allow_short"` | Needs new tests |
| ADV-01c | Cointegration validated on post-2022 data only | unit | `uv run pytest tests/unit/test_pairs_strategy.py -x -k "post_2022"` | Needs new tests |
| ADV-02a | 7 new CBR/FX/Brent features computed with 2-bar lag | unit | `uv run pytest tests/unit/test_features_moex.py -x -k "cbr or usdrub_return or brent_return"` | Needs new tests |
| ADV-02b | CBR direction one-hot encoding (cut/hike/hold) | unit | `uv run pytest tests/unit/test_features_moex.py -x -k "cbr_direction"` | Needs new tests |
| ADV-02c | Feature schema version bumped to v3 | unit | `uv run pytest tests/unit/test_ml_loader.py -x -k "schema_version"` | Existing tests (update expected value) |
| ADV-03a | ru_blue_chips preset has ml_ensemble enabled | unit | `uv run pytest tests/unit/test_ml_strategy.py -x -k "ru_blue_chips"` | Needs new tests |
| ADV-03b | Quality gates pass on 2024-2025 validation data | integration | `uv run python scripts/train_models.py --segment ru_blue_chips --walk-forward --excess-returns --sequential-bootstrap --force-save` | Manual validation |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/unit/test_pairs_strategy.py tests/unit/test_features_moex.py tests/unit/test_ml_loader.py -x`
- **Per wave merge:** `uv run pytest tests/ -x --timeout=120`
- **Phase gate:** Full suite green + backtest-iteration skill before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/test_pairs_strategy.py` -- add tests for `allow_short`, ru_blue_chips pairs config
- [ ] `tests/unit/test_features_moex.py` -- add tests for 7 new CBR/FX/Brent features
- [ ] `tests/unit/test_ml_loader.py` -- update schema version expectation to v3

## Sources

### Primary (HIGH confidence)
- `src/finalayze/strategies/pairs.py` -- Full PairsStrategy implementation reviewed (290 lines)
- `src/finalayze/ml/features/technical.py` -- All MOEX feature functions reviewed (lines 358-680)
- `src/finalayze/markets/instruments.py` -- SBER/SBERP/TATN/TATNP instrument registrations verified
- `src/finalayze/strategies/presets/ru_blue_chips.yaml` -- Current pairs config (disabled, empty pairs list)
- `src/finalayze/strategies/presets/us_tech.yaml` -- ML enablement reference pattern
- `src/finalayze/ml/loader.py` -- FEATURE_SCHEMA_VERSION=2, schema mismatch handling
- `src/finalayze/core/schemas.py` -- MoexMarketData schema (fx_rates, key_rates, commodity_candles, turnover)
- `src/finalayze/strategies/combiner.py` -- Pairs classified as MR strategy, ADX routing confirmed

### Secondary (MEDIUM confidence)
- TATNP FIGI: Registration exists but without FIGI value (line 521-527). Needs runtime lookup.
- IMOEX benchmark wiring: `run_iteration.py` populates `market_context.benchmark_candles` but unclear if IMOEX specifically for ru_* segments.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already in use, zero new dependencies
- Architecture: HIGH -- all patterns established by prior phases (pairs, ML enablement, MOEX features)
- Pitfalls: HIGH -- identified from direct code inspection (TATNP FIGI, schema version, allow_short, look-ahead bias)

**Research date:** 2026-03-20
**Valid until:** 2026-04-20 (stable domain, no external library changes expected)

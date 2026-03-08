# ML Impact Improvement Plan

**Date:** 2026-03-07
**Branch:** `feature/ml-deep-overhaul`
**Goal:** Fix 3 root causes blocking ML from adding value, raise ML grade from D to B

**Reviewed by:** quant-analyst, risk-officer, ml-engineer, systems-architect (2026-03-07)

## Current State

ML ensemble is disabled in all presets. Training pipeline exists but models produce
near-random predictions:

- **Best model accuracy:** 56.8% (LightGBM) — barely above 50% coin flip
- **Walk-forward:** FAIL — only 33% of folds pass accuracy gate (need 60%)
- **Backtest impact:** +1.7% Sharpe without calibrator (marginal)
- **With calibrator:** negative impact (calibrator compresses [0,1] → [0.41, 0.61])

### Prediction Analysis (synthetic scenarios)

| Scenario | XGB   | LGBM  | Cat   | Avg   |
|----------|-------|-------|-------|-------|
| neutral  | 0.609 | 0.570 | 0.501 | 0.560 |
| bull     | 0.500 | 0.592 | 0.497 | 0.530 |
| bear     | 0.603 | 0.562 | 0.499 | 0.555 |
| crash    | 0.598 | 0.562 | 0.499 | 0.553 |
| rally    | 0.668 | 0.592 | 0.504 | 0.588 |

XGBoost predicts ~0.60 even in crashes. CatBoost flat ~0.50 everywhere. Models
cannot distinguish market regimes.

## Root Causes

### RC1: Over-Aggressive MI Feature Selection

`select_features_mi()` with `mi_threshold=0.05` selects only **2 out of 28** features.
Pairwise MI deduplication further removes correlated features. Models train on
insufficient information.

**File:** `src/finalayze/ml/training/feature_selection.py`

### RC2: Label Imbalance + Non-Neutral Labels

Training labels have 54.2% positive rate. Combined with direction-based labeling
(not excess returns), models learn a permanent bullish bias that doesn't
differentiate regimes.

**File:** `scripts/train_models.py` (`_build_dataset_direction`, `_build_dataset_triple_barrier`)

### RC3: Platt Calibrator Over-Compression

Platt calibrator (LogisticRegression) maps the full [0,1] probability range to
[0.41, 0.61]. With `threshold=0.08` and `base_rate=0.50`, the deadzone is
[0.42, 0.58] — the calibrator's entire output range falls inside the deadzone,
killing all signals.

**File:** `src/finalayze/ml/models/ensemble.py` (calibrator in `predict_proba`)

---

## Pre-Phase: Refactor train_models.py

> **Review feedback (systems-architect):** `train_models.py` is 1193 lines and the
> plan adds 4 more phases of complexity. Split before adding.

**Changes:**

1. Extract `src/finalayze/ml/training/dataset.py` — data loading, label construction
   (direction, triple-barrier, excess-return)
2. Extract `src/finalayze/ml/training/trainer.py` — model fit + walk-forward loop
3. Extract `src/finalayze/ml/training/evaluator.py` — metrics computation, reporting
4. Keep `scripts/train_models.py` as thin CLI wrapper

**Acceptance Criteria:**
- Each module < 400 lines
- All existing tests pass without modification
- CLI interface unchanged

---

## Implementation Plan

### Phase 1: Feature Selection Fix (Priority: CRITICAL)

**Problem:** MI threshold 0.05 too aggressive, selects 2/28 features.

**Changes:**

1. **Lower MI threshold** from 0.05 to 0.02 in `feature_selection.py`
   > Review: 0.01 is noise territory (ml-engineer). 0.02 balances signal vs noise.
2. **Set minimum feature count** — floor of 8 features regardless of MI scores.
   This is the primary safeguard; if MI filtering is too aggressive, the floor catches it.
3. **Relax deduplication** — raise redundancy threshold from median to 75th percentile
   of pairwise MI, so correlated-but-useful features survive
4. **Add feature count logging** — log how many features selected vs total available

**Files:**
- `src/finalayze/ml/training/feature_selection.py` — threshold + min count + dedup fix
- `tests/unit/test_feature_selection.py` — test minimum feature count, relaxed dedup

**Acceptance Criteria:**
- Feature selection retains >= 8 features for us_tech dataset
- Tests pass with new thresholds
- Walk-forward accuracy improves (target: >52% on average)

### Phase 2: Market-Neutral Labels (Priority: CRITICAL)

**Problem:** Labels reflect absolute returns, not excess returns. In bull markets,
54% of labels are positive regardless of stock quality.

**Changes:**

1. **Subtract benchmark return** from per-stock returns before labeling
   - US segments: use SPY as benchmark (fetched via YFinanceFetcher)
   - MOEX segments: use IMOEX as benchmark (fetched via TinkoffFetcher)
   - **Alignment by timestamp join, not index position** — prevents silent corruption
     when stock has missing trading days (halts, holidays)
     > Review (quant-analyst): Index-based alignment will silently use wrong benchmark
     > bar. Must use timestamp join.
2. **Add `label_mode: "excess"` parameter** to training config
3. **Use class weights (NOT SMOTE)** — if class imbalance > 55/45 after excess-return
   adjustment, rely on existing `scale_pos_weight` in all three models
   > Review (quant-analyst + ml-engineer): SMOTE creates synthetic time-series samples
   > by interpolating between neighbors — fundamentally wrong for temporal data.
   > All three models already support `scale_pos_weight`. Use that.
4. **Log label distribution** per fold in walk-forward training

**Files:**
- `scripts/train_models.py` (or `ml/training/dataset.py` after refactor) — excess return
  calculation in `_build_dataset_*` functions, timestamp-based benchmark alignment
- `src/finalayze/ml/training/quality_gates.py` — class balance gate already exists (45-55%)
- `tests/unit/test_walk_forward.py` — test excess return label generation

**Acceptance Criteria:**
- Label distribution within 48-52% per class after excess return adjustment
- Models no longer predict >0.55 for all scenarios
- Bear/crash scenarios produce predictions <0.45

### Phase 3: Calibrator Gating (Priority: HIGH)

**Problem:** Platt calibrator destroys signal by compressing output range.

**Changes:**

1. **Add calibrator quality check** — after fitting, measure output range.
   If `max(calibrated) - min(calibrated) < 0.30`, skip calibration and use raw
   probabilities with a warning log
2. **Clamp raw probabilities to [0.30, 0.70] when calibrator is bypassed** —
   prevents uncalibrated overconfidence from producing outsized positions
   > Review (risk-officer): Without clamp, XGBoost's 0.668 rally prediction produces
   > near-maximum positions through MetaLabeler's sizing_factor().
   > Log every clamp activation. If clamping activates on >50% of predictions,
   > auto-disable ml_ensemble for that segment.
3. **Store calibrator quality metrics** in model artifacts for monitoring
4. **Add `use_calibrator` flag** to ensemble config (default: True, but auto-disabled
   when quality check fails)
5. **Consider temperature scaling** as alternative if Platt continues to compress
   after upstream fixes. Temperature scaling has 1 DOF (vs 2 for Platt), cannot
   overfit on small calibration sets.
   > Review (ml-engineer + quant-analyst): Temperature scaling preserves relative
   > ordering while adjusting confidence. If T > 5, calibrator is compressing too much.

**Files:**
- `src/finalayze/ml/models/ensemble.py` — calibrator quality gate + raw prob clamp
- `tests/unit/test_ml_ensemble.py` — test calibrator bypass, test clamp behavior

**Acceptance Criteria:**
- When calibrator compresses range below 0.30, raw probabilities are used instead
- Raw probabilities clamped to [0.30, 0.70] when calibrator bypassed
- ML signals pass through the deadzone threshold
- Ensemble logs calibrator status (active/bypassed) per segment

### Phase 4: Per-Regime Model Training (Priority: MEDIUM)

**Problem:** Single model for all market regimes. Trending and mean-reverting
markets have different feature importance patterns.

**Changes:**

1. **Add VIX-based regime feature** to technical features
   - `vix_level` (raw VIX or percentile rank over 252 days)
   - `vix_change_5d` (5-day VIX change as regime shift indicator)
   - `realized_vol_ratio` (realized vol / implied vol proxy)
   - **Lag all VIX features by 1 bar** to avoid look-ahead bias
     > Review (risk-officer): Current-bar VIX in a prediction targeting current-bar
     > return is look-ahead.
   - For MOEX: use 20-day annualized realized vol of IMOEX, z-scored over 252-day
     lookback, as VIX proxy
     > Review (quant-analyst): Specify MOEX proxy calculation explicitly.
   - Note: VIX is market-wide, identical for all US stocks on same day. Useful for
     temporal regime detection but not cross-sectional ranking.
2. **Add market breadth features**
   - `sma_cross_ratio` (% of bars where price > SMA200)
   - `drawdown_from_high` (current drawdown from 252-day high)
3. **Regime-aware sample weighting** — weight recent samples higher (exponential
   decay with half-life = 126 bars) to adapt to regime changes
   - **Regime-transition guard:** when VIX percentile crosses 80th percentile (or
     realized vol doubles within 20 bars), freeze to equal weights for that fold
     > Review (risk-officer): During regime transitions, recency weighting trains
     > on dying regime's data. ML reinforcer could boost BUY signals during crashes.

**Files:**
- `src/finalayze/ml/features/technical.py` — add 5 new features (lagged by 1 bar)
- `scripts/train_models.py` — exponential decay sample weights with regime guard
- `tests/unit/test_technical_features.py` — test new features, verify lag

**Acceptance Criteria:**
- Feature count increases from 28 to 33
- Models show differentiated predictions across regimes (crash < neutral < rally)
- Walk-forward fold pass rate improves to >50%

### Phase 5: Hyperparameter Tuning with Optuna (Priority: MEDIUM)

**Problem:** Default hyperparameters for all models. No tuning performed.

**Changes:**

1. **Optuna already in pyproject.toml** (>=4.7.0) — no dependency change needed
2. **Create `src/finalayze/ml/training/tuner.py`** with per-model tuning functions
   - XGBoost: tune max_depth (3-8), n_estimators (100-500), learning_rate (0.01-0.1),
     subsample (0.6-1.0), colsample_bytree (0.6-1.0), **min_child_weight (1-20)**
   - LightGBM: tune num_leaves (15-63), n_estimators (100-500), learning_rate,
     feature_fraction, bagging_fraction, **min_child_samples (10-100)**
   - CatBoost: tune depth (3-6), iterations (100-500), learning_rate
   > Review (ml-engineer): `min_child_weight` and `min_child_samples` are critical
   > regularization params for noisy financial data. Must be in search space.
3. **Objective = mean Brier score across WF folds** (not accuracy)
   > Review (ml-engineer + quant-analyst): Brier score is strictly more informative
   > for probability-calibrated outputs. Penalizes overconfident wrong predictions.
4. **Cap trials at 30-50** — with ~3-5 folds, more trials risk overfitting
   > Review (quant-analyst): Bayesian optimization easily overfits ~3 evaluation points.
5. **Add `--tune` flag** to `train_models.py`
6. **Store best params** in model artifacts directory
7. **Parameter sensitivity check** — after tuning, vary each param +/-20% and measure
   Sharpe degradation. Flag params where >30% Sharpe degradation as fragile.
   > Review (quant-analyst): Standard practice for detecting cliff-edge overfitting.

**Files:**
- `src/finalayze/ml/training/tuner.py` — new file
- `scripts/train_models.py` — `--tune` flag integration
- `tests/unit/test_tuner.py` — test tuning objective function

**Acceptance Criteria:**
- Optuna finds params that improve WF Brier score by >= 2% over defaults
- Best params saved to `models/{segment}/best_params.json`
- Tuning completes in <30 minutes per segment on single CPU
- No cliff-edge parameters (all survive +/-20% variation)

### Phase 6: Meta-Labeling Activation (Priority: MEDIUM)

> **SEQUENCING REQUIREMENT (risk-officer):** Phase 6 must run AFTER Phase 2
> (market-neutral labels) is validated AND Phase 4 (regime features) is complete.
> Meta-labeling on biased, regime-blind models creates directional concentration risk.

**Problem:** MetaLabeler exists (Phase E) but is not wired into the trading pipeline.

**Changes:**

1. **Wire MetaLabeler into StrategyCombiner** — after rule-based strategies generate
   signals, MetaLabeler filters/sizes them
2. **Add `meta_labeling.enabled` flag** to YAML presets
3. **Train meta-labeling model** using rule-based strategy signals as input features
4. **Track filter rate by direction** — log BUY vs SELL filter asymmetry. If asymmetry
   exceeds 2:1, auto-disable MetaLabeler with warning
   > Review (risk-officer): ML with bullish bias will systematically filter SELL signals,
   > creating long-only portfolio in choppy markets.
5. **Cache predictions per (symbol, bar_index)** — predict once, reuse if multiple
   strategies fire for same symbol on same bar
   > Review (systems-architect): Avoids 3 redundant model inference calls per signal.
6. **Backtest with meta-labeling** to measure trade quality improvement

**Files:**
- `src/finalayze/strategies/combiner.py` — integrate MetaLabeler
- `src/finalayze/strategies/presets/*.yaml` — add meta_labeling config
- `scripts/train_models.py` — meta-label training mode
- `tests/unit/test_combiner.py` — test meta-labeling integration

**Acceptance Criteria:**
- Meta-labeling filters out >20% of losing trades
- Win rate improves by >= 3% with meta-labeling active
- Profit factor improves vs baseline
- BUY/SELL filter asymmetry < 2:1

### Phase 7: Quality Gate Tuning & Walk-Forward Improvement (Priority: LOW)

**Problem:** Walk-forward gates may be too strict or too lenient after other fixes.

**Changes:**

1. **Lower accuracy gate z-score from 2.5 to 2.0** — current z=2.5 requires ~58.8%
   accuracy for n=200 fold, which is at the boundary of the success criteria
   > Review (quant-analyst): 58% target sits right at quality gate boundary.
   > z=2.0 is still significant at p<0.02.
2. **Add ensemble diversity gate** — require minimum disagreement between models
   (if all models agree perfectly, likely overfitting)
3. **Add feature importance stability tracking** — Jaccard index of top-5 features
   between consecutive WF folds. Target Jaccard > 0.4.
   > Review (quant-analyst): Unstable feature importance signals overfitting to noise.
4. **Use inverse Brier score for ensemble weighting** — replace `max(0, acc-0.50)^2`
   with `max(0, 0.25 - brier_score)` normalized to sum to 1
   > Review (ml-engineer): Accuracy-based weighting ignores calibration quality.
   > Brier-based weighting rewards well-calibrated models.
5. **Implement adaptive enablement** — automatically enable/disable ml_ensemble per
   segment based on walk-forward results
6. **Add WF metrics to iteration history** for tracking progress
7. **MOEX fold count:** For MOEX segments with <3 years data, reduce to
   8mo train / 1mo cal / 2mo test / 2mo step. Document that MOEX ML models
   require at minimum 24 months of history.
   > Review (ml-engineer): Standard fold structure yields only 1-2 folds for MOEX.
8. **Runtime ML kill switch** — track ML-boosted win rate vs baseline over 50-trade
   window. If ML-boosted drops 5pp below baseline, auto-disable + Telegram alert.
   > Review (risk-officer): Training-time gates alone don't catch runtime degradation.

**Files:**
- `src/finalayze/ml/training/quality_gates.py` — diversity gate, z-score adjustment
- `src/finalayze/ml/models/ensemble.py` — Brier-based weighting
- `scripts/train_models.py` — adaptive enablement, MOEX fold config
- `tests/unit/test_quality_gates.py` — test diversity gate, z-score

**Acceptance Criteria:**
- Walk-forward pass rate >= 60% for enabled segments
- Auto-enablement correctly enables ml_ensemble only for passing segments
- Feature importance Jaccard > 0.4 across consecutive folds
- Iteration history tracks WF metrics

---

## Additional Review Items

### Thread-Safety Fix (from systems-architect)

StalenessDetector E3 methods (`update_features`, `update_output`, `update_brier`,
`get_top_drifting_features`, `get_output_kl_score`, `get_rolling_brier`) do not
acquire `self._lock`. While individual deque operations are GIL-protected,
`get_rolling_brier()` reads two deques (probas + actuals) non-atomically.

**Fix:** Extend `self._lock` to all E3 update/read methods. Minimal perf impact.
**File:** `src/finalayze/ml/staleness.py`

### Model Versioning (from systems-architect)

Current artifact layout is flat: `models/{segment}/xgb.pkl`. Retraining overwrites
previous model with no rollback capability.

**Fix:** Change to `models/{segment}/{timestamp}/` with `latest` symlink. Update
`loader.py` to follow symlink.

### CatBoost Ordered Boosting (from ml-engineer)

CatBoost defaults to `Plain` boosting in recent versions, not `Ordered`. Must
explicitly set `boosting_type='Ordered'` for the anti-overfitting benefit on
small datasets. Also use `auto_class_weights='Balanced'` instead of manual
`scale_pos_weight` when ordered mode is active.

**File:** `src/finalayze/ml/models/catboost_model.py`

### Look-Ahead Bias Audit (from risk-officer)

- **Phase 2 excess returns:** Benchmark return over triple-barrier forward window
  is future data by definition. Same as stock return — this is the label, not a
  feature. No look-ahead issue.
- **Phase 4 VIX features:** Must be lagged by 1 bar. Already specified.
- **Phase 5 Optuna:** Verify train+cal only for Optuna, test fold reserved for
  final evaluation. Do not use test fold in optimization objective.

### Feature Mismatch Guard (from systems-architect)

When MI-selected features don't match inference-time features (e.g., new feature
added to `technical.py` but model trained without it), the current code silently
filters. Add a warning log when `selected` contains names not in computed features.

**File:** `src/finalayze/strategies/ml_strategy.py`

---

## Execution Order (Updated)

```
Pre-Phase (Refactor train_models.py)
         │
         ▼
Phase 1 (Feature Selection)  ─┐
Phase 2 (Market-Neutral)      ├── Critical path, parallel
Phase 3 (Calibrator Gating)  ─┘
         │
         ▼
Phase 4 (Regime Features)    ─┐
Phase 5 (Optuna Tuning)       ├── Medium priority, parallel
                              ─┘
         │
         ▼
Phase 6 (Meta-Labeling)      ── After Phase 2+4 validated (sequential)
         │
         ▼
Phase 7 (Quality Gates)      ── Final tuning after all improvements
```

Phases 1-3 are independent and can be implemented in parallel.
Phases 4-5 depend on Phases 1-3 (need better features/labels for tuning to matter).
**Phase 6 depends on Phase 2 (neutral labels) + Phase 4 (regime features)**
— meta-labeling on biased models creates concentration risk.
Phase 7 comes last as final calibration.

## Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| Walk-forward fold pass rate | 33% | >= 60% |
| Model accuracy (best) | 56.8% | >= 60% |
| Backtest Sharpe delta | +1.7% | >= +5% |
| Prediction range | [0.50, 0.61] | [0.30, 0.70] |
| Crash scenario prediction | 0.555 | < 0.45 |
| ML grade | D | B |
| Feature importance Jaccard (consecutive folds) | unknown | > 0.4 |
| BUY/SELL filter asymmetry (meta-labeling) | N/A | < 2:1 |

## Risks

1. **Excess-return labels may reduce accuracy initially** — fewer clear signals when
   benchmark is subtracted. Mitigate with class weighting (`scale_pos_weight`).
2. **Optuna tuning may overfit** — mitigate with WF cross-validation objective,
   cap at 30-50 trials, require parameter sensitivity check.
3. **Meta-labeling requires sufficient rule-based signal volume** — with only 626 trades,
   meta-labeling training data may be sparse. Mitigate by using all signal attempts
   (not just executed trades) as training data. Verify >= 500 signal attempts per segment.
4. **VIX data availability** — VIX not available through yfinance for MOEX. Use
   20-day annualized realized vol of IMOEX, z-scored over 252-day lookback.
5. **Calibrator bypass with raw probabilities** — uncalibrated probs can reach 0.90+.
   Mitigate with [0.30, 0.70] clamp. Log clamp activation rate.
6. **Regime-transition sample weighting** — recency weighting trains on dying regime.
   Mitigate with VIX-percentile guard (freeze to equal weights above 80th percentile).
7. **MOEX insufficient fold count** — standard WF structure yields 1-2 folds for MOEX.
   Mitigate with reduced fold sizes or documented limitation.

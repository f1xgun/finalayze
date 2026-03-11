# ML Pipeline Design

## 1. Overview

The ML pipeline trains and deploys per-segment model ensembles for directional
prediction. Each market segment (e.g. `us_tech`, `ru_blue_chips`) gets its own
trained models, feature selection, calibration, and metadata. The pipeline
supports two training modes: single temporal split and walk-forward validation.

Models are trained per-segment on multi-symbol candle data. At inference time,
the `MLEnsembleStrategy` fetches features for a given symbol, filters to the
MI-selected feature subset, queries the segment's `EnsembleModel`, and emits
a directional signal with a confidence score. The confidence is adjusted for
base rate, epistemic uncertainty, and calibrator output.

**Current status (2026-03-08):** Grade D. Models are untrained, `ml_ensemble`
is disabled in all preset YAMLs. The full training pipeline, feature
engineering, and model architecture exist but have not been deployed.

### Source layout

```
src/finalayze/ml/
  features/
    technical.py          # Feature engineering (40+ features)
    corporate_actions.py  # Stock split detection and backward adjustment
  training/
    __init__.py           # build_windows, build_dataset, validate_ensemble
    labeling.py           # Triple barrier labeling + market-neutral labels
    feature_selection.py  # MI-based and correlation-based feature selection
    quality_gates.py      # Per-fold quality gates (accuracy, Brier, PF, ...)
    sample_weights.py     # Uniqueness + decay weighting (AFML Ch. 4)
    splitter.py           # Temporal train/test split (no shuffling)
    cpcv.py               # Combinatorial Purged Cross-Validation
  models/
    base.py               # BaseMLModel ABC
    xgboost_model.py      # XGBoostModel
    lightgbm_model.py     # LightGBMModel
    catboost_model.py     # CatBoostModel (ordered boosting)
    lstm_model.py         # LSTMModel (2-layer LSTM)
    ensemble.py           # EnsembleModel (weighted averaging)
    stacking.py           # StackingEnsemble (LogisticRegression meta-learner)
  calibration.py          # EnsembleCalibrator (Platt scaling)
  registry.py             # MLModelRegistry (segment -> EnsembleModel)
  loader.py               # Load/save ensembles atomically with HMAC signing
  staleness.py            # KL divergence drift detection
  integrity.py            # HMAC-SHA256 model file signing/verification
  meta_labeler.py         # Meta-labeling: P(profitable) for rule-based signals
scripts/
  train_models.py         # CLI training script
```

---

## 2. Feature Engineering

**Source:** `src/finalayze/ml/features/technical.py`

The `compute_features()` function requires a minimum of 80 candles
(`_MIN_CANDLES = 80`) and returns a flat `dict[str, float]`. All features use
only data at or before time T (no look-ahead bias). NaN and infinity values
are replaced with 0 as a safety net.

### 2.1 Core features (5)

| Feature | Description | Source |
|---------|-------------|--------|
| `rsi_14` | RSI with period 14 | `pandas_ta.rsi` |
| `macd_hist_pct` | MACD histogram (12,26,9) normalized by price | `pandas_ta.macd` |
| `bb_pct_b` | Bollinger Bands %B (20, 2.0 std) | `pandas_ta.bbands` |
| `volume_ratio_20d` | Current volume / 20-day prior mean (shifted to avoid look-ahead) | Rolling mean with `shift(1)` |
| `atr_14_pct` | ATR(14) normalized by price | `pandas_ta.atr` |

### 2.2 Extended technical features (7)

| Feature | Description |
|---------|-------------|
| `roc_10` | Rate of Change over 10 bars |
| `willr_14` | Williams %R with period 14 |
| `adx_14` | Average Directional Index (14) |
| `hist_vol_20` | Historical volatility (20-bar stdev of returns) |
| `gk_vol_20` | Garman-Klass volatility (20-bar, uses OHLC) |
| `obv_slope_10` | OBV slope over 10 bars, normalized by volume mean |
| `rsi_divergence` | Z-score normalized price-RSI divergence over 14 bars |

### 2.3 Predictive features (9)

| Feature | Description |
|---------|-------------|
| `ret_1d`, `ret_5d`, `ret_21d` | Lagged returns (1, 5, 21 days) |
| `skew_20d`, `kurt_20d` | Return distribution moments (Harvey & Siddique 2000) |
| `max_ret_20d`, `min_ret_20d` | Extreme returns in 20-bar window |
| `rsi_2`, `rsi_5` | Short-period RSI (Connors RSI family) |

### 2.4 Microstructure features (3)

| Feature | Description |
|---------|-------------|
| `proximity_rolling_high` | Close / rolling 252-day high (52-week high proximity) |
| `amihud_20d` | Amihud illiquidity ratio (log-transformed, 20-day rolling) |
| `corwin_schultz_spread` | Corwin-Schultz (2012) bid-ask spread estimator from high/low |

### 2.5 Z-score features (4)

Relative strength indicators, each computed as `(value - rolling_mean) / rolling_std`.

| Feature | Window |
|---------|--------|
| `price_zscore_60d` | Close vs 60-day SMA/std |
| `volume_zscore_20d` | Volume vs 20-day mean/std |
| `rsi_zscore_60d` | RSI(14) vs 60-day RSI mean/std |
| `atr_zscore_60d` | ATR(14) vs 60-day ATR mean/std |

### 2.6 Calendar features (4)

Cyclical encoding of the last candle's timestamp (no look-ahead).

| Feature | Encoding |
|---------|----------|
| `dow_sin`, `dow_cos` | Day-of-week (Mon=0, Fri=4) as sin/cos pair |
| `month_sin`, `month_cos` | Month (1-12) as sin/cos pair |

### 2.7 Regime features (4)

VIX-based features for US segments; MOEX segments get zeros for VIX and only
the realized volatility ratio.

| Feature | Description |
|---------|-------------|
| `vix_level` | Lagged VIX close (uses `[-2]` to avoid look-ahead) |
| `vix_percentile_252d` | Percentile rank of VIX over 252 trading days (min 63 warmup) |
| `vix_change_5d` | 5-day VIX percentage change |
| `realized_vol_ratio` | Short-term vol (20d) / long-term vol (60d) ratio |

### 2.8 Cross-asset features (4)

Compare stock to benchmark (SPY for US, IMOEX for MOEX). When benchmark is
unavailable, returns domain-aware defaults (beta=1.0, corr=0.5, others=0.0).

| Feature | Description |
|---------|-------------|
| `relative_strength_21d` | Stock 21d return minus benchmark 21d return |
| `rolling_beta_63d` | 63-day rolling beta (cov/var) |
| `rolling_corr_63d` | 63-day rolling correlation with benchmark |
| `excess_momentum_score` | (stock_ret_63d - bench_ret_63d) / max(stock_vol_63d, 0.01) |

### 2.9 Wavelet features (4)

Daubechies-4 wavelet decomposition of log returns (3 detail levels).
Requires `pywt`; degrades gracefully if unavailable.

| Feature | Description |
|---------|-------------|
| `wavelet_approx_energy` | Fraction of energy in approximation level |
| `wavelet_detail1_energy` | Fraction in detail level 1 (high frequency) |
| `wavelet_detail2_energy` | Fraction in detail level 2 (medium frequency) |
| `wavelet_detail3_energy` | Fraction in detail level 3 (low frequency) |

### 2.10 Corporate action detection

**Source:** `src/finalayze/ml/features/corporate_actions.py`

`detect_splits()` identifies suspected stock splits when single-bar return
exceeds 40% AND the bar's high-low range is less than half the gap (ruling
out genuine crashes/rallies). `adjust_for_splits()` applies backward ratio
adjustment to all bars before the split. Windows spanning detected splits
are excluded from training data.

---

## 3. Label Generation

**Source:** `src/finalayze/ml/training/labeling.py`

### 3.1 Triple barrier labeling (default)

The `triple_barrier_label()` function applies three barriers to each entry point:

- **Upper barrier (profit target):** ATR-scaled by default (`upper_atr_mult * ATR / entry_price`),
  fallback to fixed percentage. Label = 1 (profit).
- **Lower barrier (stop loss):** ATR-scaled similarly. Label = 0 (loss).
- **Vertical barrier (timeout):** After `max_hold` bars (default 20), label = sign of PnL.
  Noise filter: vertical hits with PnL < 0.5 * ATR% are discarded.

Default ATR multipliers: 2.0x upper, 2.0x lower. MOEX segments get 1.2x uplift
(wider barriers for higher volatility).

### 3.2 Market-neutral labels

When `benchmark_candles` are provided, barriers are checked against excess
return (stock return - benchmark return) instead of raw return. This produces
market-neutral labels that isolate alpha from beta.

- US segments: SPY benchmark
- MOEX segments: IMOEX benchmark (requires `FINALAYZE_TINKOFF_TOKEN`)

### 3.3 Simple direction labels (legacy)

The `build_windows()` function in `training/__init__.py` provides the original
next-bar direction labels: `label = 1 if next_close > current_close else 0`.
Available via `--label-mode direction` CLI flag.

### 3.4 Dataset construction

`build_triple_barrier_dataset()` iterates through candles, computes features
at each entry point using the full history up to that bar (no look-ahead),
applies triple barrier labeling, and collects:
- Feature dicts
- Binary labels (0/1)
- Sample weights (abs(pnl_pct))
- Entry timestamps (for temporal ordering)
- Hold bars (for uniqueness weighting)

Windows spanning detected stock splits are excluded.

---

## 4. Training Pipeline

**Source:** `scripts/train_models.py`

### 4.1 CLI usage

```bash
uv run python scripts/train_models.py                                    # all segments, triple barrier
uv run python scripts/train_models.py --segment us_tech                  # single segment
uv run python scripts/train_models.py --segment us_tech --output-dir models/
uv run python scripts/train_models.py --label-mode direction             # old next-bar labels
uv run python scripts/train_models.py --walk-forward                     # walk-forward validation
uv run python scripts/train_models.py --walk-forward --excess-returns    # market-neutral labels
```

### 4.2 Data fetching

Per-symbol candle fetching follows a priority chain:
1. Database (PostgreSQL/TimescaleDB) via SQLAlchemy async
2. Tinkoff Invest API (for MOEX segments, requires `FINALAYZE_TINKOFF_TOKEN`)
3. YFinance fallback (US segments only)

Lookback periods: 5 years (1825 days) for US segments, 2 years (730 days) for
MOEX segments (post-sanctions structural break).

Benchmark candles (SPY/IMOEX) and VIX candles are fetched once per segment and
aligned per-symbol using date-based forward-fill.

### 4.3 Segment symbols

Training uses representative symbols per segment:
- `us_tech`: AAPL, MSFT, GOOGL, NVDA, META, AMZN, TSLA, CRM, ADBE, INTC, AMD, AVGO, CSCO, ORCL, QCOM
- `us_healthcare`: JNJ, PFE, UNH, ABBV, MRK, LLY, TMO, ABT, BMY, AMGN, GILD, MDT
- `us_finance`: JPM, BAC, GS, MS, WFC, C, BLK, SCHW, AXP, USB, PNC, TFC
- `us_broad`: SPY, QQQ, DIA, IWM, VTI
- `ru_blue_chips`: SBER, GAZP, LKOH, GMKN, ROSN, NVTK, PLZL, MGNT
- `ru_energy`: ROSN, TATN, NVTK, LKOH, SNGS, SIBN
- `ru_tech`: YNDX, OZON, VKCO, CIAN
- `ru_finance`: SBER, VTBR, TCSG, MOEX, CBOM

### 4.4 Single-split training

Three-way temporal split with purge gaps (80 bars = window + max_hold):
- **Train:** 70% of samples
- **Calibration:** 15% (purge gap after train)
- **Test:** 15% (purge gap after calibration)

Steps:
1. Build dataset (multi-symbol, sorted by timestamp)
2. MI-based feature selection on train set only (max 15 features US, 10 MOEX)
3. Compute sample weights (decay * uniqueness * barrier_weights)
4. Train XGBoost, LightGBM, CatBoost
5. Fit EnsembleCalibrator on calibration set
6. Evaluate on test set (accuracy, Brier score, log loss)
7. Save models, selected features, model weights, calibrator, segment metadata

### 4.5 Walk-forward training

Calendar-date-based folds with purge gaps between splits:
- **Train window:** 12 months
- **Calibration window:** 2 months
- **Test window:** 4 months
- **Step size:** 3 months
- **Purge gap:** 80 bars (100 days) between train/cal and cal/test

Each fold:
1. Feature selection (MI) on train data only
2. Compute sample weights (decay * uniqueness * barrier_weights)
3. Train XGBoost, LightGBM, CatBoost
4. Evaluate quality gates on test fold
5. Track best fold by accuracy

After all folds:
- Evaluate per-gate pass rates across folds (60% threshold)
- Save best fold's models, selected features, model weights
- Apply BH (Benjamini-Hochberg) multiple testing correction across segments
  (FDR = 0.10) to control false discovery rate

### 4.6 Feature selection

**Source:** `src/finalayze/ml/training/feature_selection.py`

Two approaches are available:

**MI-based (primary, used in training script):**
1. Compute Mutual Information between each feature and target
2. Remove features with MI < 0.02 (uninformative)
3. Greedy deduplication via pairwise MI (75th percentile redundancy threshold)
4. Floor: minimum 8 features
5. Cap: max 15 features (US) or 10 features (MOEX, for 50:1 sample-to-feature ratio)

**Correlation-based (alternative):**
1. Train quick XGBoost, extract gain-based importances
2. Drop features with importance < 1%
3. Deduplicate pairs with abs(correlation) > 0.85

### 4.7 Sample weighting

**Source:** `src/finalayze/ml/training/sample_weights.py`

Three weight components are multiplied together:

- **Exponential decay weights:** More importance to recent samples.
  `exp(0.5 * i / (n-1))` normalized to sum to n.
- **Sample uniqueness (AFML Ch. 4):** Inverse of average label overlap
  concurrency. Reduces overfitting to clustered events (earnings, crises).
  Computed from hold bar counts via O(n * max_hold) sliding window.
- **Barrier weights:** `sqrt(abs(pnl_pct))` dampened PnL magnitudes from
  triple barrier labeling.

### 4.8 Optuna-tuned hyperparameters

The training script checks `results/tuned_params/{segment_id}/{model_type}.json`
for Optuna-tuned hyperparameters. If found, they override the default model
parameters for XGBoost and LightGBM.

---

## 5. Model Architecture

### 5.1 BaseMLModel

**Source:** `src/finalayze/ml/models/base.py`

Abstract base class with two required methods:
- `predict_proba(features: dict[str, float]) -> float` -- returns BUY probability in [0, 1]
- `fit(X, y, *, sample_weight=None)` -- trains on feature dicts and binary labels

All models validate feature names at prediction time and raise
`InsufficientDataError` on mismatch.

### 5.2 XGBoostModel

**Source:** `src/finalayze/ml/models/xgboost_model.py`

Binary classifier using `xgboost.XGBClassifier`. Returns raw (uncalibrated)
probabilities.

**Default parameters:**
- `max_depth=5` (US), `3` (MOEX)
- `n_estimators=200`
- `learning_rate=0.05`
- `subsample=0.8`, `colsample_bytree=0.8`
- `min_child_weight=5`, `gamma=0.1`
- `reg_alpha=0.1`, `reg_lambda=1.0`
- `eval_metric="logloss"`, `early_stopping_rounds=20`
- `scale_pos_weight` auto-computed from label distribution

**Training:** Temporal validation split (last 10%) for early stopping.
Uses sample weights. Saves via `joblib.dump`.

### 5.3 LightGBMModel

**Source:** `src/finalayze/ml/models/lightgbm_model.py`

Binary classifier using `lightgbm.LGBMClassifier`. Returns raw probabilities.

**Default parameters:**
- `n_estimators=200`, `max_depth=5`
- `learning_rate=0.05`, `num_leaves=15`
- `subsample=0.8`, `colsample_bytree=0.8`
- `min_child_samples=20`
- `reg_alpha=0.1`, `reg_lambda=1.0`
- `scale_pos_weight` auto-computed
- Early stopping: 20 rounds via `lgb.early_stopping` callback

### 5.4 CatBoostModel

**Source:** `src/finalayze/ml/models/catboost_model.py`

Binary classifier using `CatBoostClassifier` with ordered boosting, designed
for small financial datasets (~3500 samples).

**Default parameters:**
- `iterations=300`
- `depth=4` (US), `3` (MOEX)
- `learning_rate=0.03`
- `l2_leaf_reg=5.0`, `random_strength=2.0`, `bagging_temperature=1.0`
- `boosting_type="Ordered"` (specifically designed for small datasets)
- `auto_class_weights="Balanced"`
- `early_stopping_rounds=25`
- `random_seed=42`

### 5.5 LSTMModel

**Source:** `src/finalayze/ml/models/lstm_model.py`

2-layer LSTM classifier using PyTorch. Thread-safe inference via
`threading.Lock` on the per-symbol feature buffer.

**Architecture:**
- 2 LSTM layers (`num_layers=2`) with `hidden_size=64`
- Dropout: 0.2 (between LSTM layers and before linear head)
- Linear head: `hidden_size -> 1` with sigmoid activation
- Sequence length: 20 (default)
- Per-symbol feature buffers (deque) to avoid cross-contamination
- StandardScaler fitted on training data, applied during inference

**Training:**
- 50 epochs max, early stopping with patience=5
- Adam optimizer: lr=0.001, weight_decay=1e-4
- BCE loss, gradient clipping at max_norm=1.0
- 10% temporal validation split for early stopping
- Saves state dict + scaler atomically via temp + rename

**Note:** In the current training script, CatBoost has replaced LSTM as the
primary third model. LSTM is still supported for backward compatibility in
the loader and ensemble.

### 5.6 EnsembleModel

**Source:** `src/finalayze/ml/models/ensemble.py`

Combines multiple `BaseMLModel` instances plus an optional `LSTMModel`.

**Aggregation:**
- **Weighted average** when `model_weights` are provided:
  Weight = `max(0, accuracy - 0.50)^2` (squared edge above coin flip).
  Weight lookup handles key format differences (e.g. `"xgboost"` vs
  `"XGBoostModel"`) via alias mapping.
- **Equal average** when no weights provided.
- **Stacking** via `StackingEnsemble` (LogisticRegression meta-learner on
  holdout sub-model predictions). Mutually exclusive with calibrator to
  prevent double-calibration.

**Graceful degradation:**
- Untrained models (where `_model is None`) are skipped entirely.
- If a trained model raises an exception during prediction, it is logged
  and skipped.
- If ALL trained models fail, raises `PredictionError`.
- If no models are trained at all, returns 0.5 (neutral).

**Calibrator integration:**
- If `EnsembleCalibrator` is fitted and not bypassed: applies Platt scaling.
- If calibrator is bypassed (output range < 0.30): clamps raw probability
  to [0.30, 0.70] with monitoring of clamp rate.
- If stacking is fitted: uses stacking output (already calibrated).

**Epistemic uncertainty:** `prediction_uncertainty` property returns std dev
of per-model probabilities. Used by `MLEnsembleStrategy` to discount
confidence when models disagree.

**Per-prediction audit:** `last_model_probas` dict records each model's output
after every prediction (keyed by class name).

### 5.7 StackingEnsemble

**Source:** `src/finalayze/ml/models/stacking.py`

Meta-learner using `LogisticRegression` (lbfgs solver, 1000 max iterations).
Trained on holdout sub-model predictions (minimum 10 samples). Falls back to
simple mean averaging when not fitted.

---

## 6. Quality Gates

**Source:** `src/finalayze/ml/training/quality_gates.py`

Seven quality gates evaluated per walk-forward fold:

| Gate | Threshold | Description |
|------|-----------|-------------|
| `accuracy` | `0.50 + 2.5 * sqrt(0.25 / n_effective)` | N-adjusted accuracy (accounts for sample size and label overlap) |
| `brier_score` | < 0.25 | Must beat coin flip calibration |
| `profit_factor` | >= 1.10 | Minimum simulated profit factor |
| `signal_count` | >= 50 | Minimum signals per fold |
| `class_balance` | >= 0.30 | min(buy_ratio, 1 - buy_ratio) prevents all-buy/all-sell |
| `sensitivity` | >= 0.45 | True positive rate |
| `specificity` | >= 0.45 | True negative rate |

**Walk-forward evaluation:** A model passes overall if each gate passes in
>= 60% of folds. Gate pass rates are saved to `wf_gate_results.json`.

**BH multiple testing correction:** After walk-forward across all segments,
Benjamini-Hochberg correction (FDR = 0.10) is applied using binomial test
p-values derived from accuracy. Segments that fail are marked
`bh_passed: false` in their results file.

### 6.1 Validation gates (single-split)

**Source:** `src/finalayze/ml/training/__init__.py`

The `validate_ensemble()` function checks three thresholds:
- Accuracy >= 0.54 (2 SE above coin-flip for n=500)
- Brier score <= 0.235 (meaningfully below 0.250)
- Log loss <= 0.680 (below ln(2) ~ 0.693)

### 6.2 CPCV screening

**Source:** `src/finalayze/ml/training/cpcv.py`

Combinatorial Purged Cross-Validation generates C(n_groups, n_test_groups)
splits with purge gaps (minimum 60 bars). Uses XGBoost for fast screening.
Acceptance criteria: median Brier <= 0.25 AND negative folds <= 40%.

---

## 7. Confidence Calibration

**Source:** `src/finalayze/ml/calibration.py`

### 7.1 EnsembleCalibrator (Platt scaling)

Single Platt scaler (`LogisticRegression`) fitted on raw ensemble probabilities
vs true labels from the calibration split (out-of-sample).

**Fitting requirements:**
- Minimum 50 samples
- Both classes must be present in labels

**Over-compression detection:** After fitting, the calibrator measures the
output range on calibration data. If the range is below 0.30, the calibrator
is flagged as "bypassed" (`calibrator_bypassed = True`). When bypassed, the
ensemble clamps raw probabilities to [0.30, 0.70] instead of applying the
calibrator, and tracks the clamping rate.

### 7.2 Integration flow

```
Raw sub-model probabilities
  -> Weighted or equal average
  -> If stacking is fitted: stacking meta-learner output (done)
  -> If calibrator is fitted and not bypassed: Platt scaling (done)
  -> If calibrator is bypassed: clamp to [0.30, 0.70] (done)
  -> Otherwise: raw average (done)
```

---

## 8. Model Registry

**Source:** `src/finalayze/ml/registry.py`

`MLModelRegistry` maps segment IDs to `EnsembleModel` instances. Thread-safe
via `threading.Lock` on both `get()` and `register()`, supporting hot-swap
during automated retraining.

- `register(segment_id, model)` -- register or replace
- `get(segment_id) -> EnsembleModel | None` -- returns None if not registered
- `create_ensemble(segment_id)` -- factory creating XGBoost + LightGBM + LSTM (untrained)

---

## 9. Model Persistence

**Source:** `src/finalayze/ml/loader.py`

### 9.1 Storage format

Per-segment directory under `models/{segment_id}/`:

| File | Contents |
|------|----------|
| `xgb.pkl` | Serialized XGBoostModel (joblib) |
| `lgbm.pkl` | Serialized LightGBMModel (joblib) |
| `catboost.pkl` | Serialized CatBoostModel (joblib) |
| `lstm.pkl` | PyTorch state dict + config (backward compat) |
| `lstm.pkl.scaler.pkl` | StandardScaler for LSTM (pickle) |
| `selected_features.json` | MI-selected feature names list |
| `calibrator.pkl` | Fitted EnsembleCalibrator (joblib) |
| `model_weights.json` | Performance-weighted model weights |
| `segment_meta.json` | Segment metadata (base_rate) |
| `wf_gate_results.json` | Walk-forward quality gate results |

### 9.2 Atomic writes

All model saves use temp file + rename pattern to prevent corruption on
interrupted writes. `_atomic_save()` creates a temporary file in the target
directory, writes content, then atomically renames.

### 9.3 HMAC integrity

**Source:** `src/finalayze/ml/integrity.py`

When `ml_model_hmac_key` is configured in settings:
- `sign_model(path, key)` writes a `.sha256` digest file alongside the model
- `verify_model(path, key)` checks the digest before loading, raises
  `ModelIntegrityError` on mismatch

### 9.4 Loading

`load_registry(model_dir, segments)` iterates segments, loads individual model
files, assembles `EnsembleModel` with calibrator, selected features, model
weights, and base rate. Missing or corrupt models are logged and skipped.

CatBoost is the preferred third model; LSTM is loaded as fallback for backward
compatibility.

---

## 10. Staleness Detection

**Source:** `src/finalayze/ml/staleness.py`

`StalenessDetector` monitors distribution drift via KL divergence between
training data and recent market data.

**Capabilities:**
- **Input KL divergence:** `compute_kl_divergence(train_dist, recent_dist)` via
  histogram discretization (50 bins, epsilon-smoothed). Threshold: 0.3.
- **Per-feature drift:** `get_top_drifting_features(n=3)` returns features
  with highest KL divergence.
- **Output distribution KL:** Tracks drift in model prediction distribution.
- **Rolling Brier score:** 60-day rolling window of prediction vs actual outcome.

Thread-safe via `threading.Lock`. Requires minimum 50 recent data points before
computing KL scores. Window size: 252 (1 year of trading days).

---

## 11. Meta-Labeling

**Source:** `src/finalayze/ml/meta_labeler.py`

Instead of predicting market direction, the `MetaLabeler` predicts
P(profitable) for signals from rule-based strategies. This is more tractable
and eliminates calibration bias.

- Adds `signal_confidence` and `signal_direction_buy` to feature dict
- Threshold: 0.40 (trades above threshold, skip below)
- Position sizing: linear map from [threshold, 1.0] to [0.0, 1.0]
- Returns None for untrained models (prob exactly 0.5)

---

## 12. Integration with MLEnsembleStrategy

**Source:** `src/finalayze/strategies/ml_strategy.py`

The `MLEnsembleStrategy` wraps `MLModelRegistry` + `EnsembleModel`:

1. Fetch features for the symbol using `compute_features(candles, benchmark, vix)`
2. Filter to `ensemble.selected_features` if available
3. Call `ensemble.predict_proba(features, symbol=symbol)`
4. Map probability to direction using base-rate-adjusted thresholds:
   - BUY if `prob > base_rate + threshold`
   - SELL if `prob < base_rate - threshold`
   - Deadzone in between (no signal)
5. Adjust confidence by epistemic uncertainty (`prediction_uncertainty`)
6. Apply minimum confidence filter (`min_confidence` param)
7. Emit `Signal` with confidence and reasoning

**Default thresholds:** `threshold=0.30`, `min_confidence=0.15`.
Currently disabled in all segment YAML presets.

---

## 13. Current Status and Known Issues

### Status: Grade D (2026-03-08)

- Models are untrained; `ml_ensemble` is disabled in all strategy presets
- Full pipeline exists: features (40+), labeling, training, calibration, quality gates
- CatBoost has replaced LSTM as the primary third model (better for small datasets)
- Walk-forward training with quality gates and BH correction implemented
- Feature selection (MI-based with greedy deduplication) implemented
- Market-neutral labels via benchmark alignment implemented
- Model integrity verification (HMAC-SHA256) implemented

### Known limitations

1. **No live training data:** Models have not been trained on real market data
2. **VIX features unavailable for MOEX:** VIX is US-only; MOEX segments get
   zeros for VIX features, relying on realized_vol_ratio
3. **MOEX data dependency:** MOEX training requires `FINALAYZE_TINKOFF_TOKEN`
4. **LSTM unused in practice:** CatBoost preferred; LSTM maintained for
   backward compatibility only
5. **Sample weights untested at scale:** Uniqueness * decay * barrier weight
   interaction not validated on real training runs

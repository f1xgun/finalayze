# ML Ensemble in Finalayze Trading System

> **Purpose:** Complete reference for the ML subsystem — architecture, training pipeline,
> current limitations, and improvement roadmap. Use this doc to onboard into the ML
> codebase for a parallel research/implementation session.

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TRAINING (offline)                          │
│                                                                     │
│  scripts/train_models.py                                            │
│    ├─ Fetch candles: DB → Tinkoff API → yfinance                   │
│    ├─ Triple barrier labeling (ATR-scaled)                          │
│    ├─ Feature engineering (22 features)                              │
│    ├─ MI feature selection (max 15, train-only)                     │
│    ├─ Sample weights: decay × barrier_pnl                           │
│    ├─ Train: XGBoost + LightGBM + LSTM                             │
│    ├─ Evaluate on OOS test set                                      │
│    ├─ Fit EnsembleCalibrator (Platt scaling)                        │
│    └─ Save: models/{segment_id}/*.pkl + selected_features.json      │
└─────────────────────────┬───────────────────────────────────────────┘
                          │ models/ directory
┌─────────────────────────▼───────────────────────────────────────────┐
│                       INFERENCE (live/backtest)                     │
│                                                                     │
│  ml/loader.py → load_registry(models/, segments)                    │
│    └─ MLModelRegistry (thread-safe, per-segment EnsembleModel)      │
│                                                                     │
│  strategies/ml_strategy.py → MLStrategy.generate_signal()           │
│    ├─ compute_features(candles) → 22 features                       │
│    ├─ Filter to selected_features                                   │
│    ├─ ensemble.predict_proba(features)                              │
│    │   ├─ XGBoost prob + LightGBM prob + LSTM prob                 │
│    │   ├─ Average (skip untrained)                                  │
│    │   └─ Calibrate (Platt) OR stack (LogisticRegression)          │
│    ├─ Deadzone filter: |prob - 0.5| > threshold                    │
│    ├─ Min confidence filter: confidence > min_confidence            │
│    └─ Return Signal(direction, confidence) or None                  │
│                                                                     │
│  strategies/combiner.py → StrategyCombiner                          │
│    ├─ ml_ensemble is REINFORCER-ONLY (cannot create standalone)     │
│    ├─ ADX routing gates momentum/MR pools (ML is neutral)           │
│    └─ Weighted combination under "firing" normalization             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Current State (2026-03-07)

**Status: ML disabled in all 9 presets.** Models degraded trading in every tested configuration.

### Training Results (latest, triple barrier labels)

| Segment | XGB acc | LGBM acc | LSTM acc | Best Brier | Samples |
|---------|---------|----------|----------|------------|---------|
| us_tech | 0.499 | 0.531 | 0.581 | 0.240 | ~2100 |
| us_broad | 0.661 | 0.593 | 0.571 | 0.245 | ~2100 |
| us_finance | 0.708 | 0.708 | 0.708 | 0.229 | ~2100 |
| us_healthcare | 0.441 | 0.349 | 0.465 | 0.250 | ~2100 |
| ru_blue_chips | 0.380 | 0.403 | 0.416 | 0.281 | ~700 |
| ru_energy | 0.574 | 0.442 | 0.442 | 0.243 | ~500 |

### Backtest Experiment Results

| Configuration | PF | Trades | WF Sharpe |
|---|---|---|---|
| Baseline (no ML) | 1.23 | 867 | -0.0008 |
| ML standalone (threshold=0.08, min_conf=0.15) | 1.07 | 2384 | -0.0009 |
| ML standalone (threshold=0.08, min_conf=0.30) | 1.09 | 2194 | +0.0007 |
| ML reinforcer-only, min_conf=0.30 | 1.28 | 949 | -0.0005 |
| ML reinforcer-only, min_conf=0.15 | 1.30 | 938 | -0.0016 |
| ML disabled | 1.23 | 867 | -0.0008 |

**Core problem:** Calibrated probabilities cluster at 0.55-0.65 (persistent bullish bias).
ML fires on 93% of bars with weak confidence, adding noise trades.

---

## 3. File Map

### Training Pipeline

| File | Purpose |
|---|---|
| `scripts/train_models.py` | Main training script, data fetching, orchestration |
| `src/finalayze/ml/training/labeling.py` | Triple barrier label generation |
| `src/finalayze/ml/training/sample_weights.py` | Decay weights, uniqueness weights |
| `src/finalayze/ml/training/cpcv.py` | Combinatorial purged cross-validation |
| `src/finalayze/ml/training/__init__.py` | Quality gates (accuracy, brier, logloss) |
| `src/finalayze/ml/calibration.py` | EnsembleCalibrator (Platt scaling) |

### Feature Engineering

| File | Purpose |
|---|---|
| `src/finalayze/ml/features/technical.py` | `compute_features()` — 22 technical features |

### Models

| File | Purpose |
|---|---|
| `src/finalayze/ml/models/xgboost_model.py` | XGBClassifier wrapper |
| `src/finalayze/ml/models/lightgbm_model.py` | LGBMClassifier wrapper |
| `src/finalayze/ml/models/lstm_model.py` | 2-layer LSTM (PyTorch) |
| `src/finalayze/ml/models/ensemble.py` | EnsembleModel (average + calibrate) |
| `src/finalayze/ml/models/stacking.py` | Stacking meta-learner (LogisticRegression) |

### Infrastructure

| File | Purpose |
|---|---|
| `src/finalayze/ml/loader.py` | Save/load models, `load_registry()` |
| `src/finalayze/ml/registry.py` | Thread-safe MLModelRegistry |
| `src/finalayze/ml/staleness.py` | KL divergence drift detection |

### Integration

| File | Purpose |
|---|---|
| `src/finalayze/strategies/ml_strategy.py` | MLStrategy (BaseStrategy subclass) |
| `src/finalayze/strategies/combiner.py` | Reinforcer-only gate, weighted combination |
| `src/finalayze/backtest/journaling_combiner.py` | Records ML probas for backtest journal |

---

## 4. Feature Engineering Details

`compute_features(candles, sentiment_score=0.0) → dict[str, float]`

Requires `_MIN_CANDLES = 30`. Raises `InsufficientDataError` otherwise.

### Core Features (5)

| Feature | Description | Range |
|---|---|---|
| `rsi_14` | Relative Strength Index | 0-100 |
| `macd_hist_pct` | MACD histogram / price | ~-0.05 to +0.05 |
| `bb_pct_b` | Bollinger %B (position in bands) | 0-1 (can exceed) |
| `volume_ratio_20d` | Volume / 20-day MA volume | 0+ |
| `atr_14_pct` | ATR(14) / price | 0+ |

### Extra Features (10)

| Feature | Description |
|---|---|
| `roc_10` | Rate of Change (10 bars) |
| `willr_14` | Williams %R (-100 to 0) |
| `adx_14` | Average Directional Index (0-100) |
| `ma_slope_20` | SMA(20) slope / price |
| `hist_vol_20` | Historical volatility (stdev of returns) |
| `gk_vol_20` | Garman-Klass volatility (OHLC-based) |
| `dow_sin` | Day-of-week sin component |
| `dow_cos` | Day-of-week cos component |
| `obv_slope_10` | OBV slope / volume mean |
| `rsi_divergence` | Price ROC(14) - RSI ROC(14) |

### Microstructure Features (3)

| Feature | Description |
|---|---|
| `proximity_52wk` | Close / 252-day rolling max |
| `amihud_20d` | Amihud illiquidity ratio |
| `corwin_schultz_spread` | Bid-ask spread estimator |

### Wavelet Features (4)

| Feature | Description |
|---|---|
| `wavelet_approx_energy` | Daubechies-4 approximation energy fraction |
| `wavelet_detail{1,2,3}_energy` | Detail coefficient energy fractions (3 levels) |

### Dead Feature

| Feature | Issue |
|---|---|
| `sentiment` | Always 0.0 in training (no historical sentiment data) |

---

## 5. Model Details

### XGBoost

```python
# Default hyperparameters
max_depth = 5       # 3 for MOEX
n_estimators = 200
learning_rate = 0.05
subsample = 0.8
colsample_bytree = 0.8
reg_alpha = 0.1     # L1
reg_lambda = 1.0    # L2

# Class imbalance: scale_pos_weight = n_neg / n_pos
# Early stopping: 20 rounds on last 10% of training data (temporal)
```

### LightGBM

```python
# Same structure as XGBoost, differences:
num_leaves = 31
min_child_samples = 20
is_unbalance = True   # Auto class weight balancing

# Early stopping: 20 rounds via lgb.early_stopping callback
```

### LSTM

```python
# Architecture
sequence_length = 20   # Input: (20, n_features)
hidden_size = 64       # 2-layer LSTM
num_layers = 2
dropout = 0.2          # Between LSTM layers

# Training
epochs = 50
learning_rate = 0.001
early_stopping_patience = 5
gradient_clipping = 1.0

# ~30K-40K parameters on ~1500 sequences = overfitting territory
# BUG: sample_weight parameter accepted but IGNORED
```

### Ensemble

```python
# Averaging: mean of trained model probabilities (equal weight)
# Calibration: Platt scaling OR stacking meta-learner (never both)
# Stacking: LogisticRegression on holdout predictions (NOT fitted in training)
```

---

## 6. Triple Barrier Labeling

```
Entry price ─────────────────────────────────
              ↑ upper_pct = ATR × 2.0          → label = 1 (profit)
              │
    ──────────┼──────────────────── max_hold = 20 bars
              │
              ↓ lower_pct = ATR × 2.0          → label = 0 (loss)

Vertical barrier (timeout): label = 1 if close > entry, else 0
Noise filter: discard if |pnl_pct| < 0.5 × ATR%
```

**Constants:**
```python
_TB_UPPER_ATR_MULT = 2.0    # Symmetric (causes bullish bias!)
_TB_LOWER_ATR_MULT = 2.0
_TB_MAX_HOLD = 20
_TB_ATR_PERIOD = 14
_MOEX_ATR_UPLIFT = 1.2      # MOEX: 2.4x / 2.4x
```

---

## 7. Training Pipeline Details

### Data Flow

```python
# scripts/train_models.py
_LOOKBACK_DAYS = 1825       # 5 years US
_MOEX_LOOKBACK_DAYS = 730   # 2 years MOEX
_WINDOW_SIZE = 60           # Feature window
_TRAIN_RATIO = 0.8
_SEQUENCE_LENGTH = 20       # LSTM

# Symbol lists (CRITICALLY SMALL — only 3 per segment)
_SEGMENT_SYMBOLS = {
    "us_tech": ["AAPL", "MSFT", "GOOGL"],
    "us_healthcare": ["JNJ", "PFE", "UNH"],
    "us_finance": ["JPM", "BAC", "GS"],
    "us_broad": ["SPY", "QQQ", "IWM"],
    "ru_blue_chips": ["SBER.ME", "GAZP.ME", "LKOH.ME"],
    "ru_energy": ["NVTK.ME", "ROSN.ME"],
    "ru_tech": ["YNDX.ME", "OZON.ME"],
    "ru_finance": ["VTBR.ME", "MOEX.ME"],
}
# vs config/segments.py which defines 4-6 symbols per segment
```

### Training Steps

1. Fetch candles for each symbol (DB → API → yfinance)
2. Build triple barrier dataset: `(features, labels, barrier_weights, timestamps)`
3. Temporal split: 80% train, gap of `_WINDOW_SIZE` bars, rest = test
4. MI feature selection on train data only (max 15 features)
5. Sample weights: `decay_weights(0.5) × normalized_barrier_weights`
6. Train XGBoost, LightGBM, LSTM (each with early stopping)
7. Evaluate on test set (accuracy, brier, logloss)
8. Fit EnsembleCalibrator on test set probabilities
9. Save all artifacts to `models/{segment_id}/`

### Quality Gates

```python
_MIN_ACCURACY = 0.54
_MAX_BRIER_SCORE = 0.235
_MAX_LOG_LOSS = 0.680
```

### Model Persistence (per segment)

```
models/{segment_id}/
  ├── xgb.pkl                 # XGBoostModel (joblib)
  ├── lgbm.pkl                # LightGBMModel (joblib)
  ├── lstm.pkl                # LSTM state dict (torch.save)
  ├── lstm.pkl.scaler.pkl     # StandardScaler (pickle)
  ├── calibrator.pkl          # EnsembleCalibrator (joblib)
  └── selected_features.json  # MI-selected feature names
```

---

## 8. Inference Flow

```python
# MLStrategy.generate_signal(symbol, candles, segment_id)
ensemble = registry.get(segment_id)         # → EnsembleModel | None
features = compute_features(candles)         # → 22 features
features = filter(selected_features)         # → max 15 features
prob = ensemble.predict_proba(features)      # → calibrated 0-1

# Deadzone: skip if |prob - 0.5| <= threshold (default 0.08)
# Confidence: (|prob - 0.5|) * 2
# Min confidence: skip if confidence <= 0.15

# Signal passes to StrategyCombiner:
#   - ml_ensemble is in _REINFORCER_STRATEGIES
#   - If ONLY ml_ensemble fires → signal suppressed
#   - If ml_ensemble + rule-based fire → combined weighted score
```

---

## 9. Identified Problems & Root Causes

### P0: Critical (Must Fix First)

#### 9.1 Insufficient Training Data
- **Problem:** 3 symbols × ~700 samples = ~2100 total per segment
- **Impact:** Tree models underfit, LSTM overfits
- **Fix:** Expand to 12-15 symbols per segment (read from `config/segments.py` + add sector peers)
- **Expected gain:** 3-5x more samples → meaningful model capacity utilization

#### 9.2 Persistent Bullish Bias (Symmetric Barriers)
- **Problem:** ATR barriers 2.0/2.0 are symmetric, but markets drift up ~7% annually
- **Impact:** Upper barrier hit more often → labels skew 55-60% bullish → model learns "always buy"
- **Impact:** Platt calibration fitted on skewed test data amplifies rather than fixes the bias
- **Fix:** Market-neutral labels — subtract SPY return from PnL before labeling:
  ```python
  excess_return = pnl_pct - spy_return_over_same_period
  label = 1 if excess_return > 0 else 0
  ```
- **Alternative:** Asymmetric barriers (upper=2.5, lower=1.5)

### P1: High Impact

#### 9.3 Missing Predictive Features
- **No lagged returns** — the single most predictive feature class in financial ML (1d, 5d, 21d, 63d, 126d momentum)
- **No cross-asset features** — no SPY relative strength, rolling beta, rolling correlation
- **No regime features** — no VIX level, yield curve, market breadth
- **No multi-timeframe** — only daily indicators, no weekly/monthly context

#### 9.4 LSTM Overfitting
- ~30K parameters on ~1500 sequences
- Sample weights are ignored (accepted for API but never used)
- **Fix:** Disable LSTM when `len(train_features) < 5000`, implement weighted BCELoss

### P2: Medium Impact

#### 9.5 Equal-Weight Ensemble
- XGBoost (65% acc) and LSTM (50% acc) contribute equally
- Stacking meta-learner exists (`stacking.py`) but is never fitted during training
- **Fix:** Fit stacking on holdout set, or use validation-performance-weighted averaging

#### 9.6 No Hyperparameter Tuning
- Optuna code path exists (loads from `results/tuned_params/`) but no tuning script
- CPCV exists (`cpcv.py`) but never called from training
- **Fix:** Create `scripts/tune_hyperparams.py` using CPCV + Optuna

#### 9.7 No Walk-Forward Retraining
- Single temporal split, model trained once
- `StalenessDetector` detects drift but no automated retraining
- **Fix:** Walk-forward training loop with rolling windows

### P3: Lower Priority

#### 9.8 Dead Sentiment Feature
- `sentiment` always 0.0 during training, model learns nothing
- **Fix:** Remove from features OR populate with historical FinBERT scores

#### 9.9 Sample Uniqueness Unused
- `compute_sample_uniqueness()` implemented but never called
- Triple barrier labels overlap (hold period up to 20 bars)
- **Fix:** Pass `label_spans` from labeling to weight computation

#### 9.10 Calibrator/Evaluation Same Test Set
- Calibrator fitted on same OOS test set used for metric reporting
- Reported metrics are pre-calibration but deployed model uses post-calibration
- **Fix:** Three-way split (train/calibration/test) or report post-cal metrics

---

## 10. Improvement Roadmap (Priority Order)

### Phase A: Data & Labels (Expected: models go from "harmful" to "neutral")

| # | Task | Effort | Files |
|---|---|---|---|
| A1 | Expand `_SEGMENT_SYMBOLS` to 12-15 per segment | 30 min | `scripts/train_models.py` |
| A2 | Read symbols from `config/segments.py` instead of hardcoded dict | 30 min | `scripts/train_models.py` |
| A3 | Market-neutral labels (subtract SPY return from PnL) | 2-4 hrs | `ml/training/labeling.py`, `scripts/train_models.py` |
| A4 | Asymmetric barriers as fallback (upper=2.5, lower=1.5) | 15 min | `scripts/train_models.py` |
| A5 | Extend US lookback to 7-8 years | 5 min | `scripts/train_models.py` |
| A6 | Disable LSTM when train samples < 5000 | 15 min | `scripts/train_models.py` |

### Phase B: Features (Expected: accuracy improvement 3-8%)

| # | Task | Effort | Files |
|---|---|---|---|
| B1 | Add lagged returns (1d, 5d, 21d, 63d, 126d) | 1 hr | `ml/features/technical.py` |
| B2 | Add SPY relative strength + rolling beta/correlation | 3-4 hrs | `ml/features/technical.py`, `scripts/train_models.py` |
| B3 | Add VIX regime features (level, percentile, change) | 2 hrs | `ml/features/technical.py`, `scripts/train_models.py` |
| B4 | Add multi-timeframe RSI (28, 50 period) | 30 min | `ml/features/technical.py` |
| B5 | Add month-of-year cyclical encoding + quarter-end flag | 30 min | `ml/features/technical.py` |
| B6 | Remove dead `sentiment` feature | 15 min | `ml/features/technical.py` |

### Phase C: Model Quality (Expected: better calibration, less overfitting)

| # | Task | Effort | Files |
|---|---|---|---|
| C1 | Fit stacking meta-learner during training | 2 hrs | `scripts/train_models.py` |
| C2 | Validation-set threshold optimization (max Sharpe, not accuracy) | 1 hr | `scripts/train_models.py` |
| C3 | Implement weighted BCELoss for LSTM | 2 hrs | `ml/models/lstm_model.py` |
| C4 | Three-way split (train/calibration/test) | 1 hr | `scripts/train_models.py` |
| C5 | Integrate CPCV into training for model selection | 3 hrs | `scripts/train_models.py` |

### Phase D: Tuning & Automation (Expected: optimized parameters)

| # | Task | Effort | Files |
|---|---|---|---|
| D1 | Create `scripts/tune_hyperparams.py` (Optuna + CPCV) | 4-6 hrs | new script |
| D2 | Walk-forward retraining loop | 4-6 hrs | `scripts/train_models.py` |
| D3 | Implement sample uniqueness weighting | 3 hrs | `scripts/train_models.py`, `ml/training/labeling.py` |
| D4 | Auto-retrain on staleness detection | 4 hrs | new module |

### Phase E: External Data (Expected: orthogonal signal sources)

| # | Task | Effort | Files |
|---|---|---|---|
| E1 | FinBERT historical sentiment (offline batch) | 6-8 hrs | new module |
| E2 | Fundamental data (P/E, earnings dates) via yfinance | 4-6 hrs | `ml/features/`, `scripts/train_models.py` |
| E3 | Options data (put/call ratio, implied vol) | 4-6 hrs | new module |

---

## 11. Quick Commands

```bash
# Train all segments
uv run python scripts/train_models.py

# Train specific segment
uv run python scripts/train_models.py --segment us_tech

# Train with direction labels (simpler, for comparison)
uv run python scripts/train_models.py --label-mode direction

# Custom output directory
uv run python scripts/train_models.py --output-dir models_v2/

# Run backtest with ML enabled
# (first enable ml_ensemble in relevant presets, then:)
uv run python scripts/run_iteration.py \
  --name "ml-test" \
  --description "ML enabled with new features" \
  --segments us_tech,us_broad,us_finance,us_healthcare

# Run strategy isolation (no ML)
uv run python scripts/run_strategy_isolation.py --segment us_tech --all

# Run tests
uv run pytest tests/unit/test_ml_*.py tests/unit/test_train_*.py -q --no-cov
```

---

## 12. Key Constants Reference

```python
# Training (scripts/train_models.py)
_LOOKBACK_DAYS = 1825           # 5 years US
_MOEX_LOOKBACK_DAYS = 730       # 2 years MOEX
_WINDOW_SIZE = 60               # Feature window
_TRAIN_RATIO = 0.8
_SEQUENCE_LENGTH = 20           # LSTM
_TB_UPPER_ATR_MULT = 2.0       # Triple barrier upper
_TB_LOWER_ATR_MULT = 2.0       # Triple barrier lower
_TB_MAX_HOLD = 20              # Max hold bars
_TB_ATR_PERIOD = 14
_MOEX_ATR_UPLIFT = 1.2

# Inference (strategies/ml_strategy.py)
_DEFAULT_THRESHOLD = 0.08       # Deadzone width
_DEFAULT_MIN_CONFIDENCE = 0.15  # Min confidence to emit signal

# Quality gates (ml/training/__init__.py)
_MIN_ACCURACY = 0.54
_MAX_BRIER_SCORE = 0.235
_MAX_LOG_LOSS = 0.680

# Features (ml/features/technical.py)
_MIN_CANDLES = 30
# Total features: 22 (5 core + 10 extra + 3 microstructure + 4 wavelet)
# MI selection: max 15 passed to models

# Combiner (strategies/combiner.py)
_REINFORCER_STRATEGIES = {"ml_ensemble"}
_MIN_COMBINED_CONFIDENCE = 0.50
_MIN_EXIT_CONFIDENCE = 0.25
```

---

## 13. Testing

```bash
# ML-specific tests (currently 2142 total, ML subset ~60)
uv run pytest tests/unit/test_ml_strategy.py -v       # 15 tests
uv run pytest tests/unit/test_ml_pipeline.py -v        # 10 tests (early stopping)
uv run pytest tests/unit/test_ml_ensemble.py -v        # selected_features, calibrator
uv run pytest tests/unit/test_ml_loader.py -v          # save/load round-trip
uv run pytest tests/unit/test_ml_validation.py -v      # quality gates
uv run pytest tests/unit/test_triple_barrier.py -v     # labeling
uv run pytest tests/unit/test_train_gap.py -v          # train/test gap
uv run pytest tests/unit/test_train_models_script.py -v # training script
uv run pytest tests/unit/test_cpcv.py -v               # CPCV splits
uv run pytest tests/unit/test_labeling_params.py -v    # barrier params
uv run pytest tests/unit/test_strategy_combiner.py -v  # reinforcer-only
```

---

## 14. Dependency Layers

```
Layer 0: core/schemas.py, core/exceptions.py
Layer 1: config/settings.py, config/segments.py
Layer 2: data/  (fetchers)
Layer 3: ml/    ← ALL ML CODE LIVES HERE
Layer 4: strategies/ml_strategy.py  (uses ml/)
Layer 4: strategies/combiner.py     (uses ml_strategy)
```

ML code (`src/finalayze/ml/`) can import from layers 0-2 only.
`ml_strategy.py` (layer 4) imports from `ml/` (layer 3).
Never import `strategies/` from `ml/`.

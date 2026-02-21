# ML Pipeline Design

## Overview

ML models are trained and deployed per-segment. Each segment has its own
feature set, training data, and model ensemble.

## Model Ensemble

| Model | Type | Horizon | Use |
|-------|------|---------|-----|
| XGBoost | Gradient boosting | Short-term (1-3 days) | Primary direction prediction |
| LightGBM | Gradient boosting | Short-term (1-3 days) | Ensemble diversity |
| LSTM | Deep learning | Multi-day (3-5 days) | Sequence patterns |

## Feature Sets

### Common Features (all segments)
- Technical indicators: RSI, MACD, Bollinger Bands, ATR, OBV
- Price features: returns, volatility, momentum
- Volume features: relative volume, volume trend

### US-Specific Features
- VIX (volatility index)
- S&P 500 index level
- Treasury yields
- Sector ETF performance

### MOEX-Specific Features
- USD/RUB exchange rate
- Oil price (Brent)
- MOEX index level
- RTS index level

## Per-Segment Training

- Each segment trains separate XGBoost/LightGBM/LSTM models
- Time-series split (no look-ahead bias)
- Walk-forward validation
- Models stored with versioning in model registry

## Confidence Calibration

Raw model outputs are calibrated to produce reliable confidence scores:
- Platt scaling for tree models
- Temperature scaling for LSTM
- Ensemble averaging across models

## Status

**Phase 2:** XGBoost + LightGBM per segment.
**Phase 3:** LSTM added, model registry with versioning.

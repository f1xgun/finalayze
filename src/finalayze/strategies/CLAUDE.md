# Strategies

## Purpose
Trading strategy implementations and signal combination. Eight strategies (5 enabled) generating BUY/SELL/HOLD signals, combined via per-segment weighted voting with ADX regime routing.

## Layer
Layer 4 -- Strategy / Risk. Can import from layers 0-3. Never import from layers 5-6.

## Key Files
- `base.py` -- BaseStrategy ABC: name, supported_segments(), generate_signal(), get_parameters()
- `combiner.py` -- StrategyCombiner: per-segment YAML weights, ADX routing, HRP allocation, turn-of-month boost, reinforcer gating
- `adx.py` -- ADX(14) computation and regime classification (trend/MR/mixed)
- `dual_momentum.py` -- Dual momentum (absolute + relative). Primary strategy, Sharpe +0.137.
- `mean_reversion.py` -- Bollinger band mean reversion. PF 1.98 but low trade count.
- `rsi2_connors.py` -- RSI(2) Connors strategy. Short-hold (5 bars max).
- `momentum.py` -- Single momentum. Enabled at reduced weight.
- `ou_mean_reversion.py` -- Ornstein-Uhlenbeck mean reversion. Enabled for us_tech.
- `ml_strategy.py` -- ML ensemble strategy (reinforcer-only, weight=0.10).
- `event_driven.py` -- News event strategy (DISABLED, no real-time feed).
- `pead.py` -- Post-earnings announcement drift (DISABLED).
- `pairs.py` -- Pairs trading (cointegration-based).
- `bond_carry.py`, `bond_duration_rotation.py` -- Bond strategies for MOEX.
- `dividend_gap.py` -- Dividend gap trading.
- `ichimoku.py` -- Ichimoku cloud strategy.
- `presets/` -- Per-segment YAML weight files (us_tech.yaml, us_broad.yaml, etc.)
- `adaptive_combiner.py` -- Adaptive weight adjustment (experimental).
- `hrp.py` -- Hierarchical Risk Parity weight computation.
- `vol_targeting.py` -- Volatility targeting overlay.

## Public API
- `BaseStrategy` -- abstract interface for all strategies
- `StrategyCombiner` -- `generate_combined_signal(symbol, candles, segment_id, ...) -> Signal | None`
- `DualMomentumStrategy`, `OUMeanReversionStrategy`, `RSI2ConnorsStrategy` -- re-exported from `__init__.py`

## Contracts
- Input: symbol, `list[Candle]`, segment_id, sentiment_score, has_open_position flag
- Output: `Signal | None` (None = no trade). Combined signal confidence >= 0.50 for entry, >= 0.25 for exit.
- Invariants: ADX > 35 = trend strategies only, ADX < 15 = MR strategies only, 15-35 = dominant pool wins. Reinforcer strategies (ml_ensemble) cannot create standalone trades. Preset YAML weights must sum to approximately 1.0 per segment.

## Testing
- Test location: `tests/unit/test_strategies.py`, `tests/unit/test_combiner.py`, `tests/unit/test_adx_routing.py`
- Run: `uv run pytest tests/unit/test_strategies.py tests/unit/test_combiner.py tests/unit/test_adx_routing.py -v`

## Common Patterns
- Each strategy returns Signal with confidence in [0.0, 1.0] and direction BUY/SELL/HOLD
- Combiner loads weights from `presets/<segment_id>.yaml` on first call per segment
- StrategyCombiner has 4 hook methods (`_on_generate_start`, `_on_strategy_signal`, `_on_normalized`, `_on_final_signal`) for extensibility (JournalingStrategyCombiner overrides these)
- Market context propagated to strategies via duck-typed `set_market_context()`

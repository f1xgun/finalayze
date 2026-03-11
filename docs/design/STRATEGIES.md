# Trading Strategies Design

This document describes the trading strategy system as of 2026-03-08.
All strategy code lives under `src/finalayze/strategies/`.

---

## 1. Strategy Interface (BaseStrategy ABC)

Every strategy must subclass `BaseStrategy` from `src/finalayze/strategies/base.py`.

```python
from abc import ABC, abstractmethod
from finalayze.core.schemas import Candle, Signal

class BaseStrategy(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def supported_segments(self) -> list[str]: ...

    @abstractmethod
    def generate_signal(
        self,
        symbol: str,
        candles: list[Candle],
        segment_id: str,
        sentiment_score: float = 0.0,
        has_open_position: bool = False,
    ) -> Signal | None: ...

    @abstractmethod
    def get_parameters(self, segment_id: str) -> dict[str, object]: ...
```

### Abstract Method Contracts

| Method | Return type | Purpose |
|---|---|---|
| `name` | `str` | Unique strategy identifier (used as dict key in combiner) |
| `supported_segments` | `list[str]` | Segment IDs where this strategy is enabled (reads YAML presets) |
| `generate_signal` | `Signal \| None` | Core signal generation; returns `None` when no signal fires |
| `get_parameters` | `dict[str, object]` | Load per-segment params from YAML preset; returns `{}` if not found |

Note: `generate_signal` accepts `sentiment_score` and `has_open_position` arguments beyond
the basic symbol/candles/segment_id. Individual strategies may use or ignore these parameters.

### Signal Schema

`Signal` is a Pydantic v2 model defined in `src/finalayze/core/schemas.py`:

```python
class Signal(BaseModel):
    strategy_name: str
    symbol: str
    market_id: str
    segment_id: str
    direction: SignalDirection   # BUY | SELL | HOLD
    confidence: float            # [0.0, 1.0]
    features: dict[str, float]   # indicator values used for decision
    reasoning: str               # human-readable explanation
```

---

## 2. Momentum Strategy

File: `src/finalayze/strategies/momentum.py`

### Algorithm

Uses RSI (Relative Strength Index) and MACD (Moving Average Convergence Divergence)
with a **lookback window** approach. A signal fires when both conditions align within
the recent lookback:

**BUY signal:**
- RSI was **recently** below `rsi_oversold` within the lookback window (default 5-8 bars)
- MACD histogram is rising (current > previous, current > 0) OR a bullish MACD line
  crossover occurred within the lookback window
- Current RSI is below `rsi_overbought` (sanity gate)

**SELL signal:**
- RSI was **recently** above `rsi_overbought` within the lookback window
- MACD histogram is falling (current < previous, current < 0) OR a bearish MACD line
  crossover occurred within the lookback window
- Current RSI is above `rsi_oversold` (sanity gate)

**HOLD:** Any other RSI/MACD state -- no signal returned.

### Signal Deduplication

A `_SignalState` tracker prevents emitting the same direction on consecutive bars.
State resets after `neutral_reset_bars` (default 20) bars without a new signal.

### Filters (Entry Signals Only)

Exit (SELL) signals bypass all filters below to avoid suppressing legitimate exits.

- **Trend SMA filter**: When `trend_filter: true`, suppresses counter-trend entries.
  BUY is blocked when price < SMA - buffer; SELL blocked when price > SMA + buffer.
  `trend_sma_period` configurable (typically 50 or 100), not the 200-period regime SMA.
- **ADX filter**: When `adx_filter: true`, suppresses entries when ADX < `adx_threshold`
  (range-bound market detection within the strategy itself).
- **Volume filter**: When `volume_filter: true`, requires `volume_ratio > volume_min_ratio`
  (current volume vs. SMA of volume).
- **Ichimoku Cloud filter**: When `ichimoku_filter: true`, suppresses counter-trend entries
  using Ichimoku cloud bullish/bearish state.

### Confidence Calculation

```python
hist_strength = min(1.0, abs(current_hist) / avg_hist_range)
rsi_component = min(1.0, rsi_distance * 1.5)
hist_component = hist_strength
crossover_bonus = 0.15 if macd_crossover else 0.0
confidence = min(1.0, 0.4 + rsi_component * 0.3 + hist_component * 0.2 + crossover_bonus)
```

Additional confidence modifiers:
- **Sentiment**: Boosts confidence when sentiment aligns with direction, penalizes when opposed.
- **Volatility targeting**: When `vol_target_enabled: true`, scales confidence by
  `target_vol / realized_vol`.
- **Ichimoku cloud thickness**: Thicker cloud in the direction of trade adds up to +0.10.

Signals below `min_confidence` (from YAML) are discarded.

### Minimum Data Requirement

At least 30 candles required (`_MIN_CANDLES = 30`).

### Parameters (loaded from YAML)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `rsi_period` | int | 14 | RSI lookback window |
| `rsi_oversold` | float | -- | RSI level considered oversold |
| `rsi_overbought` | float | -- | RSI level considered overbought |
| `macd_fast` | int | 12 | MACD fast EMA period |
| `macd_slow` | int | 26 | MACD slow EMA period |
| `macd_hist_lookback` | int | 3 | Multi-bar histogram window for momentum detection |
| `min_confidence` | float | -- | Discard signals below this threshold |
| `lookback_bars` | int | 5 | RSI regime window + MACD crossover detection window |
| `neutral_reset_bars` | int | 20 | Bars after which deduplication state resets |
| `trend_filter` | bool | false | Enable SMA trend filter |
| `trend_sma_period` | int | 50 | SMA period for trend filter |
| `trend_sma_buffer_pct` | float | 2.0 | Buffer around SMA (percent) |
| `adx_filter` | bool | false | Enable internal ADX filter |
| `volume_filter` | bool | false | Enable volume confirmation |
| `ichimoku_filter` | bool | false | Enable Ichimoku cloud filter |
| `vol_target_enabled` | bool | false | Enable volatility targeting |
| `vol_target` | float | 0.15 | Target annualized volatility |
| `sentiment_boost` | float | 0.10 | Confidence boost for aligned sentiment |
| `sentiment_penalty` | float | 0.05 | Confidence penalty for opposed sentiment |

---

## 3. Mean Reversion Strategy

File: `src/finalayze/strategies/mean_reversion.py`

### Algorithm

Uses Bollinger Bands (BB): a middle band (simple moving average) surrounded by
upper and lower bands at `bb_std_dev` standard deviations.

**BUY signal:** Price closes below the lower Bollinger Band -- the stock has
moved too far from its mean and is expected to revert upward.

**SELL signal:** Price closes above the upper Bollinger Band -- the stock is
extended to the upside and is expected to revert downward.

**Exit at mean:** When `exit_at_mean: true` and price returns inside the bands,
an exit signal is emitted (SELL to close a BUY, BUY to close a SELL).

**HOLD:** Price is between the bands (when exit_at_mean is false).

### Filters

- **Squeeze filter**: Skips signals when bandwidth `(upper - lower) / mid` is below
  `squeeze_threshold` (default 0.02). Low volatility squeezes produce unreliable signals.
- **Minimum band distance**: Requires price to be at least `min_band_distance_pct`
  (default 0.5%) beyond the band.
- **RSI confirmation**: BUY blocked when RSI > `rsi_oversold_mr` (default 35);
  SELL blocked when RSI < `rsi_overbought_mr` (default 65).
- **Trend filter**: When `trend_filter: true`, suppresses entries in strong counter-trends
  (falling knife protection). Uses SMA or EMA with configurable period and buffer.

### Signal Deduplication

Tracks active signal direction per symbol. Repeated signals in the same direction while
price stays outside the band are suppressed. State clears when price returns inside bands.

### Confidence Calculation

```python
confidence = min(0.95, 0.45 + distance * 2.5)
```

Where `distance = abs(close - band) / band_width`. Confidence is capped at 0.95.
Signals below `min_confidence` (from YAML) are discarded.

### Parameters (loaded from YAML)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `bb_period` | int | 20 | Bollinger Band lookback |
| `bb_std_dev` | float | 2.0 | Standard deviation multiplier |
| `min_confidence` | float | 0.55 | Minimum confidence to emit a signal |
| `squeeze_threshold` | float | 0.02 | Min bandwidth to avoid squeeze |
| `min_band_distance_pct` | float | 0.005 | Min % beyond band to trigger |
| `rsi_oversold_mr` | float | 35 | RSI gate for BUY signals |
| `rsi_overbought_mr` | float | 65 | RSI gate for SELL signals |
| `rsi_period` | int | 14 | RSI computation period |
| `exit_at_mean` | bool | false | Emit exit when price returns to mean |
| `trend_filter` | bool | false | Enable trend filter |
| `trend_sma_period` | int | 50 | SMA/EMA period for trend filter |
| `trend_sma_buffer_pct` | float | 2.0 | Buffer around trend indicator (%) |
| `trend_indicator_type` | str | "sma" | "sma" or "ema" |

---

## 4. Dual Momentum Strategy

File: `src/finalayze/strategies/dual_momentum.py`

### Algorithm

Combines **relative** and **absolute** momentum using weighted returns across three
lookback periods. This is a cross-asset momentum scoring approach.

**Momentum score:**
```
score = ret_1m * weight_1m + ret_3m * weight_3m + ret_6m * weight_6m
```

Default weights: 40% for 1-month (21 bars), 30% for 3-month (63 bars), 30% for 6-month
(126 bars). All weights and lookback periods are configurable via YAML.

**BUY signal:** `score > 0` (absolute momentum gate)

**SELL signal:** `score <= sell_threshold` (default -0.05). Scores between -0.05 and 0
return no signal.

**Position cap:** Maximum 5 simultaneous positions (`_MAX_POSITIONS = 5`). New BUY signals
are suppressed when the cap is reached and no open position exists.

### Signal Deduplication

Same `_SignalState` pattern as Momentum: prevents emitting the same direction on consecutive
bars, resets after `neutral_reset_bars` (default 8) bars.

### Confidence Calculation

```python
confidence = min(0.95, 0.4 + abs(score) * 1.0)
```

### Volatility Targeting

When `vol_target_enabled: true`, confidence is scaled by `target_vol / realized_vol`
using the `compute_vol_scale` helper.

### Minimum Data Requirement

At least `max(lookback_1m, lookback_3m, lookback_6m)` candles required (default 126).

### Parameters (loaded from YAML)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `lookback_1m` | int | 21 | 1-month lookback (trading days) |
| `lookback_3m` | int | 63 | 3-month lookback |
| `lookback_6m` | int | 126 | 6-month lookback |
| `weight_1m` | float | 0.4 | Weight for 1-month return |
| `weight_3m` | float | 0.3 | Weight for 3-month return |
| `weight_6m` | float | 0.3 | Weight for 6-month return |
| `min_confidence` | float | 0.4 | Minimum confidence to emit |
| `sell_threshold` | float | -0.05 | Score below which SELL fires |
| `neutral_reset_bars` | int | 8 | Deduplication reset window |
| `vol_target_enabled` | bool | false | Enable volatility targeting |
| `vol_target` | float | 0.15 | Target annualized volatility |

---

## 5. RSI2 Connors Strategy

File: `src/finalayze/strategies/rsi2_connors.py`

### Algorithm

Short-term mean-reversion strategy using a 2-period RSI, popularized by Larry Connors.

**BUY signal:** RSI(2) < 10 AND price > SMA(200) -- deeply oversold in an uptrend.

**SELL signal:** RSI(2) > 90 AND price < SMA(200) -- deeply overbought in a downtrend.

The SMA(200) acts as a long-term trend gate, ensuring entries are taken only in the
direction of the larger trend (buying pullbacks in uptrends, selling rallies in downtrends).

### Confidence Calculation

```python
# BUY:  (10 - rsi2) / 10 * 0.8 + 0.2  -> range [0.2, 1.0]
# SELL: (rsi2 - 90) / 10 * 0.8 + 0.2  -> range [0.2, 1.0]
```

Signals below `min_confidence` (from YAML, default 0.35) are discarded.

### Minimum Data Requirement

At least `sma_trend_period + 1` candles required (default 201).

### Parameters (loaded from YAML)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `rsi_period` | int | 2 | RSI computation period |
| `rsi_buy_threshold` | float | 10.0 | RSI level for BUY signal |
| `rsi_sell_threshold` | float | 90.0 | RSI level for SELL signal |
| `sma_trend_period` | int | 200 | SMA period for trend gate |
| `sma_exit_period` | int | 5 | SMA period for exit (documented but not used in signal gen) |
| `min_confidence` | float | 0.35 | Minimum confidence to emit |

---

## 6. OU Mean Reversion Strategy

File: `src/finalayze/strategies/ou_mean_reversion.py`

### Algorithm

Models log-prices as an Ornstein-Uhlenbeck stochastic process and trades deviations
from the fitted long-run mean.

**OU Process:** `dX = mu * (theta - X) * dt + sigma * dW`
- `mu`: mean reversion speed
- `theta`: long-run mean (in log-price space)
- `sigma`: volatility
- `half_life`: `ln(2) / mu` -- time for deviations to halve

**Fitting methods:**
1. **OLS regression** (default): `dX = a + b*X + epsilon`, where `mu = -b`, `theta = -a/b`
2. **Exact MLE** (when `use_mle: true`): Maximizes the log-likelihood of the exact
   discrete-time OU transition density using scipy.optimize. Falls back to OLS if
   optimization fails.

Fitting uses only historical data (excludes current bar) to avoid look-ahead bias.

**Signal logic:**
```python
z_score = (current_log_price - theta) / ou_std
# where ou_std = sigma / sqrt(2 * mu)

if z_score < -entry_threshold:   # BUY: price far below mean
if z_score > exit_threshold:     # SELL: only when has_open_position
```

**Half-life filter:** Signals are suppressed when `half_life` falls outside
`half_life_range` (e.g., [5, 60] days). Too short = noise, too long = no mean reversion.

**Regime gate:** When a `regime_state` is passed via kwargs:
- `CRISIS` regime: all signals suppressed
- `ELEVATED` regime: entry threshold raised to at least 2.0

### Confidence Calculation

```python
confidence = min(0.95, 0.4 + abs(z_score) * 0.15)
```

### Parameters

Parameters are resolved in priority order: constructor overrides > YAML preset >
hardcoded `_SEGMENT_PARAMS` class defaults.

| Parameter | Type | Default (us_tech) | Description |
|---|---|---|---|
| `ou_window` | int | 90 | Lookback window for OU fitting |
| `entry_threshold` | float | 1.5 | Z-score threshold for entry |
| `exit_threshold` | float | 0.0 | Z-score threshold for exit |
| `half_life_range` | (int, int) | (5, 60) | Acceptable half-life range |
| `use_mle` | bool | false | Use exact MLE instead of OLS |
| `min_confidence` | float | -- | Minimum confidence (from YAML) |

---

## 7. Pairs Trading Strategy

File: `src/finalayze/strategies/pairs.py`

### Algorithm

Statistical arbitrage via Engle-Granger cointegration and spread z-score trading.

**Workflow:**
1. Peer candles are injected via `set_peer_candles(symbol, candles)` before signal generation.
2. For each configured pair involving the target symbol:
   a. **Cointegration gate**: ADF test on historical log-prices (excluding current bar).
      Pairs with p-value > 0.05 are rejected.
   b. **Hedge ratio estimation**: OLS beta from covariance matrix, or Kalman filter when
      `use_kalman: true` (requires >= 20 data points).
   c. **Spread construction**: `spread = log(A) - beta * log(B)`. Mean and std computed
      from historical spread. `spread.std()` uses `ddof=1`.
   d. **Z-score**: `z = (current_spread - spread_mean) / spread_std`

**Signal logic:**
- `z < -z_entry`: BUY (spread compressed, expect reversion)
- `z > z_entry`: SELL (spread extended, expect reversion)
- `abs(z) < z_exit`: No signal (spread already closed)
- Between `z_exit` and `z_entry`: No signal (ambiguous zone)

**Confidence:** `min(1.0, abs(z) / z_entry)`, gated by `min_confidence`.

### Kalman Filter Hedge Ratio

When `use_kalman: true`, a Kalman filter with state `[alpha, beta]` tracks the
time-varying hedge ratio using observation model `y = alpha + beta * x + noise`.
This adapts the hedge ratio to regime changes.

### Parameters (loaded from YAML)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `pairs` | list[list[str]] | -- | Configured pairs, e.g. `[[AAPL, MSFT]]` |
| `z_entry` | float | 2.0 | Z-score threshold for entry |
| `z_exit` | float | 0.5 | Z-score threshold for exit |
| `min_confidence` | float | 0.6 | Minimum confidence to emit |
| `use_kalman` | bool | false | Use Kalman filter for hedge ratio |

---

## 8. Event-Driven Strategy

File: `src/finalayze/strategies/event_driven.py`

**Status: Currently disabled in all YAML presets (weight: 0.00).**

### Algorithm

Generates signals based on news sentiment scores provided by the LLM-powered
sentiment analysis pipeline.

**BUY signal:** `sentiment_score > min_sentiment` (default 0.5)

**SELL signal:** `sentiment_score < -min_sentiment`

**Confidence:** `min(1.0, abs(sentiment) * credibility)`, where `credibility` is
a source quality score in [0.0, 1.0].

### Filters

- **Price-move guard**: If the price has already moved more than `max_price_move`
  (default 5%) since the previous candle, the news is considered already priced in
  and the signal is suppressed.
- **Sanctions proximity**: For Russian-listed equities with sanctions/geopolitical
  event types, confidence is reduced proportionally to a per-ticker sanctions proximity
  score (e.g., GAZP: 0.8, SBER: 0.3).

### Why Disabled

No real-time news feed is wired in production. The strategy requires a live
`NewsAnalyzer` pipeline producing sentiment scores, which is not yet operational.

---

## 9. ML Ensemble Strategy

File: `src/finalayze/strategies/ml_strategy.py`

**Status: Currently disabled in all YAML presets (weight: 0.00). Models untrained.**

### Algorithm

Wraps `MLModelRegistry` + `EnsembleModel.predict_proba()` as a `BaseStrategy` so
the `StrategyCombiner` can include ML predictions alongside rule-based strategies.

1. Retrieves trained `EnsembleModel` for the segment from the registry.
2. Computes technical features via `compute_features()` (sentiment intentionally
   passed as 0.0 for train/inference consistency).
3. Filters to MI-selected features if the ensemble was trained with feature selection.
4. Obtains BUY probability from `predict_proba()`.

**Direction mapping:**
```python
if prob > base_rate + threshold:   BUY,  confidence = (prob - base_rate) * 2
if prob < base_rate - threshold:   SELL, confidence = (base_rate - prob) * 2
else: None (deadzone)
```

`base_rate` defaults to 0.50 but can be overridden by trained model metadata for
base-rate correction.

### Uncertainty Discount

When ensemble models disagree (std of per-model probabilities > 0.10), confidence
is reduced: `confidence *= 1.0 - uncertainty`.

### Reinforcer Role

The ML strategy is classified as a **reinforcer** in the combiner (`_REINFORCER_STRATEGIES`).
This means it can boost signals from other strategies but cannot create standalone trades.
When only reinforcer strategies fire, the combined signal is suppressed.

### Why Disabled

Models are untrained. The training pipeline (`scripts/train_models.py`) exists but has
not been run with sufficient data to produce reliable models.

---

## 10. PEAD Strategy

File: `src/finalayze/strategies/pead.py`

**Status: Currently disabled in all YAML presets (weight: 0.00).**

### Algorithm

Post-Earnings Announcement Drift: stocks drift in the direction of earnings surprises
for 60-90 days post-announcement. The effect is stronger in emerging markets and
mid-caps due to lower institutional coverage.

**Workflow:**
1. Earnings surprises are registered via `add_earnings_surprise(surprise)` with SUE
   (Standardized Unexpected Earnings) scores.
2. For each candle within `drift_window_bars` (default 60) of an announcement:
   - `sue_score > positive_threshold` (default 1.0): BUY
   - `sue_score < negative_threshold` (default -1.0): SELL

**Confidence:**
```python
excess = abs(sue) - abs(threshold)
confidence = min(0.90, 0.35 + excess * 0.10)
```

### Why Disabled

Requires an earnings surprise data source that is not yet wired into the production
pipeline.

---

## 11. ADX Regime Routing

File: `src/finalayze/strategies/adx.py` (compute helper)
Routing logic: `src/finalayze/strategies/combiner.py` (`_compute_adx_regime`)

### Purpose

Separates strategies into trend-following and mean-reversion pools, using ADX (Average
Directional Index) to determine which pool should be active.

### Pool Classification

| Pool | Strategies |
|---|---|
| **Trend** | `momentum`, `dual_momentum` |
| **Mean Reversion** | `mean_reversion`, `pairs`, `ou_mean_reversion`, `rsi2_connors` |
| **Neutral** | `event_driven`, `ml_ensemble`, `pead` (not in either pool) |

### Regime Thresholds

Configurable per segment via YAML `regime_routing` block:

| Regime | Condition | Effect |
|---|---|---|
| **Trend** | ADX > `trend_threshold` (default 35) | Only trend-pool strategies fire; MR pool skipped |
| **Mean Reversion** | ADX < `mr_threshold` (default 15) | Only MR-pool strategies fire; trend pool skipped |
| **Ambiguous** | `mr_threshold` <= ADX <= `trend_threshold` | Both pools fire; **dominant-pool-wins** logic applies |

### Dominant-Pool-Wins (Ambiguous Regime)

When ADX is in the ambiguous zone and both pools produce signals:
- Per-pool weighted scores are tracked separately.
- The pool with the larger absolute score wins; the losing pool's contributions
  are removed from the final weighted score.
- Neutral strategies (not in either pool) always contribute.

### ADX Computation

`compute_adx()` in `adx.py` wraps `pandas_ta.adx()`. Requires at least `2 * period`
bars. Returns `None` on insufficient data, which causes fallback to "ambiguous" regime.

### YAML Configuration

```yaml
regime_routing:
  enabled: true
  adx_period: 14
  trend_threshold: 35
  mr_threshold: 15
```

When `enabled: false`, all strategies fire regardless of ADX (regime is always "ambiguous").

---

## 12. Strategy Combiner

File: `src/finalayze/strategies/combiner.py`

### Purpose

`StrategyCombiner` aggregates signals from multiple strategies into a single
combined signal using per-segment weights loaded from YAML presets.

### Weighted Ensemble Algorithm

```python
weighted_score = Decimal(0)
total_weight = Decimal(0)

for strategy_name, strategy_cfg in strategies_cfg.items():
    if not strategy_cfg.get("enabled", True):
        continue
    weight = resolve_weight(strategy_name, strategy_cfg, overrides)
    signal = strategy.generate_signal(symbol, candles, segment_id, ...)
    if signal is None:
        continue
    score = +1 if signal.direction == BUY else -1
    weighted_score += score * Decimal(str(signal.confidence)) * weight
    total_weight += weight   # only accumulated when a signal fired

net = weighted_score / denominator   # see Normalization Modes
```

### Normalization Modes

The `normalize_mode` setting (per-segment YAML or constructor) controls the denominator:

| Mode | Denominator | Effect |
|---|---|---|
| `"firing"` (default) | Sum of weights for strategies that fired | Net score scales with firing strategies only |
| `"active"` | Sum of weights for registered + enabled strategies | More conservative -- unfired strategies dilute the score |
| `"total"` | Sum of all enabled weights in config | Most conservative -- includes strategies not even registered |

### Confidence Thresholds

- **Entry threshold**: `min_combined_confidence` (default 0.50, configurable per segment in YAML).
  Combined signals with `abs(net) < threshold` are discarded.
- **Exit threshold**: `min_exit_confidence` (default 0.25). When `has_open_position` is true
  and net < 0 (SELL direction), the threshold is lowered to `min(entry_threshold, exit_threshold)`
  to allow easier exits.

### Reinforcer-Only Suppression

When all firing strategies belong to `_REINFORCER_STRATEGIES` (currently only `ml_ensemble`),
the combined signal is suppressed. This prevents ML-only signals from creating trades
without support from at least one rule-based strategy.

### Turn-of-Month Effect

For US segments (`us_*`), BUY confidence receives a +0.05 boost during the last 1 and
first 3 calendar days of the month.

### HRP Allocation Mode

When `allocation_mode="hrp"`, strategy weights are dynamically computed using
Hierarchical Risk Parity based on recorded strategy returns. HRP weights override
YAML weights once sufficient history (20+ observations per strategy) is available.

### Key Design Choices

- Uses `Decimal` arithmetic throughout to avoid floating-point accumulation errors.
- Strategies absent from `self._strategies` dict are silently skipped.
- The output `Signal` has `strategy_name` set to the **dominant strategy** (the one with
  the largest absolute contribution), not always "combined".
- `features` dict carries per-strategy confidence, direction, ADX value/regime,
  turn-of-month flag, and optionally HRP weights.

### DRY Hook Architecture

`StrategyCombiner` defines four hook methods that subclasses can override:

```python
def _on_generate_start(self, symbol, segment_id) -> None: ...
def _on_strategy_signal(self, name, strategy, signal, weight) -> None: ...
def _on_normalized(self, net, features) -> None: ...
def _on_final_signal(self, signal, contributions) -> None: ...
```

### JournalingStrategyCombiner

File: `src/finalayze/backtest/journaling_combiner.py`

A subclass used by the backtest engine that overrides the hook methods to capture
per-strategy signals, weights, features, and ML model probabilities. This provides
an audit trail without duplicating the `generate_signal()` loop.

Exposed properties: `last_signals`, `last_weights`, `last_net_score`, `last_features`,
`last_model_probas`.

---

## 13. Per-Segment YAML Presets

Strategy parameters are stored in `src/finalayze/strategies/presets/<segment_id>.yaml`.
Each file defines:
- `segment_id` -- must match the segment's ID in `config/segments.py`
- `normalize_mode` -- normalization mode for the combiner (typically "firing")
- `min_combined_confidence` -- entry threshold for combined signals
- `min_exit_confidence` -- lowered threshold for exit signals
- `regime_routing` -- ADX regime routing configuration
- `strategies` -- a map of strategy name to `{enabled, weight, params}`

### Example: us_tech (2026-03-08)

```yaml
segment_id: us_tech
normalize_mode: "firing"
min_combined_confidence: 0.30
min_exit_confidence: 0.25
regime_routing:
  enabled: true
  adx_period: 14
  trend_threshold: 35
  mr_threshold: 15
strategies:
  momentum:       { enabled: true,  weight: 0.20 }
  mean_reversion: { enabled: true,  weight: 0.25 }
  dual_momentum:  { enabled: true,  weight: 0.25 }
  pairs:          { enabled: true,  weight: 0.10 }
  ou_mean_reversion: { enabled: true, weight: 0.10 }
  rsi2_connors:   { enabled: true,  weight: 0.10 }
  event_driven:   { enabled: false, weight: 0.00 }
  ml_ensemble:    { enabled: false, weight: 0.00 }
  pead:           { enabled: false, weight: 0.00 }
```

### Russian Market Presets

Russian segments (`ru_blue_chips`, `ru_energy`, `ru_tech`, `ru_finance`) generally
use tighter RSI thresholds and wider Bollinger Bands to account for higher volatility.
Event-driven strategy includes sanctions-related event types for geopolitical risk scaling.

---

## 14. Signal Quality Gates

### Confidence Thresholds

Every strategy has a `min_confidence` parameter that gates individual signal emission.
Typical values range from 0.35 to 0.65 depending on strategy and segment.

### Combined Signal Gates

| Gate | Value | Applies To |
|---|---|---|
| Per-strategy `min_confidence` | 0.35 - 0.65 | Individual signals |
| `min_combined_confidence` | 0.30 (us_tech) | Combined entry signals |
| `min_exit_confidence` | 0.25 (us_tech) | Combined exit signals |
| Reinforcer-only suppression | -- | ML-only signals |
| ADX regime routing | -- | Wrong-pool strategies |
| Signal deduplication | -- | Repeated same-direction signals |

### Pipeline Interactions

After the combiner emits a signal, it passes through the backtest engine's pre-trade
pipeline which applies additional gates:
- Half-Kelly position sizing
- ATR-based stop-loss (strategy-specific multipliers)
- Pipeline floor (15% of base position prevents cascade to zero)
- Currency-aware sizing (RUB 5000 / USD 500 base)

---

## 15. Extension Guide

### Adding a New Strategy

1. Create `src/finalayze/strategies/my_strategy.py`:

```python
from __future__ import annotations
from finalayze.core.schemas import Candle, Signal
from finalayze.strategies.base import BaseStrategy

class MyStrategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "my_strategy"

    def supported_segments(self) -> list[str]:
        # Read from presets dir, same pattern as MomentumStrategy
        ...

    def get_parameters(self, segment_id: str) -> dict[str, object]:
        # Load from presets/<segment_id>.yaml, cache results
        ...

    def generate_signal(
        self,
        symbol: str,
        candles: list[Candle],
        segment_id: str,
        sentiment_score: float = 0.0,
        has_open_position: bool = False,
    ) -> Signal | None:
        # Compute indicators and return Signal or None
        ...
```

2. Add `my_strategy` block to each relevant `presets/<segment_id>.yaml`:

```yaml
strategies:
  my_strategy:
    enabled: true
    weight: 0.3
    params:
      some_param: 42
```

3. Register with the combiner at startup:

```python
combiner = StrategyCombiner([
    MomentumStrategy(),
    MeanReversionStrategy(),
    DualMomentumStrategy(),
    RSI2ConnorsStrategy(),
    OUMeanReversionStrategy(),
    PairsStrategy(),
    MyStrategy(),   # add here
])
```

4. If the strategy is trend-following, add its name to `_MOMENTUM_STRATEGIES` in
   `combiner.py`. If mean-reverting, add to `_MR_STRATEGIES`. Neutral strategies
   (neither set) are not gated by ADX regime routing.

5. Write tests in `tests/unit/` following the existing patterns. Pass params directly
   in tests -- do not depend on YAML files.

---

## 16. Supporting Modules

| Module | File | Purpose |
|---|---|---|
| ADX computation | `adx.py` | `compute_adx()` wrapping pandas_ta |
| Ichimoku Cloud | `ichimoku.py` | `compute_ichimoku()` for trend confirmation |
| Volatility targeting | `vol_targeting.py` | `compute_vol_scale()` for confidence scaling |
| HRP weights | `hrp.py` | `compute_hrp_weights()` for dynamic allocation |
| Hurst exponent | `hurst.py` | Hurst exponent for mean-reversion regime detection |

---

## Status

| Strategy | Name Key | Status | Notes |
|---|---|---|---|
| Momentum (RSI + MACD) | `momentum` | Enabled | All segments; weight 0.15-0.25 |
| Mean Reversion (Bollinger Bands) | `mean_reversion` | Enabled | All segments; weight 0.20-0.30 |
| Dual Momentum | `dual_momentum` | Enabled | All segments; highest trade count (414 us_tech) |
| RSI2 Connors | `rsi2_connors` | Enabled | All segments; weight 0.10 |
| OU Mean Reversion | `ou_mean_reversion` | Enabled | Most segments; disabled in some via low weight |
| Pairs Trading | `pairs` | Enabled | Segments with configured pairs; weight 0.10 |
| Event-Driven (News) | `event_driven` | **Disabled** | No live news feed; weight 0.00 |
| ML Ensemble | `ml_ensemble` | **Disabled** | Models untrained; weight 0.00; reinforcer role |
| PEAD | `pead` | **Disabled** | No earnings data source; weight 0.00 |

### Isolated Strategy Performance (us_tech, 2022-2025)

| Strategy | Sharpe | Profit Factor | Trades |
|---|---|---|---|
| dual_momentum | +0.137 | 1.29 | 414 |
| mean_reversion | +0.034 | 1.98 | 27 |
| rsi2_connors | +0.020 | 0.94 | 73 |
| momentum | -0.014 | 1.46 | 27 |
| ou_mean_reversion | -0.038 | 0.91 | 67 |

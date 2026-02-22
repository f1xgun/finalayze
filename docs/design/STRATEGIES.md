# Trading Strategies Design

This document describes the trading strategy system implemented in Phase 1.
All strategy code lives under `src/finalayze/strategies/`.

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
        self, symbol: str, candles: list[Candle], segment_id: str
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

## 2. Momentum Strategy

File: `src/finalayze/strategies/momentum.py`

### Algorithm

Uses two indicators: RSI (Relative Strength Index) and MACD (Moving Average
Convergence Divergence). A signal fires only when **both** conditions align:

**BUY signal:**
- RSI is below the `rsi_oversold` threshold (stock is oversold)
- MACD histogram crossed from negative to positive (momentum turning up)

**SELL signal:**
- RSI is above the `rsi_overbought` threshold (stock is overbought)
- MACD histogram crossed from positive to negative (momentum turning down)

**HOLD:** Any other RSI/MACD state — no signal returned.

### Confidence Calculation

```python
# BUY: distance as fraction of the oversold threshold
rsi_distance = (rsi_oversold - current_rsi) / rsi_oversold

# SELL: distance as fraction of the remaining range above overbought
rsi_distance = (current_rsi - rsi_overbought) / (100.0 - rsi_overbought)

confidence = min(1.0, 0.5 + rsi_distance * 0.3 + abs(macd_hist) * 0.1)
```

The confidence combines how far RSI is from the threshold with the magnitude
of the MACD histogram. The denominators differ: BUY normalizes against the
oversold level itself, while SELL normalizes against the remaining headroom
above the overbought level (i.e. `100 - rsi_overbought`).
Signals below `min_confidence` (from YAML) are discarded.

### Minimum Data Requirement

At least 30 candles are required (`_MIN_CANDLES = 30`). With fewer candles the
strategy returns `None` immediately.

### Parameters (loaded from YAML)

| Parameter | Type | Description |
|---|---|---|
| `rsi_period` | int | RSI lookback window (default 14) |
| `rsi_oversold` | float | RSI level considered oversold (e.g. 30) |
| `rsi_overbought` | float | RSI level considered overbought (e.g. 70) |
| `macd_fast` | int | MACD fast EMA period (default 12) |
| `macd_slow` | int | MACD slow EMA period (default 26) |
| `min_confidence` | float | Discard signals below this threshold |

## 3. Mean Reversion Strategy

File: `src/finalayze/strategies/mean_reversion.py`

### Algorithm

Uses Bollinger Bands (BB): a middle band (simple moving average) surrounded by
upper and lower bands at `bb_std_dev` standard deviations.

**BUY signal:** Price closes below the lower Bollinger Band — the stock has
moved too far from its mean and is expected to revert upward.

**SELL signal:** Price closes above the upper Bollinger Band — the stock is
extended to the upside and is expected to revert downward.

**HOLD:** Price is between the bands.

### Confidence Calculation

```python
distance = (lower - last_close) / band_width   # for BUY
confidence = min(1.0, 0.5 + distance * 2.0)
```

The `distance` measures how far outside the band the price is, normalized
by band width. Signals below `min_confidence` are discarded.

### Bollinger Band Column Detection

`pandas_ta` column names vary by version. The implementation detects
columns by prefix (`BBL_`, `BBU_`, `BBM_`) to remain version-agnostic:

```python
def _find_bb_column(bb: pd.DataFrame, prefix: str) -> str | None:
    for col in bb.columns:
        if str(col).startswith(prefix):
            return str(col)
    return None
```

### Parameters (loaded from YAML)

| Parameter | Type | Description |
|---|---|---|
| `bb_period` | int | Bollinger Band lookback (default 20) |
| `bb_std_dev` | float | Standard deviation multiplier (default 2.0) |
| `min_confidence` | float | Minimum confidence to emit a signal |

## 4. Strategy Combiner

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
    weight = Decimal(str(strategy_cfg["weight"]))
    signal = strategy.generate_signal(symbol, candles, segment_id)
    if signal is None:
        continue   # weight is NOT added for strategies that return no signal
    score = +1 if signal.direction == BUY else -1
    weighted_score += score * Decimal(str(signal.confidence)) * weight
    total_weight += weight   # only accumulated when a signal fired

net = weighted_score / total_weight   # normalized over strategies that fired
```

- `net > 0` → combined BUY with `confidence = abs(net)`
- `net < 0` → combined SELL with `confidence = abs(net)`
- `abs(net) < 0.50` (`_MIN_COMBINED_CONFIDENCE`) → no combined signal
- `total_weight == 0` (no strategy fired) → `None` returned immediately

### Key Design Choices

- Uses `Decimal` arithmetic throughout to avoid floating-point accumulation errors.
- Strategies absent from `self._strategies` dict are silently skipped (e.g.
  `event_driven` is listed in YAML but not yet implemented).
- The output `Signal` has `strategy_name="combined"` to distinguish it from
  individual strategy signals.
- `features` dict carries per-strategy confidence and direction values for
  downstream ML audit trail.

## 5. Per-Segment YAML Presets

Strategy parameters are stored in `src/finalayze/strategies/presets/<segment_id>.yaml`.
Each file defines:
- `segment_id` — must match the segment's ID in `config/segments.py`
- `strategies` — a map of strategy name to `{enabled, weight, params}`

### Example: us_tech vs us_broad

`us_tech.yaml` — Tech stocks, high-growth, event-sensitive:
```yaml
segment_id: us_tech
strategies:
  momentum:
    enabled: true
    weight: 0.4
    params:
      rsi_oversold: 30
      rsi_overbought: 70
      min_confidence: 0.6
  mean_reversion:
    enabled: true
    weight: 0.2
    params:
      min_confidence: 0.65
  event_driven:          # placeholder, not yet implemented
    enabled: true
    weight: 0.4
```

`us_broad.yaml` — ETFs (SPY, QQQ), equal weight between strategies:
```yaml
segment_id: us_broad
strategies:
  momentum:
    enabled: true
    weight: 0.5
    params:
      min_confidence: 0.55
  mean_reversion:
    enabled: true
    weight: 0.5
    params:
      min_confidence: 0.6
```

Key differences:
- `us_tech` splits weight 40/20/40 across momentum/mean_reversion/event_driven;
  momentum threshold `min_confidence: 0.6` is higher (more selective).
- `us_broad` uses only momentum + mean_reversion at 50/50; thresholds are
  slightly lower reflecting lower volatility ETFs.

### Russian Market Presets

Russian segments (`ru_blue_chips`, `ru_energy`, `ru_tech`, `ru_finance`) all
use tighter RSI thresholds (oversold: 25-28, overbought: 72-75) and wider
Bollinger Bands (`bb_std_dev: 2.2-2.5`) to account for higher market volatility.
Event-driven weight is dominant (0.5) for `ru_blue_chips` and `ru_energy`
reflecting geopolitical and commodity event sensitivity.

## 6. Extension Guide

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
        # Load from presets/<segment_id>.yaml
        ...

    def generate_signal(
        self, symbol: str, candles: list[Candle], segment_id: str
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
    MyStrategy(),   # add here
])
```

4. Write tests in `tests/unit/strategies/test_my_strategy.py` following
   the existing `test_momentum.py` / `test_mean_reversion.py` patterns.

## Status

| Strategy | Phase 1 | Notes |
|---|---|---|
| Momentum (RSI + MACD) | Implemented | All 8 segments have presets |
| Mean Reversion (Bollinger Bands) | Implemented | All 8 segments have presets |
| Event-Driven (News LLM) | Planned (Phase 2) | YAML weight slots reserved |
| Pairs Trading | Planned (Phase 3) | Statistical arbitrage |

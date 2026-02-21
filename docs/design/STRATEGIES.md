# Trading Strategy Design

## Overview

Finalayze implements four trading strategies. Each strategy operates per-segment
with configurable parameters loaded from YAML presets.

## Strategy Interface

All strategies implement `BaseStrategy` (see `src/finalayze/strategies/base.py`):

```python
class BaseStrategy(ABC):
    def name(self) -> str: ...
    def supported_segments(self) -> list[str]: ...
    async def generate_signals(self, symbol, candles, features, sentiment, segment_config) -> Signal | None: ...
    def get_parameters(self, segment_id: str) -> dict: ...
```

## Strategies

### 1. Momentum (RSI + MACD)

- **Signals:** BUY when RSI oversold + MACD bullish cross; SELL when RSI overbought + MACD bearish cross
- **Key params:** `rsi_period`, `rsi_oversold`, `rsi_overbought`, `macd_fast`, `macd_slow`
- **Best for:** Trending markets (`us_tech`, `us_broad`, `ru_blue_chips`)

### 2. Mean Reversion (Bollinger Bands)

- **Signals:** BUY when price touches lower band; SELL when price touches upper band
- **Key params:** `bb_period`, `bb_std_dev`, `min_confidence`
- **Best for:** Range-bound markets (`us_finance`, `ru_finance`)

### 3. Event-Driven (News-Based)

- **Signals:** Generated from LLM sentiment analysis of relevant news
- **Key params:** `min_sentiment`, `event_types`
- **Best for:** News-sensitive segments (`us_healthcare`, `ru_blue_chips`, `ru_energy`)

### 4. Pairs Trading (Statistical Arbitrage)

- **Signals:** Based on cointegration spread deviation
- **Key params:** `lookback_period`, `entry_zscore`, `exit_zscore`
- **Best for:** Correlated pairs within same market
- **Status:** Planned for Phase 3

## Signal Combination

The `combiner.py` module produces a weighted ensemble signal per segment.
Weights are defined in YAML presets (e.g., `us_tech`: momentum 0.4, mean_reversion 0.2, event_driven 0.4).

## Status

**Phase 1:** Momentum + Mean Reversion implemented for US segments.
**Phase 2:** Event-Driven added.
**Phase 3:** Pairs Trading added.

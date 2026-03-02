# Strategy & Infrastructure Improvement Plan

**Date:** 2026-03-02
**Based on:** `docs/research/2026-03-02-deep-strategy-research.md`
**Baseline:** 48 symbols, 61% win rate, +0.07 avg Sharpe, 22/48 positive Sharpe
**Target:** Sharpe > 0.50 avg, 35+ symbols positive Sharpe, max drawdown < 15%

---

## Phase A: Quick Wins + Regime Detection (Weeks 1-4)

Expected aggregate Sharpe improvement: +0.30 to +0.60

---

### A.1 VIX Regime Filter

**Priority:** P0 (highest impact-to-effort ratio)
**Expected Sharpe gain:** +0.15 to +0.30
**Dependencies:** None
**Estimated effort:** 3-5 days

#### Rationale

The research identifies VIX-based regime filtering as the easiest high-impact improvement.
The current system trades identically regardless of market regime, which means it takes
full-size positions during crisis periods when most strategies have negative expectancy.

#### Files to Create

1. **`src/finalayze/risk/regime.py`** (new file)

```python
"""Market regime detection (Layer 4)."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from finalayze.core.schemas import Candle


class MarketRegime(StrEnum):
    LOW_VOL = "low_vol"        # VIX < 15
    NORMAL = "normal"          # VIX 15-20
    ELEVATED = "elevated"      # VIX 20-30
    CRISIS = "crisis"          # VIX > 30


@dataclass(frozen=True, slots=True)
class RegimeState:
    regime: MarketRegime
    vix_value: float
    sma200_above: bool       # Is SPY above its 200-day SMA?
    position_scale: Decimal  # Multiplier for position sizing [0.0, 1.0]
    allow_new_longs: bool
    allow_new_shorts: bool


# VIX thresholds
_VIX_LOW = 15.0
_VIX_NORMAL = 20.0
_VIX_ELEVATED = 30.0


def classify_regime(vix_value: float) -> MarketRegime:
    """Classify market regime from VIX value."""
    if vix_value < _VIX_LOW:
        return MarketRegime.LOW_VOL
    if vix_value < _VIX_NORMAL:
        return MarketRegime.NORMAL
    if vix_value < _VIX_ELEVATED:
        return MarketRegime.ELEVATED
    return MarketRegime.CRISIS


_REGIME_SCALES: dict[MarketRegime, Decimal] = {
    MarketRegime.LOW_VOL: Decimal("1.0"),
    MarketRegime.NORMAL: Decimal("1.0"),
    MarketRegime.ELEVATED: Decimal("0.50"),
    MarketRegime.CRISIS: Decimal("0.25"),
}


def compute_sma(candles: list[Candle], period: int = 200) -> float | None:
    """Compute SMA from candle close prices. Returns None if insufficient data."""
    if len(candles) < period:
        return None
    closes = [float(c.close) for c in candles[-period:]]
    return sum(closes) / len(closes)


def compute_regime_state(
    vix_value: float,
    spy_candles: list[Candle] | None = None,
) -> RegimeState:
    """Compute full regime state from VIX and optional SPY candle data.

    Args:
        vix_value: Current VIX level.
        spy_candles: SPY candles for SMA200 computation. If None, sma200_above defaults True.

    Returns:
        RegimeState with position scaling and trade permission flags.
    """
    regime = classify_regime(vix_value)
    scale = _REGIME_SCALES[regime]

    sma200_above = True
    if spy_candles is not None:
        sma_val = compute_sma(spy_candles, 200)
        if sma_val is not None:
            current_price = float(spy_candles[-1].close)
            sma200_above = current_price > sma_val

    # When SPY is below SMA200, halve the position scale for longs
    if not sma200_above:
        scale = scale * Decimal("0.5")

    allow_longs = regime != MarketRegime.CRISIS or not sma200_above
    allow_shorts = True  # Shorts are always allowed

    # In crisis + below SMA200: no new longs
    if regime == MarketRegime.CRISIS and not sma200_above:
        allow_longs = False

    return RegimeState(
        regime=regime,
        vix_value=vix_value,
        sma200_above=sma200_above,
        position_scale=scale,
        allow_new_longs=allow_longs,
        allow_new_shorts=allow_shorts,
    )
```

#### Files to Modify

2. **`src/finalayze/backtest/engine.py`** -- add regime awareness to `BacktestEngine`

**Insertion point:** `BacktestEngine.__init__()` -- add `regime_state` parameter.

```python
# In __init__ signature, add:
regime_state: RegimeState | None = None,

# Store it:
self._regime_state = regime_state
```

**Insertion point:** `_handle_buy()` method, after position_value computation (~line 858),
before confidence scaling:

```python
# Apply regime-based position scaling
if self._regime_state is not None:
    if signal is not None and signal.direction == SignalDirection.BUY:
        if not self._regime_state.allow_new_longs:
            self._journal_skip(...)
            return
    position_value = position_value * self._regime_state.position_scale
```

3. **`config/settings.py`** -- add regime configuration fields

```python
# Under "# Risk" section, add:
vix_regime_enabled: bool = True
vix_low_threshold: float = 15.0
vix_normal_threshold: float = 20.0
vix_elevated_threshold: float = 30.0
sma200_filter_enabled: bool = True
```

4. **`src/finalayze/strategies/presets/*.yaml`** -- add regime section to each preset

```yaml
regime:
  vix_regime_enabled: true
  sma200_filter_enabled: true
  crisis_allow_longs: false
  elevated_scale: 0.5
  crisis_scale: 0.25
```

#### Tests to Write

5. **`tests/unit/test_regime.py`**

| Test Case | Description |
|-----------|-------------|
| `test_classify_regime_low_vol` | VIX=12 returns LOW_VOL |
| `test_classify_regime_normal` | VIX=17 returns NORMAL |
| `test_classify_regime_elevated` | VIX=25 returns ELEVATED |
| `test_classify_regime_crisis` | VIX=35 returns CRISIS |
| `test_regime_state_full_risk` | VIX=12 + SPY above SMA200 -> scale=1.0, longs=True |
| `test_regime_state_elevated` | VIX=25 -> scale=0.5 |
| `test_regime_state_crisis_below_sma` | VIX=35 + SPY below SMA200 -> allow_longs=False |
| `test_compute_sma_insufficient_data` | <200 candles returns None |
| `test_compute_sma_correct_value` | Known candle set produces expected SMA |
| `test_engine_regime_scales_position` | BacktestEngine with regime_state scales position_value |
| `test_engine_regime_blocks_longs_in_crisis` | Crisis regime skips BUY signals |

#### Acceptance Criteria

- [ ] `classify_regime()` covers all 4 VIX thresholds
- [ ] `compute_regime_state()` combines VIX + SMA200 correctly
- [ ] BacktestEngine applies `position_scale` to all BUY orders when regime_state is set
- [ ] BacktestEngine skips BUY orders when `allow_new_longs=False`
- [ ] Backtest with VIX>30 data shows reduced position sizes (verifiable in trades)
- [ ] All preset YAML files have regime section
- [ ] 100% test coverage on `src/finalayze/risk/regime.py`

---

### A.2 SMA200 Trend Filter at Engine Level

**Priority:** P0
**Expected Sharpe gain:** +0.05 to +0.10
**Dependencies:** A.1 (shares SMA computation)
**Estimated effort:** 2-3 days

#### Rationale

Currently, trend filtering (SMA-based) is implemented per-strategy in `mean_reversion.py`
(line 211) and `momentum.py` (line 358). This means some strategies use it, others do not.
Moving the trend filter to engine level ensures ALL strategies benefit from not trading
against the primary trend.

#### Files to Modify

1. **`src/finalayze/backtest/engine.py`**

**Add to `__init__`:**
```python
trend_filter_enabled: bool = False,
trend_sma_period: int = 200,
```

**Add to `run()` method, after signal generation (line 356), before acting on signal:**
```python
# Engine-level trend filter: suppress counter-trend entries
if self._trend_filter_enabled and signal is not None and signal.direction == SignalDirection.BUY:
    sma_val = _compute_sma(history, self._trend_sma_period)
    if sma_val is not None:
        current_close = float(candles[i].close)
        if current_close < sma_val * 0.98:  # 2% buffer below SMA
            signal = None  # Suppress buy below SMA200
```

**Add private helper (or import from regime.py):**
```python
def _compute_sma(candles: list[Candle], period: int) -> float | None:
    if len(candles) < period:
        return None
    return sum(float(c.close) for c in candles[-period:]) / period
```

2. **Optionally remove per-strategy trend filters** from `mean_reversion.py` and
`momentum.py` to avoid double-filtering. Alternatively, keep them as additional
per-strategy refinements but document that engine-level filter takes priority.

#### Tests to Write

| Test Case | Description |
|-----------|-------------|
| `test_engine_trend_filter_suppresses_buy_below_sma` | BUY signal suppressed when close < SMA200*0.98 |
| `test_engine_trend_filter_allows_buy_above_sma` | BUY signal passes when close > SMA200 |
| `test_engine_trend_filter_allows_sell_always` | SELL signals are never suppressed by trend filter |
| `test_engine_trend_filter_disabled_by_default` | No filtering when trend_filter_enabled=False |
| `test_engine_trend_filter_insufficient_data` | No filtering when <200 candles available |

#### Acceptance Criteria

- [ ] Engine-level SMA200 filter is configurable via constructor parameter
- [ ] BUY signals below SMA200 (with 2% buffer) are suppressed
- [ ] SELL signals are never suppressed
- [ ] Backtest shows fewer losing trades in downtrending periods
- [ ] Per-strategy trend filters still work when engine filter is disabled

---

### A.3 Volatility-Targeted Position Sizing

**Priority:** P1
**Expected Sharpe gain:** +0.10 to +0.20
**Dependencies:** None (vol-adjusted sizing already exists, needs enhancement)
**Estimated effort:** 3-4 days

#### Rationale

The system already has `compute_vol_adjusted_position_size()` in `position_sizer.py`
and it is optionally applied in `_handle_buy()` (engine.py line 846). However:
- The `target_vol` parameter defaults to `None` (disabled)
- There is no VIX-based dynamic adjustment
- The scale bounds (`min_scale=0.25`, `max_scale=2.0`) are hardcoded

This task enhances vol-targeting with regime awareness and proper defaults.

#### Files to Modify

1. **`src/finalayze/risk/position_sizer.py`** -- add hybrid Kelly-VIX function

**Add new function after `compute_vol_adjusted_position_size()`:**

```python
def compute_hybrid_kelly_vix_size(
    base_position: Decimal,
    vix_value: float | None = None,
    vix_history: list[float] | None = None,
    lookback: int = 252,
) -> Decimal:
    """Scale position by inverse VIX-Rank (percentile in lookback window).

    VIX-Rank = percentile of current VIX in {lookback}-day window.
    Scale = 1 / (1 + VIX-Rank).

    When VIX-Rank > 0.8 (80th pctile): effectively Quarter-Kelly.
    When VIX-Rank < 0.2 (20th pctile): effectively Half-Kelly (full allocation).

    Args:
        base_position: Base position from Kelly sizer.
        vix_value: Current VIX value.
        vix_history: Historical VIX values (at least lookback+1 values).
        lookback: Number of days for VIX percentile computation.

    Returns:
        Scaled position size.
    """
    if vix_value is None or vix_history is None or len(vix_history) < lookback:
        return base_position

    recent_vix = vix_history[-lookback:]
    vix_rank = sum(1 for v in recent_vix if v <= vix_value) / len(recent_vix)

    scale = Decimal(str(1.0 / (1.0 + vix_rank)))
    # Floor at 0.25 (quarter-Kelly), cap at 1.0
    scale = max(Decimal("0.25"), min(Decimal("1.0"), scale))
    return base_position * scale
```

2. **`src/finalayze/backtest/engine.py`** -- change default `target_vol` to `Decimal("0.10")`
   in constructor and add VIX-rank sizing.

**In `__init__`:**
```python
target_vol: Decimal | None = Decimal("0.10"),  # 10% annualized target
```

**In `_handle_buy()`, after vol-adjusted sizing block (~line 853), add:**
```python
# Apply VIX-rank scaling if regime state provides VIX data
if self._regime_state is not None:
    position_value = compute_hybrid_kelly_vix_size(
        base_position=position_value,
        vix_value=self._regime_state.vix_value,
        vix_history=self._vix_history,  # new instance variable
    )
```

3. **`config/settings.py`** -- add settings:
```python
target_vol: float = 0.10           # 10% annualized target
vol_lookback: int = 20             # days for realized vol
vix_rank_lookback: int = 252       # days for VIX percentile
```

#### Tests to Write

| Test Case | Description |
|-----------|-------------|
| `test_hybrid_kelly_vix_low_rank` | VIX in 10th percentile -> scale near 1.0 |
| `test_hybrid_kelly_vix_high_rank` | VIX in 90th percentile -> scale near 0.25 |
| `test_hybrid_kelly_vix_no_history` | Returns base_position unchanged |
| `test_hybrid_kelly_vix_insufficient_data` | <252 values returns base unchanged |
| `test_vol_target_default_enabled` | target_vol defaults to 0.10, not None |
| `test_engine_vol_sizing_reduces_high_vol_positions` | High-vol asset gets smaller position |
| `test_engine_vol_sizing_increases_low_vol_positions` | Low-vol asset gets larger position (capped) |

#### Acceptance Criteria

- [ ] `compute_hybrid_kelly_vix_size()` scales correctly across VIX percentile range
- [ ] Vol-targeting is enabled by default (target_vol=0.10)
- [ ] Position sizes are inversely proportional to asset realized volatility
- [ ] High VIX rank (>80th pctile) reduces to ~quarter-Kelly
- [ ] No position exceeds `max_position_pct` after all adjustments
- [ ] Portfolio-level annualized vol stays near 10% target in backtest

---

### A.4 Chandelier Exit (Replace Trailing Stop)

**Priority:** P1
**Expected Sharpe gain:** +0.08 to +0.15
**Dependencies:** None
**Estimated effort:** 2-3 days

#### Rationale

The current trailing stop in `SimulatedBroker` uses a fixed ATR distance from current price.
The Chandelier Exit instead anchors to the highest high since entry, which lets winners run
longer during strong trends while still protecting against reversals.

#### Files to Create

1. **`src/finalayze/risk/chandelier_exit.py`** (new file)

```python
"""Chandelier Exit stop-loss computation (Layer 4)."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from finalayze.core.schemas import Candle


def compute_chandelier_stop(
    candles_since_entry: list[Candle],
    atr_period: int = 22,
    atr_multiplier: Decimal = Decimal("3.0"),
) -> Decimal | None:
    """Compute Chandelier Exit: Highest High(N) - ATR(N) * Multiplier.

    Unlike standard trailing stops that anchor to current price, Chandelier
    anchors to the highest high since entry, allowing stronger trends to run.

    Args:
        candles_since_entry: Candles from entry bar to current bar.
        atr_period: Lookback for ATR and highest high computation.
        atr_multiplier: Multiplier applied to ATR.

    Returns:
        Chandelier stop price, or None if insufficient data.
    """
    if len(candles_since_entry) < atr_period + 1:
        return None

    recent = candles_since_entry[-(atr_period + 1):]
    highest_high = max(c.high for c in recent)

    # ATR calculation
    true_ranges: list[Decimal] = []
    for i in range(1, len(recent)):
        prev_close = recent[i - 1].close
        high = recent[i].high
        low = recent[i].low
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        true_ranges.append(tr)

    atr = sum(true_ranges, Decimal(0)) / Decimal(len(true_ranges))
    stop = highest_high - atr * atr_multiplier
    return max(stop, Decimal(0))
```

#### Files to Modify

2. **`src/finalayze/execution/simulated_broker.py`** -- integrate Chandelier option

Add a `stop_loss_mode` parameter ("trailing" or "chandelier") to SimulatedBroker.
When mode is "chandelier", use `compute_chandelier_stop()` to update stop levels.

3. **`src/finalayze/backtest/engine.py`** -- add `stop_loss_mode` parameter

**In `__init__`:**
```python
stop_loss_mode: str = "chandelier",  # "trailing" or "chandelier"
```

Pass through to SimulatedBroker and update stop computation in `_handle_buy()`.

#### Tests to Write

| Test Case | Description |
|-----------|-------------|
| `test_chandelier_stop_basic` | Known candle series produces correct stop |
| `test_chandelier_stop_rises_with_highs` | Stop rises as new highs are made |
| `test_chandelier_stop_never_decreases` | Stop should not decrease (monotonic) |
| `test_chandelier_stop_insufficient_data` | Returns None with <23 candles |
| `test_chandelier_vs_trailing` | Chandelier lets winner run longer than trailing |
| `test_engine_chandelier_default` | Engine uses chandelier by default |

#### Acceptance Criteria

- [ ] Chandelier stop correctly computes highest_high - ATR * multiplier
- [ ] Stop never decreases (ratchet up only)
- [ ] Backtest with Chandelier shows fewer premature exits in trending markets
- [ ] Chandelier is the default stop-loss mode
- [ ] Trailing mode still available as fallback

---

### A.5 Strategy-Specific Time Exits

**Priority:** P2
**Expected Sharpe gain:** +0.05 to +0.10
**Dependencies:** None
**Estimated effort:** 1-2 days

#### Rationale

Currently `max_hold_bars=30` is hardcoded for all strategies. Research shows optimal
holding periods vary significantly by strategy type.

#### Files to Modify

1. **`src/finalayze/backtest/engine.py`**

**Replace `max_hold_bars: int = 30` with a dict-based approach:**

```python
# In __init__:
max_hold_bars: int | dict[str, int] = 30,

# Add mapping:
_DEFAULT_STRATEGY_HOLD_BARS: dict[str, int] = {
    "momentum": 45,
    "mean_reversion": 10,
    "pairs": 20,
    "event_driven": 60,
    "ml_ensemble": 30,
    "rsi2_connors": 5,
    "combined": 30,  # combiner default
}
```

**In the time-exit check (~line 306 in `run()`, line 568 in `run_portfolio()`),
look up the strategy name:**

```python
# Determine strategy-specific max_hold_bars
if isinstance(self._max_hold_bars, dict):
    strategy_name = getattr(self._strategy, 'name', 'combined')
    effective_max_hold = self._max_hold_bars.get(strategy_name, 30)
else:
    effective_max_hold = self._max_hold_bars
```

2. **`src/finalayze/strategies/presets/*.yaml`** -- add `max_hold_bars` per strategy:

```yaml
strategies:
  momentum:
    params:
      max_hold_bars: 45
  mean_reversion:
    params:
      max_hold_bars: 10
  pairs:
    params:
      max_hold_bars: 20
```

#### Tests to Write

| Test Case | Description |
|-----------|-------------|
| `test_time_exit_momentum_45_bars` | Momentum position exits at 45 bars |
| `test_time_exit_mean_reversion_10_bars` | MR position exits at 10 bars |
| `test_time_exit_default_fallback` | Unknown strategy uses default 30 |
| `test_time_exit_dict_config` | Dict-based config applies per strategy |
| `test_time_exit_int_config_backward_compat` | Int config still works (all strategies same) |

#### Acceptance Criteria

- [ ] Each strategy type can have its own max_hold_bars
- [ ] Backward compatibility: int config applies to all strategies uniformly
- [ ] Mean reversion exits faster (10 bars) than momentum (45 bars)
- [ ] Backtest shows improved Sharpe for MR strategy due to faster exits

---

### A.6 HMM Regime Detection (Advanced)

**Priority:** P1 (high impact, medium effort)
**Expected Sharpe gain:** +0.25 to +0.50
**Dependencies:** A.1 (regime framework)
**Estimated effort:** 7-10 days

#### Rationale

HMM is the single highest-impact improvement identified in the research. It detects
regime shifts before they become obvious through VIX alone.

#### Files to Create

1. **`src/finalayze/risk/hmm_regime.py`** (new file)

```python
"""Hidden Markov Model regime detection (Layer 4)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from finalayze.core.schemas import Candle
    from finalayze.risk.regime import MarketRegime


@dataclass(frozen=True, slots=True)
class HMMRegimeResult:
    """Result of HMM regime classification."""
    regime_id: int           # 0, 1, or 2
    regime_label: str        # "low_vol_bull", "high_vol_bull", "bear"
    confidence: float        # Posterior probability of assigned state
    position_scale: Decimal  # [0.0, 1.0]


class HMMRegimeDetector:
    """Detect market regimes using a Gaussian HMM.

    Trained on 3 features from SPY (or market index):
    - Daily log returns
    - 20-day realized volatility
    - Volume ratio (current volume / 20-day avg volume)

    States:
    - 0: Low-volatility bull (positive mean, low variance)
    - 1: High-volatility bull (positive mean, high variance)
    - 2: Bear/crisis (negative mean, high variance)
    """

    def __init__(
        self,
        n_states: int = 3,
        train_window: int = 504,  # 2 years
        retrain_frequency: int = 21,  # monthly
    ) -> None:
        self._n_states = n_states
        self._train_window = train_window
        self._retrain_frequency = retrain_frequency
        self._model = None
        self._bars_since_retrain = 0
        self._state_labels: dict[int, str] = {}
        self._state_scales: dict[int, Decimal] = {}

    def _compute_features(self, candles: list[Candle]) -> np.ndarray:
        """Build feature matrix: [log_return, realized_vol_20d, volume_ratio]."""
        n = len(candles)
        features = []
        for i in range(20, n):
            # Log return
            if float(candles[i - 1].close) <= 0:
                continue
            log_ret = math.log(float(candles[i].close) / float(candles[i - 1].close))

            # 20-day realized vol
            rets = [
                math.log(float(candles[j].close) / float(candles[j - 1].close))
                for j in range(i - 19, i + 1)
                if float(candles[j - 1].close) > 0
            ]
            realized_vol = float(np.std(rets)) * math.sqrt(252) if len(rets) > 1 else 0.0

            # Volume ratio
            vol_window = [candles[j].volume for j in range(i - 19, i + 1)]
            avg_vol = sum(float(v) for v in vol_window) / len(vol_window) if vol_window else 1.0
            vol_ratio = float(candles[i].volume) / avg_vol if avg_vol > 0 else 1.0

            features.append([log_ret, realized_vol, vol_ratio])

        return np.array(features) if features else np.array([]).reshape(0, 3)

    def fit(self, candles: list[Candle]) -> None:
        """Train HMM on candle data. Requires hmmlearn."""
        from hmmlearn.hmm import GaussianHMM  # noqa: PLC0415

        features = self._compute_features(candles[-self._train_window:])
        if len(features) < 60:
            return

        model = GaussianHMM(
            n_components=self._n_states,
            covariance_type="full",
            n_iter=1000,
            random_state=42,
        )
        model.fit(features)
        self._model = model
        self._bars_since_retrain = 0

        # Label states by mean return and variance
        self._label_states(model)

    def _label_states(self, model) -> None:
        """Assign human-readable labels based on state means/covariances."""
        means = model.means_[:, 0]  # Mean return per state
        covars = np.array([model.covars_[i][0, 0] for i in range(self._n_states)])

        # Sort by mean return descending
        state_order = np.argsort(-means)

        labels = ["low_vol_bull", "high_vol_bull", "bear"]
        scales = [Decimal("1.0"), Decimal("0.5"), Decimal("0.25")]

        # Assign: highest mean -> bull, lowest -> bear
        # If 2 states have positive mean, assign by variance
        for rank, state_idx in enumerate(state_order):
            self._state_labels[int(state_idx)] = labels[min(rank, len(labels) - 1)]
            self._state_scales[int(state_idx)] = scales[min(rank, len(scales) - 1)]

    def predict(self, candles: list[Candle]) -> HMMRegimeResult | None:
        """Predict current regime from recent candles."""
        if self._model is None:
            return None

        features = self._compute_features(candles)
        if len(features) == 0:
            return None

        # Predict state for the last observation
        states = self._model.predict(features)
        posteriors = self._model.predict_proba(features)

        current_state = int(states[-1])
        confidence = float(posteriors[-1, current_state])

        self._bars_since_retrain += 1

        return HMMRegimeResult(
            regime_id=current_state,
            regime_label=self._state_labels.get(current_state, "unknown"),
            confidence=confidence,
            position_scale=self._state_scales.get(current_state, Decimal("1.0")),
        )

    @property
    def needs_retrain(self) -> bool:
        return self._bars_since_retrain >= self._retrain_frequency
```

#### Files to Modify

2. **`src/finalayze/risk/regime.py`** (from A.1) -- add `from_hmm()` factory method
   to `RegimeState` that converts `HMMRegimeResult` into the standard `RegimeState`
   interface used by the engine.

3. **`src/finalayze/backtest/engine.py`** -- add optional `hmm_detector` parameter.
   When provided, retrain periodically and use HMM prediction instead of static
   VIX regime.

4. **`pyproject.toml`** -- add `hmmlearn` to dependencies:
```toml
"hmmlearn>=0.3",
```

#### Tests to Write

| Test Case | Description |
|-----------|-------------|
| `test_hmm_feature_computation` | Features matrix has correct shape and values |
| `test_hmm_fit_3_states` | Model trains successfully on synthetic data |
| `test_hmm_predict_returns_result` | Prediction returns valid HMMRegimeResult |
| `test_hmm_state_labeling` | States are labeled by mean return correctly |
| `test_hmm_retrain_flag` | `needs_retrain` True after N bars |
| `test_hmm_insufficient_data` | Returns None with <60 data points |
| `test_hmm_regime_to_engine` | HMM result correctly scales position in engine |

#### Acceptance Criteria

- [ ] HMM trains on 2-year window of SPY data
- [ ] 3 states consistently map to bull/transition/bear
- [ ] Monthly retrain trigger works
- [ ] Position scaling reduces by 50-75% during detected bear states
- [ ] Backtest on 2020 COVID crash shows early regime shift detection
- [ ] Test coverage >= 90% on `hmm_regime.py`

---

## Phase B: Strategy Upgrades (Weeks 5-10)

Expected aggregate Sharpe improvement: +0.20 to +0.40 additional

---

### B.1 Ornstein-Uhlenbeck Mean Reversion

**Priority:** P1
**Expected Sharpe gain:** +0.15 to +0.25
**Dependencies:** None
**Estimated effort:** 7-10 days

#### Rationale

The current Bollinger Band approach in `mean_reversion.py` uses fixed standard deviation
bands. OU process modeling provides statistically grounded entry/exit thresholds based on
the actual mean-reversion speed of each asset.

#### Files to Create

1. **`src/finalayze/strategies/ou_mean_reversion.py`** (new file)

```python
"""Ornstein-Uhlenbeck mean reversion strategy (Layer 4).

Replaces Bollinger Band z-score with OU process parameters
fitted via Maximum Likelihood Estimation on rolling windows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal

import numpy as np

from finalayze.core.schemas import Candle, Signal, SignalDirection
from finalayze.strategies.base import BaseStrategy


@dataclass(frozen=True, slots=True)
class OUParameters:
    """Fitted OU process parameters."""
    theta: float     # Long-term mean (equilibrium price level)
    mu: float        # Mean-reversion speed (higher = faster reversion)
    sigma: float     # Volatility of the OU process
    half_life: float # Days until half of deviation from mean is recovered


def fit_ou_mle(prices: list[float], dt: float = 1.0 / 252.0) -> OUParameters | None:
    """Fit OU process parameters via Maximum Likelihood Estimation.

    The OU process: dX = mu*(theta - X)*dt + sigma*dW

    MLE for discrete observations:
    X[t+1] = a + b*X[t] + epsilon
    where:
        b = exp(-mu*dt)
        a = theta*(1 - b)
        var(epsilon) = sigma^2 * (1 - b^2) / (2*mu)

    Args:
        prices: Price series (log prices recommended).
        dt: Time step (1/252 for daily data).

    Returns:
        OUParameters or None if fitting fails.
    """
    if len(prices) < 30:
        return None

    x = np.array(prices)
    n = len(x) - 1

    # OLS regression: X[t+1] = a + b * X[t]
    x_prev = x[:-1]
    x_next = x[1:]

    # b = cov(x_next, x_prev) / var(x_prev)
    mean_prev = np.mean(x_prev)
    mean_next = np.mean(x_next)
    cov = np.sum((x_prev - mean_prev) * (x_next - mean_next)) / n
    var_prev = np.sum((x_prev - mean_prev) ** 2) / n

    if var_prev == 0:
        return None

    b = cov / var_prev
    a = mean_next - b * mean_prev

    # Prevent invalid log
    if b <= 0 or b >= 1:
        return None

    # Extract OU parameters
    mu = -math.log(b) / dt
    theta = a / (1.0 - b)

    residuals = x_next - (a + b * x_prev)
    var_eps = float(np.var(residuals, ddof=1))
    sigma_sq = 2 * mu * var_eps / (1 - b**2) if (1 - b**2) > 0 else 0
    sigma = math.sqrt(max(0, sigma_sq))

    half_life = math.log(2) / mu if mu > 0 else float('inf')

    return OUParameters(theta=theta, mu=mu, sigma=sigma, half_life=half_life)


class OUMeanReversionStrategy(BaseStrategy):
    """Mean reversion using OU process z-score.

    Entry: OU z-score exceeds entry_threshold (default 1.5 sigma)
    Exit: OU z-score returns to 0 (mean)
    Filter: Only trade when half-life is between 5 and 60 days.
    """

    _OU_WINDOW = 90        # Rolling window for OU fitting (days)
    _ENTRY_THRESHOLD = 1.5  # Sigma units
    _EXIT_THRESHOLD = 0.0   # Mean
    _MIN_HALF_LIFE = 5.0    # Days
    _MAX_HALF_LIFE = 60.0   # Days
    _MIN_CONFIDENCE = 0.55

    def __init__(self) -> None:
        self._active_signal: dict[str, SignalDirection] = {}
        self._params_cache: dict[str, dict[str, object]] = {}

    @property
    def name(self) -> str:
        return "ou_mean_reversion"

    def supported_segments(self) -> list[str]:
        # Enabled wherever mean_reversion is enabled
        # (share the same YAML config section initially)
        return []  # populated from presets

    def get_parameters(self, segment_id: str) -> dict[str, object]:
        return self._params_cache.get(segment_id, {})

    def generate_signal(
        self,
        symbol: str,
        candles: list[Candle],
        segment_id: str,
        sentiment_score: float = 0.0,
        has_open_position: bool = False,
    ) -> Signal | None:
        if len(candles) < self._OU_WINDOW + 1:
            return None

        # Fit OU on rolling window of log prices
        log_prices = [math.log(float(c.close)) for c in candles[-self._OU_WINDOW:]]
        params = fit_ou_mle(log_prices)
        if params is None:
            return None

        # Half-life filter: only trade when mean-reversion is neither
        # too fast (noise) nor too slow (no reversion)
        if not (self._MIN_HALF_LIFE <= params.half_life <= self._MAX_HALF_LIFE):
            return None

        # Compute OU z-score
        current_log_price = math.log(float(candles[-1].close))
        if params.sigma <= 0:
            return None
        ou_std = params.sigma / math.sqrt(2 * params.mu) if params.mu > 0 else params.sigma
        z_score = (current_log_price - params.theta) / ou_std if ou_std > 0 else 0.0

        # Signal logic
        direction: SignalDirection | None = None
        if z_score < -self._ENTRY_THRESHOLD:
            direction = SignalDirection.BUY   # Price below equilibrium -> buy
        elif z_score > self._ENTRY_THRESHOLD:
            direction = SignalDirection.SELL  # Price above equilibrium -> sell
        elif abs(z_score) < abs(self._EXIT_THRESHOLD + 0.3):
            # Near mean: exit signal
            active = self._active_signal.pop(symbol, None)
            if active is not None:
                direction = (
                    SignalDirection.SELL if active == SignalDirection.BUY
                    else SignalDirection.BUY
                )
                confidence = min(0.9, 0.6 + (1.0 - abs(z_score)) * 0.3)
                return Signal(
                    strategy_name=self.name,
                    symbol=symbol,
                    market_id=candles[0].market_id,
                    segment_id=segment_id,
                    direction=direction,
                    confidence=confidence,
                    features={"ou_z_score": round(z_score, 4), "half_life": round(params.half_life, 1)},
                    reasoning=f"OU exit: z={z_score:.2f}, hl={params.half_life:.0f}d",
                )
            return None

        if direction is None:
            return None

        # Suppress repeated signals
        if self._active_signal.get(symbol) == direction:
            return None

        confidence = min(0.95, 0.5 + abs(z_score) / (self._ENTRY_THRESHOLD * 2))
        if confidence < self._MIN_CONFIDENCE:
            return None

        self._active_signal[symbol] = direction

        return Signal(
            strategy_name=self.name,
            symbol=symbol,
            market_id=candles[0].market_id,
            segment_id=segment_id,
            direction=direction,
            confidence=confidence,
            features={
                "ou_z_score": round(z_score, 4),
                "half_life": round(params.half_life, 1),
                "ou_theta": round(params.theta, 4),
                "ou_mu": round(params.mu, 4),
                "ou_sigma": round(params.sigma, 4),
            },
            reasoning=(
                f"OU mean reversion: z={z_score:.2f}, "
                f"half_life={params.half_life:.0f}d, "
                f"theta={math.exp(params.theta):.2f}"
            ),
        )
```

#### Files to Modify

2. **`src/finalayze/strategies/presets/*.yaml`** -- add `ou_mean_reversion` config:

```yaml
ou_mean_reversion:
  enabled: true
  weight: 0.25
  params:
    ou_window: 90
    entry_threshold: 1.5
    exit_threshold: 0.0
    min_half_life: 5
    max_half_life: 60
    min_confidence: 0.55
    max_hold_bars: 10  # Strategy-specific
```

3. **`src/finalayze/strategies/__init__.py`** -- add export

4. **`src/finalayze/strategies/combiner.py`** -- no changes needed (dynamically loads
   strategies by name from preset YAML)

#### Tests to Write

| Test Case | Description |
|-----------|-------------|
| `test_ou_fit_mle_basic` | Known mean-reverting series produces valid parameters |
| `test_ou_fit_mle_trending_series` | Trending series returns b>=1 -> None |
| `test_ou_fit_mle_insufficient_data` | <30 points returns None |
| `test_ou_half_life_filter_too_fast` | Half-life=2 days -> no signal |
| `test_ou_half_life_filter_too_slow` | Half-life=100 days -> no signal |
| `test_ou_z_score_buy` | z < -1.5 produces BUY signal |
| `test_ou_z_score_sell` | z > 1.5 produces SELL signal |
| `test_ou_exit_at_mean` | z returns to ~0 produces exit signal |
| `test_ou_no_repeated_signals` | Same direction suppressed for same symbol |
| `test_ou_backtest_vs_bollinger` | OU strategy Sharpe >= BB strategy Sharpe on test data |

#### Acceptance Criteria

- [ ] MLE fitting produces valid parameters on mean-reverting series
- [ ] Half-life filter excludes noise (<5d) and non-reverting (>60d) assets
- [ ] OU z-score signals have correct direction (buy below mean, sell above)
- [ ] Exit signals fire when price returns to equilibrium
- [ ] Backtest Sharpe for OU strategy >= Bollinger Band strategy Sharpe
- [ ] Strategy integrates with existing combiner framework

---

### B.2 Dual Momentum Strategy

**Priority:** P1
**Expected Sharpe gain:** +0.15 to +0.25
**Dependencies:** None
**Estimated effort:** 5-7 days

#### Files to Create

1. **`src/finalayze/strategies/dual_momentum.py`** (new file)

Core logic:
```python
class DualMomentumStrategy(BaseStrategy):
    """Dual momentum: absolute + relative momentum.

    Absolute momentum: Asset 12-month return > 0 (time-series momentum)
    Relative momentum: Rank assets by 12-month return, buy top N

    Combined: Only buy top-ranked assets that also have positive absolute momentum.
    When no assets pass, go to cash (emit no signals).

    Weighted momentum variant: 40% * 1m + 30% * 3m + 30% * 6m returns.
    """

    def _compute_weighted_momentum(self, candles: list[Candle]) -> float | None:
        """Compute weighted momentum score.

        Returns:
            Weighted return score, or None if insufficient data.
        """
        if len(candles) < 252:  # ~12 months
            return None

        current = float(candles[-1].close)
        ret_1m = (current / float(candles[-21].close) - 1) if len(candles) >= 21 else 0
        ret_3m = (current / float(candles[-63].close) - 1) if len(candles) >= 63 else 0
        ret_6m = (current / float(candles[-126].close) - 1) if len(candles) >= 126 else 0

        return 0.4 * ret_1m + 0.3 * ret_3m + 0.3 * ret_6m

    def generate_signal(self, symbol, candles, segment_id, **kwargs):
        score = self._compute_weighted_momentum(candles)
        if score is None:
            return None

        # Absolute momentum gate: only trade when score > 0
        if score <= 0:
            return None  # Negative absolute momentum -> cash

        # Relative momentum produces a BUY with confidence proportional to score
        confidence = min(0.95, 0.5 + abs(score) * 2)
        if confidence < 0.40:
            return None

        return Signal(
            strategy_name=self.name,
            symbol=symbol,
            market_id=candles[0].market_id,
            segment_id=segment_id,
            direction=SignalDirection.BUY,
            confidence=confidence,
            features={"momentum_score": round(score, 4)},
            reasoning=f"Dual momentum: score={score:.3f}",
        )
```

The cross-sectional ranking (relative momentum) is applied at the portfolio level in
the engine's `run_portfolio()` method: rank all symbols by their momentum score and
only allow BUY signals for the top N (configurable, default top 10).

#### Files to Modify

2. **`src/finalayze/strategies/presets/*.yaml`** -- add `dual_momentum` config
3. **`src/finalayze/strategies/__init__.py`** -- add export

#### Tests to Write

| Test Case | Description |
|-----------|-------------|
| `test_weighted_momentum_bull` | Rising prices -> positive score |
| `test_weighted_momentum_bear` | Falling prices -> negative score -> no signal |
| `test_absolute_momentum_gate` | Negative absolute momentum -> no BUY |
| `test_dual_momentum_signal_confidence` | Higher score -> higher confidence |
| `test_dual_momentum_insufficient_data` | <252 candles -> None |

#### Acceptance Criteria

- [ ] Weighted momentum uses 40/30/30 weighting across 1m/3m/6m
- [ ] Absolute momentum gate prevents buying assets in downtrends
- [ ] Strategy integrates with combiner framework
- [ ] Backtest shows positive Sharpe with reasonable turnover (< monthly rebalance)

---

### B.3 ML Pipeline: Triple Barrier Labeling

**Priority:** P1
**Expected Sharpe gain:** +0.10 to +0.20
**Dependencies:** None
**Estimated effort:** 5-7 days

#### Rationale

The current labeling in `build_windows()` (`ml/training/__init__.py`, line 107-109)
is a simple binary label: `1 if next_close > cur_close else 0`. This is noisy and
does not reflect actual trading outcomes. Triple barrier labeling assigns labels
based on which barrier (profit target, stop loss, or time limit) is hit first.

#### Files to Create

1. **`src/finalayze/ml/training/labeling.py`** (new file)

```python
"""Triple barrier labeling for ML training (Layer 3).

Implements Lopez de Prado's triple barrier method:
- Upper barrier: profit take (e.g., 2x ATR)
- Lower barrier: stop loss (e.g., 1x ATR)
- Vertical barrier: time expiration (e.g., 20 bars)

Label = which barrier is hit first:
  +1 = upper barrier (profitable trade)
  -1 = lower barrier (losing trade)
   0 = vertical barrier (time exit, effectively neutral)
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from finalayze.core.schemas import Candle


@dataclass(frozen=True, slots=True)
class TripleBarrierLabel:
    label: int        # +1, 0, -1
    barrier_type: str # "upper", "lower", "vertical"
    bars_held: int    # Number of bars until barrier hit
    pnl_pct: float    # Realized PnL percentage


def compute_atr(candles: list[Candle], period: int = 14) -> float | None:
    """Compute ATR for barrier sizing."""
    if len(candles) < period + 1:
        return None
    true_ranges = []
    for i in range(1, min(period + 1, len(candles))):
        prev_close = float(candles[-(i + 1)].close)
        high = float(candles[-i].high)
        low = float(candles[-i].low)
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        true_ranges.append(tr)
    return sum(true_ranges) / len(true_ranges) if true_ranges else None


def triple_barrier_label(
    candles: list[Candle],
    entry_idx: int,
    upper_atr_mult: float = 2.0,
    lower_atr_mult: float = 1.0,
    max_bars: int = 20,
    atr_period: int = 14,
) -> TripleBarrierLabel | None:
    """Assign a triple barrier label starting from entry_idx.

    Args:
        candles: Full candle series.
        entry_idx: Index of the entry bar.
        upper_atr_mult: Profit target as multiple of ATR.
        lower_atr_mult: Stop loss as multiple of ATR.
        max_bars: Maximum holding period (vertical barrier).
        atr_period: Period for ATR computation.

    Returns:
        TripleBarrierLabel or None if insufficient data.
    """
    if entry_idx < atr_period or entry_idx >= len(candles) - 1:
        return None

    entry_price = float(candles[entry_idx].close)
    atr = compute_atr(candles[:entry_idx + 1], atr_period)
    if atr is None or atr <= 0 or entry_price <= 0:
        return None

    upper_barrier = entry_price + upper_atr_mult * atr
    lower_barrier = entry_price - lower_atr_mult * atr

    for offset in range(1, max_bars + 1):
        bar_idx = entry_idx + offset
        if bar_idx >= len(candles):
            break

        high = float(candles[bar_idx].high)
        low = float(candles[bar_idx].low)
        close = float(candles[bar_idx].close)

        # Check upper barrier (profit target)
        if high >= upper_barrier:
            pnl_pct = (upper_barrier - entry_price) / entry_price
            return TripleBarrierLabel(
                label=1, barrier_type="upper", bars_held=offset, pnl_pct=pnl_pct
            )

        # Check lower barrier (stop loss)
        if low <= lower_barrier:
            pnl_pct = (lower_barrier - entry_price) / entry_price
            return TripleBarrierLabel(
                label=-1, barrier_type="lower", bars_held=offset, pnl_pct=pnl_pct
            )

    # Vertical barrier: time exit
    final_idx = min(entry_idx + max_bars, len(candles) - 1)
    final_price = float(candles[final_idx].close)
    pnl_pct = (final_price - entry_price) / entry_price
    label = 1 if pnl_pct > 0 else (-1 if pnl_pct < 0 else 0)
    return TripleBarrierLabel(
        label=label, barrier_type="vertical", bars_held=final_idx - entry_idx, pnl_pct=pnl_pct
    )
```

#### Files to Modify

2. **`src/finalayze/ml/training/__init__.py`** -- add `build_windows_triple_barrier()`
   function that uses triple barrier labeling instead of simple binary labels.

**Add after `build_windows()` (~line 114):**

```python
def build_windows_triple_barrier(
    candles: list[Candle],
    window_size: int = DEFAULT_WINDOW_SIZE,
    upper_atr_mult: float = 2.0,
    lower_atr_mult: float = 1.0,
    max_bars: int = 20,
) -> tuple[list[dict[str, float]], list[int], list[datetime]]:
    """Build windows with triple barrier labels instead of simple binary."""
    from finalayze.ml.training.labeling import triple_barrier_label

    features_list, label_list, ts_list = [], [], []
    sorted_candles = sorted(candles, key=lambda c: c.timestamp)

    for i in range(len(sorted_candles) - window_size - max_bars):
        window = sorted_candles[i : i + window_size]
        try:
            row_features = compute_features(window)
        except Exception:
            continue

        entry_idx = i + window_size - 1
        result = triple_barrier_label(
            sorted_candles, entry_idx,
            upper_atr_mult=upper_atr_mult,
            lower_atr_mult=lower_atr_mult,
            max_bars=max_bars,
        )
        if result is None:
            continue

        # Convert to binary: +1 -> 1, -1 or 0 -> 0
        binary_label = 1 if result.label == 1 else 0
        features_list.append(row_features)
        label_list.append(binary_label)
        ts_list.append(sorted_candles[entry_idx].timestamp)

    return features_list, label_list, ts_list
```

3. **`scripts/train_models.py`** -- add `--labeling=triple_barrier` CLI option

#### Tests to Write

| Test Case | Description |
|-----------|-------------|
| `test_triple_barrier_upper_hit` | Price hits profit target -> label=+1 |
| `test_triple_barrier_lower_hit` | Price hits stop loss -> label=-1 |
| `test_triple_barrier_vertical` | Neither hit within max_bars -> label from close |
| `test_triple_barrier_atr_scaling` | Barrier widths scale with ATR |
| `test_triple_barrier_insufficient_data` | Returns None at edge of candle array |
| `test_build_windows_triple_barrier` | Integration: produces correct feature/label pairs |
| `test_triple_barrier_pnl_pct` | PnL percentage is correctly computed |

#### Acceptance Criteria

- [ ] Triple barrier labeling correctly identifies which barrier was hit first
- [ ] Labels are more balanced than simple binary (fewer extreme imbalance)
- [ ] ML model trained on triple barrier labels has lower Brier score
- [ ] `build_windows_triple_barrier()` produces same feature format as `build_windows()`
- [ ] Train script supports `--labeling` flag

---

### B.4 CPCV (Combinatorial Purged Cross-Validation)

**Priority:** P1
**Expected Sharpe gain:** Included in B.3 estimate
**Dependencies:** B.3 (triple barrier labeling)
**Estimated effort:** 5-7 days

#### Files to Create

1. **`src/finalayze/ml/training/cpcv.py`** (new file)

```python
"""Combinatorial Purged Cross-Validation (Layer 3).

Implements Lopez de Prado's CPCV method for time-series ML validation.
Generates multiple chronology-respecting train/test splits with purging
and embargo to prevent data leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import combinations


@dataclass(frozen=True, slots=True)
class CPCVSplit:
    """A single train/test split with indices."""
    train_indices: list[int]
    test_indices: list[int]
    fold_id: str  # e.g., "fold_0_1" for groups 0 and 1 in test


def generate_cpcv_splits(
    n_samples: int,
    n_groups: int = 5,
    n_test_groups: int = 2,
    purge_window: int = 5,
    embargo_window: int = 10,
    timestamps: list[datetime] | None = None,
) -> list[CPCVSplit]:
    """Generate CPCV splits.

    Divides data into n_groups contiguous blocks. For each combination of
    n_test_groups blocks used as test set, the remaining blocks form the
    training set with purging and embargo applied at boundaries.

    Args:
        n_samples: Total number of samples.
        n_groups: Number of contiguous groups to split data into.
        n_test_groups: Number of groups used for test in each split.
        purge_window: Number of samples to remove around train/test boundary.
        embargo_window: Number of samples to skip after test set.
        timestamps: Optional timestamps for time-aware purging.

    Returns:
        List of CPCVSplit objects.
    """
    group_size = n_samples // n_groups
    groups: list[list[int]] = []
    for g in range(n_groups):
        start = g * group_size
        end = start + group_size if g < n_groups - 1 else n_samples
        groups.append(list(range(start, end)))

    splits: list[CPCVSplit] = []
    for test_combo in combinations(range(n_groups), n_test_groups):
        test_indices: list[int] = []
        for g in test_combo:
            test_indices.extend(groups[g])

        test_set = set(test_indices)

        # Purge: remove samples near train/test boundary
        purge_set: set[int] = set()
        for g in test_combo:
            group_start = groups[g][0]
            group_end = groups[g][-1]
            for offset in range(1, purge_window + 1):
                purge_set.add(group_start - offset)
                purge_set.add(group_end + offset)

        # Embargo: skip samples after each test group
        embargo_set: set[int] = set()
        for g in test_combo:
            group_end = groups[g][-1]
            for offset in range(1, embargo_window + 1):
                embargo_set.add(group_end + offset)

        excluded = test_set | purge_set | embargo_set
        train_indices = [i for i in range(n_samples) if i not in excluded]

        fold_id = f"fold_{'_'.join(str(g) for g in test_combo)}"
        splits.append(CPCVSplit(
            train_indices=train_indices,
            test_indices=sorted(test_indices),
            fold_id=fold_id,
        ))

    return splits
```

#### Files to Modify

2. **`src/finalayze/ml/training/splitter.py`** -- add `cpcv_train_test_split()` function
   that wraps `generate_cpcv_splits()` for the `LabelledRow` interface.

3. **`scripts/train_models.py`** -- add `--validation=cpcv` option (default: keep
   temporal split, CPCV as secondary validation).

#### Tests to Write

| Test Case | Description |
|-----------|-------------|
| `test_cpcv_split_count` | C(5,2)=10 splits generated |
| `test_cpcv_no_overlap` | Train and test indices never overlap |
| `test_cpcv_purge_removes_boundary` | Boundary samples excluded from train |
| `test_cpcv_embargo_after_test` | Samples after test group excluded from train |
| `test_cpcv_all_samples_covered` | Every sample appears in test at least once |
| `test_cpcv_temporal_order` | Test groups are contiguous blocks |

#### Acceptance Criteria

- [ ] CPCV generates C(n_groups, n_test_groups) splits
- [ ] No data leakage between train and test (purge + embargo)
- [ ] CPCV Sharpe distribution is more conservative than single split
- [ ] Models that pass CPCV (Sharpe > 0.5 in >50% of folds) are marked as validated
- [ ] CPCV is secondary validation; temporal split remains primary

---

### B.5 ML Pipeline: Model Calibration

**Priority:** P2
**Expected Sharpe gain:** +0.05 to +0.10
**Dependencies:** B.3 (triple barrier improves calibration)
**Estimated effort:** 3-4 days

#### Files to Create

1. **`src/finalayze/ml/calibration.py`** (new file)

```python
"""Model probability calibration (Layer 3).

Implements Platt scaling (logistic regression on model outputs)
for XGBoost and LightGBM probability calibration.
"""

from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression


class PlattScaler:
    """Platt scaling: fit logistic regression on model outputs."""

    def __init__(self) -> None:
        self._lr: LogisticRegression | None = None

    def fit(self, raw_probas: list[float], labels: list[int]) -> None:
        """Fit Platt scaler on validation set outputs."""
        X = np.array(raw_probas).reshape(-1, 1)
        y = np.array(labels)
        self._lr = LogisticRegression()
        self._lr.fit(X, y)

    def calibrate(self, raw_proba: float) -> float:
        """Return calibrated probability."""
        if self._lr is None:
            return raw_proba
        return float(self._lr.predict_proba([[raw_proba]])[0, 1])
```

#### Files to Modify

2. **`src/finalayze/ml/models/ensemble.py`** -- add optional `PlattScaler` to
   `predict_proba()` that calibrates the averaged probability.

3. **`src/finalayze/ml/training/__init__.py`** -- fit PlattScaler on validation set
   after ensemble training.

#### Acceptance Criteria

- [ ] Calibration reduces Brier score by >0.02 on validation set
- [ ] Calibrated 70% predictions actually win ~70% of the time
- [ ] PlattScaler is trained on validation set (not training set)

---

## Phase C: Portfolio & Infrastructure (Weeks 11-18)

Expected aggregate Sharpe improvement: +0.15 to +0.30 additional

---

### C.1 Correlation-Aware Position Sizing

**Priority:** P1
**Expected Sharpe gain:** +0.05 to +0.10
**Dependencies:** A.3 (position sizing framework)
**Estimated effort:** 5-7 days

#### Files to Create

1. **`src/finalayze/risk/correlation.py`** (new file)

```python
"""Correlation-aware portfolio risk management (Layer 4)."""

from __future__ import annotations

import math
from decimal import Decimal
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from finalayze.core.schemas import Candle


def compute_correlation_matrix(
    candles_by_symbol: dict[str, list[Candle]],
    lookback: int = 60,
) -> tuple[np.ndarray, list[str]]:
    """Compute rolling pairwise correlation matrix from candle data.

    Args:
        candles_by_symbol: Candles keyed by symbol.
        lookback: Number of days for rolling correlation.

    Returns:
        Tuple of (correlation_matrix, symbol_list).
    """
    symbols = sorted(candles_by_symbol.keys())
    returns_matrix = []
    valid_symbols = []

    for sym in symbols:
        candles = candles_by_symbol[sym]
        if len(candles) < lookback + 1:
            continue
        closes = [float(c.close) for c in candles[-(lookback + 1):]]
        rets = [
            math.log(closes[i] / closes[i - 1])
            for i in range(1, len(closes))
            if closes[i - 1] > 0
        ]
        if len(rets) >= lookback:
            returns_matrix.append(rets[-lookback:])
            valid_symbols.append(sym)

    if len(returns_matrix) < 2:
        return np.array([]), valid_symbols

    corr = np.corrcoef(returns_matrix)
    return corr, valid_symbols


def compute_correlation_scale(
    avg_correlation: float,
    low_threshold: float = 0.3,
    high_threshold: float = 0.5,
    min_scale: float = 0.3,
) -> Decimal:
    """Compute position scale factor based on average pairwise correlation.

    When avg correlation > high_threshold, reduce position sizes.
    Scale factor = max(min_scale, 1 - (avg_corr - low_threshold))

    Args:
        avg_correlation: Average pairwise correlation of current portfolio.
        low_threshold: Below this, full sizing.
        high_threshold: Above this, maximum reduction.
        min_scale: Floor for the scale factor.

    Returns:
        Position scale factor as Decimal [min_scale, 1.0].
    """
    if avg_correlation <= low_threshold:
        return Decimal("1.0")
    scale = 1.0 - (avg_correlation - low_threshold)
    return Decimal(str(max(min_scale, min(1.0, scale))))
```

#### Files to Modify

2. **`src/finalayze/backtest/engine.py`** -- in `run_portfolio()`, compute correlation
   matrix periodically (every 20 bars) and apply correlation scaling to new entries.

#### Tests to Write

| Test Case | Description |
|-----------|-------------|
| `test_correlation_matrix_basic` | Known correlated series produce expected matrix |
| `test_correlation_scale_low` | avg_corr=0.1 -> scale=1.0 |
| `test_correlation_scale_high` | avg_corr=0.7 -> scale=0.3 |
| `test_correlation_scale_medium` | avg_corr=0.4 -> scale=0.9 |
| `test_portfolio_engine_applies_corr_scale` | Engine reduces positions when corr high |

#### Acceptance Criteria

- [ ] Correlation matrix computed correctly from log returns
- [ ] Position sizes reduce when average correlation > 0.5
- [ ] Scale never goes below 0.3 (min_scale)
- [ ] Portfolio drawdown is reduced during high-correlation regimes

---

### C.2 Dynamic Strategy Weighting

**Priority:** P1
**Expected Sharpe gain:** +0.08 to +0.15
**Dependencies:** None (enhances existing combiner)
**Estimated effort:** 5-7 days

#### Rationale

The current `StrategyCombiner` uses static weights from YAML presets. Dynamic weighting
allocates more capital to strategies that are currently performing well (positive Sharpe)
and pauses strategies with negative Sharpe.

#### Files to Create

1. **`src/finalayze/strategies/adaptive_combiner.py`** (new file)

```python
"""Adaptive strategy combiner with dynamic weighting (Layer 4).

Extends StrategyCombiner to adjust weights based on rolling strategy Sharpe ratios.
"""

from __future__ import annotations

from collections import defaultdict, deque
from decimal import Decimal
from typing import TYPE_CHECKING

from finalayze.core.schemas import Candle, Signal
from finalayze.strategies.combiner import StrategyCombiner

if TYPE_CHECKING:
    from finalayze.strategies.base import BaseStrategy

_SHARPE_LOOKBACK = 63  # ~3 months of daily bars
_REWEIGHT_INTERVAL = 21  # Monthly reweighting


class AdaptiveStrategyCombiner(StrategyCombiner):
    """Strategy combiner that adjusts weights by rolling Sharpe ratio.

    weight_i = max(0, Sharpe_i) / sum(max(0, Sharpe_j))

    Strategies with negative Sharpe get zero allocation (paused).
    Reweighting occurs every _REWEIGHT_INTERVAL bars.
    """

    def __init__(
        self,
        strategies: list[BaseStrategy],
        normalize_mode: str = "firing",
        sharpe_lookback: int = _SHARPE_LOOKBACK,
        reweight_interval: int = _REWEIGHT_INTERVAL,
    ) -> None:
        super().__init__(strategies, normalize_mode)
        self._sharpe_lookback = sharpe_lookback
        self._reweight_interval = reweight_interval
        self._strategy_returns: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=sharpe_lookback)
        )
        self._dynamic_weights: dict[str, Decimal] = {}
        self._bars_since_reweight = 0

    def record_strategy_return(self, strategy_name: str, daily_return: float) -> None:
        """Record a daily return for a strategy."""
        self._strategy_returns[strategy_name].append(daily_return)
        self._bars_since_reweight += 1

        if self._bars_since_reweight >= self._reweight_interval:
            self._recompute_weights()
            self._bars_since_reweight = 0

    def _recompute_weights(self) -> None:
        """Recompute dynamic weights based on rolling Sharpe ratios."""
        import statistics

        sharpe_values: dict[str, float] = {}
        for name, returns in self._strategy_returns.items():
            if len(returns) < 20:
                continue
            ret_list = list(returns)
            mean_ret = statistics.mean(ret_list)
            std_ret = statistics.stdev(ret_list) if len(ret_list) > 1 else 1.0
            sharpe = (mean_ret / std_ret * (252 ** 0.5)) if std_ret > 0 else 0.0
            sharpe_values[name] = sharpe

        # Weight = max(0, Sharpe_i) / sum(max(0, Sharpe_j))
        positive_sharpes = {k: max(0, v) for k, v in sharpe_values.items()}
        total = sum(positive_sharpes.values())

        if total > 0:
            self._dynamic_weights = {
                k: Decimal(str(v / total)) for k, v in positive_sharpes.items()
            }
        # else: keep previous weights
```

#### Files to Modify

2. **`src/finalayze/backtest/engine.py`** -- use `AdaptiveStrategyCombiner` when
   configured (add `adaptive_weights: bool = False` parameter).

3. **`src/finalayze/strategies/presets/*.yaml`** -- add `adaptive_weights: true` option.

#### Tests to Write

| Test Case | Description |
|-----------|-------------|
| `test_adaptive_weights_positive_sharpe` | Strategy with Sharpe=1.0 gets higher weight |
| `test_adaptive_weights_negative_sharpe_paused` | Strategy with Sharpe=-0.5 gets weight=0 |
| `test_adaptive_reweight_interval` | Weights recomputed every 21 bars |
| `test_adaptive_insufficient_data` | <20 returns -> use static weights |
| `test_adaptive_all_negative` | All strategies negative -> keep previous weights |

#### Acceptance Criteria

- [ ] Dynamic weights reflect rolling 63-day Sharpe ratios
- [ ] Strategies with negative Sharpe receive zero allocation
- [ ] Reweighting occurs monthly, not daily (avoid overfitting)
- [ ] Backward compatible: static weights used when adaptive is disabled
- [ ] Backtest shows improved Sharpe vs static weighting

---

### C.3 Earnings Surprise Signal (Alternative Data)

**Priority:** P2
**Expected Sharpe gain:** +0.10 to +0.20
**Dependencies:** None
**Estimated effort:** 7-10 days

#### Files to Create

1. **`src/finalayze/data/earnings.py`** (new file)

```python
"""Earnings data fetcher and PEAD signal generator (Layer 2).

Fetches earnings surprise data from Finnhub API.
Computes Standardized Unexpected Earnings (SUE) for PEAD signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True, slots=True)
class EarningsSurprise:
    symbol: str
    report_date: date
    actual_eps: float
    estimated_eps: float
    surprise_pct: float   # (actual - estimated) / abs(estimated) * 100
    sue_score: float      # Standardized Unexpected Earnings


async def fetch_earnings_surprises(
    symbols: list[str],
    api_key: str,
) -> list[EarningsSurprise]:
    """Fetch recent earnings surprises from Finnhub."""
    ...  # Implementation uses httpx to call Finnhub earnings calendar


def compute_sue(
    surprises: list[EarningsSurprise],
    lookback_quarters: int = 4,
) -> float:
    """Compute Standardized Unexpected Earnings.

    SUE = (EPS_actual - EPS_expected) / std(surprise over last N quarters)
    """
    ...
```

2. **`src/finalayze/strategies/pead_strategy.py`** (new file)

Core logic: Buy stocks in top SUE quintile on day +1 after earnings, hold 63 days.

#### Files to Modify

3. **`config/settings.py`** -- add `pead_hold_days: int = 63`
4. **`src/finalayze/strategies/presets/*.yaml`** -- add pead section

#### Tests to Write

| Test Case | Description |
|-----------|-------------|
| `test_sue_computation` | Known surprise data produces correct SUE |
| `test_pead_buy_signal_on_positive_surprise` | Top SUE quintile -> BUY |
| `test_pead_no_signal_on_negative_surprise` | Bottom SUE quintile -> no signal |
| `test_pead_hold_period_63_days` | Position exits after 63 bars |

#### Acceptance Criteria

- [ ] SUE correctly standardizes earnings surprises
- [ ] Only top quintile SUE scores generate BUY signals
- [ ] Hold period is fixed at 63 calendar days
- [ ] Strategy works with Finnhub API (or mock data in backtest)

---

### C.4 Short Interest Data Integration

**Priority:** P3
**Expected Sharpe gain:** +0.05 to +0.10
**Dependencies:** None
**Estimated effort:** 3-5 days

#### Files to Create

1. **`src/finalayze/data/short_interest.py`** (new file)

Fetch FINRA short interest reports, compute short interest ratio (SI/float),
SI momentum (change in SI over 3 months).

#### Files to Modify

2. **`src/finalayze/ml/features/technical.py`** -- add `short_interest_ratio` and
   `short_interest_momentum` as ML features.

3. **`src/finalayze/strategies/presets/*.yaml`** -- add short interest filter:
   - Avoid going long when SI > 20% (unless specific contrarian setup)
   - SI momentum increasing over 3 months = bearish signal

#### Acceptance Criteria

- [ ] Short interest ratio is available as ML feature
- [ ] High SI (>20%) triggers caution flag in pre-trade checks
- [ ] SI momentum is computed correctly from bimonthly FINRA data

---

## Cross-Cutting Concerns

### Configuration Integration

All new features must be configurable via:
1. `config/settings.py` -- global defaults
2. `src/finalayze/strategies/presets/*.yaml` -- per-segment overrides
3. Constructor parameters -- for testing

### Testing Standards

- TDD mandatory: RED test first, then GREEN implementation
- Coverage: >= 90% for new modules, >= 80% for modified modules
- Integration tests: backtest with each new feature shows measurable Sharpe improvement
- Regression tests: existing test suite must continue to pass

### Dependency Layering

All new files must respect the layering rules in `docs/architecture/DEPENDENCY_LAYERS.md`:
```
Layer 0: Types & Schemas
Layer 1: Configuration
Layer 2: Data / Repository      (earnings.py, short_interest.py)
Layer 3: Analysis / ML          (labeling.py, cpcv.py, calibration.py)
Layer 4: Strategy / Risk        (regime.py, hmm_regime.py, chandelier_exit.py,
                                  ou_mean_reversion.py, dual_momentum.py,
                                  correlation.py, adaptive_combiner.py)
```

### New Dependencies

| Package | Phase | Purpose |
|---------|-------|---------|
| `hmmlearn>=0.3` | A.6 | HMM regime detection |
| (none else) | -- | All other features use existing deps (numpy, pandas, sklearn, pandas_ta) |

---

## Summary: Expected Outcomes

| Phase | Timeline | Expected Sharpe Gain | Key Deliverables |
|-------|----------|---------------------|-----------------|
| A | Weeks 1-4 | +0.30 to +0.60 | VIX regime filter, SMA200 trend filter, vol-targeting, Chandelier exit, strategy-specific time exits, HMM regime |
| B | Weeks 5-10 | +0.20 to +0.40 | OU mean reversion, dual momentum, triple barrier labeling, CPCV, model calibration |
| C | Weeks 11-18 | +0.15 to +0.30 | Correlation-aware sizing, dynamic strategy weighting, PEAD strategy, short interest data |

**Total projected Sharpe: 0.07 (current) + 0.65 to 1.30 = 0.72 to 1.37**

### Task Dependency Graph

```
A.1 VIX Regime ──────┬──> A.6 HMM Regime
                      │
A.2 SMA200 Filter     │
                      │
A.3 Vol-Targeting ────┼──> C.1 Correlation Sizing
                      │
A.4 Chandelier Exit   │
                      │
A.5 Strategy Time Exits
                      │
B.1 OU Mean Reversion │
                      │
B.2 Dual Momentum     │
                      │
B.3 Triple Barrier ───┼──> B.4 CPCV
                      │         │
                      │         v
                      │    B.5 Model Calibration
                      │
C.2 Dynamic Weighting │
                      │
C.3 PEAD Strategy     │
                      │
C.4 Short Interest ───┘
```

Tasks without arrows are independent and can be developed in parallel.

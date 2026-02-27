# Phase 4 Track A — Strategy & Backtest Hardening Design

**Version:** 1.0 | **Date:** 2026-02-27 | **Status:** Draft

---

## Motivation

Backtest validation of the MomentumStrategy (regime lookback fix) across 10 US
large-caps (2022-2025) was reviewed by three independent expert perspectives:
a quantitative systematic trader, a discretionary swing trader, and a chief risk
officer. All three converged on the same structural problems that must be fixed
before any live capital deployment.

### Expert Consensus (unanimous)

1. The strategy is accidentally **counter-trend** — buying oversold dips is
   mean-reversion, not momentum. A trend filter is the #1 priority.
2. **Confidence-weighted sizing is premature/harmful** — the confidence formula
   is uncalibrated and doesn't predict outcomes.
3. **39 trades on 10 hand-picked FAANG stocks** proves nothing — survivorship
   bias, no out-of-sample validation, statistically insignificant sample.
4. **Stops must be trailing, not just wider** — fixed stops systematically kill
   winners during normal retracements (NVDA: bought $18.93, stopped $16.39,
   stock went to $130+).
5. The strategy **has potential** — winners (TSLA +62%, AMZN +31%) show it can
   catch real momentum moves. The problem is signal quality and exit logic.

### What's Already Designed But Not Implemented

| Item | Where Designed | Gap |
|------|---------------|-----|
| Trailing stop (+1 ATR profit) | RISK.md | SimulatedBroker only has fixed stops |
| Max correlated positions (r > 0.7) | RISK.md | Not implemented anywhere |
| Walk-forward validation | ML_PIPELINE.md, Roadmap | No framework exists |
| Confidence calibration (Platt scaling) | ML_PIPELINE.md | Only for ML models, not strategies |
| Circuit breakers in backtest | RISK.md | BacktestEngine ignores them |

---

## Goal

Make the strategy + backtest pipeline **statistically rigorous and production-
ready** by fixing signal quality, exit logic, risk calibration, and validation
methodology. This track runs in parallel with Track B (Observability).

---

## Architecture Changes

```
                        BEFORE                              AFTER
                    ┌──────────┐                      ┌──────────────┐
  Candles ─────────▶│ RSI+MACD │──▶ Signal            │  Trend Gate  │
                    │ (regime  │                       │  (SMA filter)│
                    │ lookback)│               ┌──────▶│              │──┐
                    └──────────┘               │       └──────────────┘  │
                                               │       ┌──────────────┐  │
                                    Candles ───┤──────▶│ Regime Gate  │  ├──▶ Signal
                                               │       │ (ADX filter) │  │    (or None)
                                               │       └──────────────┘  │
                                               │       ┌──────────────┐  │
                                               └──────▶│ RSI+MACD     │──┘
                                                       │ + state      │
                                                       │   machine    │
                                                       └──────────────┘
```

Signal generation becomes a **three-gate pipeline**: trend → regime → indicator.
Each gate can suppress signals independently. The state machine prevents
duplicate same-direction signals.

---

## PR Split

| PR | Contents | Dependency |
|----|----------|------------|
| **A-1** | Backtest infrastructure — trailing stops, transaction costs, benchmark, circuit breakers, batch runner | none |
| **A-2** | Strategy signal quality — trend filter, signal state machine, ADX filter, volume filter, YAML updates | none |
| **A-3** | Risk calibration — rolling Kelly, daily/weekly loss limits, confidence calibration | A-1 merged |
| **A-4** | Statistical validation — walk-forward framework, symbol universe, Monte Carlo, out-of-sample pipeline | A-1 merged |

A-1 and A-2 can be developed in parallel. A-3 and A-4 depend on A-1 (they need
the improved backtest engine to validate against).

---

## PR A-1: Backtest Infrastructure

### 1.1 Trailing Stops in SimulatedBroker

**Problem:** RISK.md specifies "trailing stop activates at +1 ATR profit" but
`SimulatedBroker` only supports fixed stop-loss. Winners get stopped out on
normal retracements (AAPL: 4/4 trades lost at -3% to -7%).

**Design:**

```python
# src/finalayze/execution/simulated_broker.py

@dataclass
class StopLossState:
    initial_stop: Decimal       # entry - N * ATR
    current_stop: Decimal       # may trail upward
    highest_price: Decimal      # high-water mark since entry
    trail_activated: bool       # True once price reaches entry + activation_atr * ATR
    activation_atr: Decimal     # ATR multiplier to activate trailing (default 1.0)
    trail_atr: Decimal          # ATR multiplier for trailing distance (default 1.5)
```

Trailing logic per bar:
1. Update `highest_price = max(highest_price, candle.high)`
2. If not yet activated: activate when `highest_price >= entry + activation_atr * ATR`
3. Once activated: `trail_stop = highest_price - trail_atr * ATR`
4. `current_stop = max(current_stop, trail_stop)` — stop only moves up
5. Trigger if `candle.low <= current_stop`

**YAML params** (added to `momentum.params` in all presets):

```yaml
stop_atr_multiplier: 2.5      # initial stop distance (was 2.0)
trail_activation_atr: 1.0     # activate trailing after +1 ATR profit
trail_distance_atr: 1.5       # trailing distance once active
```

### 1.2 Transaction Cost Model

**Problem:** No commissions, spread, or slippage in backtest. A ~5% gross return
can disappear with even modest friction.

**Design:**

```python
# src/finalayze/backtest/costs.py

@dataclass(frozen=True)
class TransactionCosts:
    commission_per_share: Decimal = Decimal("0.005")  # $0.005/share (Alpaca-like)
    min_commission: Decimal = Decimal("1.00")          # $1 minimum per order
    spread_bps: Decimal = Decimal("5")                 # 5 bps half-spread
    slippage_bps: Decimal = Decimal("3")               # 3 bps market impact

    def total_cost(self, price: Decimal, quantity: Decimal) -> Decimal:
        commission = max(self.min_commission, self.commission_per_share * quantity)
        spread = price * self.spread_bps / Decimal(10000)
        slippage = price * self.slippage_bps / Decimal(10000)
        return commission + (spread + slippage) * quantity
```

Applied in `BacktestEngine`: deduct cost from PnL on both entry and exit fills.

### 1.3 Benchmark Comparison

**Problem:** No alpha measurement. SPY returned ~30% over 2022-2025; best
single-name result is +6.44%.

**Design:** `PerformanceAnalyzer.analyze()` accepts an optional `benchmark_candles`
parameter. When provided, computes:

- **Alpha** = strategy return - benchmark return
- **Beta** = covariance(strategy, benchmark) / variance(benchmark)
- **Information ratio** = alpha / tracking error
- **Max relative drawdown** = worst underperformance vs benchmark

Output added to `BacktestResult` schema as optional fields.

### 1.4 Circuit Breakers in Backtest

**Problem:** Live system has 3-level circuit breakers (Phase 3), but
`BacktestEngine` ignores them. Backtest results are unrealistically optimistic
for drawdown scenarios.

**Design:** Inject `CircuitBreaker` into `BacktestEngine.__init__()`. Before
generating signals, check circuit breaker level. At L2+ suppress new entries.
At L3 trigger liquidation. Reset logic follows same rules as live (2 profitable
days for L2 auto-reset).

### 1.5 Multi-Symbol Batch Backtest

**Problem:** Current `run_backtest.py` tests one symbol at a time. Need 50+
symbols for statistical significance.

**Design:** New script `scripts/run_batch_backtest.py`:

```
uv run python scripts/run_batch_backtest.py \
    --universe sp500_sample \
    --segment us_tech \
    --start 2018-01-01 --end 2025-01-01 \
    --output results/batch_2025.csv
```

**Symbol universes** (JSON files in `config/universes/`):

| Universe | Size | Description |
|----------|------|-------------|
| `sp500_sample.json` | 50 | Stratified random sample from S&P 500 (10 per sector) |
| `us_mega.json` | 10 | Current FAANG test set (for regression) |
| `us_losers.json` | 20 | Stocks that underperformed (INTC, PFE, DIS, BA, etc.) |
| `moex_blue_chips.json` | 8 | Existing MOEX instruments |

Output: CSV with per-symbol metrics + aggregate portfolio stats + benchmark
comparison. Summary printed to stdout.

### 1.6 Acceptance Criteria (PR A-1)

- [ ] Trailing stops: unit test showing trailing activates and ratchets up
- [ ] Trailing stops: NVDA-like scenario — entry at $18.93 stays open past $16.39
- [ ] Transaction costs: backtest returns decrease vs zero-cost baseline
- [ ] Benchmark: alpha/beta/IR computed and printed in backtest report
- [ ] Circuit breaker: backtest halts trading at L2, liquidates at L3
- [ ] Batch runner: produces CSV output for 50+ symbols
- [ ] All existing tests pass, ruff clean, mypy clean

---

## PR A-2: Strategy Signal Quality

### 2.1 Trend Filter (SMA Gate)

**Problem:** Strategy is counter-trend by construction. JPM: 2 BUY vs 40 SELL
in a 3-year uptrend. NVDA: 8 BUY vs 61 SELL.

**Design:**

New YAML params per segment:

```yaml
momentum:
  params:
    trend_filter: true           # enable/disable
    trend_sma_period: 50         # SMA period (50 or 200, configurable)
    trend_sma_buffer_pct: 2.0    # suppress signals within 2% of SMA (avoid whipsaw)
```

Logic in `_evaluate_signal()`:

```python
sma = closes.rolling(trend_sma_period).mean().iloc[-1]
buffer = sma * trend_sma_buffer_pct / 100

if current_close > sma + buffer:
    # Uptrend: suppress SELL signals
    if direction == SignalDirection.SELL:
        return None
elif current_close < sma - buffer:
    # Downtrend: suppress BUY signals
    if direction == SignalDirection.BUY:
        return None
# Within buffer zone: allow both directions
```

**Why 50-day SMA, not 200-day:** The swing trader expert noted 200-day is too
slow for regime changes — misses the first 3-6 months of a new trend. 50-day
is more responsive. Made configurable so segments can choose: `us_broad` might
use 200, `us_tech` might use 50.

### 2.2 Signal State Machine

**Problem:** TSLA generates 100 raw signals in 3 years — clusters of 5-12
consecutive same-direction signals. Only the first matters.

**Design:**

Replace arbitrary cooldown with a **state machine** that tracks signal
transitions:

```python
class _SignalState:
    """Tracks signal state per symbol to prevent duplicate signals."""

    def __init__(self) -> None:
        self._last_direction: dict[str, SignalDirection] = {}

    def should_emit(self, symbol: str, direction: SignalDirection) -> bool:
        last = self._last_direction.get(symbol)
        if last == direction:
            return False  # duplicate suppressed
        self._last_direction[symbol] = direction
        return True
```

Rules:
- After BUY fires, suppress all subsequent BUYs until a SELL fires (or vice versa)
- Neutral reset: if neither BUY nor SELL condition is met for N bars
  (`neutral_reset_bars: 20`), reset state to allow re-entry
- State is per-symbol, per-strategy instance

**Effect:** TSLA goes from 100 signals → ~12-15 alternating BUY/SELL signals.
Each signal is a genuine regime transition.

### 2.3 ADX Regime Filter

**Problem:** Same parameters for TSLA (high volatility, strong trends) and AAPL
(low volatility, slow grind). RSI + MACD whipsaw in range-bound markets.

**Design:**

New YAML params:

```yaml
momentum:
  params:
    adx_filter: true
    adx_period: 14
    adx_threshold: 25           # only trade when ADX > 25 (directional market)
```

Logic: compute ADX via `pandas_ta.adx()`. If `ADX < adx_threshold`, return
`None` — the market is range-bound and momentum signals are unreliable.

**Expected impact:** Filters out the AAPL and SPY scenarios where the strategy
repeatedly enters/exits for small losses in a choppy, directionless market.

### 2.4 Volume Confirmation

**Problem:** Momentum without volume is a trap. A MACD crossover on declining
volume is a false signal.

**Design:**

New YAML params:

```yaml
momentum:
  params:
    volume_filter: true
    volume_sma_period: 20
    volume_min_ratio: 1.0       # require current volume >= 1x average
```

Logic: compute `volume_sma = volumes.rolling(20).mean()`. If current bar's
volume is below `volume_min_ratio * volume_sma`, suppress the signal.

### 2.5 Update All 8 YAML Presets

Add new params to all presets with segment-appropriate defaults:

| Segment | trend_sma | adx_thresh | vol_ratio | Notes |
|---------|-----------|------------|-----------|-------|
| us_tech | 50 | 25 | 1.0 | Fast SMA for high-momentum names |
| us_broad | 200 | 20 | 0.8 | Slower SMA for ETFs, lower ADX bar |
| us_finance | 50 | 25 | 1.0 | Standard |
| us_healthcare | 50 | 25 | 1.0 | Standard |
| ru_tech | 50 | 25 | 1.0 | Standard |
| ru_finance | 50 | 20 | 0.8 | Lower ADX — MOEX is less directional |
| ru_energy | 50 | 20 | 0.8 | Commodity-driven, regime-dependent |
| ru_blue_chips | 50 | 20 | 0.8 | Mixed, lower bar |

### 2.6 Update STRATEGIES.md

Update the design doc to reflect:
- New signal pipeline (trend → regime → indicator → state machine)
- New parameters table
- Regime lookback logic (replaces same-bar coincidence description)

### 2.7 Acceptance Criteria (PR A-2)

- [ ] Trend filter: JPM generates <5 SELL signals (was 40) when price > 50 SMA
- [ ] State machine: TSLA raw signals drop from 100 to <20 alternating signals
- [ ] ADX filter: SPY flat-market bars suppressed (no signals when ADX < 25)
- [ ] Volume filter: signal suppressed when volume < 1x SMA
- [ ] Backtest AAPL: fewer trades, better win rate (no more 0/4)
- [ ] Backtest NVDA: no premature sells in the 2023-2024 uptrend
- [ ] All 8 YAML presets updated
- [ ] STRATEGIES.md updated
- [ ] All existing tests pass, 7+ new unit tests, ruff clean, mypy clean

---

## PR A-3: Risk Calibration

### 3.1 Rolling Kelly Estimation

**Problem:** `BacktestEngine` uses hardcoded `win_rate=0.5, avg_win_ratio=1.5`.
Actual win rate is ~38%. With real numbers, Kelly says bet ZERO — the assumed
edge doesn't exist.

**Design:**

```python
# src/finalayze/risk/kelly.py

class RollingKelly:
    """Estimate Kelly fraction from a rolling window of recent trades."""

    def __init__(self, window: int = 50, fraction: float = 0.25) -> None:
        self._window = window
        self._fraction = fraction  # quarter-Kelly by default
        self._trades: deque[TradeResult] = deque(maxlen=window)

    def update(self, trade: TradeResult) -> None:
        self._trades.append(trade)

    def optimal_fraction(self) -> Decimal:
        if len(self._trades) < 20:  # minimum sample
            return Decimal("0.01")  # 1% fixed fractional until enough data

        wins = [t for t in self._trades if t.pnl > 0]
        losses = [t for t in self._trades if t.pnl <= 0]
        if not wins or not losses:
            return Decimal("0.01")

        win_rate = len(wins) / len(self._trades)
        avg_win = sum(float(t.pnl_pct) for t in wins) / len(wins)
        avg_loss = abs(sum(float(t.pnl_pct) for t in losses) / len(losses))

        if avg_loss == 0:
            return Decimal("0.01")

        ratio = avg_win / avg_loss
        kelly = (win_rate * ratio - (1 - win_rate)) / ratio

        if kelly <= 0:
            return Decimal("0.005")  # minimum 0.5% even with no edge

        return Decimal(str(round(kelly * self._fraction, 4)))
```

**Integration:** `BacktestEngine` uses `RollingKelly` instead of the hardcoded
constants. The first 20 trades use fixed fractional (1% risk per trade), then
Kelly kicks in with quarter-Kelly dampening.

### 3.2 Daily / Weekly Loss Limits

**Problem:** No mechanism to halt trading after a string of stop-outs. Losses
can compound.

**Design:**

New config params:

```yaml
risk:
  daily_loss_limit_pct: 3.0     # halt if down 3% in a single day
  weekly_loss_limit_pct: 5.0    # halt if down 5% in a rolling week
  cooldown_after_halt_days: 1   # wait 1 day before resuming
```

Logic in `BacktestEngine`: track daily/weekly equity change. If limit breached,
suppress all new entries for `cooldown_after_halt_days`. Existing positions and
stop-losses remain active.

### 3.3 Confidence Calibration

**Problem:** The confidence formula `0.5 + rsi_distance * 0.3 + (hist/close * 100) * 0.1`
is arbitrary. Weights are not empirically derived.

**Design (Phase 1 — simple):** Replace the arbitrary formula with a single-
factor confidence: RSI distance from threshold, linearly scaled.

```python
# BUY confidence: how deeply oversold the RSI window was
confidence = min(1.0, 0.5 + (rsi_oversold - min(rsi_window)) / rsi_oversold * 0.5)

# SELL confidence: how deeply overbought the RSI window was
confidence = min(1.0, 0.5 + (max(rsi_window) - rsi_overbought) / (100 - rsi_overbought) * 0.5)
```

**Design (Phase 2 — after enough trades):** Use Platt scaling calibrated on
out-of-sample backtest results. Requires 200+ trades from PR A-4 validation.

### 3.4 Acceptance Criteria (PR A-3)

- [ ] Rolling Kelly: first 20 trades use 1% fixed fractional
- [ ] Rolling Kelly: after 50 trades, sizing adapts to measured win rate
- [ ] Kelly with 38% win rate sizes smaller than hardcoded 50% assumption
- [ ] Daily loss limit: trading halts after -3% intraday drawdown
- [ ] Weekly loss limit: trading halts after -5% weekly drawdown
- [ ] Simplified confidence: single-factor RSI-based (no histogram weight)
- [ ] All existing tests pass, ruff clean, mypy clean

---

## PR A-4: Statistical Validation

### 4.1 Walk-Forward Optimization Framework

**Problem:** All 39 trades are in-sample. Parameters were chosen with knowledge
of the data. Zero evidence the strategy generalizes.

**Design:**

```python
# src/finalayze/backtest/walk_forward.py

@dataclass
class WalkForwardConfig:
    train_years: int = 3         # train on 3 years
    test_years: int = 1          # test on 1 year
    step_months: int = 6         # slide window by 6 months
    param_grid: dict             # parameters to optimize over

class WalkForwardOptimizer:
    def run(self, symbol: str, segment: str, candles: list[Candle]) -> WalkForwardResult:
        """Split data into rolling train/test windows. For each window:
        1. Optimize parameters on train set
        2. Lock parameters
        3. Evaluate on test set (out-of-sample)
        4. Collect OOS trades
        Return aggregate OOS metrics.
        """
```

**Windows example (2018-2025):**

| Window | Train | Test (OOS) |
|--------|-------|------------|
| 1 | 2018-01 to 2020-12 | 2021-01 to 2021-12 |
| 2 | 2018-07 to 2021-06 | 2021-07 to 2022-06 |
| 3 | 2019-01 to 2021-12 | 2022-01 to 2022-12 |
| 4 | 2019-07 to 2022-06 | 2022-07 to 2023-06 |
| 5 | 2020-01 to 2022-12 | 2023-01 to 2023-12 |
| 6 | 2020-07 to 2023-06 | 2023-07 to 2024-06 |
| 7 | 2021-01 to 2023-12 | 2024-01 to 2024-12 |

Only OOS trades are counted in final metrics.

### 4.2 Survivorship-Bias-Free Symbol Universe

**Problem:** Testing only on FAANG winners inflates results.

**Design:** Create `config/universes/sp500_sample.json` with 50 symbols sampled
stratified by GICS sector as of 2018-01-01 (the earliest train window start).
Include companies that subsequently underperformed or were removed from the
index. Source: historical S&P 500 constituents.

| Sector | Count | Example tickers |
|--------|-------|-----------------|
| Technology | 8 | AAPL, MSFT, INTC, IBM, CRM, ADBE, CSCO, TXN |
| Healthcare | 6 | JNJ, PFE, UNH, ABBV, MRK, BMY |
| Financials | 6 | JPM, BAC, GS, C, WFC, BRK-B |
| Consumer Disc. | 5 | AMZN, TSLA, HD, NKE, DIS |
| Industrials | 5 | BA, CAT, UPS, HON, GE |
| Communication | 4 | META, GOOGL, VZ, T |
| Energy | 4 | XOM, CVX, COP, SLB |
| Consumer Staples | 4 | PG, KO, WMT, COST |
| Utilities | 3 | NEE, DUK, SO |
| Materials | 3 | LIN, APD, NEM |
| Real Estate | 2 | AMT, PLD |

### 4.3 Monte Carlo Bootstrap

**Problem:** With 39 trades, Sharpe 0.62 has 95% CI of [-0.26, 1.50]. Cannot
distinguish from zero.

**Design:**

```python
# src/finalayze/backtest/monte_carlo.py

def bootstrap_metrics(
    trades: list[TradeResult],
    n_simulations: int = 10_000,
    confidence_level: float = 0.95,
) -> BootstrapResult:
    """Resample trades with replacement, compute metrics for each sample.
    Return point estimates + confidence intervals for:
    - Total return
    - Sharpe ratio
    - Max drawdown
    - Win rate
    - Profit factor
    """
```

Output: printed as CI ranges in the backtest report. A strategy is considered
to have a real edge only if the lower bound of the Sharpe CI is > 0.

### 4.4 Out-of-Sample Validation Script

**Design:** New script `scripts/run_validation.py`:

```
uv run python scripts/run_validation.py \
    --universe sp500_sample \
    --segment us_tech \
    --train-start 2018-01-01 --train-end 2022-12-31 \
    --test-start 2023-01-01 --test-end 2025-01-01 \
    --bootstrap 10000 \
    --output results/validation_2025.csv
```

Output:
1. Walk-forward OOS metrics (Sharpe, return, drawdown, win rate)
2. Bootstrap 95% CIs for all metrics
3. Benchmark comparison (vs SPY buy-and-hold)
4. Per-sector breakdown
5. PASS/FAIL verdict: Sharpe CI lower bound > 0 AND positive alpha

### 4.5 Acceptance Criteria (PR A-4)

- [ ] Walk-forward: produces OOS metrics across 7 rolling windows
- [ ] Walk-forward: OOS Sharpe documented (may be negative — that's valid data)
- [ ] Symbol universe: 50 symbols across 11 sectors, including underperformers
- [ ] Monte Carlo: 10k simulations produce CI ranges for Sharpe/return/drawdown
- [ ] Validation script: single command produces full report
- [ ] Report clearly states PASS/FAIL based on statistical significance
- [ ] All existing tests pass, ruff clean, mypy clean

---

## Dependency Graph

```
         A-1 (Backtest Infra)          A-2 (Signal Quality)
              │    │                         │
              │    └──────────┬──────────────┘
              │               │
              ▼               ▼
         A-3 (Risk Cal.)   A-4 (Validation)
              │               │
              └───────┬───────┘
                      ▼
              Go/No-Go Decision
              (deploy to sandbox)
```

- A-1 and A-2 are independent, can be developed in parallel
- A-3 needs A-1 (rolling Kelly uses improved backtest engine)
- A-4 needs A-1 (walk-forward uses trailing stops + transaction costs)
- A-4 should run AFTER A-2 (validates the improved signal quality)
- Final Go/No-Go: strategy deployed to sandbox only if A-4 validation passes

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Filters reduce trade count below statistical significance | High | Medium | Lower thresholds; test on longer history (2015-2025) |
| Walk-forward shows no out-of-sample edge | Medium | High | This is a valid outcome — better to know before deploying capital |
| ADX/volume data quality varies by source | Low | Medium | Use yfinance (consistent); validate against Alpaca data |
| Parameter explosion (too many knobs) | Medium | Medium | All new params have defaults; segment-level overrides only |
| MOEX symbols lack sufficient history for walk-forward | Medium | Low | Use shorter train windows; validate separately |

---

## Success Criteria

The strategy is approved for sandbox paper trading when ALL of the following
are met:

1. **Walk-forward OOS Sharpe > 0.3** across 50-symbol universe
2. **Bootstrap Sharpe 95% CI lower bound > 0** (statistically significant edge)
3. **Positive alpha vs SPY** over the test period
4. **Win rate > 35%** with profit factor > 1.5
5. **Max drawdown < 15%** across all OOS windows
6. **Trailing stops demonstrated** — winning trades hold through normal retracements
7. **No survivorship bias** — performance holds on underperforming stocks

If criteria are NOT met, the strategy remains in sandbox-only mode and the
findings are documented for further iteration.

---

## Relationship to Track B

Track A (this document) and Track B (Observability) are independent and can be
developed in parallel:

- **Track A** fixes what the strategy does (signal quality, exits, validation)
- **Track B** fixes what the operator sees (API, dashboard, monitoring)

Both must be complete before Phase 4 go-live. Track B's Prometheus metrics
(e.g., `finalayze_strategy_win_rate`, `finalayze_drawdown_pct`) will consume
the improved metrics produced by Track A's risk calibration.

---

## Estimated Effort

| PR | Effort | Parallelizable with |
|----|--------|---------------------|
| A-1 | 2-3 days | A-2 |
| A-2 | 2-3 days | A-1 |
| A-3 | 1-2 days | Nothing (needs A-1) |
| A-4 | 2-3 days | A-3 (partially) |
| **Total** | **7-11 days** | |

# Phase 3 Design: Hardening & Advanced

**Date:** 2026-02-23
**Status:** Approved
**Phase:** 3 of 4

---

## Goal

Add a live trading loop (test/paper mode), per-market circuit breakers, Telegram alerting,
LSTM model per segment, pairs trading strategy, and integration + E2E tests.

---

## Parallel Implementation Strategy

Three tracks, A and B developed simultaneously:

| Track | Branch | Scope |
|-------|--------|-------|
| A — Trading Loop | `feature/phase3-trading-loop` | Settings additions, circuit breakers, Telegram alerting, TradingLoop (APScheduler), integration tests |
| B — ML & Strategies | `feature/phase3-ml-strategies` | LSTM model, training script, pairs trading strategy, statsmodels dep |
| C — E2E Tests | `feature/phase3-e2e` | End-to-end paper trading tests (after A + B merge) |

**Merge order:** Track A first, Track B second, Track C last.

---

## Track A: Trading Loop

### A1 — Settings Additions

New fields in `config/settings.py`:

```python
# Cycle intervals (restart required to apply changes)
news_cycle_minutes: int = 30          # FINALAYZE_NEWS_CYCLE_MINUTES
strategy_cycle_minutes: int = 60      # FINALAYZE_STRATEGY_CYCLE_MINUTES
daily_reset_hour_utc: int = 0         # FINALAYZE_DAILY_RESET_HOUR_UTC

# Telegram alerting
telegram_bot_token: str = ""          # FINALAYZE_TELEGRAM_BOT_TOKEN
telegram_chat_id: str = ""            # FINALAYZE_TELEGRAM_CHAT_ID
```

Also update `.env.example` with these new variables.

### A2 — Circuit Breakers

```
src/finalayze/risk/circuit_breaker.py
```

**Per-market circuit breaker:**

```python
class CircuitLevel(StrEnum):
    NORMAL    = "normal"     # trade freely
    CAUTION   = "caution"    # -5% daily: halve size, raise min confidence
    HALTED    = "halted"     # -10% daily: no new trades
    LIQUIDATE = "liquidate"  # -15% daily: close all positions immediately

class CircuitBreaker:
    def __init__(
        self,
        market_id: str,
        l1_threshold: float = 0.05,   # from settings.circuit_breaker_l1
        l2_threshold: float = 0.10,   # from settings.circuit_breaker_l2
        l3_threshold: float = 0.15,   # from settings.circuit_breaker_l3
    ) -> None: ...

    def check(
        self, current_equity: Decimal, baseline_equity: Decimal
    ) -> CircuitLevel: ...

    def reset_daily(self, new_baseline: Decimal) -> None: ...
        # Resets CAUTION and HALTED back to NORMAL.
        # LIQUIDATE is NOT auto-reset — requires reset_manual().

    def reset_manual(self) -> None: ...
        # Resets LIQUIDATE back to NORMAL (operator action only).

    @property
    def level(self) -> CircuitLevel: ...

    @property
    def market_id(self) -> str: ...
```

**Cross-market circuit breaker:**

```python
class CrossMarketCircuitBreaker:
    def __init__(self, halt_threshold: float = 0.10) -> None: ...
        # halt_threshold from settings.max_cross_market_exposure_pct logic

    def check(
        self,
        market_equities: dict[str, Decimal],   # market_id -> current equity
        baseline_equities: dict[str, Decimal], # market_id -> baseline equity
    ) -> bool: ...
        # Returns True if combined drawdown exceeds threshold → halt ALL markets

    def reset_daily(self, new_baselines: dict[str, Decimal]) -> None: ...
```

**Integration with trading loop:**
- Before any `submit_order`: call `CircuitBreaker.check()` for the order's market
- `CAUTION`: multiply `position_size * 0.5`, require `confidence >= min_confidence * 1.2`
- `HALTED`: skip order, log reason
- `LIQUIDATE`: submit SELL for all open positions in that market, then alert
- After each strategy cycle: call `CrossMarketCircuitBreaker.check()`

### A3 — Telegram Alerting

```
src/finalayze/core/alerts.py
```

```python
class TelegramAlerter:
    def __init__(self, bot_token: str, chat_id: str) -> None: ...
        # If bot_token is empty → no-op on all methods (safe default)

    def on_trade_filled(self, result: OrderResult, market_id: str, broker: str) -> None: ...
    def on_trade_rejected(self, order: OrderRequest, reason: str) -> None: ...
    def on_circuit_breaker_trip(
        self, market_id: str, level: CircuitLevel, drawdown_pct: float
    ) -> None: ...
    def on_circuit_breaker_reset(self, market_id: str) -> None: ...
    def on_daily_summary(
        self,
        market_pnl: dict[str, Decimal],  # market_id -> daily P&L in local currency
        total_equity_usd: Decimal,
    ) -> None: ...
    def on_error(self, component: str, message: str) -> None: ...
```

**Implementation details:**
- Uses `httpx.post()` (sync) to `https://api.telegram.org/bot{token}/sendMessage`
- Fire-and-forget: errors are logged but never propagate (never crash the trading loop)
- If `bot_token` is empty string, all methods return immediately

**Message format examples:**
```
🟢 BUY AAPL ×10 @ $150.00 (Alpaca paper)
⚠️ AAPL BUY rejected: insufficient funds
🔴 [US] Circuit breaker L2 — trading halted (-10.3% daily)
✅ [US] Circuit breaker reset — trading resumed
📊 Daily: US +$342 | MOEX +₽1,200 | Equity $51,200
🚨 TinkoffFetcher error: gRPC timeout
```

### A4 — TradingLoop

```
src/finalayze/core/trading_loop.py
```

```python
class TradingLoop:
    def __init__(
        self,
        settings: Settings,
        fetchers: dict[str, BaseFetcher],     # market_id -> fetcher
        news_fetcher: NewsApiFetcher,
        news_analyzer: NewsAnalyzer,
        event_classifier: EventClassifier,
        impact_estimator: ImpactEstimator,
        strategy: StrategyCombiner,
        broker_router: BrokerRouter,
        circuit_breakers: dict[str, CircuitBreaker],  # market_id -> breaker
        cross_market_breaker: CrossMarketCircuitBreaker,
        alerter: TelegramAlerter,
        instrument_registry: InstrumentRegistry,
    ) -> None: ...

    def start(self) -> None: ...
        # Starts APScheduler, blocks until stop() is called

    def stop(self) -> None: ...
        # Graceful shutdown — completes current cycle, then stops

    def _news_cycle(self) -> None: ...
        # Fetches latest news, analyzes, stores sentiment in self._sentiment_cache
        # self._sentiment_cache: dict[str, float]  # symbol -> latest sentiment score

    def _strategy_cycle(self) -> None: ...
        # For each market, for each symbol:
        #   1. Fetch recent candles (last 60 bars)
        #   2. Get cached sentiment score (default 0.0 if absent)
        #   3. combiner.generate_signal(symbol, candles, segment_id, sentiment)
        #   4. Check circuit breaker level
        #   5. If signal and breaker allows: size position, pre-trade check, submit
        #   6. Alert on fill or rejection

    def _daily_reset(self) -> None: ...
        # Reset all circuit breakers
        # Send daily P&L summary alert

    def _liquidate_market(self, market_id: str) -> None: ...
        # Close all open positions for a market (L3 circuit breaker response)
```

**APScheduler setup:**
```python
scheduler = BackgroundScheduler(timezone="UTC")
scheduler.add_job(
    self._news_cycle,
    "interval",
    minutes=settings.news_cycle_minutes,
)
scheduler.add_job(
    self._strategy_cycle,
    "interval",
    minutes=settings.strategy_cycle_minutes,
)
scheduler.add_job(
    self._daily_reset,
    "cron",
    hour=settings.daily_reset_hour_utc,
    minute=0,
)
```

**Sentiment cache:** In-memory `dict[str, float]` mapping `symbol → sentiment_score`.
Updated by `_news_cycle`, read by `_strategy_cycle`. Thread-safe via `threading.Lock`.

### A5 — Integration Tests

```
tests/integration/test_trading_loop.py
tests/integration/test_circuit_breaker_integration.py
tests/integration/test_news_to_signal.py
```

| Test file | What it covers |
|---|---|
| `test_trading_loop.py` | Full strategy cycle: mock fetcher + mock broker, signal → pre-trade check → circuit breaker → order submitted → alert fired |
| `test_circuit_breaker_integration.py` | L1→L2→L3 escalation, cross-market trip, auto daily reset, manual reset |
| `test_news_to_signal.py` | NewsAnalyzer → EventClassifier → ImpactEstimator → EventDrivenStrategy chain, mocked LLM |

---

## Track B: ML & Strategies

### B1 — statsmodels Dependency

Add to `pyproject.toml`:
```toml
"statsmodels>=0.14.0",
```

Add to `[[tool.mypy.overrides]]`:
```toml
"statsmodels.*",
```

### B2 — LSTM Model

```
src/finalayze/ml/models/lstm_model.py
```

```python
class LSTMModel(BaseMLModel):
    segment_id: str

    def __init__(
        self,
        segment_id: str,
        sequence_length: int = 20,
        hidden_size: int = 64,
        num_layers: int = 2,
    ) -> None: ...

    def predict_proba(self, features: dict[str, float]) -> float: ...
        # Returns 0.5 when untrained (graceful degradation)
        # Expects features dict with keys sorted (same as XGBoost/LightGBM)

    def fit(self, X: list[dict[str, float]], y: list[int]) -> None: ...
        # Trains for 50 epochs, Adam optimizer, BCE loss
        # Reshapes X into (batch, sequence_length, n_features) tensor
        # If len(X) < sequence_length: raises InsufficientDataError

    def save(self, path: Path) -> None: ...
        # torch.save(self._model.state_dict(), path)

    def load(self, path: Path) -> None: ...
        # torch.load state_dict, sets self._trained = True
```

**PyTorch model (internal):**
```python
class _LSTMNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int) -> None:
        # nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # nn.Linear(hidden_size, 1)
        # nn.Sigmoid()
```

**EnsembleModel update** (`src/finalayze/ml/models/ensemble.py`):
```python
class EnsembleModel:
    def predict_proba(self, features: dict[str, float]) -> float:
        # Average of all trained models (1, 2, or 3)
        # At least one must be trained, else returns 0.5
```

### B3 — Model Training Script

```
scripts/train_models.py
```

```
usage: train_models.py [--segment SEGMENT_ID] [--output-dir models/]

For each segment (or specified segment):
  1. Load candles from DB (falls back to yfinance if DB empty)
  2. Generate features via compute_features() for each window of 30+ candles
  3. Create labels: y=1 if close[t+1] > close[t], else 0
  4. Time-series split: 80% train / 20% test (no shuffle)
  5. Train XGBoostModel, LightGBMModel, LSTMModel
  6. Evaluate on test split: print accuracy + AUC per model
  7. Save to models/<segment_id>/{xgb,lgbm,lstm}.pkl
```

### B4 — Pairs Trading Strategy

```
src/finalayze/strategies/pairs.py
```

```python
class PairsStrategy(BaseStrategy):
    @property
    def name(self) -> str: return "pairs"

    def supported_segments(self) -> list[str]:
        # Reads YAML presets — returns segments with pairs.enabled = true

    def generate_signal(
        self, symbol: str, candles: list[Candle], segment_id: str
    ) -> Signal | None:
        # For each configured pair containing symbol:
        #   1. Requires 60+ candles for both symbols
        #   2. statsmodels.coint(log_prices_A, log_prices_B) → skip if p > 0.05
        #   3. OLS regression: beta = cov(A,B) / var(B)
        #   4. spread = log(A) - beta * log(B)
        #   5. z = (spread[-1] - mean(spread)) / std(spread)
        #   6. z < -z_entry → BUY; z > z_entry → SELL; |z| < z_exit → HOLD (close)
        #   7. confidence = min(1.0, abs(z) / z_entry)

    def get_parameters(self, segment_id: str) -> dict[str, object]: ...
```

**YAML preset additions:**
```yaml
# In strategies section of e.g. ru_blue_chips.yaml:
pairs:
  enabled: true
  weight: 0.2
  params:
    pairs: [[SBER, VTBR], [GAZP, LKOH]]
    z_entry: 2.0
    z_exit: 0.5
    min_confidence: 0.6
```

---

## Track C: E2E Tests

```
tests/e2e/test_paper_trading_cycle.py
tests/e2e/test_circuit_breaker_e2e.py
```

| Test file | What it covers |
|---|---|
| `test_paper_trading_cycle.py` | `TradingLoop` runs 3 strategy cycles on US + MOEX with mock fetchers and mock brokers; verifies orders submitted to both, Telegram messages captured |
| `test_circuit_breaker_e2e.py` | Inject -15% equity drop → L3 fires → all positions closed → alert sent → manual reset → trading resumes |

**Test mode:** All tests run with `WorkMode.TEST`. No live API calls — brokers and LLM are mocked.

---

## Testing Approach

| Component | Strategy | Mock target |
|-----------|----------|-------------|
| CircuitBreaker | Unit | pure logic, no mocks |
| CrossMarketCircuitBreaker | Unit | pure logic, no mocks |
| TelegramAlerter | Unit | `httpx.post` |
| TradingLoop._news_cycle | Unit | `NewsApiFetcher.fetch_news`, `LLMClient.complete` |
| TradingLoop._strategy_cycle | Unit | `BaseFetcher.fetch_candles`, `BrokerRouter.submit` |
| LSTMModel | Unit | synthetic numpy arrays |
| PairsStrategy | Unit | synthetic candle lists |
| train_models.py | Unit | DB query + yfinance fallback |
| Integration: trading_loop | Integration | mock fetcher + mock broker |
| E2E: paper trading | E2E | mock broker + mock LLM |

Coverage target: ≥80% per new module.

---

## Dependency Notes

- `statsmodels>=0.14.0` — for `coint()` in pairs trading
- `torch` already in `pyproject.toml` — LSTM uses existing dep
- `apscheduler>=3.10.4` already in `pyproject.toml` — TradingLoop scheduler
- `httpx>=0.28.0` already in `pyproject.toml` — TelegramAlerter HTTP calls

---

## What's Deferred to Phase 4

- Prometheus metrics
- Grafana dashboards
- Walk-forward optimization
- Online learning per segment
- A/B testing framework
- Production Docker setup
- Real mode trading

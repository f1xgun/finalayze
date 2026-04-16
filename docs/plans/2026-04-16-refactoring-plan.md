# Refactoring Plan: Large File Decomposition

**Date**: 2026-04-16
**Goal**: Разбить файлы >800 LOC на модули с единой ответственностью. Снизить merge-conflict surface, улучшить тестируемость, ускорить параллельную разработку.

**Правило**: Если файл >500 строк и содержит >2 зон ответственности — он кандидат на split.

---

## Phase 1: orchestration/trading_loop.py (2883 → ~500 строк)

**Приоритет**: CRITICAL — god-object, 13 зон ответственности, вызвал 90% проблем при мерже v10.0.

### Порядок извлечения (от низкого риска к высокому)

Каждый шаг — отдельный PR, с тестами, без изменения поведения.

#### 1.1 DB Persistence Layer → `orchestration/db_persistence.py` (~200 LOC)

**Методы для извлечения:**
- `_get_bg_session_factory()` (lines 2499-2528)
- `_persist_to_db()` / `_persist_to_db_async()` (lines 2530-2550)
- `_persist_news_article_async()` (lines 2552-2584)
- `_persist_sentiment_batch_async()` (lines 2586-2608)
- `_persist_order_async()` (lines 2610-2638)
- `_persist_signal_async()` (lines 2640-2659)
- `_persist_equity_snapshots()` / `_persist_snapshots_async()` (lines 2661-2702)

**Shared state**: Нет (fire-and-forget, изолированный async engine).
**Риск**: Минимальный — чистая запись в БД, без обратных зависимостей.

**Интерфейс:**
```python
class TradingPersistence:
    def __init__(self, db_url: str, async_loop: asyncio.AbstractEventLoop): ...
    def persist_news_article(self, article, impact_result): ...  # fire-and-forget
    def persist_sentiment_batch(self, scores, market_id, confidence): ...
    def persist_order(self, order, result, market_id): ...
    def persist_signal(self, signal): ...
    def persist_equity_snapshots(self, baselines, now): ...
```

#### 1.2 ML Retraining → `orchestration/ml_retraining.py` (~130 LOC)

**Методы для извлечения:**
- `_retrain_cycle()` (lines 2250-2279)
- `_retrain_segment()` (lines 2281-2382)

**Shared state**: Читает `_fetchers`, `_registry`, `_ml_registry`. Пишет только в `_ml_registry` (hot-swap).
**Риск**: Низкий — self-contained ML pipeline.

**Интерфейс:**
```python
class MLRetrainingService:
    def __init__(self, fetchers, registry, ml_registry, settings, alerter): ...
    def retrain_all_segments(self): ...
```

#### 1.3 Sentiment Manager → `orchestration/sentiment_manager.py` (~150 LOC)

**Методы для извлечения:**
- `_get_segment_tickers()` (lines 1168-1175)
- `_collect_active_segments()` (lines 1177-1187)
- `_read_decayed_sentiment()` (lines 1189-1219)
- `_get_sentiment()` (lines 1221-1232)
- `_any_event_driven_enabled()` (lines 1234-1263)

**Shared state**: `_sentiment_cache` (dict + Lock), `_event_driven_active` (cached bool).
**Риск**: Средний — читается из News pipeline и Signal execution.

**Интерфейс:**
```python
class SentimentManager:
    def __init__(self, registry, cache: RedisCache | None, sentiment_lock: Lock): ...
    def get_sentiment(self, seg_id, ticker=None) -> float: ...
    def update_sentiment(self, seg_id, ticker, score, ts): ...  # called by News
    def is_event_driven_active(self) -> bool: ...
```

**Ключевое решение**: `_sentiment_cache` перестаёт быть атрибутом TradingLoop — владеет им SentimentManager. News pipeline и Signal execution получают SentimentManager как зависимость.

#### 1.4 Daily Reporting → `orchestration/daily_reporting.py` (~250 LOC)

**Методы для извлечения:**
- `_daily_reset()` (lines 2384-2475)
- `_compute_top_movers()` (lines 2477-2495)
- `_load_baseline_from_db()` / `_load_baseline_async()` (lines 2706-2774)
- `_weekly_digest()` (lines 2776-2847)

**Shared state**: `_baseline_equities` (read/write из нескольких мест).
**Риск**: Средний — baseline_equities shared с liquidation и strategy cycle.

**Интерфейс:**
```python
class DailyReportingService:
    def __init__(self, broker_router, circuit_breakers, alerter, persistence, ...): ...
    def daily_reset(self) -> dict[str, Decimal]:  # returns new baselines
    def weekly_digest(self, baselines): ...
    def load_baselines_from_db(self) -> dict[str, Decimal]: ...
```

**Ключевое решение**: `_baseline_equities` остаётся на TradingLoop, но DailyReportingService принимает и возвращает их как параметры.

#### 1.5 News Pipeline → `orchestration/news_pipeline.py` (~280 LOC)

**Методы для извлечения:**
- `_news_cycle()` (lines 890-955)
- `_is_article_duplicate()` (lines 957-983)
- `_analyze_impact_batch()` (lines 985-1066)
- `_apply_impact_result()` (lines 1068-1127)
- `_persist_sentiment_scores()` / `_persist_sentiment_scores_async()` (lines 1129-1166)

**Shared state**: `_seen_article_hashes` (dedup), пишет в SentimentManager.
**Риск**: Средний — глубокий async, semaphore, circuit breaker.

**Интерфейс:**
```python
class NewsPipeline:
    def __init__(self, rss_fetcher, telegram_reader, news_fetcher,
                 impact_analyzer, sector_mapper, sentiment_mgr,
                 persistence, async_loop, settings): ...
    def run_news_cycle(self): ...
```

#### 1.6 Position Manager → `orchestration/position_manager.py` (~200 LOC)

**Методы для извлечения:**
- `_check_stop_losses()` (lines 2173-2237)
- `_update_kelly()` (lines 2239-2248)
- Entry/exit state: `_entry_prices`, `_entry_strategy`, `_stop_states`, `_cycle_exited_symbols`

**Shared state**: Тесная связь с Signal Execution (BUY записывает, SELL/SL читает и чистит).
**Риск**: Высокий — bidirectional state с signal execution, `_stop_loss_lock`.

**Интерфейс:**
```python
class PositionTracker:
    """Owns entry prices, stop states, exit tracking, Kelly updates."""
    def __init__(self, kelly_sizer, stop_loss_lock): ...
    def register_entry(self, symbol, price, strategy, stop_state): ...
    def check_stop_losses(self, market_id, symbol, price, broker) -> bool: ...
    def register_exit(self, symbol): ...
    @property
    def exited_symbols(self) -> set[str]: ...
```

#### 1.7 Signal Executor → `orchestration/signal_executor.py` (~450 LOC)

**Методы для извлечения:**
- `_process_instrument()` (lines 1576-1854)
- `_build_order()` (lines 1952-2027)
- `_submit_order()` (lines 2069-2171)
- `_build_sizing_pipeline()` (lines 1855-1896)
- `_compute_asset_vol()` (lines 1899-1913)
- Вспомогательные: `_get_regime_scale/state()`, `_has_pending_order()`, `_get_segment_min_confidence()`

**Shared state**: Самый сложный — cycle counters, position tracker, broker, strategy, sentiment.
**Риск**: Высокий — ядро торговой логики.

**Интерфейс:**
```python
class SignalExecutor:
    def __init__(self, strategy, broker_router, position_tracker,
                 sentiment_mgr, persistence, pre_trade_checker,
                 loss_limit_tracker, macro_cache, ...): ...
    def process_instrument(self, instrument, market_id, level,
                          fetcher, now, cycle_ctx) -> CycleResult: ...
```

**Ключевое решение**: Ввести `CycleContext` dataclass для per-cycle counters вместо self._cycle_*.

#### 1.8 Emergency Liquidation → оставить в trading_loop.py (~40 LOC)

Слишком мала для отдельного модуля. `_liquidate_market()` и `_close_positions()` остаются.

### Итоговая структура после Phase 1:

```
orchestration/
├── trading_loop.py          ~500  Orchestrator: init, start/stop, scheduling, cycle dispatch
├── news_pipeline.py         ~280  News fetch, impact analysis, sentiment updates
├── signal_executor.py       ~450  Signal generation, sizing, order execution
├── position_manager.py      ~200  Stop-loss, Kelly, entry/exit state
├── sentiment_manager.py     ~150  Decay cache, Redis reads, segment inventory
├── db_persistence.py        ~200  Fire-and-forget DB writes
├── daily_reporting.py       ~250  Daily reset, weekly digest, baselines
├── ml_retraining.py         ~130  Model retrain cycle
├── bond_cycle.py            ~170  (already exists)
└── preset_applicator.py     ~453  (already exists)
```

---

## Phase 2: backtest/engine.py (1486 → ~600 строк)

**Приоритет**: HIGH — second-largest class, mixed portfolio + execution + risk + journaling.

### 2.1 Decision Journal → `backtest/journal.py` (~150 LOC)

**Извлечь**: `_journal_decision()`, `_journal_skip()`, journal init logic.

**Интерфейс:**
```python
class BacktestJournal:
    def __init__(self): ...
    def record_decision(self, signal, order, result, context: dict): ...
    def record_skip(self, symbol, reason, context: dict): ...
    def get_entries(self) -> list[dict]: ...
```

### 2.2 Position Executor → `backtest/position_executor.py` (~350 LOC)

**Извлечь**: `_handle_buy()`, `_handle_sell()`, `_close_position()`, `_record_trade()`.

**Интерфейс:**
```python
class BacktestPositionExecutor:
    def __init__(self, broker, sizing_pipeline, config): ...
    def handle_buy(self, signal, candles, portfolio_state) -> TradeResult: ...
    def handle_sell(self, signal, position, candles) -> TradeResult: ...
```

### 2.3 Risk Evaluator → `backtest/risk_evaluator.py` (~100 LOC)

**Извлечь**: `_compute_segment_exposure()`, `_compute_correlations()`, concentration checks из `run_portfolio()`.

### Результат:

```
backtest/
├── engine.py                ~600  Orchestrator: run(), run_portfolio() loop
├── position_executor.py     ~350  Buy/sell/close execution
├── journal.py               ~150  Decision logging
├── risk_evaluator.py        ~100  Concentration, correlation checks
├── config.py                      (exists)
├── iteration_tracker.py           (exists)
└── ...
```

---

## Phase 3: ml/features/technical.py (1232 → ~300 строк)

**Приоритет**: HIGH — 29 функций, 12 доменов, чистые функции без state → самый безопасный split.

### Разделение по доменам:

```
ml/features/
├── technical.py             ~300  compute_features() + core/extra/predictive
├── regime.py                ~60   _compute_regime_features (VIX-based)
├── cross_asset.py           ~90   _compute_cross_asset_features (benchmark)
├── macro.py                 ~170  _compute_macro_features, _compute_cbr_features, _TRAILING_CPI
├── microstructure.py        ~150  _compute_microstructure_features, amihud, corwin_schultz
├── moex_external.py         ~200  _compute_fx_*, _compute_commodity_*, _compute_brent_*, turnover
├── wavelet.py               ~40   _compute_wavelet_features
├── calendar.py              ~20   _compute_calendar_features
├── zscore.py                ~50   _compute_zscore_features, _safe_zscore, _rolling_zscore_clipped
└── constants.py             ~50   All shared constants (_MIN_CANDLES, lookback windows, etc.)
```

**Риск**: Минимальный — все функции pure, без shared state. `compute_features()` остаётся entry point, импортирует из субмодулей.

---

## Phase 4: scripts/train_models.py (2073 → ~400 строк)

**Приоритет**: MEDIUM — скрипт, не импортируется другими модулями. Не блокирует мерж.

### Разделение:

```
scripts/training/
├── __init__.py
├── cli.py                   ~100  _parse_args, main()
├── data_loader.py           ~250  _fetch_*, _orm_to_candle, _align_benchmark_candles
├── dataset_builder.py       ~200  _build_dataset_*, _get_barrier_params, _compute_uniqueness
├── walk_forward.py          ~200  _generate_walk_forward_folds, train_walk_forward, _apply_bh_*
├── model_trainer.py         ~150  train_one_segment, _train_and_evaluate_models, _evaluate_model
├── calibration.py           ~100  _fit_and_save_calibrator, _fit_and_save_meta_learner
└── quality.py               ~50   compute_n_eff, compute_accuracy_threshold, compute_brier_threshold
```

**Обратная совместимость**: `scripts/train_models.py` становится тонким wrapper:
```python
from scripts.training.cli import main
if __name__ == "__main__":
    main()
```

---

## Phase 5: orchestration/bond_cycle.py (877 → ~400 строк)

**Приоритет**: LOW-MEDIUM — один класс, но mixed orchestration + pricing + execution.

### Разделение:

```
orchestration/
├── bond_cycle.py            ~400  BondCycleProcessor: run_cycle(), orchestration
├── bond_pricing.py          ~200  Bond pricing calculations, yield stops
├── bond_executor.py         ~150  Bond order execution, LayerLedger interaction
```

---

## Phase 6: main.py (570 → ~200 строк)

**Приоритет**: LOW-MEDIUM — `_build_trading_loop()` занимает 300 строк.

### Разделение:

```
src/finalayze/
├── main.py                  ~200  FastAPI app, lifespan, middleware
├── bootstrap.py             ~300  _build_trading_loop(), component wiring
```

---

## Dependency & Ordering

```
Phase 1.1 (DB Persistence)     ──┐
Phase 1.2 (ML Retraining)      ──┤── Независимые, можно параллельно
Phase 3   (technical.py split)  ──┘
                                  │
Phase 1.3 (Sentiment Manager)  ──── Зависит от интерфейса, но не от кода Phase 1.1
                                  │
Phase 1.4 (Daily Reporting)    ──── Зависит от Persistence (Phase 1.1)
Phase 1.5 (News Pipeline)     ──── Зависит от Sentiment (Phase 1.3) + Persistence (Phase 1.1)
                                  │
Phase 1.6 (Position Manager)  ──── Зависит от интерфейса Signal Executor
Phase 1.7 (Signal Executor)   ──── Зависит от Position + Sentiment + Persistence
                                  │
Phase 2   (backtest engine)    ──── Независим от Phase 1
Phase 4   (train_models.py)    ──── Независим
Phase 5   (bond_cycle.py)      ──── Независим
Phase 6   (main.py)            ──── После Phase 1 (bootstrap wires new components)
```

---

## Критерии завершения каждого шага

1. **Тесты проходят**: `uv run pytest` — 0 failures
2. **Lint/Type clean**: `ruff check . && ruff format --check . && mypy src/`
3. **Поведение не изменилось**: Docker sandbox запускается, news cycle и strategy cycle работают
4. **Строк в оригинале уменьшилось** на ожидаемое количество
5. **Нет circular imports**: проверка `python -c "from finalayze.orchestration.trading_loop import TradingLoop"`

---

## Оценка

| Phase | Файл | Было | Станет | Шагов | Сложность |
|-------|------|------|--------|-------|-----------|
| 1 | trading_loop.py | 2883 | ~500 | 7 | High |
| 2 | backtest/engine.py | 1486 | ~600 | 3 | Medium |
| 3 | ml/features/technical.py | 1232 | ~300 | 1 | Low |
| 4 | scripts/train_models.py | 2073 | ~400 | 1 | Low |
| 5 | bond_cycle.py | 877 | ~400 | 2 | Medium |
| 6 | main.py | 570 | ~200 | 1 | Low |
| **Total** | | **9121** | **~2400** | **15** | |

**Суммарное сокращение**: ~6700 строк перераспределены в ~15 новых модулей по 100-450 строк.

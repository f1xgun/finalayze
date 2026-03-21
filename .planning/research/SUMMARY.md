# Project Research Summary

**Project:** Finalayze MOEX MVP — v3.0 Production Readiness
**Domain:** Sandbox monitoring, go/no-go gate automation, gradual rollout, production health operations
**Researched:** 2026-03-21
**Confidence:** HIGH

## Executive Summary

Finalayze v3.0 is a production readiness milestone for an existing autonomous MOEX trading system. The system already has a rich v2.0 foundation — Prometheus metrics, structured logging, CircuitBreaker, PreTradeChecker, TelegramAlerter, APScheduler cycles, and a Streamlit dashboard — so the work is not building an observability stack from scratch. It is building the decision layer on top of that stack: formalized go/no-go gate logic, slippage capture to feed that gate, gradual rollout configuration, and emergency kill switch capability. The recommended approach is to add a new `monitoring/` module at Layer 6 (same plane as `api/`), implement the gate as a pure function for testability, configure rollout phases as frozen dataclasses at Layer 1, and make minimal additive modifications to existing components. At most 2-3 new pip packages are required (Jinja2, WeasyPrint for PDF reports; aiogram for Telegram bot commands) — zero if PDF archiving is deferred.

The critical risk for v3.0 is false confidence from sandbox metrics. Tinkoff's sandbox fills orders at the last price with 100% synthetic fill rate, so sandbox P&L will appear 0.3-1.5% better than live execution on illiquid MOEX instruments. The go/no-go gate must focus on what sandbox CAN validate reliably — uptime, signal-direction correctness, absence of critical errors — and thresholds must be derived from walk-forward backtest distribution percentiles, not round numbers. The go-live decision is always a human decision: the system generates a structured report with PROCEED/DEFER/ABORT recommendation; the operator reviews it and initiates the mode transition with an existing confirmation token.

Research also surfaced a parallel v2.0 MOEX strategy improvement workstream with data-integrity prerequisites that must be resolved before any new strategy backtesting is valid. The current codebase has three confirmed problems invalidating all MOEX backtest results: vol_target 0.19 is US-calibrated (MOEX needs 0.35-0.40); the dividend calendar has only 43 events and excludes cancelled dividends, introducing look-ahead bias; and Feb-Mar 2022 MOEX closure contaminates walk-forward training with artificially distorted volatility. These are quick fixes with high impact and must come before strategy work. The highest architectural risk in the v2.0 stream is sector rotation — it must be implemented as a SectorAllocationStep in the PositionSizingPipeline, not as a combiner strategy; implementing it in the wrong layer carries a HIGH recovery cost.

## Key Findings

### Recommended Stack

v3.0 requires minimal new dependencies. The entire monitoring and health-pulse infrastructure already exists. See `.planning/research/STACK.md` for the full audit of existing capabilities.

**Core technologies (new additions only):**
- `Jinja2 >= 3.1.4`: HTML template rendering for gate reports — already a transitive dependency via FastAPI/Starlette; just make it explicit in pyproject.toml
- `weasyprint >= 68.1`: HTML-to-PDF conversion for archiving go/no-go reports — best Python option in 2025, no Chromium, CSS3 support; can be omitted entirely if PDF archiving is not required
- `aiogram >= 3.17.0`: Async-native Telegram bot command handling for /kill, /rollout, /gonogo — optional; raw httpx POST is sufficient if interactive conversation FSM is not needed
- `types-Jinja2 >= 3.1.0`: mypy stubs for Jinja2 (dev dependency only)

**Stack decision rule:** If PDF report archiving is not required, v3.0 adds zero new pip packages. All monitoring primitives (metrics, alerting, scheduling, health checks, kill switch) are already installed.

**What NOT to add:** OpenTelemetry (overkill for a single-process system), Celery (mismatches APScheduler single-process model), external feature flag services (rollout is capital/risk limit progression, not code path toggling), Datadog/NewRelic/Sentry APMs (external SaaS with sensitive trading data egress), ADTK/darts for anomaly detection (unmaintained or too heavy; rolling z-score with numpy is sufficient).

### Expected Features

See `.planning/research/FEATURES.md` for the full feature table with prioritization matrix and dependency graph.

**Must have (table stakes — blocks go-live without these):**
- Formalized go/no-go gate with configurable thresholds — current gate has 4 hardcoded criteria; need 8 configurable criteria including uptime %, fill rate, mean slippage, and signal frequency
- Slippage capture in sandbox — expected_price at signal generation time vs. actual fill price; required to populate slippage gate criterion
- Gradual rollout tightened risk limits config — `RolloutPhase` profiles (minimal_50k: 3% max position, 1% daily loss, 2% DD auto-stop) consumed by PreTradeChecker and CircuitBreaker
- Production health heartbeat alert — silent TradingLoop crash undetected for hours without this; timer check + Telegram CRITICAL alert
- REST API kill switch POST /system/stop — Telegram /stop exists but CI and scripts need a programmatic equivalent

**Should have (P2 — add during or after sandbox validation period):**
- Sandbox dashboard validation progress page — Streamlit page showing per-criterion progress bars and per-day equity chart; manual JSONL report is acceptable for MVP
- Automated gate evaluation on daily schedule — APScheduler job at 19:00 MSK; manual run is acceptable for MVP
- Strategy signal-frequency anomaly alert — detects silent strategy failure via Prometheus strategy_signal_count counters
- Post-live slippage report vs. backtest comparison — weekly Telegram message; needs 5-10 real trades before first run

**Defer to v3.1+:**
- Signal divergence tracker (shadow backtest engine per cycle) — too compute-intensive for the trading loop hot path; needs separate architectural design
- Capital scaling confirmation flow — manual confirmation token is acceptable for the first 30 days live; automate after stable live data

**Anti-features (specifically avoid):**
- Fully automated sandbox-to-live promotion — any system that promotes itself to real money without a human checkpoint is a liability
- Automated position liquidation on gate FAIL — adds state management complexity with no benefit in sandbox; high-risk in live (slippage, incomplete fills)
- Per-cycle shadow backtest for signal divergence — adds 30-60 seconds compute per cycle, starves the main trading loop
- Multi-account capital splitting — Tinkoff Invest does not support multi-account management via one API token

**Go/no-go gate expansion (current hardcoded vs. v3.0 configurable target):**

| Criterion | Current (hardcoded) | v3.0 Target (configurable) |
|-----------|---------------------|----------------------------|
| Minimum trading days | >= 5 | >= 10 (two full MOEX weeks) |
| Max drawdown | < 5.0% | < 5.0% configurable; tighten to 3% for phase 1 live |
| Round-trip fills | >= 10 | >= 20 (statistical minimum) |
| Critical errors | == 0 | == 0 (keep) |
| Uptime % | Not measured | >= 99% (actual cycles / expected cycles) |
| Fill rate | Not measured | >= 95% (orders_filled / orders_submitted) |
| Mean slippage | Not measured | < 30 bps |
| Signal frequency | Not measured | >= 1 signal per enabled strategy per 5 days |

### Architecture Approach

The architecture adds a new `monitoring/` top-level module at Layer 6 and makes minimal additive modifications to existing components. See `.planning/research/ARCHITECTURE.md` for full data-flow diagrams, component specifications, and anti-pattern analysis.

**New module structure:**
```
src/finalayze/monitoring/   # NEW — Layer 6, same plane as api/
    sandbox_monitor.py      # SandboxMonitorService: collect metrics, persist to TimescaleDB
    gonogo.py               # GoNoGoReporter: pure evaluation function, no side effects
    anomaly.py              # AnomalyDetector: z-score alerting on equity/slippage/signals
    health_monitor.py       # ProductionHealthMonitor: 5-minute health pulse
    kill_switch.py          # KillSwitch: halt-all with order cancellation sequence

config/rollout.py           # NEW — Layer 1: RolloutPhase frozen dataclasses, env var resolver
api/v1/sandbox.py           # NEW: GET /sandbox/metrics, GET /sandbox/gonogo
alembic/versions/XXX.py     # NEW migration: sandbox_metrics TimescaleDB hypertable
```

**Major components and their responsibilities:**
1. `SandboxMonitorService` — reads ValidationLogger JSONL (fast append-only write path), aggregates daily metric snapshots, persists SandboxMetricSnapshot to TimescaleDB (queryable read path)
2. `GoNoGoReporter` — pure function: takes list of SandboxMetricSnapshot and configurable thresholds, applies 8 gate rules, returns frozen GoNoGoReport with PROCEED/DEFER/ABORT recommendation; no side effects, trivially testable
3. `RolloutPhase` in `config/rollout.py` — frozen dataclasses with named profiles (minimal_50k, scale_200k, target_500k) and per-phase risk parameter overrides; active phase resolved from FINALAYZE_ROLLOUT_PHASE env var at startup; validated by Pydantic
4. `ProductionHealthMonitor` — 5-minute APScheduler job: broker ping, ValidationLogger feed freshness, Redis ping, circuit breaker level; sends Telegram IMPORTANT alert on degradation
5. `KillSwitch.activate(reason)` — strict sequence: (1) set _stop_event to halt scheduler, (2) escalate all CircuitBreakers to LIQUIDATE, (3) cancel pending orders via BrokerRouter, (4) send CRITICAL Telegram alert, (5) set kill_switch_active Prometheus gauge

**Existing components modified (all additive, None-default to preserve existing behavior):**
- `CycleLogEntry` — 3 optional None-default fields: fill_rate_pct, avg_slippage_bps, signal_divergence_pct
- `PreTradeChecker` — accept optional `rollout_phase: RolloutPhase | None = None` for position limit overrides
- `CircuitBreaker` — accept optional `rollout_phase: RolloutPhase | None = None` for threshold overrides
- `TradingLoop` — add `_health_pulse_cycle()` APScheduler job; add `activate_kill_switch()` method
- `TelegramBotHandler` — add /gonogo, /health, /killswitch to existing command dispatch table
- `MetricsCollector` — add ~5 new Prometheus gauges (fill_rate, kill_switch_active, rollout_phase, health_pulse_status, signal_divergence)

**Suggested build order (dictated by dependency chain):**
1. Schema and config foundation (GoNoGoReport, SandboxMetricSnapshot, HealthCheckResult schemas; RolloutPhase dataclasses; DB migration — zero behavior change)
2. Risk layer rollout wiring (CircuitBreaker and PreTradeChecker accept optional rollout_phase)
3. Monitoring services (SandboxMonitorService, GoNoGoReporter, AnomalyDetector, ProductionHealthMonitor)
4. API endpoints (GET /sandbox/metrics, GET /sandbox/gonogo; thin wrappers over Phase 3 services)
5. TradingLoop and Telegram extensions (health pulse job, kill switch, new bot commands — highest regression risk, done last)
6. Streamlit dashboard pages (optional — Telegram commands provide equivalent operational visibility)

### Critical Pitfalls

PITFALLS.md covers two research domains. See `.planning/research/PITFALLS.md` for full analysis including detection signals, recovery costs, and phase-specific warnings.

**Top pitfalls — v3.0 production readiness:**

1. **Sandbox gives false confidence because Tinkoff fills are synthetic** — Sandbox always fills at last price with 100% fill rate. Live MOEX execution on illiquid instruments will be 0.3-1.5% worse. Sandbox P&L must NOT be the primary gate criterion. Track "simulated slippage" (sandbox fill price vs. MOEX ISS mid-price at signal time) to measure the gap explicitly. Zero fill rejections over 30 sandbox days is a warning sign, not a positive indicator.

2. **Go/no-go thresholds invented, not calibrated against backtest distributions** — A 5% DD gate will permanently block go-live if the system naturally produces 6% drawdown during sideways MOEX markets. Derive each threshold from walk-forward percentiles from existing PortfolioBacktestOrchestrator results. Separate "blocking" gates (uptime, signal direction) from "advisory" gates (slippage, fill rate).

3. **Kill switch that only stops the scheduler leaves pending broker orders live** — Pending MOEX limit orders remain valid until market close. Strict sequence required: stop accepting cycles, escalate CircuitBreakers to LIQUIDATE, cancel all pending orders via BrokerRouter, then shut down scheduler. If order cancellation fails, log and continue — the LIQUIDATE circuit state prevents new orders.

4. **Monitoring logic embedded in TradingLoop** — TradingLoop is already 500+ lines with 20+ injected dependencies. Adding monitoring creates scheduler contention (a slow health check delays trade execution) and untestable circular coupling. Monitoring services must be standalone with separate APScheduler jobs.

5. **Automated go/no-go block at real mode startup** — Creates a chicken-and-egg problem and hides root cause of failures. The existing `real_confirmed` environment variable guard is the correct automated safety. Go/no-go is an on-demand report that the operator reviews before initiating mode transition.

**Top pitfalls — v2.0 MOEX strategy improvements:**

1. **Look-ahead bias in dividend gap backtests** — The dividend YAML has only paid dividends; cancelled events (GAZP 2022: 52.53 RUB recommended, rejected by shareholders) are missing. Backtest win rate appears >85% when the real rate including cancellations is ~65-75%. Fix: expand from 43 to 150+ events with a `status: paid|cancelled|reduced` field; include ALL board recommendations, not just successful payments.

2. **Survivorship bias from 2022 MOEX sanctions structural break** — Feb-Mar 2022 included a 25-day trading halt, artificial circuit breakers, and government-supported price floors. Any strategy calibrated on this data learns false patterns. Fix: add `exclude_periods: [("2022-02-24", "2022-04-01")]` to BacktestConfig; remove toxic symbols (GAZP, VTBR, SNGS) from active universe; never train walk-forward across the structural break.

3. **Vol target 0.19 systematically undersizes all MOEX positions** — US-calibrated target vs. MOEX's 0.35-0.60 annualized volatility causes VolTargetStep to hit the 0.25x floor on 60-70% of MOEX trades. Positions are too small to overcome transaction costs. Fix: set MOEX vol_target to 0.35-0.40 for ru_blue_chips in preset YAMLs. Quick config change, high impact.

4. **CBR rate regime timing error** — The market prices CBR decisions 1-3 weeks before announcement via the OFZ yield curve. Buying on the announcement buys AFTER the rally, then holds through mean-reversion. Fix: use OFZ yield curve slope (2Y-10Y spread) and RUONIA-OIS spread as leading indicators; only trade the surprise component (actual minus market-implied rate) on announcement day.

5. **Sector rotation forced into per-symbol combiner** — Sector rotation operates at portfolio level; the combiner operates per-symbol per-bar. Forcing it in creates contradictory signals, monthly whipsaw at rebalance, and backtest overfitting to macro events. Fix: implement as SectorAllocationStep in PositionSizingPipeline. Recovery cost if built in the wrong layer: HIGH — requires architectural refactor.

## Implications for Roadmap

Research across both streams points to a sequential structure within each stream and a clear dependency ordering between them. The v3.0 production readiness path has a well-specified 5-6 phase build order driven by dependency chains. The v2.0 MOEX strategy path has a 4-phase structure where data integrity must precede all strategy work.

### Phase 1: Schema, Config Foundation, and Data Integrity

**Rationale:** Schema definitions at Layer 0/1 must exist before any service can produce or consume them (zero behavior change, all existing tests pass). MOEX data integrity problems invalidate all backtest results regardless of strategy quality — they must be resolved before any strategy work produces reliable signal.

**Delivers (v3.0):** `GoNoGoReport`, `GoNoGoGateResult`, `SandboxMetricSnapshot`, `HealthCheckResult` frozen Pydantic schemas; `CycleLogEntry` extended with 3 optional None-default fields; `RolloutPhase` dataclasses with named profiles; `sandbox_metrics` TimescaleDB hypertable migration

**Delivers (v2.0):** Toxic symbols removed from MOEX segments (GAZP, VTBR, SNGS, IRAO, ALRS); dividend calendar expanded from 43 to 150+ events with `status` field and cancelled events; `exclude_periods` in BacktestConfig for Feb-Mar 2022; `vol_target` recalibrated to 0.35-0.40 in ru_*.yaml presets; `event_driven` disabled in backtest configs; `DEFAULT_STRATEGY_HOLD_BARS["dividend_gap"]` aligned to 60

**Avoids:** Look-ahead bias (Pitfall 1 v2.0), survivorship bias (Pitfall 2 v2.0), vol undersizing (Pitfall 3 v2.0), phantom event_driven signals in backtest (Pitfall 9 v2.0)

**Research flag:** Standard patterns — Pydantic models, Alembic migration, YAML config updates, T-Invest API calls against documented endpoints. No research phase needed.

### Phase 2: Risk Layer Wiring and Core Monitoring Services

**Rationale:** Risk layer rollout wiring (Phase 2 in ARCHITECTURE.md build order) must be validated in isolation before TradingLoop integration. Monitoring services depend on Phase 1 schemas and can be built and tested as standalone units before touching the live trading path. DividendGapStrategy wiring belongs here because it has data dependencies from Phase 1 and is the highest-confidence MOEX alpha source.

**Delivers (v3.0):** CircuitBreaker and PreTradeChecker accept optional rollout_phase param (None preserves existing behavior); SandboxMonitorService with TimescaleDB persistence and JSONL read logic; GoNoGoReporter pure function with configurable threshold evaluation; AnomalyDetector z-score alerting on equity moves and fill rate drops

**Delivers (v2.0):** DividendGapStrategy with corrected T+1 settlement date handling (last_buy_date vs. ex-date renamed and logic fixed); combiner `_EVENT_STRATEGIES` frozenset bypass so dividend signals are not diluted by ADX routing; rub_oil_regime.py wired into PositionSizingPipeline as a regime scale step

**Addresses:** Formalized go/no-go gate (table stakes), gradual rollout tightened risk limits (table stakes), dividend gap correctness

**Avoids:** Combiner diluting dividend signals (Pitfall 5 v2.0), T+1 settlement timing confusion (Pitfall 13 v2.0), monitoring logic in TradingLoop (anti-pattern from ARCHITECTURE.md)

**Research flag:** Standard patterns — sizing pipeline step extension, combiner frozenset addition, pure service with existing JSONL and TimescaleDB infrastructure. No research phase needed.

### Phase 3: API Endpoints, Kill Switch, and Health Monitoring

**Rationale:** API endpoints are thin wrappers over Phase 2 services — lowest-risk API change, follows established sub-router pattern. Kill switch and health pulse extend TradingLoop (the highest-regression-risk change) and are done after all dependencies are tested. CBR regime wiring belongs here because it requires a new OFZ data source and is a Phase 3 deliverable per PITFALLS.md.

**Delivers (v3.0):** GET /sandbox/metrics and GET /sandbox/gonogo REST endpoints; ProductionHealthMonitor 5-minute APScheduler job with broker ping and feed freshness checks; KillSwitch with strict order-cancellation sequence; TradingLoop extended with `_health_pulse_cycle()` and `activate_kill_switch()`; TelegramBotHandler extended with /gonogo, /health, /killswitch commands; MetricsCollector extended with ~5 new Prometheus gauges; REST API kill switch POST /api/v1/system/killswitch

**Delivers (v2.0):** CBR regime integration using OFZ yield curve slope as leading indicator (not CBR announcement date); MacroSnapshot extended with brent_close and usdrub_daily_change fields; BacktestEngine passes MacroSnapshot through `_process_bar()` to strategy kwargs

**Addresses:** REST API kill switch (table stakes), production health heartbeat (table stakes), CBR regime timing error (Pitfall 4 v2.0)

**Avoids:** Kill switch that only stops the scheduler (v3.0 Pitfall P3), CBR timing error (v2.0 Pitfall 4)

**Research flag:** CBR leading indicator design needs `/gsd:research-phase`. OFZ yield curve slope data source (MOEX ISS endpoint availability), RUONIA-OIS spread availability, and integration approach are open questions before implementation.

### Phase 4: Sector Rotation, Sizing Pipeline, and Preferred Share Arbitrage

**Rationale:** Sector rotation requires an explicit architectural decision (sizing step, not combiner strategy) and depends on clean universe (Phase 1) and calibrated vol target (Phase 1). Must be designed carefully — building it in the wrong layer has HIGH recovery cost. SectorAllocationStep must be analyzed alongside existing multiplicative sizing reduction to avoid MOEX positions clustering at the pipeline floor.

**Delivers (v2.0):** SectorAllocationStep in PositionSizingPipeline (NOT in StrategyCombiner) with 20-day linear weight transition; SectorClassifier at Layer 2 with static YAML symbol-to-sector mapping; Brent-in-RUB gate (BZ=F * USDRUB, 1-day lag) for energy sector; PreferredShareArbStrategy (long-only spread convergence on SBER/SBERP, TATN/TATNP; entry threshold z > 2.0; window excludes 2022 crisis period); multiplicative sizing reduction cap at pipeline level

**Addresses:** Sector rotation alpha on MOEX blue chips

**Avoids:** Sector rotation in wrong layer (Pitfall 6 v2.0 — HIGH recovery cost), Brent gate wrong currency/lag (Pitfall 11 v2.0), multiplicative sizing floor (Pitfall 14 v2.0), preferred share constant spread assumption (Pitfall 10 v2.0)

**Research flag:** Quantitative research needed on MOEX sector momentum validity and whether 3 years of post-2022 data is sufficient to avoid Pitfall 7 (overfitting to crisis). Preferred share cointegration test on post-2022 data must be run before implementing PreferredShareArbStrategy — if cointegration fails, skip the strategy.

### Phase 5: Dashboard, Automated Gate, and MOEX ML

**Rationale:** Dashboard pages and automated gate scheduling are read-only consumers of Phase 2-3 services and can be built any time after Phase 3 without affecting trading functionality. MOEX ML enablement depends on clean data (Phase 1), regime infrastructure (Phase 3), and a demonstrated positive equity baseline to validate against.

**Delivers (v3.0):** Streamlit sandbox validation progress page with per-criterion progress bars and per-day equity chart; automated daily gate evaluation via APScheduler job at 19:00 MSK; strategy signal-frequency anomaly alert via Prometheus counters; post-live slippage vs. backtest comparison weekly Telegram report

**Delivers (v2.0):** 10 new MOEX macro ML features (cbr_key_rate_level, cbr_rate_delta_3m, cbr_rate_direction, usdrub_return_20d, usdrub_zscore_60d, usdrub_vol_20d, brent_return_20d, brent_rub_spread, imoex_relative_21d, moex_turnover_zscore) gated by quality gates; ML walk-forward training excluding Feb-Mar 2022 with validation on 2024-2025 calm data

**Addresses:** Automated gate evaluation (P2 feature), signal anomaly detection (P2 feature), MOEX ML alpha

**Avoids:** Overfitting to 2022 crisis regime in ML (Pitfall 7 v2.0), MOEX ML on insufficient training samples

**Research flag:** MOEX ML with < 3 years of clean post-2022 data has insufficient sample volume. Research on transfer learning from US model and pooled-sector feature approaches is needed before implementation. Portfolio orchestrator design (for OFZ/equity combined backtest) also needs research on monthly rebalancing mechanics.

### Phase Ordering Rationale

- Schema and config must precede all service code — Layer 0/1 before Layer 4/6 (fundamental dependency direction)
- Data integrity prerequisites (Phase 1 v2.0) are not optional — backtesting with dirty data invalidates all iteration results and makes them incomparable to v2.0 baselines; 104 rejected iterations partially traced to these problems
- Risk layer wiring in isolation (Phase 2) before TradingLoop integration (Phase 3) limits regression surface to the most critical path changes
- Sector rotation (Phase 4) is deferred because its architectural decision must not be made under time pressure and has the highest recovery cost if built incorrectly — the codebase must have a confirmed positive MOEX equity baseline before adding this complexity
- Dashboard and automation (Phase 5) are genuinely optional for go-live — Telegram commands provide equivalent operational visibility for a single-operator system

### Research Flags

Phases needing `/gsd:research-phase` during planning:

- **Phase 3 (CBR leading indicator):** OFZ yield curve slope data source (MOEX ISS vs. separate endpoint), RUONIA-OIS spread availability, and pre-meeting consensus rate input mechanism are all open questions before CBR regime implementation
- **Phase 3 (Kill switch order cancellation):** BrokerRouter.cancel_all_pending() interface needs verification; BrokerBase may need extension; needs targeted API research before implementation
- **Phase 4 (Sector rotation alpha validation):** Whether MOEX sector momentum is statistically significant on 3 years of post-2022 data is unvalidated; academic threshold sources needed to avoid overfitting to 2022 crisis
- **Phase 5 (MOEX ML sample size):** Transfer learning from US model to MOEX to address the 3-year vs. 10-year training data gap; pooled-sector feature approach; validation methodology on limited clean data

Phases with standard patterns (skip research-phase):

- **Phase 1:** Pydantic schema additions, Alembic migration, YAML config updates, T-Invest API calls — all patterns established in codebase
- **Phase 2:** Sizing pipeline step addition, combiner frozenset extension, pure service with JSONL and TimescaleDB — all precedented in codebase
- **Phase 4 (preferred share arb):** Extends existing PairsStrategy pattern with MOEX-specific config — standard if cointegration test passes

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack (v3.0) | HIGH | Direct codebase audit. Every existing capability verified against source. New package versions confirmed on PyPI. Zero ambiguity on what exists vs. what is new. |
| Features (v3.0) | HIGH | Based on direct codebase audit of existing validation scripts + industry sources on trading system go-live requirements. Anti-features clearly identified with explicit rationale. |
| Architecture (v3.0) | HIGH | Layer-by-layer analysis of all 6 layers. Component boundaries verified against existing interfaces (TradingLoop, CircuitBreaker, PreTradeChecker, TelegramBotHandler). Dependency chain confirmed. |
| Pitfalls (v3.0) | HIGH (confirmed) / MEDIUM (severity quantification) | Pitfalls confirmed against codebase. Severity of sandbox false confidence on live MOEX execution quality (0.3-1.5% worse fills) is estimated from market microstructure knowledge, not measured. |
| Stack (v2.0) | HIGH | Zero new packages required; all building blocks confirmed in codebase against pyproject.toml. |
| Features (v2.0) | MEDIUM | Dividend closure statistics from financial media; CBR sector impact magnitudes from Forbes.ru/RBC; these require live empirical validation. |
| Architecture (v2.0) | HIGH | Derived from direct inspection of combiner.py, position_sizing_pipeline.py, engine.py, dividend_gap.py, cbr_calendar.py, rub_oil_regime.py; all layer boundaries confirmed. |
| Pitfalls (v2.0) | HIGH (structural) / MEDIUM (MOEX domain) | Structural pitfalls (vol target, T+1 settlement, combiner routing, sector rotation layer) verified in code. MOEX domain pitfalls (CBR pricing timing, Brent correlation lag) are research-informed estimates. |

**Overall confidence:** HIGH for v3.0 execution plan. MEDIUM-HIGH for v2.0 MOEX strategy alpha projections.

### Gaps to Address

- **Slippage budget quantification:** The 0.3-1.5% live slippage estimate for MOEX mid-caps needs validation against actual MOEX ISS order book data before finalizing go/no-go slippage threshold. This is a Phase 2 task, not a planning blocker.
- **Go/no-go threshold calibration:** The specific thresholds (uptime >= 99%, fill rate >= 95%, DD < 5%) need validation against walk-forward backtest distribution percentiles from existing PortfolioBacktestOrchestrator results. This is a Phase 2 deliverable.
- **OFZ yield curve data source:** CBR regime overlay (Phase 3 v2.0) requires OFZ 2Y-10Y yield data. MOEX ISS provides OFZ yield data but integration is unverified. This is the key open question for Phase 3 planning.
- **Preferred share spread stationarity post-2022:** SBER/SBERP spread was non-stationary in 2022 (preferred briefly exceeded common). Cointegration test on post-2022 data must be run before Phase 4 preferred arb implementation — if cointegration fails, skip the strategy entirely.
- **MOEX sector index ticker availability via ISS:** MOEX ISS confirmed for IMOEX; specific sector tickers (MOEXOG, MOEXFN, MOEXMM) need live API validation before Phase 4 sector rotation implementation.

## Sources

### Primary (HIGH confidence)

- Codebase direct inspection: `src/finalayze/` all 6 layers — `api/metrics.py`, `core/validation_logger.py`, `core/trading_loop.py`, `core/telegram_bot.py`, `risk/circuit_breaker.py`, `risk/pre_trade_check.py`, `risk/position_sizing_pipeline.py`, `execution/sandbox_tracker.py`, `strategies/combiner.py`, `strategies/dividend_gap.py`, `strategies/presets/ru_blue_chips.yaml`, `risk/rub_oil_regime.py`, `backtest/config.py`, `config/segments.py`, `pyproject.toml`
- WeasyPrint v68.1: https://pypi.org/project/weasyprint/ (released 2025-01-30)
- Jinja2 v3.1.6: https://pypi.org/project/Jinja2/
- APScheduler 4.x pre-release status: https://github.com/agronholm/apscheduler/issues/465
- aiogram v3: https://docs.aiogram.dev/en/latest/
- CBR key rate history: https://cbr.ru/eng/hd_base/KeyRate/
- MOEX ISS API: https://iss.moex.com/iss/reference/
- T-Invest API: t-tech-investments gRPC SDK proto definitions

### Secondary (MEDIUM confidence)

- FIA Best Practices for Automated Trading Risk Controls: https://www.fia.org/fia/articles/fia-releases-best-practices-automated-trading-risk-controls-and-system-safeguards
- NYIF: Trading System Kill Switch: https://www.nyif.com/articles/trading-system-kill-switch-panacea-or-pandoras-box
- Eventus: Algo Monitoring Real-Time Oversight: https://www.eventus.com/algo-monitoring-real-time-oversight-for-automated-ever-evolving-markets/
- MOEX dividend calendar: https://www.moex.com/ru/listing/dividend-yield.aspx
- Smart-Lab dividend calendar: https://smart-lab.ru/dividends/
- Dividend gap closure statistics (2024-2025): https://www.finam.ru/publications/item/istoricheski-lukoyl-i-tatneft-obladayut-potentsialom-bystrogo-vosstanovleniya-posle-dividendnogo-gepa-20250604-0900/
- CBR rate cut sector impact: https://www.forbes.ru/investicii/543288-raduznye-nadezdy-kakie-akcii-vyrastut-iz-za-snizenia-stavki-cb
- Common pitfalls of sector rotation: https://www.gwcindia.in/gigapro/blog/common-pitfalls-of-sector-rotation-and-how-to-avoid-them/
- Sector Rotation Myth (Molchanov 2024): https://onlinelibrary.wiley.com/doi/10.1002/ijfe.2882
- MOEX 2022 crisis structural break: Bloomberg (Russian Stocks Slump Most on Record, 2022-02-24)

### Tertiary (LOW confidence)

- Live MOEX slippage estimates (0.3-1.5% for illiquid instruments) — estimated from market microstructure knowledge; needs empirical validation during sandbox period
- Brent-MOEX energy correlation magnitude (0.6-0.8) and 1-3 day lag — general market knowledge; needs empirical validation against training data
- Preferred share spread ranges (SBER/SBERP: 5-15%) — historical patterns; current levels require live cointegration check on post-2022 data

---
*Research completed: 2026-03-21*
*Ready for roadmap: yes*

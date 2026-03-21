# Feature Research

**Domain:** Production trading monitoring, sandbox validation, go/no-go gates, gradual rollout
**Researched:** 2026-03-21
**Confidence:** HIGH — based on direct codebase audit + industry sources

---

## Context: What Already Exists

Before listing new features, understanding the existing baseline is essential to avoid overlap.

### Already Shipped (do not rebuild)

| Component | What it Does |
|-----------|-------------|
| `ValidationLogger` + `CycleLogEntry` | Appends JSONL per cycle: equity, drawdown, orders_submitted, orders_filled, errors, circuit_breaker_level |
| `generate_validation_report.py` | Reads JSONL, produces markdown with 4 criteria: trading_days >= 5, max_dd < 5%, fills >= 10, errors == 0 |
| `run_sandbox_validation.py` | Manual checklist script — NOT automated; prints steps, optionally queries /health |
| `SandboxPortfolioTracker` | Shadow accounting for coupon/dividend income not provided by Tinkoff sandbox |
| `CircuitBreaker` (3-level) | CAUTION/HALTED/LIQUIDATE on 5/10/15% daily drawdown. Sticky escalation, 2-day recovery for L2 |
| `TelegramAlerter` + `TelegramBotHandler` | Priority queue alerts; /status, /breakers, /stop commands |
| `MetricsCollector` (Prometheus) | Gauges for equity, drawdown, open positions, circuit level; histograms for slippage and fill latency |
| `PreTradeChecker` (11 checks) | Pre-trade pipeline: exposure, drawdown, market hours, PDT, correlation limits |
| `WorkMode` + `ModeManager` | debug/sandbox/test/real state machine with REAL guard token |
| REST API /health, /system/status, /system/errors | Component health (DB, Redis, Tinkoff), uptime, last 100 errors |
| Dashboard pages | portfolio, trades, signals, risk, system_status — 5 Streamlit pages |

### The Gaps (what v3.0 needs to add)

1. Validation report gates are too thin — only 4 criteria, no fill rate, no uptime %, no signal divergence
2. No automated go/no-go gate — report is generated manually, requires human to run script and read markdown
3. No slippage capture in sandbox mode — SandboxPortfolioTracker fills at last price, no slippage comparison
4. No sandbox-specific dashboard page showing validation progress over time
5. No gradual rollout configuration — tightened limits for Phase 1 live not configurable
6. No capital scaling mechanism — no safe path from 50K RUB to full capital based on validation period
7. No anomaly detection on live signal quality — expected vs actual signal ratio per strategy
8. No production health heartbeat — silent crash can run undetected; system goes quiet without alert
9. No REST API kill switch — Telegram /stop exists but needs programmatic equivalent
10. No post-go-live equity curve and slippage report vs backtest expectations

---

## Feature Landscape

### Table Stakes (Users Expect These)

Features that any production trading system must have. Missing = system cannot go live safely.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Formalized go/no-go gate with configurable thresholds | Any live trading system needs explicit criteria before committing real capital; current 4 criteria are insufficient and hardcoded | MEDIUM | Extend existing `generate_validation_report.py`; add uptime %, fill rate, signal divergence, slippage vs backtest thresholds. Keep as script + report file, make thresholds configurable via settings |
| Slippage capture in sandbox validation | Cannot gate on slippage quality without comparing sandbox fill_price to expected signal-time price | MEDIUM | Capture expected_price at signal generation in TradingLoop; compare to actual fill in TinkoffBroker; add slippage_bps field to CycleLogEntry or a separate trade-level log |
| Gradual rollout tightened risk limits config | First live run at 50K RUB must be more conservative than backtested 500K params; current config has no phase-aware limits | MEDIUM | New `RolloutPhase` config section in settings: max_position_pct, daily_loss_limit_rub, auto_stop_dd. Read in PreTradeChecker and CircuitBreaker thresholds |
| Production health heartbeat alert | If trading loop crashes silently (no new cycle for 30+ min during market hours), operator must know | LOW | Timer check in TradingLoop or separate watchdog; Telegram CRITICAL alert on silence. Depends on: TelegramAlerter CRITICAL tier (already exists) |
| REST API kill switch (POST /system/stop) | Telegram /stop is operational but requires mobile access; scripts and CI need a programmatic stop | LOW | Add authenticated POST /system/stop that transitions mode to halted state; already have ModeManager and auth middleware |
| Sandbox dashboard validation progress page | Operator must see validation status live without reading JSONL manually | MEDIUM | New Streamlit page reading ValidationLogger; show per-criterion progress bar, per-day equity chart, pass/fail verdict. Depends on: ValidationLogger (exists) |

### Differentiators (Competitive Advantage)

Features that set this system's production ops apart. Not required for go-live, but provide meaningful additional value.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Automated gate evaluation on daily schedule | Gate evaluates each evening at 19:00 MSK and sends Telegram summary with pass/fail delta vs yesterday; no manual report generation | MEDIUM | APScheduler job in TradingLoop daily_reset cycle; calls extracted `ValidationGate` class; sends formatted Telegram message. Directly extends existing scheduler pattern |
| Strategy signal-frequency anomaly alert | If a strategy fires 0 signals for 2+ consecutive trading days, alert operator — likely a data feed or configuration issue | MEDIUM | Per-strategy signal counters already in Prometheus; add threshold watcher in daily_reset; Telegram IMPORTANT alert |
| Post-live-launch slippage report vs backtest | Weekly Telegram message comparing realized slippage (bps) vs what BacktestCosts modeled; validates cost assumptions | LOW | Read from Prometheus trade_slippage_bps histogram; compare to BacktestConfig.slippage_bps constant; send formatted weekly summary |
| Capital scaling confirmation flow | After 10-day live validation period passes thresholds, operator receives Telegram confirmation request; one-tap approve unlocks higher position limits | MEDIUM | Gate evaluator checks live metrics; sends approval request; waits for /approve command; updates rollout config. Depends on: formalized gate, rollout config, TelegramBotHandler extension |

### Anti-Features (Commonly Requested, Often Problematic)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Fully automated live launch after sandbox PASS | Removing all human steps from sandbox to live transition sounds efficient | Any system that promotes itself to real-money trading without a human checkpoint is a liability. A PASS verdict does not account for exceptional market conditions (CBR rate shock, MOEX circuit breaker day) | Gate generates report + Telegram PASS notification; human explicitly runs capital-scaling script with confirmation token |
| Per-cycle backtest shadow for signal divergence | Complete real-time observability of whether strategies fire as expected | Running full backtest engine against live candles per-cycle adds 30-60 seconds compute per 5-minute cycle; starves the main trading loop; backtest engine requires full candle history loaded in memory | Log signals and candles during live; run divergence analysis on a 1-hour cron job separate from the trading loop |
| Automatic position liquidation on gate FAIL | If go/no-go fails, automatically unwind sandbox positions | In Tinkoff sandbox, positions have no real value. Auto-liquidation adds state management complexity with no benefit. In live, auto-unwind on a validation failure is high-risk (slippage, incomplete fills, gaps) | Report FAIL, send Telegram CRITICAL alert, require operator to review and decide on action |
| Multi-account capital splitting for rollout | Run 25% on account A, 75% on account B to de-risk rollout | Tinkoff Invest does not support multi-account management via same API token; adds reconciliation complexity with no clear benefit for a single-operator system | Single account gradual rollout via tighter position size limits (3% max position -> 5% -> 8%) controlled in RolloutPhase config |
| Dynamic risk limit adjustment based on real-time volatility during rollout | Auto-scale max position size based on MOEX volatility during live period | The pipeline already has VolTarget sizing step. Adding a second volatility gate at the config level creates conflicting signals — position gets adjusted by both pipeline and rollout config | Use VolTarget step for dynamic sizing; rollout config provides only a hard ceiling, not a dynamic one |

---

## Feature Dependencies

```
[Formalized Go/No-Go Gate]
    └──requires──> [Slippage Capture in Sandbox]
                       └──requires──> [expected_price logging in TradingLoop]
                       └──requires──> [slippage_bps field in ValidationLogger]
    └──requires──> [ValidationGate class] (extracted from generate_validation_report.py)
    └──enhances──> [Sandbox Dashboard Page]
    └──enhances──> [Automated Gate Evaluation on Schedule]

[Sandbox Dashboard Page]
    └──requires──> [ValidationLogger JSONL] (already exists)
    └──requires──> [Formalized Gate thresholds] (to show progress bars)

[Gradual Rollout Config]
    └──requires──> [RolloutPhase in Settings]
    └──requires──> [PreTradeChecker reads rollout limits]
    └──requires──> [CircuitBreaker tighter thresholds for phase 1]

[Capital Scaling Confirmation Flow]
    └──requires──> [Formalized Go/No-Go Gate] (gate must pass first)
    └──requires──> [Gradual Rollout Config] (config to update)
    └──requires──> [TelegramBotHandler] (new /approve command)

[Production Health Heartbeat]
    └──requires──> [TradingLoop last_cycle_timestamp tracking]
    └──enhances──> [TelegramAlerter CRITICAL tier] (already exists)

[Automated Gate Evaluation on Schedule]
    └──requires──> [ValidationGate class] (extracted gate logic)
    └──requires──> [APScheduler daily_reset job] (already exists in TradingLoop)

[Signal Anomaly Alert]
    └──requires──> [Prometheus strategy_signal_count counter] (already exists)
    └──requires──> [Per-strategy baseline signal rate config]
```

### Dependency Notes

- **Slippage capture requires expected_price logging:** TradingLoop must snapshot mid-price at signal time before order submission. Fill price comes back from TinkoffBroker. Difference goes to ValidationLogger as a new trade-level field. This touches 3 layers.
- **Formalized gate requires ValidationGate class extraction:** Currently gate logic is inline in a script. Extract to `finalayze.core.validation_gate.ValidationGate` (or `backtest/` layer) so the scheduler can call it programmatically.
- **Capital scaling requires gate to pass first:** Hard dependency. Capital scaling script should re-evaluate gate before allowing limit changes, not trust a stale report file.
- **Gradual rollout config must not conflict with VolTarget:** RolloutPhase provides a hard ceiling (e.g., 3% max position); VolTarget step provides dynamic sizing within that ceiling. They must compose, not compete.

---

## MVP Definition

This milestone is "production readiness" — the system has v2.0 shipped, so MVP means the minimum to safely go live.

### Launch With (v3.0 MVP — must have before go-live)

- [ ] Formalized go/no-go gate with configurable thresholds — without this there are no agreed pass criteria
- [ ] Slippage capture in sandbox — without this fill-rate and slippage gate criteria cannot be computed
- [ ] Gradual rollout tightened risk limits config — without this first live run uses backtested params at full capital scale, unsafe
- [ ] Production health heartbeat alert — without this a silent crash runs undetected for hours
- [ ] REST API kill switch (POST /system/stop) — Telegram /stop exists but CI/scripts need programmatic equivalent

### Add After Sandbox Validation Period (v3.0.x — during live phase 1)

- [ ] Sandbox dashboard page with gate progress — useful during sandbox period; manual report acceptable for MVP
- [ ] Automated gate evaluation on daily schedule — manual run is acceptable for MVP; automate for ongoing monitoring
- [ ] Post-live slippage report vs backtest — needs at least 5-10 real trades; add after first week live

### Future Consideration (v3.1+)

- [ ] Signal divergence tracker — high value but high complexity; needs architectural design before implementation
- [ ] Capital scaling confirmation flow — manual with confirmation token is acceptable for first 30 days; automate after stable live data

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Formalized go/no-go gate (configurable thresholds) | HIGH | LOW — extends existing script + extracts class | P1 |
| Slippage capture in sandbox | HIGH | MEDIUM — touches TradingLoop, TinkoffBroker, ValidationLogger | P1 |
| Gradual rollout tightened risk limits | HIGH | MEDIUM — new config section + PreTradeChecker + CircuitBreaker | P1 |
| Production health heartbeat alert | HIGH | LOW — timer check + existing Telegram alerter | P1 |
| REST API kill switch | MEDIUM | LOW — single endpoint + existing ModeManager | P1 |
| Sandbox dashboard validation page | MEDIUM | MEDIUM — new Streamlit page with progress bars | P2 |
| Automated gate evaluation on daily schedule | MEDIUM | LOW — APScheduler job + extracted ValidationGate | P2 |
| Strategy signal-frequency anomaly alert | MEDIUM | LOW — Prometheus + daily_reset watcher | P2 |
| Post-live slippage report vs backtest | LOW | LOW — Prometheus query + Telegram weekly job | P2 |
| Capital scaling confirmation flow | MEDIUM | MEDIUM — needs 10-day live data first | P3 |
| Signal divergence tracker (shadow backtest) | HIGH | HIGH — cron shadow engine + candle history store | P3 |

---

## What the Existing Validation Report Misses

The current `generate_validation_report.py` gates on only 4 criteria. The formalized gate for v3.0 should add:

| Criterion | Current State | v3.0 Target |
|-----------|--------------|-------------|
| Minimum trading days | >= 5 days (hardcoded) | >= 10 trading days (configurable); 2 calendar weeks covers both trend and MR conditions |
| Max drawdown | < 5.0% (hardcoded) | < 5.0% (configurable); tighten to 3% for first live phase |
| Round-trip fills | >= 10 (hardcoded) | >= 20 (configurable); 20 is minimum for statistical validity |
| Critical errors | == 0 (hardcoded) | == 0 (keep) |
| Uptime % | Not measured | >= 99% computed as (actual_cycles / expected_cycles) per trading day |
| Fill rate | Not measured | >= 95% computed as (orders_filled / orders_submitted) |
| Mean slippage | Not measured | < 30 bps; MOEX daily-bar systems typically see 10-20 bps |
| Strategy signal frequency | Not measured | >= 1 signal per enabled strategy per 5 days; detects silent strategy failure |

The 5-day minimum is also too short. Industry standard for sandbox validation before real capital is 10-15 trading days to cover at least 2 full MOEX trading weeks including any holiday-adjacent sessions. This gives ADX routing a chance to encounter both trend (ADX > 30) and mean-reversion (ADX < 20) conditions.

---

## Sources

- [FIA Best Practices for Automated Trading Risk Controls and System Safeguards](https://www.fia.org/fia/articles/fia-releases-best-practices-automated-trading-risk-controls-and-system-safeguards) — MEDIUM confidence (referenced from search; PDF requires access)
- [Eventus: Algo Monitoring Real-Time Oversight](https://www.eventus.com/cat-article/algo-monitoring-real-time-oversight-for-automated-ever-evolving-markets/) — MEDIUM confidence
- [LuxAlgo: Risk Management Strategies for Algo Trading](https://www.luxalgo.com/blog/risk-management-strategies-for-algo-trading/) — MEDIUM confidence
- [NYIF: Trading System Kill Switch](https://www.nyif.com/articles/trading-system-kill-switch-panacea-or-pandoras-box) — MEDIUM confidence
- [5 Key Metrics to Monitor in Automated Trading Systems](https://nurp.com/wisdom/5-key-metrics-to-monitor-in-automated-trading-systems/) — MEDIUM confidence
- Codebase audit of `validation_logger.py`, `generate_validation_report.py`, `run_sandbox_validation.py`, `sandbox_tracker.py`, `trading_loop.py`, `circuit_breaker.py`, `metrics.py`, `system.py`, `telegram_bot.py` — HIGH confidence (direct source)

---
*Feature research for: Production trading monitoring and go-live validation (v3.0 milestone)*
*Researched: 2026-03-21*

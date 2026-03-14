# Feature Research

**Domain:** Autonomous MOEX trading — stocks, OFZ bonds, coupons, LLM news, Telegram
**Researched:** 2026-03-14
**Confidence:** HIGH (codebase inspection) / MEDIUM (market landscape via web search)

---

## Feature Landscape

### Table Stakes (Users Expect These)

Features that any autonomous MOEX trading system must have. Missing these = system feels broken.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Autonomous order execution (MOEX stocks) | Core value proposition — "hands-off operation" | MEDIUM | TinkoffBroker exists; needs MOEX segment tuning + RUB sizing fix |
| Position sizing in RUB | MOEX positions sized wrong (USD math gives ~0.02% instead of 15%) | MEDIUM | Known blocker in PROJECT.md; RUB base in `position_sizing_pipeline.py` needed |
| MOEX-specific strategy parameters | Russian market has different volatility; US params produce poor signals on RU segments | MEDIUM | ru_* YAML presets exist but may need tuning; MOEX backtest results required |
| Positive MOEX backtest (walk-forward) | Can't deploy real money without validated backtests | HIGH | Walk-forward engine exists; needs MOEX candle data via TinkoffFetcher + Russian segment results |
| 3-level circuit breaker for MOEX portfolio | Prevent runaway losses in autonomous mode | LOW | `CircuitBreaker` exists and is wired; MOEX-specific thresholds to verify |
| ATR-based stop-losses (MOEX 1.2x uplift) | Standard risk control; MOEX is more volatile than US | LOW | Already implemented with 1.2x MOEX multiplier in backtest config |
| Telegram alert on trade fill | User needs to know trades happened without watching dashboard | LOW | `TelegramAlerter.on_trade_filled()` exists and fires correctly |
| Telegram alert on circuit breaker trip | User must be notified immediately when autopilot halts | LOW | `on_circuit_breaker_trip()` exists |
| Daily P&L summary via Telegram | Primary monitoring signal for hands-off operation | LOW | `on_daily_summary()` exists but known bug: shows zero (PROJECT.md) |
| Sandbox validation before real money | Safety gate — must prove system works before live | MEDIUM | sandbox mode + TinkoffBroker sandbox endpoint wired; need to run N days |
| MOEX trading hours gate | MOEX main session 07:00-15:40 UTC, evening 16:05-20:50 UTC | LOW | Pre-trade check item 1; needs MOEX-specific schedule (not US hours) |
| T-Invest gRPC candle data | yfinance cannot fetch MOEX tickers (SBER, GAZP, etc.) | LOW | TinkoffFetcher fully wired; constraint is hard-enforced |
| FIGI-based instrument identification | MOEX uses FIGI codes, not ticker symbols, for orders | LOW | InstrumentRegistry with FIGI mapping exists |
| Lot-size-aware ordering | MOEX lots (e.g. SBER = 10 shares); wrong sizing = order rejection | LOW | Must be enforced in order construction |
| Autonomous OFZ bond trading (4-layer) | Bond trading is explicitly in scope; user requirement | HIGH | BondCycleProcessor + 3 bond strategies + LayerLedger exist; need TradingLoop wiring |
| Coupon schedule tracking | Bonds pay coupons; system must avoid buying in ex-coupon period | MEDIUM | `fetch_bond_coupons()` exists; ex-coupon check in bond pre-trade validation |
| Coupon Telegram alert | User wants to know when income arrives | LOW | `on_coupon_received()` exists in TelegramAlerter |
| CBR key rate as macro input | Bond strategies (BondDurationRotationStrategy, CBREventStrategy) require CBR data | LOW | CBRFetcher + MacroContextProvider exist; MacroCacheService needs wiring |
| LLM news analysis (Russian) | Event-driven strategy for MOEX needs RU sentiment; English news misses key events | HIGH | NewsAnalyzer + sentiment_ru.txt prompt exist; no live Russian news source wired |

---

### Differentiators (Competitive Advantage)

Features that set Finalayze apart from simple trading bots. These are the "AI" in "AI-powered."

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| 4-layer OFZ bond portfolio architecture | Separates capital by time horizon (Core/Strategic/Tactical/Short) with regime-adaptive rebalancing; most bots treat bonds as a single undifferentiated asset | HIGH | Architecture exists (`bond_duration_rotation`, `bond_carry`, `cbr_event` strategies, DV01 sizing, LayerLedger) |
| CBR regime classifier driving bond decisions | Automatically adjusts duration target based on RUONIA-key rate gap + last CBR meeting decision + CPI; removes subjectivity from rate-cycle positioning | MEDIUM | `classify_regime()` in `bond_duration_rotation.py` fully implemented |
| DV01-budget-aware bond position sizing | Size bonds by interest rate risk (bps impact), not just capital; avoids overconcentration in long-duration instruments | MEDIUM | `DV01BudgetStep` implemented; `EqualWeightBondSizer` for OFZ-PK floaters |
| Yield stop-loss (not ATR) for bonds | Bond price moves are driven by yield changes; ATR is wrong for fixed income | LOW | `YieldStop` with regime-adaptive thresholds implemented |
| CBR event strategy (pre-meeting momentum) | Trades OFZ around CBR rate meetings; entry 2-7 days before, mechanical exit T+1/T+2; statistically grounded arbitrage of announcement drift | MEDIUM | `CBREventStrategy` fully implemented; ~24 trades per 3-year backtest |
| Claude Sonnet LLM analysis with Russian prompt | Multi-step reasoning (extract facts → identify affected companies → assess market impact → synthesize) in Russian; far above keyword-based sentiment | MEDIUM | Prompt exists; needs live news source |
| Sanctions proximity scoring on Russian equities | Geopolitical risk of individual stocks (GAZP 0.8, SBER 0.3) reduces confidence on sanctions-related events | LOW | Implemented in `event_driven.py`; reusable once news feed is wired |
| ADX regime routing (MOEX segments) | Separates trend-following and mean-reversion strategies by market regime; prevents fighting trend with MR strategies | LOW | Exists for US; needs validation/tuning for MOEX volatility regime thresholds |
| ML ensemble as signal reinforcer | XGBoost + LightGBM + CatBoost meta-learner boosts confirmed signals rather than generating standalone trades | HIGH | Architecture exists; models need training on MOEX features |
| RUB-oil regime overlay | Russian market is strongly correlated to Brent crude; RUB/oil regime changes trading posture | LOW | `rub_oil_regime.py` + `commodity_currency.py` in risk module |
| MacroCacheService with CBR-day force-refresh | Macro data refreshes at market open + forced refresh after CBR press conference (15:30 MSK, not 13:30 spread-spike window) | LOW | Designed and documented; needs implementation |

---

### Anti-Features (Commonly Requested, Often Problematic)

Features that seem good but create more problems than they solve for this system.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Real-time tick-level MOEX trading (HFT) | More trades = more opportunity | System operates on daily/intraday bars; tick-level requires co-location, different risk model, and different API quota tier. Out of scope in PROJECT.md. | Daily + intraday candles via TinkoffFetcher; no tick data |
| Multi-account management | Separate accounts for different strategies | Complicates capital allocation, reporting, and circuit breaker logic significantly. Single T-Invest account is sufficient for MVP. | Layer-based capital allocation (LayerLedger) within one account |
| Derivatives/futures trading | Leverage boosts returns | Futures on MOEX require separate margin account, different risk model (VaR/margin calls vs stop-losses), and increase max-loss beyond the 10% hard limit. | Equity + bond portfolio with Kelly sizing provides sufficient leverage-free return |
| Mobile app | Monitoring on phone | Streamlit dashboard + Telegram alerts cover all operational needs. Building a native app is months of work with no trading value. | Telegram bot (alerts + /status command) |
| Full Telegram trading commands (/buy, /sell) | Manual override capability | Manual orders bypass the pre-trade pipeline and circuit breakers, violating the "autonomous with risk limits" design contract. Commands should be view-only. | `/status` command showing positions/P&L; manual trades via T-Invest app directly |
| Automated ML model retraining in production | "Self-learning" system | Continuous retraining without human review creates model drift risk and potential for silent degradation. Walk-forward + sequential bootstrap already handles the training/validation split. | Scheduled offline retraining via `train_models.py`, human review before deployment |
| Custom UI for ML model configuration | Power-user feature | CLI scripts (`train_models.py`, `run_iteration.py`) are sufficient; a UI adds frontend complexity with no trading value. | CLI + structured YAML presets |
| News from Telegram crypto channels | Broader news coverage | Crypto channels produce noisy signals for equity/bond trading; majority of content is irrelevant or adversarially biased. | T-Invest news API + RBC/Interfax/TASS RSS (financial-focused, regulated sources) |
| Cryptocurrency trading | MOEX users want one platform for all assets | Crypto is not available on MOEX. Integrating a separate exchange (Binance, etc.) breaks the "T-Invest only" broker constraint and the Russian regulatory framework. | Out of scope; clearly documented in PROJECT.md |

---

## Feature Dependencies

```
[MOEX stock autonomous trading]
    └──requires──> [RUB position sizing fix]
    └──requires──> [MOEX strategy tuning (ru_* YAML presets)]
    └──requires──> [Positive MOEX walk-forward backtest]
    └──requires──> [MOEX trading hours gate]
    └──requires──> [Lot-size-aware ordering]

[Positive MOEX walk-forward backtest]
    └──requires──> [TinkoffFetcher candle data (already exists)]
    └──requires──> [ru_* segment definitions (already exist)]

[OFZ bond autonomous trading]
    └──requires──> [BondCycleProcessor wired into TradingLoop]
    └──requires──> [MacroCacheService with CBR refresh schedule]
    └──requires──> [CBRFetcher (already exists)]
    └──requires──> [Bond instrument registry (OFZ-PD + OFZ-PK)]
    └──requires──> [Bond backtest showing positive PnL]

[MacroCacheService]
    └──requires──> [MacroContextProvider + CBRFetcher (already exist)]

[CBR event strategy (live)]
    └──requires──> [MacroCacheService]
    └──requires──> [BondCycleProcessor]

[LLM news analysis from Russian sources]
    └──requires──> [Russian news fetcher (RBC/Interfax/TASS RSS or T-Invest news)]
    └──requires──> [NewsAnalyzer + sentiment_ru.txt prompt (already exist)]
    └──requires──> [NewsArticle language routing (already exists)]

[Event-driven strategy (live)]
    └──requires──> [LLM news analysis from Russian sources]
    └──requires──> [Sentiment cache updated on news cycle]

[Telegram alerts]
    └──requires──> [TelegramAlerter (already exists — needs daily P&L bug fix)]

[Coupon Telegram alert]
    └──requires──> [OFZ bond autonomous trading]
    └──requires──> [Coupon detection in bond cycle (needs implementation)]

[Daily P&L summary (correct)]
    └──requires──> [MOEX portfolio equity calculation in RUB]
    └──requires──> [RUB→USD conversion for multi-market totals]

[Real money deployment]
    └──requires──> [Sandbox validation (N days without critical errors)]
    └──requires──> [All circuit breakers tested]
    └──requires──> [Telegram alerts verified working]

[ML ensemble on MOEX]
    └──requires──> [MOEX training data via TinkoffFetcher]
    └──requires──> [ML model training for ru_* segments]
    └──requires──> [Walk-forward validation + quality gates]
    └──enhances──> [MOEX stock autonomous trading]
```

### Dependency Notes

- **RUB position sizing is a blocker for everything else**: MOEX positions at 0.02% instead of 15% means the system is effectively in "paper mode" even with real money. This must be fixed before any meaningful backtest or live test.
- **MOEX backtest must precede sandbox**: Cannot validate autonomous operation without known-good historical performance on Russian segments.
- **Bond cycle wiring is an independent path**: BondCycleProcessor can be wired into TradingLoop independently of the equity MOEX fixes. Bond strategies are better isolated from equity risk.
- **LLM news conflicts with event_driven timeline**: The event_driven strategy is disabled and requires a live news feed. This is a differentiator feature (not table stakes), so it should come after the equity and bond foundations are stable.
- **MacroCacheService is a bond prerequisite**: Without daily CBR macro data, all 3 bond strategies fall back to NEUTRAL regime (sub-optimal decisions).

---

## MVP Definition

### Launch With (v1 — MOEX MVP)

Minimum to achieve "autonomous MOEX trading with acceptable risk."

- [ ] RUB position sizing fix — without this, no MOEX trade is sized correctly
- [ ] MOEX walk-forward backtest with positive PnL (ru_blue_chips + ru_energy at minimum)
- [ ] MOEX strategy parameter tuning (tighter RSI thresholds, wider BB for higher RU volatility)
- [ ] OFZ bond autonomous trading: BondCycleProcessor wired into TradingLoop with all 4 layers
- [ ] MacroCacheService with daily CBR refresh + CBR-day force-refresh
- [ ] Bond backtest with positive PnL (OFZ-PD duration rotation + OFZ-PK carry)
- [ ] Telegram daily P&L summary bug fix (currently shows zero)
- [ ] Sandbox validation: N days of autonomous operation without critical errors
- [ ] Telegram alerts for circuit breaker events working
- [ ] Coupon payment detection and alert

### Add After Validation (v1.x)

Features to add once core autonomous trading is proven in sandbox.

- [ ] Russian news feed (T-Invest API news or RBC/Interfax RSS) — trigger: sandbox stable, event_driven strategy ready to enable
- [ ] LLM-powered Russian news analysis wired to event_driven strategy — trigger: news feed working
- [ ] Real money deployment (small account, 500K RUB) — trigger: sandbox proves itself over 2+ weeks
- [ ] Telegram `/status` command — trigger: real money deployment, user wants on-demand check

### Future Consideration (v2+)

Features to defer until MVP is generating positive live returns.

- [ ] ML ensemble for MOEX (ru_* segments) — training requires 3+ years of MOEX candle history; quality gates must pass before enablement
- [ ] PEAD strategy for MOEX (requires Russian earnings surprise data source — not readily available)
- [ ] Pairs trading on MOEX (requires careful cointegration testing on RU pairs; sanctions-driven correlation breaks are a risk)
- [ ] Telegram financial channel monitoring for news — filter noise from signal is research work
- [ ] Portfolio optimization with HRP weights across MOEX segments

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| RUB position sizing fix | HIGH | LOW | P1 |
| MOEX walk-forward backtest (positive) | HIGH | MEDIUM | P1 |
| MOEX strategy parameter tuning | HIGH | MEDIUM | P1 |
| OFZ bond cycle wired into TradingLoop | HIGH | MEDIUM | P1 |
| MacroCacheService | HIGH | LOW | P1 |
| Bond backtest (positive) | HIGH | MEDIUM | P1 |
| Telegram daily P&L bug fix | HIGH | LOW | P1 |
| Sandbox validation | HIGH | LOW | P1 |
| Coupon payment detection + alert | MEDIUM | LOW | P1 |
| MOEX trading hours gate validation | HIGH | LOW | P1 |
| Russian news feed integration | HIGH | MEDIUM | P2 |
| LLM news → event_driven strategy | HIGH | MEDIUM | P2 |
| Telegram /status command | MEDIUM | LOW | P2 |
| Real money deployment | HIGH | LOW (ops) | P2 |
| ML ensemble for MOEX | MEDIUM | HIGH | P3 |
| PEAD strategy for MOEX | LOW | HIGH | P3 |
| Pairs trading for MOEX | LOW | HIGH | P3 |
| Telegram financial channel monitoring | MEDIUM | HIGH | P3 |

**Priority key:**
- P1: Must have for MOEX MVP launch
- P2: Should have, add after sandbox validation
- P3: Nice to have, defer to v2+

---

## Competitor Feature Analysis

The MOEX autonomous trading space is thin. Most Russian retail algo-trading tools are simple rule-based bots. Direct comparisons are limited.

| Feature | EIDiamond invest-bot (GitHub) | MOEX ALGOPACK (platform) | Finalayze Approach |
|---------|-------------------------------|--------------------------|-------------------|
| Bond trading | Not supported | Backtesting data only | 4-layer OFZ portfolio with CBR regime routing |
| LLM news analysis | Not supported | Not supported | Claude Sonnet with Russian financial prompt |
| Telegram alerts | Order details + daily summary | Not applicable | 9 alert types including coupon, CBR meeting, stop-loss |
| Risk controls | Min balance + lot limits | Backtesting only | 11-check pre-trade pipeline + 3-level circuit breaker + DV01 budget |
| Sandbox mode | Not mentioned | Available | T-Invest sandbox with separate endpoint |
| CBR rate sensitivity | Not addressed | Not addressed | MacroContextProvider + regime classifier drives bond strategy |
| Coupon income | Not tracked | Not addressed | `on_coupon_received()` alert + ex-coupon period filter |

---

## Sources

- Codebase inspection (HIGH confidence): `src/finalayze/strategies/bond_carry.py`, `bond_duration_rotation.py`, `cbr_event.py`, `risk/dv01_sizing.py`, `risk/yield_stop.py`, `core/alerts.py`, `data/fetchers/tinkoff_data.py`, `data/fetchers/cbr.py`, `analysis/news_analyzer.py`
- Design documents (HIGH confidence): `docs/plans/2026-03-12-bond-tradingloop-integration-design.md`, `docs/design/RISK.md`, `docs/design/NEWS_PIPELINE.md`, `docs/design/STRATEGIES.md`
- Project context (HIGH confidence): `.planning/PROJECT.md`
- [EIDiamond invest-bot — GitHub](https://github.com/EIDiamond/invest-bot) (MEDIUM confidence, community bot example)
- [T-Invest API — RussianInvestments investAPI](https://github.com/RussianInvestments/investAPI) (MEDIUM confidence, official SDK)
- [MOEX bonds market overview](https://www.moex.com/en/bondization) (MEDIUM confidence)
- [FIA Best Practices for Automated Trading Risk Controls](https://www.fia.org/sites/default/files/2024-07/FIA_WP_AUTOMATED%20TRADING%20RISK%20CONTROLS_FINAL_0.pdf) (MEDIUM confidence, general industry standard)
- Web search: MOEX algorithmic trading 2025, T-Invest Telegram integration, Russian news bots (LOW confidence, limited MOEX-specific results)

---

*Feature research for: Autonomous MOEX trading — stocks, OFZ bonds, coupons, LLM news, Telegram*
*Researched: 2026-03-14*

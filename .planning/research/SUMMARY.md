# Project Research Summary

**Project:** v2.0 MOEX Profitability
**Domain:** MOEX-native equity alpha — dividend gap, CBR regime overlay, sector rotation, preferred share arbitrage, Russian macro ML features
**Researched:** 2026-03-20
**Confidence:** MEDIUM-HIGH (stack + architecture: HIGH from direct codebase inspection; features + pitfalls: MEDIUM on MOEX-specific domain knowledge)

## Executive Summary

The v2.0 MOEX profitability milestone is a focused extension of an already-working system, not a greenfield build. The existing Python 3.12 codebase already contains all required libraries, data fetchers, strategy framework hooks, and ML pipeline infrastructure. Every planned feature — dividend gap expansion, CBR regime gating, sector rotation, preferred share arbitrage, and MOEX-specific ML features — can be implemented using the current dependency set with zero new pip packages. The only genuinely new external data is MOEX sector index candles (8 tickers: MOEXOG, MOEXFN, MOEXMM, etc.), fetched via the already-operational `MoexISSFetcher`. Work is purely at the application layer: new strategy modules, ML feature extensions, combiner configuration, and risk pipeline wiring.

The recommended build sequence is dictated by a hard dependency chain. Data and parameter foundations must be fixed first — the current codebase has three confirmed problems invalidating all MOEX backtest results: (1) `vol_target: 0.19` (US-calibrated) destroys MOEX position sizes because MOEX blue chip volatility is 35-60% annualized, not 19%; (2) the dividend calendar has only 43 events, all of which are paid dividends (cancelled events like GAZP 2022 are missing, introducing look-ahead bias); and (3) the Feb-Mar 2022 MOEX closure distorts vol estimates 3-5x and teaches false mean-reversion patterns. These problems account for a substantial portion of the 104 rejected backtest iterations and must be resolved before any new strategy work begins. After the data foundation is clean, existing-but-unconnected strategies should be wired (DividendGapStrategy calendar, CBRStrategyWrapper, rub_oil_regime), then new macro-gated strategies added, and finally portfolio-level allocation and ML features layered on.

The highest-severity architectural risk is sector rotation placement. Sector rotation is a portfolio-level signal that must NOT enter the per-symbol `StrategyCombiner` — doing so creates contradictory symbol-level signals, monthly rebalancing whipsaw, and backtest overfitting. It must be implemented as a `SectorAllocationStep` in `PositionSizingPipeline`. If built wrong, recovery cost is HIGH (full architectural refactor). CBR regime gating has a subtler risk: using the CBR announcement date as a trading signal is lagging (the market prices decisions 1-3 weeks in advance via OFZ yield curve movements); the regime signal must be OFZ curve slope and RUONIA spread, not raw rate decisions.

## Key Findings

### Recommended Stack

Zero new packages required. The codebase already provides every building block needed for v2.0. See `.planning/research/STACK.md` for full module mapping and the "What NOT to Add" table.

**Core technologies (all existing — no new pip installs):**
- `t-tech-investments` gRPC SDK: MOEX candles, dividends, instruments — the only valid source for MOEX tickers; `TinkoffFetcher.get_dividends()` already operational
- `CBRFetcher` + `MacroSnapshot` + `MacroCacheService`: CBR key rate, USDRUB, CPI, RUONIA — already operational; `MacroSnapshot` needs two new fields (`brent_close`, `usdrub_daily_change`)
- `MoexISSFetcher`: MOEX ISS API for IMOEX and any sector index — needs sector tickers added (MOEXOG, MOEXFN, MOEXMM, MOEXCN, MOEXTL, MOEXCH, MOEXEU, MOEXIT)
- `PairsStrategy` + `statsmodels`: Kalman hedge ratio + Engle-Granger cointegration — needs MOEX pref/ord pair configuration (SBER/SBERP, TATN/TATNP)
- `XGBoost` + `LightGBM` + `CatBoost` + `ml/features/technical.py`: ML ensemble with 45 features — needs 10 new MOEX macro features (`cbr_key_rate_level`, `cbr_rate_delta_3m`, `usdrub_return_20d`, `usdrub_zscore_60d`, `brent_return_20d`, `brent_rub_spread`, `imoex_relative_21d`, `moex_turnover_zscore`, plus 2 more)
- `QuantLib`: bond math already in use for OFZ allocation layer

**Only new application-layer data schema addition:** `MoexMarketData.sector_index_candles: dict[str, tuple[Candle, ...]]` field.

### Expected Features

See `.planning/research/FEATURES.md` for full details including per-symbol dividend gap closure statistics (ROSN: same-day, LKOH: ~10 days, TATN: 19-255 days bimodal) and CBR rate impact mechanics (8 meetings/year, 100-200bps moves vs. Fed's 25bps).

**Must have (table stakes for positive MOEX Sharpe — P0/P1):**
- Universe cleanup — remove GAZP, VTBR, SNGS, IRAO, ALRS from active segments (account for ~60% of negative PnL)
- `vol_target` recalibration — update ru_*.yaml from 0.19 to 0.35-0.40; quick fix, high impact
- Expanded dividend calendar — 150+ events from T-Invest API including ALL board recommendations and cancelled events (not just paid dividends)
- Dividend gap strategy tuning — per-symbol `max_hold_bars`, regime filter, closure-rate-based confidence scaling
- CBR rate regime gating — wire existing `cbr_calendar.py` into combiner (code exists, not connected)
- Brent energy gate — wire existing `rub_oil_regime.py` into combiner for energy sector (code exists, not connected)
- RUB crisis brake — halt new equity longs when RUB/oil correlation < 0.1

**Should have (competitive differentiators — P2):**
- Preferred share arbitrage (SBER/SBERP, TATN/TATNP) — configure existing `PairsStrategy` for MOEX pref/ord pairs; long-only constraint (T-Invest retail cannot short)
- Brent-conditional sector rotation — as a `SectorAllocationStep` in sizing pipeline, NOT a combiner signal
- CBR meeting pre-positioning — extend `generate_cbr_signal()` for pre-meeting entry on financials when cut is expected; requires manual consensus rate input (8x/year)

**Defer to v2.1+ (require positive equity Sharpe baseline first — P3):**
- ML ensemble with Russian macro features — 10 new features are designed and ready; defer until clean baseline data exists; overfitting risk is high on 3-year MOEX history
- OFZ PK-to-PD rotation — existing OFZ-PK carry Sharpe is already +1.14; rotation is optimization, not MVP
- Portfolio-level allocation optimizer — implement simple fixed 40/60 OFZ/equity split first

**Anti-features (explicitly excluded):**
- Intraday dividend gap scalping — gap closure is a multi-day phenomenon; daily bars are correct
- Automated CBR consensus scraping — fragile; use manual YAML input (8x/year = acceptable burden)
- Full sector rotation optimizer — overfitting trap on short post-2022 MOEX history; use binary gates instead
- Preferred share short-selling — T-Invest retail cannot short; long-only on undervalued leg only
- ML with sanctions text features — N=3 sanctions events is insufficient for ML; use rub_oil_regime as proxy

### Architecture Approach

The 7-layer dependency architecture (L0 types → L6 orchestration) requires no structural changes. All new components slot into existing layers without violating downward-only import rules. The key design constraint is that strategies receive macro context via `**kwargs` to `generate_signal()` using the existing `regime_state` kwarg pattern — never by importing from L2 directly. This is enforced by passing a `MacroSnapshot` object through the backtest engine's `_process_bar()` method. See `.planning/research/ARCHITECTURE.md` for complete component boundaries, data flow diagrams, layer violation analysis, and the explicit anti-pattern list.

**Major new and modified components:**

| Component | Layer | Type | Key Responsibility |
|-----------|-------|------|--------------------|
| `DividendCalendarLoader` | L2 | New | Bulk-load ex-div dates from Tinkoff for all ru_* symbols; populates `DividendGapStrategy` before backtest |
| `SectorClassifier` | L2 | New | Maps symbols to sectors from static YAML; avoids hardcoded `if symbol in [...]` anti-pattern |
| `PreferredShareMapper` | L2 | New | Maps common/preferred share FIGIs (SBER↔SBERP, TATN↔TATNP) |
| `MacroSnapshot` | L2 | Extend | Add `brent_close`, `usdrub_daily_change` fields for macro-gated strategies |
| `CBRRegimeStep` + `BrentGateStep` | L4 | New | Add to `PositionSizingPipeline`; scale positions by CBR cycle and Brent price — do NOT block signals entirely |
| `SectorRotationStrategy` / `SectorAllocationStep` | L4 | New | Portfolio-level sizing overlay in pipeline, NOT a combiner signal |
| `PreferredShareArbStrategy` | L4 | New | Long-only spread convergence on pref/ord pairs; extends `PairsStrategy` pattern |
| `PortfolioAllocator` | L4 | New | 40% OFZ / 60% equity capital split with RUB crisis brake |
| `PortfolioBacktestOrchestrator` | L5 | New | Runs bond + equity engines separately, merges PnL at portfolio level |

**Suggested build order within each phase** (from ARCHITECTURE.md §"Suggested Build Order"):
Phase 1: Data (DividendCalendarLoader, SectorClassifier, PreferredShareMapper, MacroSnapshot extensions) → Phase 2: Wire existing strategies (DividendGap calendar, CBRStrategyWrapper, rub_oil_regime sizing) → Phase 3: New strategies (SectorAllocationStep, PreferredShareArbStrategy, CBRRegimeStep, BrentGateStep) → Phase 4: Portfolio layer (PortfolioAllocator, RUBCrisisBrake, PortfolioBacktestOrchestrator).

### Critical Pitfalls

Full analysis with detection criteria, recovery costs, and phase mapping in `.planning/research/PITFALLS.md`. Seven critical pitfalls identified.

1. **Look-ahead bias in dividend calendar** — YAML contains only paid dividends; cancelled events (GAZP 2022: 52.53 RUB recommended, rejected) are missing. Backtest win rate appears >85% when real rate is ~65-75%. Fix: rebuild calendar with `status: paid|cancelled|reduced` field; include ALL board recommendations 2020-2025. Must resolve in Phase 1 before any dividend gap backtesting.

2. **Vol target 0.19 destroys MOEX position sizes** — US-calibrated target vs. MOEX's 35-60% annualized vol causes VolTargetStep to hit the 0.25x floor on 60-70% of MOEX trades. Positions too small to overcome transaction costs. Fix: set `vol_target: 0.35-0.40` in ru_*.yaml. Quick config change, do in Phase 1.

3. **Survivorship bias from 2022 MOEX structural break** — Feb-Mar 2022 closure + circuit breakers distort vol 3-5x and teach false mean-reversion patterns on MOEX-supported price floors. Fix: add `exclude_periods: [("2022-02-24", "2022-04-01")]` to `BacktestConfig`; remove toxic symbols first; never train walk-forward across the sanctions break.

4. **CBR rate regime timing error** — Using the CBR announcement as a trading signal is lagging; the market prices decisions 1-3 weeks in advance. Buying on "CBR cut" announcement buys AFTER the rally. Fix: use OFZ yield curve slope (2Y-10Y spread) and RUONIA-OIS spread as leading indicators; only trade the *surprise* component (actual vs. market-implied rate) on announcement day.

5. **Dividend gap signals diluted by ADX combiner routing** — `dividend_gap` is not in `_MOMENTUM_STRATEGIES` or `_MR_STRATEGIES` frozensets; its classification is accidental. On ex-div days, other strategies generate HOLD/SELL that average down the BUY below `min_combined_confidence`. Fix: create `_EVENT_STRATEGIES = frozenset({"dividend_gap", "cbr_calendar", "event_driven"})` that bypasses ADX routing; or give dividend_gap weight >= 0.40 on MOEX segments.

6. **Sector rotation in the wrong architectural layer** — Sector rotation is a portfolio-level allocation signal; forcing it into the per-symbol combiner creates contradictory signals (sector says "buy energy", technicals say "sell ROSN"), monthly whipsaw on rebalance day, and backtest overfitting to macro events. Fix: implement as `SectorAllocationStep` in `PositionSizingPipeline`. Recovery cost if built wrong: HIGH (architectural refactor).

7. **T+1 settlement date confusion** — `DividendGapStrategy` triggers on `ex_date` from YAML, but Tinkoff's "ex_date" is actually the last buy date; the actual price gap appears the NEXT bar. Fix: rename YAML field to `last_buy_date`; strategy buys one bar after. Easy fix but must be correct before any dividend gap backtest.

**Additional pitfalls to track (moderate):**
- Pitfall 7: Overfitting to 2022-2024 CBR hiking cycle — regime parameters calibrated on crisis data won't fire in normal markets
- Pitfall 10: Preferred share spread non-stationarity — SBERP briefly exceeded SBER in 2022; use z-score window excluding crisis period; entry threshold > 2 std (not 1.5)
- Pitfall 11: Brent gate must use Brent-in-RUB (Brent * USDRUB), not USD Brent; apply 1-day lag
- Pitfall 12: `DEFAULT_STRATEGY_HOLD_BARS["dividend_gap"]` = 15 conflicts with strategy's 60-bar expectation; engine will force-close positions before gap closes
- Pitfall 14: Multiplicative sizing steps (vol target × regime × sector) compound to pipeline floor; cap total scale or use additive sector adjustment

## Implications for Roadmap

Based on combined research, a 4-phase structure is strongly indicated by the dependency chain. Each phase must complete before the next can produce valid results. The architecture research (ARCHITECTURE.md §"Suggested Build Order") and pitfall research (PITFALLS.md §"Phase-Specific Warnings") both independently converge on the same ordering.

### Phase 1: Data Foundation and Parameter Cleanup

**Rationale:** All current MOEX backtest results are invalid due to vol_target miscalibration, toxic symbols, 2022 data contamination, and a dividend calendar with only paid events. This is the root cause of 104 rejected iterations. No new strategy work will produce reliable signal until these are fixed. This phase has zero dependencies on other new work and unblocks everything downstream.

**Delivers:** Clean baseline — valid walk-forward results for existing ru_* segments, properly sized positions (5-15% of equity instead of 0.5-2%), toxic symbols removed from active universe, 150+ event dividend calendar ready for Phase 2 use.

**Addresses features (from FEATURES.md):**
- Universe cleanup: GAZP, VTBR, SNGS, IRAO, ALRS removal
- `vol_target` recalibration: all ru_*.yaml presets updated (0.19 → 0.35-0.40)
- Dividend calendar expansion: 43 → 150+ events with `status: paid|cancelled|reduced` field
- `exclude_periods` for Feb-Mar 2022 in `BacktestConfig`
- `event_driven` strategy disabled in backtest presets (no live news feed; phantom signals)
- `DEFAULT_STRATEGY_HOLD_BARS["dividend_gap"]` aligned to 60 bars (not 15)

**Avoids pitfalls:** 1 (look-ahead), 2 (vol target), 3 (survivorship), 9 (phantom event_driven), 12 (hold bar mismatch)

**Research flag:** Standard patterns. All tasks are configuration updates and data wiring against documented APIs. No research phase needed.

---

### Phase 2: Wire Existing Strategies into Backtest Engine

**Rationale:** `DividendGapStrategy`, `CBRStrategyWrapper`, and `rub_oil_regime.py` all exist in the codebase but are not connected to the backtest engine or position sizing pipeline. The dividend gap strategy is the highest expected-alpha feature (documented 70%+ closure rate for ROSN/LKOH). Wiring existing code before adding new code is the correct sequencing — validate the highest-confidence signals first.

**Delivers:** Dividend gap strategy generating real trades in backtests using 150+ event calendar; CBR event contrarian signals active in combiner; RUB/oil regime signal wired into `PositionSizingPipeline`; baseline MOEX walk-forward Sharpe with positive equity component.

**Addresses features (from FEATURES.md):**
- `DividendCalendarLoader` (L2 new) + calendar population in `run_iteration.py`
- Combiner `_EVENT_STRATEGIES` bypass so dividend_gap is not diluted by ADX routing
- T+1 settlement date fix: rename YAML `ex_date` → `last_buy_date`; buy one bar after
- Per-symbol `max_hold_bars` for dividend gap (ROSN: 2, LKOH: 15, TATN: 30)
- Closure-rate-based confidence scaling (ROSN/LKOH > 80% → high confidence; GAZP → exclude)
- `CBRStrategyWrapper` registration in combiner with appropriate weight
- `rub_oil_regime.py` wired into `PositionSizingPipeline` as regime scale step

**Avoids pitfalls:** 5 (combiner dilution of dividend signals), 8 (sparse calendar → starvation), 12 (hold bar mismatch), 13 (T+1 settlement confusion)

**Research flag:** Standard patterns. Wiring follows established combiner registration and sizing pipeline step patterns. Calendar initialization follows the `populate_calendar()` pattern already in the strategy.

---

### Phase 3: New Macro-Gated Strategies

**Rationale:** With a clean baseline and working dividend gap strategy providing a positive Sharpe reference point, macro regime overlays and new strategies can be validated independently and incrementally. The architectural decision for sector rotation placement (sizing pipeline, not combiner) must be made explicit before a single line of code is written to avoid the HIGH-cost recovery scenario.

**Delivers:** Brent-conditional energy gating, CBR regime equity sizing adjustment, MOEX sector rotation as sizing overlay, preferred share arbitrage for SBER/SBERP and TATN/TATNP. `MacroSnapshot` extended with `brent_close` and `usdrub_daily_change` fields passed per-bar through backtest engine.

**Addresses features (from FEATURES.md):**
- `MacroSnapshot` extension: `brent_close`, `usdrub_daily_change` fields
- `BacktestEngine._process_bar()`: pass `macro_snapshot` to strategy `**kwargs`
- `CBRRegimeStep` in `PositionSizingPipeline`: scale equity positions 0.6x (hiking) or 1.2x (cutting)
- `BrentGateStep` in `PositionSizingPipeline`: using Brent-in-RUB (not USD Brent), 1-day lag
- `SectorAllocationStep` in `PositionSizingPipeline` (NOT in combiner): energy overweight when Brent > $75, underweight when < $60
- `SectorClassifier` (L2): static YAML mapping of symbols to sectors
- `PreferredShareArbStrategy`: long-only spread convergence on SBER/SBERP, TATN/TATNP; entry threshold z > 2.0 (not 1.5); window excludes 2022 crisis
- CBR leading indicator: OFZ yield curve slope (2Y-10Y spread) and RUONIA-OIS spread, NOT raw CBR announcement

**Avoids pitfalls:** 4 (CBR timing error), 6 (sector rotation in wrong layer), 7 (crisis overfitting), 10 (pref share constant spread assumption), 11 (Brent wrong currency/lag), 14 (multiplicative sizing floor)

**Research flag:** Needs `/gsd:research-phase` for CBR leading indicator design. OFZ yield curve slope data source (MOEX ISS or separate endpoint), RUONIA-OIS spread availability, and whether pre-meeting consensus rate input is achievable via YAML are open questions that require research before implementation.

---

### Phase 4: Portfolio Assembly and ML Extension

**Rationale:** Portfolio-level OFZ/equity allocation and ML macro features both require a working equity baseline with positive walk-forward Sharpe. The 40/60 OFZ/equity split composes existing components. ML macro features extend the operational `technical.py` pipeline but carry high overfitting risk on 3-year MOEX history — defer until there is a clean baseline to validate against.

**Delivers:** Combined OFZ + equity portfolio backtest with aggregate Sharpe, DD, PF across both engines. 10 new Russian macro ML features enabled for ru_* segments where quality gates pass. OFZ PK-to-PD rotation triggered when CBR cuts >= 2 consecutive meetings.

**Addresses features (from FEATURES.md):**
- `PortfolioAllocator`: 40% OFZ, 60% equity, with RUB crisis brake (shift to 80% OFZ on USDRUB spike > 5% in 5 days)
- `RUBCrisisBrake`: wired to existing `rub_oil_regime.py` crisis detection
- `PortfolioBacktestOrchestrator`: separate bond + equity engines run independently, monthly rebalance, merged PnL timeseries
- `_compute_moex_macro_features_extended()` in `ml/features/technical.py`: 10 new features (cbr_key_rate_level, cbr_rate_delta_3m, cbr_rate_direction, usdrub_return_20d, usdrub_zscore_60d, usdrub_vol_20d, brent_return_20d, brent_rub_spread, imoex_relative_21d, moex_turnover_zscore)
- ML walk-forward training: exclude Feb-Mar 2022; validate quality gates on 2024-2025 calm data; pool features across sectors to address sample size constraint

**Avoids pitfalls:** Pitfall 7 (crisis overfitting in ML), phase-5 warning (OFZ + equity currency double-counting; track allocations separately with independent circuit breakers)

**Research flag:** Needs `/gsd:research-phase` for portfolio orchestrator design — specifically, how monthly rebalancing is modeled in the backtest (cash transfer mechanics between bond and equity engine) and PnL merging methodology. ML transfer learning from US model to MOEX to address the 3-year vs. 10-year sample size gap is also a research question.

---

### Phase Ordering Rationale

- Phase 1 before all else: data and parameter problems invalidate results regardless of strategy quality. Cannot tune or validate any strategy on broken foundations.
- Phase 2 before Phase 3: existing-but-unconnected strategies are higher-confidence alpha sources than new strategies. Dividend gap has documented 70%+ closure rates; wire it before adding new complexity. Clean equity baseline from Phase 2 is also a prerequisite for validating Phase 3 macro overlays.
- Phase 3 before Phase 4: `MacroSnapshot` extensions (Phase 3) are direct dependencies for ML macro features (Phase 4). Portfolio allocator also requires equity strategies to demonstrate positive individual Sharpe before composing them.
- Sector rotation is a HARD architectural constraint: it must be a sizing step, not a combiner signal. This is the single most important design decision from the research and must be resolved before a line of Phase 3 code is written.

### Research Flags

Needs `/gsd:research-phase` during planning:
- **Phase 3 (CBR leading indicator):** OFZ yield curve slope data source (MOEX ISS vs. separate API), RUONIA-OIS spread availability, pre-meeting consensus rate input mechanism — all open questions before implementation
- **Phase 4 (portfolio orchestrator):** Monthly rebalancing mechanics in backtest (cash transfer between bond/equity engines), PnL merging methodology, and ML transfer learning from US to MOEX model for sample size mitigation

Standard patterns (skip research-phase):
- **Phase 1:** Pure configuration updates and T-Invest API calls against documented endpoints — all patterns established in existing codebase
- **Phase 2:** Strategy wiring follows existing combiner registration, calendar initialization, and sizing pipeline step patterns — all precedented in codebase

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Verified every feature against pyproject.toml and existing modules; zero new packages required; all building blocks confirmed in codebase |
| Features | MEDIUM | Dividend closure statistics from financial media (spydell, finam.ru); CBR sector impact from Forbes.ru/RBC; magnitude figures (0.6-0.8 Brent correlation, 5-15% pref/ord spread) need live validation |
| Architecture | HIGH | Derived from direct codebase inspection of combiner.py, position_sizing_pipeline.py, engine.py, dividend_gap.py, cbr_calendar.py, schemas.py; all layer boundaries confirmed; no layer violations in proposed design |
| Pitfalls | HIGH (structural) / MEDIUM (MOEX domain) | Structural pitfalls (vol target, T+1 settlement, combiner routing, sector rotation layer) verified in code; MOEX domain pitfalls (pref share spread dynamics, Brent correlation lag) are research-informed estimates needing empirical validation |

**Overall confidence:** MEDIUM-HIGH

### Gaps to Address

- **MOEX sector index ticker availability:** ISS API confirmed for index candles (tested with IMOEX), but specific sector tickers (MOEXOG, MOEXFN, MOEXMM) need live API validation in Phase 1 before any sector rotation work begins. If a sector index is unavailable, sector rotation must fall back to symbol-level classification.
- **T-Invest `get_dividends()` batch performance:** Per-symbol dividend fetch is operational; batch performance for 50+ symbols is untested. Rate limit is 600 requests/minute; 50 symbols = ~50 API calls, safely within limit, but must be validated empirically in Phase 1.
- **OFZ yield curve slope data source:** Pre-meeting CBR positioning and the CBR leading indicator design require OFZ 2Y-10Y yield data. It is unclear whether this is available via MOEX ISS or requires a separate source. This is the key open question for Phase 3 CBR regime design.
- **Brent-energy correlation lag:** Research indicates 1-3 day lag for MOEX energy stocks vs. Brent, but optimal lag should be verified empirically in Phase 3 before hardcoding the gate parameter.
- **Preferred share spread stationarity post-2022:** SBER/SBERP spread was non-stationary in 2022 (preferred briefly exceeded common during retail buying surge). Cointegration tests must be run on post-2022 data only before implementing pref arb in Phase 3. If cointegration fails, skip the strategy.

## Sources

### Primary (HIGH confidence — official sources and direct codebase inspection)
- Codebase: `src/finalayze/strategies/dividend_gap.py`, `cbr_calendar.py`, `rub_oil_regime.py`, `combiner.py`, `position_sizing_pipeline.py`, `backtest/engine.py`, `data/fetchers/cbr.py`, `ml/features/technical.py`, `strategies/presets/ru_*.yaml`, `config/segments.py`
- CBR official calendar: https://www.cbr.ru/eng/dkp/cal_mp/
- CBR key rate history: https://cbr.ru/eng/hd_base/KeyRate/
- MOEX ISS API reference: https://iss.moex.com/iss/reference/
- MOEX sector indices listing: https://www.moex.com/en/indices
- T-Invest API: `t-tech-investments` package proto definitions

### Secondary (MEDIUM confidence — financial media with quantitative data)
- Dividend gap closure statistics (2007-2017): https://spydell.livejournal.com/642950.html
- Dividend gap recent data (2024-2025): https://www.finam.ru/publications/item/istoricheski-lukoyl-i-tatneft-obladayut-potentsialom-bystrogo-vosstanovleniya-posle-dividendnogo-gepa-20250604-0900/
- CBR rate cut sector impact: https://www.forbes.ru/investicii/543288-raduznye-nadezdy-kakie-akcii-vyrastut-iz-za-snizenia-stavki-cb
- CBR rate and financials: https://www.rbc.ru/quote/news/article/68497aae9a794711e7402f87
- ML on MOEX stocks (multimodal approach): https://arxiv.org/html/2503.08696
- MOEX 2022 crisis: Bloomberg (Russian Stocks Slump Most on Record, 2022-02-24)

### Tertiary (LOW confidence — general context, needs live validation)
- Brent-MOEX energy correlation magnitude (0.6-0.8) — based on general market knowledge; needs empirical validation against training data
- Preferred share spread ranges (SBER/SBERP: 5-15%, TATN/TATNP: 5-10%) — historical patterns; current levels require live check
- MOEX short-selling restrictions for retail T-Invest accounts — based on T-Invest documentation; may have changed in 2025-2026

---
*Research completed: 2026-03-20*
*Ready for roadmap: yes*

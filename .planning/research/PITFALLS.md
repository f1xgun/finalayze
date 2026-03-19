# Domain Pitfalls: MOEX Equity Profitability (v2.0)

**Domain:** Adding dividend gap, CBR regime overlay, and sector rotation strategies to existing MOEX multi-strategy trading system
**Researched:** 2026-03-20
**Confidence:** HIGH (codebase analysis) / MEDIUM (MOEX-specific domain knowledge from web research)

---

## Critical Pitfalls

Mistakes that cause rewrites, persistent negative Sharpe, or invalidate entire backtest results.

### Pitfall 1: Look-Ahead Bias in Dividend Gap Backtests

**What goes wrong:**
The dividend gap strategy uses `moex_dividends.yaml` (static file with 43 events) to know ex-dividend dates and amounts. In a backtest, the strategy "sees" all future dividends at bar 0. When the engine encounters bar N matching an ex-date, it generates a BUY. This is correct *if* the dividend was publicly announced before the ex-date. But on MOEX, dividend recommendations are announced by the Board of Directors 30-50 days before the shareholder meeting, and the ex-date is set at record date - 2 business days. A strategy that buys on the ex-date is fine (the date is known by then). But if gap_pct or confidence scaling uses the dividend amount, there is look-ahead risk: the *exact* dividend amount is often finalized only at the shareholder meeting, which can be 5-10 days before record date. In 2022, GAZP's board recommended 52.53 RUB but shareholders rejected it -- the YAML file shows the actual 51.03 RUB payment from October, but the strategy would have traded on the announced 52.53 RUB in June.

**Why it happens:**
The static YAML file conflates "announced" and "paid" dividends. The DividendGapStrategy.generate_signal() uses `matching_div.amount` from the YAML to compute `gap_pct = amount / pre_exdiv_close * 100`. If the YAML contains the final paid amount but the market priced in the announced amount (which may differ), the gap_pct calculation is wrong. Worse, some MOEX companies announce dividends then cancel them entirely (GAZP 2022, VTBR 2022) -- these events should NOT generate BUY signals but will if the YAML only contains successful payments.

**Consequences:**
- Backtest shows inflated win rate because only successful dividend payments are in the dataset
- Gap closure rate appears higher than reality (cancelled dividends = no gap to close, but also no loss counted)
- Strategy appears to have ~15 clean trades when real deployment would encounter 3-5 cancelled/reduced dividends per year

**Prevention:**
1. Add a `status` field to YAML entries: `paid`, `cancelled`, `reduced`. Include ALL announced dividends, not just paid ones
2. DividendGapStrategy must only use dividend data that would have been known at the ex-date: the amount from the Board recommendation (not shareholder vote), flagged with `confidence: preliminary|confirmed`
3. Include failed dividend events (GAZP 2022 June: recommended 52.53, cancelled) as entries with `status: cancelled` -- the strategy must NOT trade these in backtest
4. Expand YAML from 43 to 150+ events, including ALL board recommendations from 2020-2025, not just successful payments
5. Add test: no DividendEntry in calendar has ex_date before board_recommendation_date

**Detection:**
- Backtest shows 0 losing dividend gap trades (unrealistic -- ~20% of announced dividends are modified or cancelled)
- All gap_pct values in backtest exactly match final paid amounts (impossible to know pre-ex-date for ~30% of events)
- Win rate > 85% on dividend gap strategy (historically 65-75% when including cancellations)

**Phase to address:** Phase 1 (data preparation) -- must be resolved before any dividend gap backtesting.

---

### Pitfall 2: Survivorship Bias from 2022 Sanctions Structural Break

**What goes wrong:**
The backtest uses 2022-2025 data for walk-forward validation. February 2022 sanctions caused: MOEX closure Feb 28 - Mar 24 (25 trading days), foreign investor freeze (40% of equity ownership trapped), index drop of ~50% in Feb 2022, and permanent delisting/restructuring of several instruments (Polymetal -> Solidcore, Yandex -> YDEX). The current universe in `segments.py` includes symbols that either didn't exist pre-2022 (YDEX) or were fundamentally different entities (POLY -> new listing). Backtesting these symbols across the structural break treats pre-break and post-break as the same instrument.

**Why it happens:**
The segment definitions in `config/segments.py` use current ticker lists. Symbols like `GAZP`, `VTBR`, `SNGS` are in the universe but the PROJECT.md already identifies them as "toxic" (60% of negative PnL). The system treats 2022 as regular data, but the Feb-Mar 2022 period had: trading halt, forced margin calls, artificial circuit breakers by MOEX (20% daily limits), and CBR-mandated short-selling ban. Any strategy calibrated on this data learns "crisis = opportunity" which is survivorship bias -- you only see the survivors.

**Consequences:**
- Walk-forward folds that include Feb-Mar 2022 in training learn distorted patterns (artificial floor from MOEX circuit breakers)
- Strategies calibrated on 2022 data overestimate recovery speed (MOEX intervention prevented natural price discovery)
- Toxic symbols (GAZP, VTBR) show "mean reversion" in 2022-2023 that was actually government-supported price floors, not a repeatable pattern
- Vol estimates from 2022 data are 3-5x higher than normal, choking VolTargetStep for all subsequent bars

**Prevention:**
1. Universe surgery FIRST: remove GAZP, VTBR, SNGS, IRAO, ALRS from active segments before any backtesting
2. Purge Feb-Mar 2022 data from walk-forward training windows entirely (treat as gap, not data)
3. Use separate walk-forward configs: pre-sanctions (2020-2022-02) and post-sanctions (2022-04-2025). Never train across the break
4. For symbols that were restructured (POLY, YNDX), start data only from post-restructuring date
5. Add `exclude_periods` parameter to BacktestConfig: `[("2022-02-24", "2022-04-01")]`
6. Vol estimates must use adaptive window that excludes the crisis period or uses exponential decay

**Detection:**
- Walk-forward Sharpe swings wildly between folds (one fold includes 2022, another doesn't)
- VolTargetStep consistently scales MOEX positions to 25% (floor) because asset_vol > 0.50 from 2022 data
- Mean reversion strategies show artificially high win rate on GAZP/VTBR in 2022-2023 folds

**Phase to address:** Phase 1 (universe cleanup and data preparation) -- must be the FIRST thing done.

---

### Pitfall 3: Vol Target Calibrated for US (0.19) Destroys MOEX Position Sizes

**What goes wrong:**
The VolTargetStep in `position_sizing_pipeline.py` computes `raw_ratio = target_vol / asset_vol` and clamps to [0.25, 1.5]. The target_vol comes from strategy presets: `vol_target: 0.19` (19% annualized). MOEX blue chip volatility is typically 0.35-0.60 (35-60% annualized). The ratio becomes 0.19/0.45 = 0.42, clamped to 0.42. Combined with RegimeStep (scale 0.50 in ELEVATED), the position becomes 0.42 * 0.50 = 0.21 of base. Combined with the pipeline floor of 15%, many signals end up at exactly the 15% floor -- effectively a fixed tiny position regardless of signal strength.

**Why it happens:**
The 0.19 vol target was calibrated for US equities (SPY vol ~0.15-0.20). MOEX equities inherently have 2-3x higher vol due to: thinner liquidity, commodity/FX exposure, geopolitical risk premium, and smaller free float. The preset YAML files (`ru_blue_chips.yaml`) copy-pasted `vol_target: 0.19` from US presets without recalibrating.

**Consequences:**
- ALL MOEX equity positions are systematically undersized (21-42% of what they should be)
- The pipeline floor (15%) fires on 60-70% of MOEX trades, eliminating any signal-strength differentiation
- Position sizes too small to overcome transaction costs (MOEX commission + slippage > expected return on tiny positions)
- 104 backtest iterations ALL showing negative Sharpe partly because positions are too small to generate meaningful P&L

**Prevention:**
1. Set MOEX-specific vol_target: 0.35-0.40 for ru_blue_chips, 0.45-0.50 for ru_energy (match the market's natural vol level)
2. Alternative: disable VolTargetStep entirely for MOEX segments and rely on Kelly + RegimeStep for sizing
3. Never copy US preset parameters to MOEX presets without recalibrating -- create a separate calibration script
4. Add a test: for each segment, assert that VolTargetStep ratio > 0.50 for the median asset_vol of that segment's universe
5. Log the VolTargetStep ratio per trade in backtest -- if > 70% of trades hit the floor, the vol_target is miscalibrated

**Detection:**
- Backtest logs show `vol_target_ratio: 0.25` (clamped to floor) for > 50% of MOEX trades
- Pipeline floor firing rate > 50% for MOEX segments (should be < 10%)
- Position sizes for MOEX equities are 0.5-2% of equity (should be 5-15%)

**Phase to address:** Phase 1 (parameter recalibration) -- quick fix, high impact, do alongside universe cleanup.

---

### Pitfall 4: CBR Rate Regime Timing Error -- Using Announcement Date Instead of Market Pricing Date

**What goes wrong:**
The CBR announces key rate decisions 8 times per year. The `rub_oil_regime.py` exists but is not wired into sizing. When adding a CBR rate regime overlay, the natural approach is: "if CBR cut rate, go risk-on; if CBR hiked, go risk-off." But the market prices in CBR decisions 1-3 weeks BEFORE the announcement based on: inflation data (released monthly), CBR forward guidance (published in monetary policy reports), and OFZ yield curve movements. If the strategy waits for the actual announcement, the move has already happened. If the strategy trades on announcement, it buys at the post-announcement price (already adjusted).

**Why it happens:**
Developers treat CBR decisions as "events" (like earnings). But unlike earnings, CBR decisions are highly predictable -- the OFZ yield curve embeds the expected rate path. The "surprise" component (actual vs. expected) is what moves prices, not the absolute decision. A strategy that buys on "CBR cut" and sells on "CBR hike" is buying/selling AFTER the information is priced in.

**Consequences:**
- CBR regime overlay shows near-zero alpha in backtests (market already priced the decision)
- Worse: if the strategy buys on "cut" announcement, it buys AFTER the rally, then holds through the mean-reversion back to fair value -- negative alpha
- Look-ahead bias if the backtest uses actual CBR decisions to size positions on dates before the decision was announced

**Prevention:**
1. CBR regime must use OFZ yield curve slope (2Y-10Y spread) and RUONIA-OIS spread as *leading* indicators, not the CBR decision itself as a *lagging* indicator
2. Implement two distinct regime signals:
   - **Pre-decision positioning** (2 weeks before meeting): based on OFZ curve steepening/flattening trend
   - **Post-decision adjustment** (day after meeting): only on *surprise* component (actual vs. market-implied rate)
3. The existing `rub_oil_regime.py` (RUB-oil decorrelation) is actually a BETTER regime signal than CBR decisions -- wire it into sizing first
4. For backtesting: CBR decision dates must be from a historical calendar (available from cbr.ru), NOT embedded in the candle data. Ensure the strategy only "sees" the decision after the announcement bar
5. Add "CBR surprise" feature for ML: `actual_rate - implied_rate_from_OFZ_curve`. This has genuine predictive value

**Detection:**
- CBR regime overlay shows < 0.05 improvement in Sharpe vs. no overlay (signal already priced in)
- Positions opened on CBR decision day show negative average return over next 5 bars (bought the top)
- All CBR-gated trades cluster around 8 dates per year with no signal between meetings

**Phase to address:** Phase 3 (CBR regime integration) -- requires careful design; do NOT just gate on rate decisions.

---

### Pitfall 5: Dividend Gap Strategy Conflicts with Combiner ADX Routing

**What goes wrong:**
The `DividendGapStrategy` generates BUY signals on ex-dividend dates. These signals enter the `StrategyCombiner` where they are subject to ADX regime routing. If ADX > 35 (trend regime), only `_MOMENTUM_STRATEGIES` (momentum, dual_momentum) are allowed. Dividend gap is NOT in `_MOMENTUM_STRATEGIES` or `_MR_STRATEGIES` -- it is not classified in either pool. This means: (a) the signal may be filtered by regime routing (depending on implementation), or (b) it always passes routing but its weight gets diluted by whichever pool is active.

The deeper problem: dividend gap is an EVENT-DRIVEN strategy (triggers on calendar date, not price pattern). It should bypass ADX routing entirely. But the combiner's `generate_combined_signal()` applies routing to ALL strategies. The current code in combiner.py lines 35-36 defines:
```
_MOMENTUM_STRATEGIES = frozenset({"momentum", "dual_momentum"})
_MR_STRATEGIES = frozenset({"mean_reversion", "pairs", "ou_mean_reversion", "rsi2_connors"})
```
`dividend_gap` is in neither set, so it falls through to "ambiguous" classification and gets included regardless of ADX -- but this is ACCIDENTAL, not intentional. A future refactor that defaults unknown strategies to "blocked" would silently kill the dividend gap strategy.

**Why it happens:**
ADX routing was designed for US equity strategies (trend vs. mean-reversion classification). Event-driven strategies (dividend_gap, pead, cbr_calendar) don't fit this taxonomy. The combiner was not designed for strategies with fundamentally different signal generation logic.

**Consequences:**
- Dividend gap signals get weighted alongside trend/MR signals, diluting a high-conviction event signal with noisy technical signals
- On ex-div days, the combined confidence may fall below `min_combined_confidence` (0.15 for ru_blue_chips) because other strategies generate HOLD/SELL signals that average down the BUY
- If another strategy generates a SELL on the ex-div day (e.g., momentum sees the gap as a bearish breakdown), the signals cancel out and no trade occurs -- missing the highest-alpha signal

**Prevention:**
1. Create an `_EVENT_STRATEGIES` frozenset: `{"dividend_gap", "pead", "cbr_calendar", "event_driven"}` that bypasses ADX routing entirely
2. Event strategies should have a "standalone" mode: if dividend_gap confidence > 0.60, it should generate a trade independently of the combiner's weighted average
3. Alternative: give dividend_gap a combiner weight of 0.50+ on MOEX segments so it dominates the combined signal on ex-div days
4. Test: on a known ex-div date with ADX > 35, verify dividend_gap signal is NOT filtered
5. Test: on a known ex-div date, verify combined signal is BUY even if momentum generates SELL

**Detection:**
- Backtest shows 0 dividend_gap trades despite 43 ex-div events in calendar (signals filtered or diluted)
- Trade log shows "combined_confidence: 0.12" on ex-div dates (below 0.15 threshold)
- Dividend gap appears in strategy signal log but never in final trade list

**Phase to address:** Phase 2 (dividend gap strategy implementation) -- combiner modification required before dividend gap can work.

---

### Pitfall 6: Sector Rotation Creates Catastrophic Signal Conflicts in Multi-Strategy Combiner

**What goes wrong:**
Sector rotation generates allocation signals at the SECTOR level (overweight energy, underweight financials). But the existing combiner operates at the SYMBOL level (BUY SBER, SELL GAZP). Mixing these two levels creates contradictions: sector rotation says "overweight energy" (all energy symbols should get larger positions), but mean_reversion says "SELL ROSN" (overbought). The combiner weights these equally, and the result is incoherent -- neither a clean sector bet nor a clean technical signal.

Worse: sector rotation typically rebalances monthly. Technical strategies generate daily signals. If sector rotation weight is significant (e.g., 0.20), it produces a constant BUY/SELL bias for 30 days that drowns out daily technical signals. If sector rotation weight is small (e.g., 0.05), it has no effect and adds complexity without alpha.

**Why it happens:**
The combiner was designed for strategies that all operate at the same level (per-symbol, per-bar). Sector rotation is a different beast -- it operates at the portfolio level across symbols. Forcing it into the per-symbol combiner framework creates an impedance mismatch.

**Consequences:**
- Sector rotation "BUY energy" generates weak BUY signals for ALL energy symbols, even toxic ones (SNGS, GAZP) that should be excluded
- The constant sector bias reduces the combiner's ability to discriminate between strong and weak signals within a sector
- Monthly rebalancing creates whipsaw at transition dates: sector rotation flips from BUY to SELL on all symbols in a sector simultaneously
- Backtest optimization overfits the sector rotation timing to known macro events (2022 sanctions, 2024 rate hikes)

**Prevention:**
1. Sector rotation must NOT go through the per-symbol combiner. Implement it as a PORTFOLIO-LEVEL overlay in the sizing pipeline, not the signal pipeline
2. Correct architecture: sector rotation modifies position sizes (via a `SectorAllocationStep` in `PositionSizingPipeline`), not signal direction
3. The `SectorAllocationStep` scales positions: if sector = overweight (1.5x), the position for symbols in that sector is 1.5x base. If sector = underweight (0.5x), the position is 0.5x
4. Sector rotation rebalancing should use a 20-day rolling transition (linear ramp from old weights to new weights), not a hard flip on rebalance day
5. Never apply sector rotation to individual toxic symbols -- if a symbol is excluded from the universe, it stays excluded regardless of sector weight

**Detection:**
- Backtest shows sector rotation generating trades on excluded symbols (GAZP, VTBR)
- Trade log shows 20+ trades on the same day at month-end (sector rebalance whipsaw)
- Sector rotation + technical signals produce opposite directions on the same symbol simultaneously

**Phase to address:** Phase 3 (sector rotation implementation) -- requires architectural decision on where sector rotation lives (signal vs. sizing).

---

### Pitfall 7: Overfitting to 2022-2024 CBR Hiking Cycle

**What goes wrong:**
The backtest period (2022-2025) covers an extraordinary CBR hiking cycle: 9.5% (Feb 2022) -> 20% (Feb 2022 emergency) -> 7.5% (Sep 2022) -> 16% (Aug 2023) -> 21% (Oct 2024) -> 15.5% (Feb 2026). This is NOT a normal rate cycle -- it includes emergency wartime hikes, sanctions-driven currency crisis, and the most aggressive tightening in CBR history. Any regime strategy calibrated on this data will be overfit to extreme events that may not recur.

**Why it happens:**
The 2022-2025 window happens to be the available data with Tinkoff API. It is also one of the most volatile periods in Russian financial history. Walk-forward optimization on this data will find parameters that work well for crisis/recovery cycles but fail in "boring" markets (e.g., 2017-2019 when CBR rate was stable at 7.25-7.75%).

**Consequences:**
- CBR regime parameters tuned to detect 200bps+ emergency hikes will never fire in normal conditions (typical hikes are 25-50bps)
- RUB/oil decorrelation thresholds calibrated on 2022 sanctions shock (correlation went negative) will never trigger in normal markets (correlation typically 0.30-0.60)
- Walk-forward results look good because train/test both cover crisis periods -- but this is overfitting to crisis, not genuine alpha
- When deployed in a "normal" 2026 market (CBR cutting from 15.5% gradually), all regime signals will be permanently in NORMAL mode and add no value

**Prevention:**
1. If possible, source pre-2022 MOEX data from MOEX ISS (free, goes back to 2010) for longer backtest windows that include calm periods (2017-2019)
2. Apply Optuna overfitting guardrails (already implemented for US): DSR haircut, holdout validation, perturbation check
3. For CBR regime parameters: use academic/historical thresholds from CBR monetary policy research, not data-mined thresholds from 2022-2025
4. Test regime parameters on "synthetic calm" data: generate 2 years of flat rate environment and verify the strategy does NOT trade (no false positives)
5. Track "regime signal firing rate" -- if it fires < 5% of bars, it's a crisis-only detector (fine for risk overlay, useless for alpha generation)
6. The RUB/oil decorrelation regime (`rub_oil_regime.py`) with thresholds 0.3/0.1 was designed for normal markets -- validate these thresholds still make sense post-2022

**Detection:**
- Walk-forward Sharpe is positive ONLY on folds that include crisis/recovery periods
- Regime signal fires 0 times in 2024-2025 data (market normalized, parameters only detect 2022-level events)
- CBR regime strategy shows Sharpe > 1.0 in backtest but all alpha comes from 3-4 crisis trades

**Phase to address:** Phase 2-3 (regime strategy development) -- design the strategy for normal markets FIRST, then add crisis detection as an overlay.

---

## Moderate Pitfalls

### Pitfall 8: Dividend Calendar Sparse Data Creates Backtest Starvation

**What goes wrong:**
The current `moex_dividends.yaml` has 43 entries across 7 symbols over 3 years. That's ~6 events per symbol per year, but some symbols have only 1-2 events. For walk-forward with 12-month training windows, a training fold may contain 0-2 dividend events for a given symbol -- far too few to estimate win rate, expected return, or calibrate any parameters. The strategy appears to "work" in backtest only because the few events happen to be winners (survivorship from only including paid dividends -- see Pitfall 1).

**Prevention:**
1. Expand to 150+ events: add ALL blue chip + mid-cap dividends from 2020-2025 using Tinkoff API `get_dividends()`
2. Include symbols beyond current universe: MTSS, MGNT, CHMF, NLMK, PHOR -- all have regular dividends
3. For symbols with < 3 events in a walk-forward fold, fall back to a "MOEX average" gap closure model (pooled parameters)
4. Add interim dividends: many MOEX companies pay twice or thrice per year (TATN has 3/year, LKOH has 2/year)
5. Test: assert that every walk-forward training fold contains >= 10 dividend events across all symbols

**Detection:**
- Walk-forward folds show dividend_gap firing 0-1 times in 6-month test window
- Gap closure parameters (hold period, confidence) cannot be calibrated due to < 5 events per fold
- Strategy appears to have infinite Sharpe (2 trades, both winners) in some folds

**Phase to address:** Phase 1 (data preparation).

---

### Pitfall 9: event_driven Strategy Generates Phantom Trades in Combined Backtests

**What goes wrong:**
The `event_driven` strategy is enabled in `ru_blue_chips.yaml` with weight 0.15. In backtests, there is no real news feed, so `event_driven` generates 0 trades (expected). But its 15% weight allocation is still present in the combiner normalization. When `normalize_mode: "firing"` is used (only normalize across strategies that actually fire), event_driven's weight is excluded. This is correct. BUT: if event_driven is accidentally configured with `normalize_mode: "all"`, every combined signal is multiplied by 0.85 (because event_driven contributes 0, taking 15% of the weight). This systematically reduces all signal confidences by 15%, pushing more signals below `min_combined_confidence`.

Separately: the PROJECT.md notes "event_driven strategy fires without real news feed in backtests" as an issue. If the strategy has cached/stale sentiment data from a previous live run, it may generate phantom signals in backtest mode.

**Prevention:**
1. Explicitly disable `event_driven` in ALL segment presets used for backtesting: `event_driven: {enabled: false}` in backtest configs
2. Ensure `normalize_mode: "firing"` is the default for all MOEX presets (it is, but verify)
3. Add a guard in BacktestEngine: if `mode != "real"` and `event_driven.enabled`, log WARNING and auto-disable
4. Clear any cached sentiment state at backtest start (DividendGapStrategy already has `reset()`, event_driven needs the same)
5. Test: backtest with event_driven enabled vs. disabled produces identical results (zero contribution)

**Detection:**
- Backtest metrics differ between `event_driven.enabled: true` and `event_driven.enabled: false` even though no news feed exists
- Trade count changes when event_driven weight changes (shouldn't if it fires 0 times)
- Combiner logs show `event_driven: confidence=0.XX` in backtest (phantom signal from cached data)

**Phase to address:** Phase 1 (backtest configuration cleanup).

---

### Pitfall 10: Preferred Share Arbitrage (SBER/SBERP) Assumes Constant Spread

**What goes wrong:**
Preferred share arbitrage (buy SBERP when discount to SBER > historical mean) assumes a mean-reverting spread. But MOEX preferred shares have structurally different dynamics: (a) SBERP gets the SAME dividend as SBER but trades at 5-15% discount -- the discount IS the dividend yield premium, (b) the spread changes with interest rate environment (higher rates -> smaller discount, because prefs have higher yield), (c) during 2022 crisis, preferred shares became MORE expensive than common (SBERP > SBER briefly) due to retail buying of high-yield instruments.

**Prevention:**
1. Model the spread as a function of CBR rate and dividend yield, not a simple mean
2. Use spread z-score relative to a rolling window that EXCLUDES the 2022 crisis period
3. Entry threshold must be > 2 standard deviations (not 1.5 as in current pairs params)
4. Always check that BOTH legs (SBER and SBERP) are in the active universe and tradeable
5. Set max_hold_bars for pairs to match dividend cycle (if entering before ex-div, close before record date to avoid dividend tax complications)

**Detection:**
- Pairs strategy shows > 50% of trades entering during crisis periods (overfitting to 2022)
- Spread mean changes significantly between walk-forward folds (non-stationary)
- Win rate drops below 40% when 2022 data is excluded from training

**Phase to address:** Phase 3 (if implementing preferred arbitrage).

---

### Pitfall 11: Brent Gate for Energy Sector Uses Wrong Correlation Lag

**What goes wrong:**
MOEX energy stocks (ROSN, TATN, LKOH) correlate with Brent crude, but with a 1-3 day lag (Russian market closes before Brent final settlement). A Brent price gate that uses same-day Brent close to gate MOEX entries introduces a timing mismatch: if you gate on today's Brent, MOEX already reacted to yesterday's Brent. The signal is stale.

Worse: Brent data comes from yfinance (BZ=F), which reports in USD. The actual exposure is Brent-in-RUB (Brent * USDRUB). When RUB weakens (Brent in USD flat), Brent-in-RUB rises and MOEX energy stocks rally -- but a USD Brent gate would miss this entirely.

**Prevention:**
1. Use Brent-in-RUB (Brent * USDRUB) as the gate variable, not USD Brent
2. Apply 1-day lag: gate on yesterday's Brent-in-RUB change, not today's
3. Brent gate should be a SIZING modifier (SectorAllocationStep), not a signal filter -- don't block entries, scale positions
4. The existing `rub_oil_regime.py` already computes RUB/oil correlation -- use this as the Brent gate rather than building a separate one
5. Test: correlation between MOEX energy returns and Brent-in-RUB returns at lag=1 vs lag=0 (expect lag=1 is higher)

**Detection:**
- Brent gate correlation with energy stock returns is < 0.20 (using wrong lag or wrong currency)
- Energy sector positions are gated off on days when energy stocks actually rally (Brent-in-RUB up but USD Brent flat)

**Phase to address:** Phase 3 (sector rotation / Brent gating).

---

## Minor Pitfalls

### Pitfall 12: Dividend Gap max_hold_bars Set Too High (60 in preset, 15 in config)

**What goes wrong:**
`ru_blue_chips.yaml` sets `dividend_gap.params.max_hold_bars: 60` but `backtest/config.py` has `DEFAULT_STRATEGY_HOLD_BARS["dividend_gap"] = 15`. These are different code paths: the preset parameter controls the strategy's internal exit logic, while the config parameter controls the engine's forced exit. The engine will force-close positions at 15 bars even though the strategy expects 60 bars for gap closure. For MOEX blue chips, typical gap closure takes 20-40 bars (30-60 calendar days). The 15-bar engine limit will cut most dividend gap positions before gap closure, converting winners into losers.

**Prevention:**
1. Align `DEFAULT_STRATEGY_HOLD_BARS["dividend_gap"]` with the strategy's `max_hold_bars` parameter (60)
2. Better: dividend_gap should NOT use the engine's generic hold bar limit at all -- it has its own exit logic (gap closure OR max_hold_bars)
3. Add test: `assert DEFAULT_STRATEGY_HOLD_BARS["dividend_gap"] >= preset_max_hold_bars`

**Phase to address:** Phase 2 (dividend gap implementation).

---

### Pitfall 13: T+1 Settlement Means Dividend Gap Entry Is One Bar Late

**What goes wrong:**
MOEX uses T+1 settlement for equities. The strategy buys on the ex-dividend date (current_candle.timestamp matches ex_date). But in MOEX terminology, the "last buy date" in the Tinkoff API is actually T-1 relative to the record date. If you buy at close on the "last buy date," you get the dividend. If you buy on the NEXT day (actual ex-date), you don't. The YAML comments confirm: "ex_date = last buy date (Tinkoff convention); actual ex-div is next trading day." But the strategy uses `div.ex_date.date() == current_date.date()` to trigger -- this triggers on the LAST BUY DATE, not the ex-date. This means the strategy buys BEFORE the gap occurs, at the pre-gap price, and the position immediately drops by the dividend amount.

**Prevention:**
1. Clarify terminology: rename YAML field from `ex_date` to `last_buy_date` to match Tinkoff convention
2. Strategy should buy ONE BAR AFTER `last_buy_date` (the actual ex-date when the gap appears)
3. OR: buy on `last_buy_date`, collect the dividend, and set gap closure target = pre-ex-date price + dividend (breakeven accounting for dividend received)
4. The current implementation uses `candles[-2].close` as pre_exdiv_close, which is correct IF triggered on the actual ex-date (bar after last_buy_date), but WRONG if triggered on last_buy_date itself

**Detection:**
- Dividend gap trades show entry price = pre-gap price (no gap at entry)
- Many trades show immediate 5-10% unrealized loss on day 1 (bought pre-gap, gap happened next bar)
- gap_pct in trade features doesn't match actual price drop

**Phase to address:** Phase 2 (dividend gap implementation) -- critical correctness issue.

---

### Pitfall 14: RegimeStep and Sector Rotation Step Compete for Position Scale

**What goes wrong:**
The sizing pipeline already has `RegimeStep` (scale by regime_scale, floor 0.15) and adding a `SectorAllocationStep` creates multiplicative scaling: final_size = base * vol_target_ratio * regime_scale * sector_scale. If regime_scale = 0.50 (ELEVATED) and sector_scale = 0.50 (underweight sector), the position is 0.25x base -- hitting the pipeline floor of 15%. Three or more multiplicative reduction steps guarantee that most MOEX positions end up at the floor, eliminating signal differentiation.

**Prevention:**
1. Cap total multiplicative reduction: `total_scale = max(product_of_all_scales, 0.25)` at the pipeline level
2. Or: make sector allocation ADDITIVE to a neutral baseline, not multiplicative: `sector_adjusted = base * (1.0 + sector_deviation)` where sector_deviation is [-0.30, +0.30]
3. Monitor floor-hit rate: if > 30% of trades hit the pipeline floor, the pipeline has too many reduction steps
4. Consider reducing the number of sizing steps for MOEX: Kelly -> MOEX-calibrated VolTarget -> HardCaps (3 steps instead of 7)

**Phase to address:** Phase 3 (sector rotation sizing integration).

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| Phase 1: Universe cleanup | Removing toxic symbols changes walk-forward fold composition; old iterations incomparable | Start iteration history fresh for v2.0; do not compare v2.0 Sharpe to v1.0 iterations |
| Phase 1: Data expansion | Tinkoff API rate limits on historical dividend fetches (600 requests/minute) | Batch dividend fetches per symbol; cache to YAML; fetch once and store |
| Phase 2: Dividend gap | T+1 settlement timing confusion between last_buy_date and ex-date (Pitfall 13) | Write explicit test with known SBER 2024 dividend; verify entry is on gap day |
| Phase 2: Dividend gap | Combiner dilutes dividend signal below threshold (Pitfall 5) | Give dividend_gap standalone mode or weight >= 0.40 on MOEX segments |
| Phase 2: Dividend gap | Only 43 events -> insufficient for walk-forward calibration (Pitfall 8) | Expand to 150+ events BEFORE any backtest tuning |
| Phase 3: CBR regime | Using CBR decisions as signals instead of leading indicators (Pitfall 4) | Wire `rub_oil_regime.py` FIRST; add OFZ curve slope as leading indicator |
| Phase 3: Sector rotation | Portfolio-level strategy forced into per-symbol combiner (Pitfall 6) | Implement as SectorAllocationStep in sizing pipeline, NOT in combiner |
| Phase 3: Brent gate | Wrong currency denomination for energy correlation (Pitfall 11) | Use Brent * USDRUB, not USD Brent; test correlation at lag=0 vs lag=1 |
| Phase 3: Parameter tuning | Overfitting to 2022 crisis regime (Pitfall 7) | Source pre-2022 data from MOEX ISS; use academic thresholds for regime |
| Phase 4: ML ensemble | MOEX features trained on 2022 crisis -> model learns crisis-only patterns | Exclude Feb-Mar 2022 from training; validate on 2024-2025 calm data |
| Phase 4: ML ensemble | Too few training samples (3 years MOEX vs 10+ years US data) | Pool features across sectors; use transfer learning from US model |
| Phase 5: Portfolio assembly | OFZ + equity allocation creates currency double-counting (both RUB) | Track OFZ and equity allocations separately with independent circuit breakers |

---

## Integration Gotchas Specific to v2.0

| Integration Point | Common Mistake | Correct Approach |
|-------------------|----------------|------------------|
| DividendGapStrategy + Combiner | Dividend signals diluted by technical strategy HOLD/SELL signals | Create `_EVENT_STRATEGIES` bypass in ADX routing; or give dividend_gap standalone trade capability |
| CBR Regime + VolTargetStep | CBR regime reduces position AND vol target reduces position -> double penalty | Choose ONE of: regime overlay OR vol target for MOEX. Not both multiplicatively |
| Sector rotation + existing segments | Sector rotation implies variable universe; existing segments have fixed symbol lists | Sector rotation modifies WEIGHTS within fixed universe, does not add/remove symbols |
| rub_oil_regime.py + RegimeStep | Both produce a regime_scale; which one wins? | Compose: `effective_regime_scale = min(rub_oil_scale, vix_regime_scale)` -- use the more conservative |
| dividend_gap + ATR stop | ATR trailing stop triggers on gap day (price drops by dividend amount) | Exempt dividend_gap positions from trailing stop for first 5 bars (grace period for gap to stabilize) |
| MOEX ISS data + Tinkoff API data | Timestamps from different sources in different timezones (ISS = MSK, Tinkoff = UTC) | Normalize all timestamps to UTC at fetch time; assert timezone-aware datetimes in all data pipelines |

---

## "Looks Done But Isn't" Checklist for v2.0

- [ ] **vol_target recalibration**: YAML files still show `vol_target: 0.19` for MOEX -- verify updated to 0.35-0.40 before any backtest
- [ ] **dividend calendar completeness**: `moex_dividends.yaml` has 43 events -- verify expanded to 150+ before dividend gap backtest
- [ ] **ex-date vs. last_buy_date alignment**: `DividendGapStrategy` triggers on `ex_date` from YAML -- verify this is the actual gap day, not the day before
- [ ] **rub_oil_regime.py wired into sizing**: file exists but is not referenced in `BacktestEngine` or `PositionSizingPipeline` -- verify wired before claiming regime overlay works
- [ ] **event_driven disabled in backtest**: `ru_blue_chips.yaml` has `event_driven.enabled: true` -- verify disabled or confirm 0 phantom signals in backtest
- [ ] **DEFAULT_STRATEGY_HOLD_BARS for dividend_gap**: config says 15, strategy expects 60 -- verify aligned
- [ ] **2022 data handling**: walk-forward includes Feb-Mar 2022 -- verify either excluded or handled with purge gap
- [ ] **toxic symbols removed**: GAZP, VTBR still in `ru_blue_chips` segments.py definition -- verify removed from MOEX equity backtests
- [ ] **sector rotation architecture**: decided as combiner strategy vs. sizing step -- verify sizing step approach before implementation
- [ ] **Brent-in-RUB not USD Brent**: Brent gate uses correct currency -- verify BZ=F * USDRUB, not raw BZ=F

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Look-ahead bias in dividend data | HIGH | Rebuild entire dividend calendar with announcement dates and cancelled events; re-run all backtests; previous iterations invalidated |
| Survivorship bias from 2022 | HIGH | Source pre-2022 data from MOEX ISS; rebuild walk-forward with separate pre/post periods; all 104 iterations invalid for comparison |
| Vol target miscalibration | LOW | Update YAML `vol_target` values; re-run backtests; quick fix, no architectural change |
| CBR regime timing error | MEDIUM | Redesign regime from "react to decision" to "lead with OFZ curve"; requires new data pipeline for OFZ yields |
| Combiner dilutes dividend signals | MEDIUM | Add `_EVENT_STRATEGIES` bypass; requires combiner refactor but isolated change |
| Sector rotation in wrong layer | HIGH | If implemented in combiner, must refactor to sizing pipeline; significant code move |
| T+1 settlement date confusion | LOW | Fix date matching in DividendGapStrategy; rename YAML field; re-run backtests |
| Multiplicative sizing reduction | MEDIUM | Reduce pipeline steps for MOEX or add total-scale floor; requires sizing pipeline redesign |

---

## Sources

- Codebase analysis: `src/finalayze/strategies/dividend_gap.py`, `src/finalayze/strategies/combiner.py`, `src/finalayze/risk/position_sizing_pipeline.py`, `src/finalayze/risk/rub_oil_regime.py`, `src/finalayze/strategies/presets/ru_blue_chips.yaml`, `src/finalayze/strategies/presets/moex_dividends.yaml`, `config/segments.py`, `src/finalayze/backtest/config.py`
- PROJECT.md: v2.0 milestone context, known issues, 104 REJECT iterations, toxic symbol identification
- MOEX dividend calendar: [MOEX Dividend Yield Listing](https://www.moex.com/ru/listing/dividend-yield.aspx) | [Smart-Lab Dividend Calendar](https://smart-lab.ru/dividends/)
- CBR key rate history: [CBR Key Rate](https://cbr.ru/eng/hd_base/KeyRate/) | [CBR Monetary Policy Guidelines 2024-2026](https://www.cbr.ru/eng/about_br/publ/ondkp/on_2024_2026/)
- MOEX 2022 crisis: [Bloomberg - Russian Stocks Slump Most on Record](https://www.bloomberg.com/news/articles/2022-02-24/russian-stocks-slump-most-on-record-on-ukraine-attack-chart)
- Sector rotation pitfalls: [Common Pitfalls of Sector Rotation - GIGAPRO](https://www.gwcindia.in/gigapro/blog/common-pitfalls-of-sector-rotation-and-how-to-avoid-them/) | [Sector Rotation Myth - Molchanov 2024](https://onlinelibrary.wiley.com/doi/10.1002/ijfe.2882)
- Dividend gap closure statistics: [T-Bank Invest - MOEX_TRADE analysis](https://www.tbank.ru/invest/social/profile/MOEX_TRADE/d9c7bf45-730e-4294-b376-557ca790fcdb/) | [Alfa Investor - Dividend Gap Closure](https://alfabank.ru/alfa-investor/t/moskovskaya-birzha-kak-bystro-zakroetsya-dividendnyy-gep/)
- MOEX market data: [Trading Economics - Russia Stock Market](https://tradingeconomics.com/russia/stock-market) | [Statista - Weekly MOEX Performance](https://www.statista.com/statistics/1254381/weekly-performance-moex/)

---
*Pitfalls research for: v2.0 MOEX Equity Profitability -- dividend gap, CBR regime, sector rotation*
*Researched: 2026-03-20*

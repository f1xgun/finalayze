# Feature Landscape: MOEX Equity Profitability (v2.0)

**Domain:** MOEX-native alpha strategies for Russian equity market
**Researched:** 2026-03-20
**Scope:** NEW features only -- dividend gap optimization, CBR regime trading, sector rotation, preferred share arbitrage, MOEX-specific ML
**Overall confidence:** MEDIUM (domain knowledge + web research; limited quantitative backtesting data available publicly)

---

## Table Stakes

Features that are minimum-viable for MOEX equity profitability. Without these, the system cannot generate positive Sharpe on Russian equities.

| Feature | Why Expected | Complexity | Existing Code | Notes |
|---------|--------------|------------|---------------|-------|
| Expanded dividend calendar (150+ events) | Current 43 events produce too few trades; dividend gap is the most documented MOEX alpha | LOW | `moex_dividends.yaml` has 43 entries; `DividendGapStrategy` exists | Must expand to cover SBER, SBERP, LKOH, TATN, TATNP, ROSN, NVTK, GMKN, CHMF, MTSS, PHOR + 2022-2025 history. T-Invest `get_dividends()` API fetches this data. |
| Dividend gap closure strategy tuning | Default params (min_gap 3%, max_hold 40 bars) are untested against real closure statistics | MEDIUM | `DividendGapStrategy` with configurable params | See Market Mechanics below. Rosneft closes same-day; Lukoil ~10 days; Tatneft varies wildly (19-255 days). Per-symbol max_hold_bars needed. |
| CBR rate regime gating for equities | 21% key rate environment suppresses equity returns; rate decisions cause 3-5% intraday moves on financials | MEDIUM | `cbr_calendar.py` exists but is NOT wired into combiner or backtest engine | CBR meets 8x/year (known schedule). Surprise hikes (+100-200bps) trigger sell-off then 3-5 day contrarian rebound. Already coded; needs integration. |
| Universe cleanup (toxic symbol removal) | GAZP, VTBR, SNGS, IRAO, ALRS account for ~60% of negative PnL per PROJECT.md | LOW | Symbols defined in `segments.py` | Remove from active universe or add negative weight bias. GAZP: sanctions-impaired, no dividends since 2022. VTBR: chronic dilution. |
| Brent price gate for energy sector | MOEX energy sector (40%+ of IMOEX) is highly correlated with Brent; trading energy without oil context is blind | MEDIUM | `rub_oil_regime.py` computes RUB/oil correlation but is NOT wired into strategy combiner | Must gate energy sector momentum: Brent > $75 = energy bullish, Brent < $60 = energy bearish. Use yfinance BZ=F for Brent data. |
| CBR rate direction as bond/equity allocation signal | In hiking cycles (2023-2024: 7.5% to 21%), equities underperform; in cutting cycles, equities rally | LOW | `CBRFetcher` provides key rate history; rate direction trivially computable | 2pp cut (21% to 18%) in Jul 2025 sent IMOEX +15% in 2 months per Forbes.ru data. |

### Market Mechanics: Dividend Gap Closure (MOEX-specific)

**Source:** spydell.livejournal.com historical analysis (2007-2017), finam.ru (2024-2025 data).

| Company | Typical Gap Size | Average Closure Time | Notes |
|---------|-----------------|---------------------|-------|
| Rosneft (ROSN) | 3-5% | Same day to 1 day | Fastest closer on MOEX. 9 of 10 last gaps closed within 1 trading day. |
| Lukoil (LKOH) | 5-8% | ~10 trading days | Consistent fast closer. 57 total trading days for ~30% returns over 5 years of gap events. |
| Sberbank (SBER) | 3-6% | 5-15 days (normal market) | Gap closure disrupted by sanctions events (2014: 226+ days). Filter by regime. |
| Tatneft (TATN) | 4-7% | 19 days (recent) to 255 days (worst) | Bimodal: either fast (under 20 days) or very slow. Requires regime filter. |
| Surgutneft pref (SNGSP) | 8-12% | Variable; 18 days (2015 for 21% yield) | Highest yields but unpredictable closure. FX-driven (USD cash pile). |
| Norilsk (GMKN) | 5-10% | 490+ total days (slow); typically 1-2 weeks normally | Macro-sensitive. Avoid during commodity downturns. |
| Gazprom (GAZP) | 0-6% | 324 days (historically slow) | EXCLUDE from dividend gap strategy. No dividends since 2022 ex-date. |

**Key insight:** Gap closure speed correlates with: (a) market expectations for future dividends, (b) no geopolitical stress, (c) overall MOEX trend. Strategy MUST include regime filter (RUB/oil correlation > 0.3 = normal).

### Market Mechanics: CBR Rate Meeting Impact

**Source:** CBR official calendar, Forbes.ru analysis, RBC Investitsii.

- CBR meets 8x per year on a published schedule (next: 2026-03-20, 2026-04-24, 2026-06-19...)
- Russian rate moves are 100-200bps (vs Fed's 25bps) -- outsized impact
- Press release at 13:30 MSK, press conference at 15:00 MSK
- Financial sector (SBER, VTBR, SBERP) reacts within minutes
- Surprise hike: -3% to -5% on financials day-of, then contrarian rebound bars 3-5
- Surprise cut: immediate +3% to +5% on financials, real estate (PIK, SMLT), and indebted companies (RUAL, MTLR)
- Rate direction matters more than level: cutting cycle = equity bullish, hiking cycle = equity bearish

---

## Differentiators

Features that provide competitive edge over simple MOEX trading bots. These are the alpha generators.

| Feature | Value Proposition | Complexity | Existing Code | Notes |
|---------|-------------------|------------|---------------|-------|
| Per-symbol adaptive max_hold for dividend gaps | Instead of fixed 40-bar hold, use historical closure statistics per symbol (ROSN: 2 bars, LKOH: 15, TATN: 30) | LOW | `DividendGapStrategy._max_hold_bars` is global | Reduces capital lock-up. ROSN gap closes same-day -- holding 40 bars wastes opportunity cost. |
| Dividend gap confidence scaling by historical closure rate | Higher confidence for symbols with >80% closure rate (ROSN, LKOH); lower for unreliable closers (GAZP, GMKN in downturns) | LOW | `_CONFIDENCE_SCALE` exists but is gap-size-based, not closure-rate-based | Add `closure_reliability` feature to confidence calculation. |
| Preferred share arbitrage (SBER/SBERP, TATN/TATNP, SNGS/SNGSP) | Pref-ordinary spread on MOEX ranges 5-30%; same dividends but different prices. Mean-reversion of spread is tradeable. | HIGH | `PairsStrategy` with Kalman hedge ratio exists; NOT configured for pref/ord pairs | SBER/SBERP spread narrowed from 30% to 5% historically. When spread widens beyond 1 sigma, buy pref + sell ordinary (or just buy pref if no short-selling). MOEX short-selling restrictions are a constraint. |
| Brent-conditional sector rotation | Rotate between energy (Brent > $75), financials (rate cut cycle), and defensive (MTSS, MGNT) sectors based on macro regime | HIGH | `rub_oil_regime.py` provides regime; sector segments defined in `segments.py` | Energy stocks (ROSN, LKOH, TATN) have 0.6-0.8 correlation with Brent. When Brent drops 20%+, rotate to defensive/consumer sectors. |
| CBR meeting pre-positioning | Enter financial sector positions 3-5 days before CBR meeting when rate cut is expected; exit T+1 after announcement | MEDIUM | `cbr_calendar.py` has `generate_cbr_signal()` with contrarian logic | Extend to pre-meeting positioning, not just post-surprise contrarian. Requires consensus rate expectation data (manual input or scraping). |
| ML ensemble with Russian macro features | Add CBR key rate, USDRUB, Brent price, IMOEX relative strength, and VIX-Russia (RVI index) as ML features | HIGH | ML pipeline exists for us_tech (XGBoost+LightGBM+CatBoost+meta-learner); 45 technical features | arxiv.org/abs/2503.08696 confirms multimodal (price + text) approach works on MOEX with 176 stocks. Add macro features: CBR rate delta, USDRUB 20d momentum, Brent/IMOEX beta, RVI level. |
| OFZ PK-to-PD rotation on CBR cutting cycle detection | When CBR starts cutting (21% to 18%), switch from floating-rate OFZ-PK to fixed-rate OFZ-PD to lock in high yields before they fall | MEDIUM | `bond_carry.py` (OFZ-PK) and `bond_duration_rotation.py` (OFZ-PD) exist separately | Rotation trigger: CBR cuts >= 2 consecutive meetings OR cumulative cut >= 200bps. Current OFZ-PK carry Sharpe +1.14; OFZ-PD would outperform in cutting cycle. |
| RUB crisis brake (portfolio-level) | When USDRUB moves >5% in 5 days or RUB/oil correlation breaks down (< 0.1), halt all new equity longs and increase OFZ allocation to 60% | LOW | `rub_oil_regime.py` provides crisis detection; circuit breaker exists | Geopolitical/sanctions events cause 10-20% RUB drops. System must protect capital. |
| Dividend reinvestment timing | After collecting dividend, immediately reinvest into the same stock if gap exists (compound the gap-closure alpha) | LOW | `DividendGapStrategy` handles entry; no reinvestment logic | Dividends credited T+25 working days on MOEX. Can pre-plan reinvestment. |

### Preferred Share Arbitrage: MOEX-Specific Mechanics

**Key pairs on MOEX with same dividends for ordinary and preferred:**

| Pair | Typical Spread | Dividend Equality | Liquidity | Tradability |
|------|---------------|-------------------|-----------|-------------|
| SBER / SBERP | 5-15% (was 30%) | Yes, same dividend per share | Both highly liquid | HIGH -- best pair for arb |
| TATN / TATNP | 5-10% | Yes, same dividend per share | Both liquid | HIGH |
| SNGS / SNGSP | 200-400% (ordinary much cheaper) | No -- SNGSP gets 10%+ yield, SNGS gets ~1% | SNGSP more liquid | MEDIUM -- not true arb, different economics |
| ROSN / ROSNP | N/A | Preferred rarely trades | Very low ROSNP liquidity | LOW -- skip |

**Constraint:** MOEX short-selling is restricted for retail investors via T-Invest. Strategy must be long-only on the undervalued leg (typically pref), or use the `PairsStrategy` spread z-score to time entries on the cheaper leg only.

### Brent-MOEX Correlation Specifics

- Energy stocks (ROSN, LKOH, TATN, SIBN) have 0.6-0.8 rolling correlation with Brent crude
- When Brent > $80: energy sector outperforms IMOEX by 5-10% annually
- When Brent < $60: energy sector underperforms by 10-15%; financials and consumer staples outperform
- Urals discount to Brent varies ($5-$25 depending on sanctions); direct Brent tracking is sufficient proxy
- MOEX Oil & Gas Index (MOEXOG) available via MOEX ISS API as sector benchmark

---

## Anti-Features

Features to explicitly NOT build for v2.0.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Intraday dividend gap scalping | Gap closure happens over days/weeks, not intraday. Intraday noise would trigger false exits. Daily bars are correct timeframe. | Keep daily bar resolution for dividend gap strategy. |
| Automated CBR consensus scraping | Consensus rate expectations change daily; scraping Russian financial sites is fragile and may violate ToS. | Manual input of expected rate before each CBR meeting (8x/year = low burden). Store in YAML. |
| Full sector rotation optimizer | Optimizing across 10+ sectors with macro inputs is an overfitting trap with MOEX's short liquid history (post-2022 sanctions). | Binary sector gates: energy ON/OFF based on Brent, financials ON/OFF based on CBR direction. Simple rules, not optimization. |
| Preferred share short-selling | T-Invest retail accounts cannot short most MOEX stocks. Building short-leg infrastructure is wasted effort. | Long-only on undervalued leg (buy pref when spread is wide). Use PairsStrategy z-score for timing only. |
| ML model with sanctions text features | Sanctions events are rare (2-3 per year), unpredictable, and regime-breaking. ML cannot learn from N=3 events. | Use RUB/oil correlation regime (already built) as sanctions proxy. News pipeline handles specific events. |
| PEAD strategy for MOEX | Russian companies have inconsistent earnings calendars; MOEX earnings surprises are not systematically measurable. | Focus on dividends (MOEX's primary corporate event with clean data) rather than earnings. |
| Cross-market arbitrage (MOEX vs London/HK listings) | Russian ADRs suspended since 2022 sanctions. No liquid cross-listed pairs available. | MOEX-only trading. |

---

## Feature Dependencies

```
[Dividend gap optimization]
    requires -> [Expanded dividend calendar: 43 -> 150+ events]
    requires -> [Per-symbol closure statistics (backtest analysis)]
    requires -> [Regime filter wired (rub_oil_regime.py)]
    enhances -> [Existing DividendGapStrategy code]

[CBR regime trading]
    requires -> [cbr_calendar.py wired into strategy combiner]
    requires -> [CBR meeting dates in YAML config]
    requires -> [Expected rate input (manual, 8x/year)]
    enhances -> [Existing cbr_calendar.py + generate_cbr_signal()]

[Brent-conditional sector rotation]
    requires -> [Brent price data (yfinance BZ=F -- already available)]
    requires -> [rub_oil_regime.py wired into combiner]
    requires -> [Sector segment allocation weights adjustable at runtime]
    depends-on -> [Universe cleanup (remove toxic symbols first)]

[Preferred share arbitrage]
    requires -> [PairsStrategy configured for SBER/SBERP, TATN/TATNP pairs]
    requires -> [Cointegration testing on pref/ord pairs (backtest)]
    requires -> [Long-only constraint enforcement (no short selling)]
    depends-on -> [Universe includes both SBER+SBERP, TATN+TATNP in same segment]

[ML with Russian macro features]
    requires -> [CBR rate, USDRUB, Brent as feature inputs to ML pipeline]
    requires -> [MOEX training data (3+ years via TinkoffFetcher)]
    requires -> [Walk-forward validation with quality gates]
    depends-on -> [Universe cleanup + dividend gap tuning (clean baseline first)]
    depends-on -> [Existing ML pipeline (us_tech architecture reused)]

[Portfolio allocation (40% OFZ + 60% equity)]
    requires -> [OFZ carry strategy (already Sharpe +1.14)]
    requires -> [Equity strategies with positive Sharpe (this milestone)]
    requires -> [RUB crisis brake (rub_oil_regime wired)]
    depends-on -> [All equity strategies tuned and validated]

[OFZ PK->PD rotation]
    requires -> [CBR cutting cycle detection]
    requires -> [bond_carry.py + bond_duration_rotation.py coordination]
    depends-on -> [CBR regime trading (uses same regime signal)]
```

---

## MVP Recommendation (v2.0 Minimum)

### Prioritize (must-have for positive MOEX Sharpe):

1. **Universe cleanup** -- remove GAZP, VTBR, SNGS, IRAO, ALRS. Immediate PnL improvement.
2. **Expanded dividend calendar** -- 150+ events from T-Invest API. Foundation for dividend gap alpha.
3. **Dividend gap strategy tuning** -- per-symbol max_hold, regime filter, closure-rate confidence.
4. **CBR regime gating** -- wire `cbr_calendar.py` into combiner. Block equity longs during hiking surprises.
5. **Brent energy gate** -- wire `rub_oil_regime.py` into combiner for energy sector positions.
6. **RUB crisis brake** -- halt new longs when RUB/oil correlation < 0.1.

### Add after baseline is positive:

7. **Preferred share arbitrage** -- configure PairsStrategy for SBER/SBERP, TATN/TATNP.
8. **Brent-conditional sector rotation** -- sector weight adjustment based on macro regime.
9. **CBR meeting pre-positioning** -- pre-meeting entry on financials when cut expected.

### Defer (only after positive Sharpe on equity):

10. **ML with Russian macro features** -- requires clean baseline data; overfitting risk without clean signals first.
11. **OFZ PK->PD rotation** -- already have Sharpe +1.14 on PK carry; rotation is optimization.
12. **Portfolio-level allocation optimizer** -- simple fixed 40/60 split first; optimize later.

---

## Feature Prioritization Matrix

| Feature | Alpha Potential | Implementation Cost | Risk of Overfitting | Priority |
|---------|----------------|--------------------|--------------------|----------|
| Universe cleanup | HIGH (remove -60% PnL drag) | LOW | NONE | P0 |
| Expanded dividend calendar | HIGH (3x more tradeable events) | LOW | NONE | P0 |
| Dividend gap tuning (per-symbol) | HIGH (documented 70%+ closure) | MEDIUM | LOW | P1 |
| CBR regime gating | MEDIUM (8 events/year) | MEDIUM | LOW | P1 |
| Brent energy gate | MEDIUM (filters bad energy trades) | LOW | LOW | P1 |
| RUB crisis brake | MEDIUM (drawdown protection) | LOW | NONE | P1 |
| Preferred share arbitrage | MEDIUM (mean-reversion of spread) | HIGH | MEDIUM | P2 |
| Sector rotation | MEDIUM (macro-driven allocation) | HIGH | HIGH | P2 |
| CBR pre-positioning | LOW-MEDIUM (8 events/year, needs consensus data) | MEDIUM | MEDIUM | P2 |
| ML with macro features | MEDIUM-HIGH (if done right) | HIGH | HIGH | P3 |
| OFZ PK->PD rotation | LOW (optimization of already-working strategy) | MEDIUM | LOW | P3 |

---

## Sources

### HIGH confidence (official, quantitative)
- CBR official calendar of rate decisions: [cbr.ru/eng/dkp/cal_mp/](https://www.cbr.ru/eng/dkp/cal_mp/)
- CBR key rate history: [cbr.ru/eng/hd_base/KeyRate/](https://cbr.ru/eng/hd_base/KeyRate/)
- MOEX Oil & Gas Index: [investing.com/indices/mcxog](https://www.investing.com/indices/mcxog)
- Codebase inspection: `dividend_gap.py`, `cbr_calendar.py`, `rub_oil_regime.py`, `pairs.py`, `segments.py`
- Project context: `.planning/PROJECT.md` (v2.0 requirements)

### MEDIUM confidence (financial media, research)
- Dividend gap closure statistics (2007-2017): [spydell.livejournal.com](https://spydell.livejournal.com/642950.html)
- Dividend gap recent data (2024-2025): [finam.ru historical analysis](https://www.finam.ru/publications/item/istoricheski-lukoyl-i-tatneft-obladayut-potentsialom-bystrogo-vosstanovleniya-posle-dividendnogo-gepa-20250604-0900/)
- CBR rate impact on financial sector: [Forbes.ru analysis](https://www.forbes.ru/investicii/543288-raduznye-nadezdy-kakie-akcii-vyrastut-iz-za-snizenia-stavki-cb)
- CBR rate cut impact on sectors: [RBC Investitsii](https://www.rbc.ru/quote/news/article/68497aae9a794711e7402f87)
- SBER/SBERP spread dynamics: [t-j.ru](https://t-j.ru/ask/sber-pref/)
- ML on MOEX stocks (multimodal): [arxiv.org/abs/2503.08696](https://arxiv.org/html/2503.08696)
- MOEX dividend calendar: [school.moex.com](https://school.moex.com/articles/dividendnyy-kalendar)
- Dividend gap general mechanics: [a2-finance.com](https://a2-finance.com/en/posts/the-dividend-gap)

### LOW confidence (general market context, needs validation)
- Brent-MOEX energy correlation magnitude (0.6-0.8) -- based on training data, not verified with recent data
- Preferred share spread ranges -- based on historical patterns, current levels need live verification
- MOEX short-selling restrictions for retail -- based on T-Invest documentation, may have changed

---

*Feature research for: MOEX Equity Profitability v2.0*
*Researched: 2026-03-20*

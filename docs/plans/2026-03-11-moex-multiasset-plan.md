# MOEX Multi-Asset Multi-Timeframe Trading System — Implementation Plan (v2)

**Date:** 2026-03-11 (v2: post-validation by quant-analyst, risk-officer, systems-architect, ml-engineer)
**Goal:** Autonomous MOEX trading across bonds (ОФЗ) and equities (акции) on multiple timeframes
**Capital:** 500K–2.5M RUB
**Max Drawdown:** 10% portfolio, 5% per layer
**Timeline:** 14 weeks to live trading (extended from 12 after review)
**Priority:** MOEX first, US later

---

## Validation History

- **v1 (2026-03-11):** Initial plan. Reviewed by 4 domain agents.
- **v2 (2026-03-11):** Addressed 9 CRITICAL + 27 HIGH issues from validation round 1.
  Key changes: BondBacktestEngine (separate from equity), per-layer broker instances,
  DV01 budgeting (not Kelly for bonds), duration cap 5Y, excess-return Sharpe,
  rule-based regime classifier with explicit thresholds, look-ahead guards,
  Phase 0 data validation, 4-week paper trading, stress scenarios.

---

## Executive Summary

The current equity-only MOEX system has negative Sharpe (-0.49). We're building a **multi-asset, multi-timeframe system** that combines:

- **Bonds (ОФЗ):** Carry + duration rotation in a CBR easing cycle (21% → 15.5%, targeting ~13%)
- **Equities (акции):** Mean reversion + dividend gap on liquid MOEX stocks
- **Multiple timeframes:** From short-term (1-5 days) to core holdings (6-12+ months)

**Performance targets (excess return over RUONIA):**
- Expected excess Sharpe: 0.3-0.8 (bond-dominated portfolio)
- Expected gross return: 15-22% nominal (vs RUONIA ~15% risk-free)
- Expected excess return: 2-7% above risk-free

*Note: The RUB risk-free rate (RUONIA/OFZ-PK) is ~15%. A portfolio returning 15% has excess Sharpe ≈ 0, not 1.5. All targets are stated as excess returns.*

### Portfolio Layer Architecture

| Layer | Horizon | Capital | Assets | Strategy | Rebalance |
|-------|---------|---------|--------|----------|-----------|
| **Core** | 6-12+ mo | 40-50% | ОФЗ-ПК (floaters) | Buy & hold, coupon harvest | Quarterly |
| **Strategic** | 1-6 mo | 25-30% | ОФЗ-ПД (fixed) | Duration rotation per CBR regime | Monthly / CBR meeting |
| **Tactical** | 1-4 wk | 15-20% | ОФЗ-ПД + Акции | CBR event trades, sector rotation | Around events |
| **Short** | 1-5 days | 10-15% | Акции | Mean reversion, дивгэпы | Daily signals |

---

## Architecture Design

### Key Decisions from Validation Review

**1. Separate BondBacktestEngine (not modifications to equity engine).**
The existing `BacktestEngine` (1426 lines) is equity-optimized with ADX routing, ML ensemble, grace bars, chandelier stops. Bond trading has fundamentally different mechanics. Creating a purpose-built `BondBacktestEngine` (~300-400 lines) is safer than a 2000-line frankenengine. Both engines share `PerformanceAnalyzer`, `IterationTracker`, and `TransactionCosts`.

**2. BondSimulatedBroker subclass (not mutations to base).**
`BondSimulatedBroker(SimulatedBroker)` overrides `_execute_buy` (adds NKD to cost), `_execute_sell` (adds NKD to proceeds), and adds `process_coupons(timestamp)` for mid-hold coupon income. Base `SimulatedBroker` stays untouched.

**3. Per-layer broker instances with explicit cash allocation.**
Each layer gets its own `SimulatedBroker`/`BondSimulatedBroker` instance with pre-allocated cash. Eliminates shared-pool ordering ambiguity and makes per-layer PnL directly observable. At periodic rebalance points, unused cash is redistributed across layers.

**4. DV01 budgeting for bonds (not Kelly).**
Kelly is calibrated on equity return distributions (unbounded, symmetric). Bond returns are bounded (par value), have known cash flows, and are duration-dependent. Replace `KellyStep` + `DurationStep` with `DV01BudgetStep` for bond layers. Each layer gets a DV01 budget; positions are sized so aggregate DV01 stays within budget.

**5. Move InstrumentType to Layer 0 (core/schemas.py).**
Currently at Layer 2 (markets/instruments.py). Both Layer 0 Signal and Layer 1 SegmentConfig need it. Move to core/schemas.py to respect dependency layering.

**6. Rule-based CBR regime classifier (not ML).**
Only 8 CBR meetings/year = ~24 data points over 3 years. ML would memorize or default to majority class. Use explicit rules with RUONIA-key rate gap as primary signal.

**7. Multi-timeframe: completed-period-only protocol.**
Weekly features for bar at time T use only the COMPLETED week ending before T. Monthly features use only the completed prior month. Apply `_EXTERNAL_DATA_LAG_BARS = 2` lag on top. No partial-period bars in features.

### New Abstractions

```
PortfolioLayer (enum): CORE, STRATEGIC, TACTICAL, SHORT
  - capital_pct: float          # target allocation of total portfolio
  - max_drawdown_pct: float     # independent peak-to-trough DD limit
  - max_positions: int          # position count limit per layer
  - rebalance_interval: str     # "daily", "weekly", "monthly", "event"
  - allowed_instrument_types: list[InstrumentType]

LayerLedger (persistent):
  - layer_id: PortfolioLayer
  - positions: dict[str, Decimal]  # symbol -> quantity
  - cash: Decimal
  - peak_equity: Decimal          # for peak-to-trough DD
  - current_dd_pct: Decimal

BondInfo (schema):
  - figi, ticker, isin: str
  - face_value: Decimal         # 1000 RUB for OFZ
  - coupon_rate: Decimal
  - coupon_frequency: int
  - maturity_date: date
  - floating_coupon: bool
  - class_code: str              # TQOB for OFZ

CouponPayment (schema):
  - bond_figi: str
  - payment_date: date
  - record_date: date            # T-2 business days before payment
  - amount_per_bond: Decimal

BondMath (utility, Layer 0 — pure computation, no I/O):
  - ytm(clean_price_pct, coupon_schedule, face_value) -> Decimal
  - modified_duration(ytm, coupon_schedule, face_value) -> Decimal
  - convexity(ytm, coupon_schedule, face_value) -> Decimal  # for deep-discount bonds
  - dv01(modified_duration, dirty_price) -> Decimal
  - dirty_price_pct(clean_price_pct, nkd_per_bond, face_value) -> Decimal
  - nkd(coupon_amount, days_since_last, coupon_period_days) -> Decimal
  Precision: YTM to 2 dp, duration to 2 dp, DV01 to 4 dp.
  Validation: test against published MOEX/CBR values for specific OFZ on specific dates.
```

### Engine Architecture

```
                    ┌─────────────────┐
                    │  run_iteration   │  orchestrator script
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
    ┌─────────────┐  ┌─────────────┐  ┌──────────────┐
    │ BacktestEngine│ │BondBacktest │  │ Portfolio     │
    │ (equities)   │ │Engine (OFZ) │  │ Aggregator    │
    └──────┬──────┘  └──────┬──────┘  └──────┬───────┘
           │                │                │
    SimulatedBroker  BondSimulatedBroker  Combines per-layer
    (per equity      (per bond layer)     PnL, computes
     layer)                               portfolio metrics
```

### Instrument Changes

```python
# Move to core/schemas.py (Layer 0)
InstrumentType = Literal["stock", "etf", "bond"]

# Extended Instrument (markets/instruments.py, Layer 2)
@dataclass(frozen=True)
class Instrument:
    symbol: str
    instrument_type: InstrumentType  # "stock" | "bond" | "etf"
    market_id: str
    currency: str
    lot_size: int
    figi: str | None = None
    segment_id: str | None = None
    # Bond-specific (None for stocks)
    face_value: Decimal | None = None
    coupon_rate: Decimal | None = None
    coupon_frequency: int | None = None
    maturity_date: date | None = None
    floating_coupon: bool = False

# Signal gets instrument_type (default "stock" for backward compat)
class Signal(BaseModel):
    ...
    instrument_type: str = "stock"  # "stock" or "bond"
```

### New Segments

```yaml
ru_ofz_pd:
  market: moex
  broker: tinkoff
  currency: RUB
  instrument_type: bond
  symbols: [SU26238RMFS4, SU26239RMFS2, SU26241RMFS8, SU26243RMFS4,
            SU26244RMFS2, SU26246RMFS7, SU26247RMFS5, SU26248RMFS3,
            SU26252RMFS5, SU26253RMFS3]
  active_strategies: [bond_duration_rotation]

ru_ofz_pk:
  market: moex
  broker: tinkoff
  currency: RUB
  instrument_type: bond
  symbols: [SU29007RMFS0, SU29008RMFS8, SU29009RMFS6, SU29010RMFS4]
  active_strategies: [bond_carry]

ru_blue_chips:  # existing, keep for equity layer
  # disable momentum/dual_momentum, focus on MR + dividend_gap
```

### Bond PnL Calculation

OFZ prices are quoted as clean price (% of face value). Actual cost includes NKD:

```
Entry cost per bond = (clean_entry_pct / 100) * face_value + nkd_entry
Exit proceeds per bond = (clean_exit_pct / 100) * face_value + nkd_exit
PnL per bond = exit_proceeds - entry_cost + coupons_received_during_hold
Total PnL = pnl_per_bond * quantity

TradeResult.coupon_income = sum of coupons received during hold
TradeResult.pnl = price PnL + coupon_income - transaction_costs
```

**NKD computation:** `nkd = coupon_amount * days_since_last_coupon / coupon_period_days`.
Computed from coupon schedule (known at issuance for OFZ-PD). For OFZ-PK, use
prevailing RUONIA (lagged) to estimate, NOT actual ex-post coupon.

**Daily equity tracking:** `broker.portfolio_value = cash + sum(qty * dirty_price_per_bond)`.
Dirty price changes daily even without market moves (NKD accrues).

### Transaction Costs

```python
# New in backtest/costs.py
MOEX_BOND_COSTS = TransactionCosts(
    commission_per_share=Decimal(0),
    commission_rate=Decimal("0.0005"),  # 0.05% (Tinkoff Trader tariff)
    min_commission=Decimal("0.01"),
    spread_bps=Decimal(5),    # On-the-run OFZ benchmark
    slippage_bps=Decimal(3),  # Normal conditions
)

MOEX_BOND_EVENT_COSTS = TransactionCosts(  # Around CBR meetings
    commission_rate=Decimal("0.0005"),
    spread_bps=Decimal(15),   # Widened around events
    slippage_bps=Decimal(10),
)

# Per-instrument spread override for off-the-run bonds
OFF_THE_RUN_SPREAD_UPLIFT = Decimal("10")  # +10 bps for 26238, 26239
```

### Tax Handling

OFZ coupon income: 13% NDFL for Russian tax residents.
Backtest deducts tax on coupon income: `net_coupon = gross_coupon * (1 - 0.13)`.
Capital gains tax (13%) deducted at year-end in backtest.

---

## Strategy Specifications

### S1: BondCarryStrategy (Core Layer)

**Purpose:** Coupon income from OFZ-PK floaters.

**Logic:**
1. Build a maturity ladder: equal face-value weight across 29007, 29008, 29009, 29010
2. As bonds approach maturity (<6 months), rotate into longest-dated floater
3. Reinvest net coupon income (after 13% NDFL) into underweight positions
4. Dynamic registry: if Minfin issues new OFZ-PK series, add them

**Expected return:** Tracks RUONIA + spread. Currently ~16-17% gross, ~14% net of tax.
**Important:** In easing scenario (CBR cuts to 13%), RUONIA drops to ~12.5-13%, and
floater yield drops to ~14% gross (~12% net). This is NOT fixed income — it tracks rates.

**Rebalance:** Quarterly + on coupon receipt dates
**Risk:** Minimal mark-to-market risk. During stress (2022), OFZ-PK lost 15-20% on MTM
even though RUONIA caught up. Core positions valued at amortized cost for DD calculation
(see Risk Management section).

### S2: BondDurationRotationStrategy (Strategic Layer)

**Purpose:** Capture price appreciation from CBR rate changes via duration management.

**Regime Classifier — Explicit Rules:**

Primary signal: RUONIA-key rate gap (highest quality real-time input).

```python
# Rule-based regime classifier
def classify_regime(
    key_rate: Decimal,
    ruonia_7d_avg: Decimal,
    cpi_yoy_latest_published: Decimal,  # ONLY published values, not estimates
    last_cbr_decision: str,             # "cut", "hold", "hike"
) -> Regime:
    gap = ruonia_7d_avg - key_rate  # negative = market prices in cuts

    # Step 1: base regime from RUONIA gap + last CBR decision
    if gap < Decimal("-0.50") and last_cbr_decision == "cut":
        regime = Regime.DOVISH       # target duration 4.0-5.0Y
    elif gap > Decimal("0.50") and last_cbr_decision == "hike":
        regime = Regime.HAWKISH      # target duration 0-1.5Y (shift to floaters)
    else:
        regime = Regime.NEUTRAL      # target duration 2.5-3.5Y

    # Step 2: CPI stagflation override (ALWAYS checked)
    # If inflation > 8% YoY, force at least NEUTRAL (never DOVISH during stagflation)
    if cpi_yoy_latest_published > Decimal("8.0"):
        regime = max(regime, Regime.NEUTRAL)

    return regime
```

*Note: HAWKISH gate uses symmetric threshold (gap > +0.50, matching DOVISH -0.50) AND
requires last_cbr_decision == "hike". This prevents false HAWKISH triggers from temporary
RUONIA spikes during liquidity events in early easing cycles.*

**CPI look-ahead guard:** Rosstat publishes monthly CPI with ~2 week lag. The strategy
uses ONLY CPI values from months whose publication date <= current bar date.
Implementation: maintain a `CPI_PUBLICATION_DATES` mapping.

**Duration targets by regime:**

| Regime | Target Modified Duration | OFZ-PD Selection | Max Position Duration |
|--------|-------------------------|-------------------|----------------------|
| Dovish | 4.0-5.0Y (capped at 5.0) | 26246, 26252, 26244 (medium-long) | 5.0Y |
| Neutral | 2.5-3.5Y | 26241, 26239 (medium) | 4.0Y |
| Hawkish | 0-1.5Y | Exit OFZ-PD → shift capital to Core (floaters) | 2.0Y |

*Duration cap reduced from 7Y to 5Y per risk review. At 5Y duration, a 200bps surprise
causes ~10% loss — right at the portfolio DD limit. At 7Y it would be 14%.*

**Signals:** `BUY` long-duration OFZ-PD in dovish, `SELL` and shorten in hawkish
**Expected excess return:** 3-8% above risk-free in favorable cycle
**Rebalance:** Around CBR meetings + monthly regime re-check

### S3: CBREventStrategy (Tactical Layer)

**Purpose:** Trade OFZ around CBR rate decisions.

**Realistic return calibration:** A 25bps surprise on duration-5Y bond = ~1.25% price move.
A 50bps surprise = ~2.5%. Expected range per correct trade: **1-3%** (not 2-5% as in v1).

**Look-ahead guard (CRITICAL):**
- Entry signal uses ONLY pre-meeting information: RUONIA gap, OFZ yield changes in prior week, publicly available CBR commentary
- Entry: 3-5 days before meeting
- Exit: mechanical T+1 or T+2 after announcement, NOT conditioned on actual decision
- The backtest MUST NOT use the actual rate decision to construct the entry signal

**Statistical note:** With 8 meetings/year and 2-3 year backtest window, this strategy
produces ~16-24 trades. This is insufficient for statistical validation (need ≥30).
CBREventStrategy will be validated via paper trading and domain judgment, not Sharpe.

**Transaction costs:** Use `MOEX_BOND_EVENT_COSTS` (wider spreads around meetings).

### S4: EquityMeanReversionStrategy (Short Layer) — EXISTING, RETUNE

**Changes from current:**
- **Disable:** momentum, dual_momentum (MOEX is mean-reverting)
- **Keep:** mean_reversion, ou_mean_reversion, rsi2_connors
- **Fix win/loss asymmetry:** widen profit targets (current too tight, cutting winners)
- **Do NOT tighten stops to 2.0 ATR** until backtest validates improvement (current -0.49 Sharpe)
- **Focus:** SBER, LKOH, YNDX, GAZP (showed some positive Sharpe in isolation)
- **Gate:** Do NOT include Short layer in live until MOEX equity Sharpe turns positive in backtest

### S5: DividendGapStrategy (Short Layer) — EXISTING

Keep dividend_gap strategy (2/2 wins in previous MOEX tests).
Focus: top-5 dividend payers. Hold 3-10 days post ex-date.

### S6: SectorRotationStrategy (Tactical Layer) — NEW, RULE-BASED

**Rules with hysteresis (not ML — only 2 features, 4 sector groups, ~52 rebalancing points):**

```python
# Enter overweight at |z| > 1.0, exit at |z| < 0.5 (hysteresis prevents whipsaw)
if brent_zscore_60d > 1.0 and not already_overweight_energy:
    overweight_energy()   # GAZP, LKOH, ROSN, TATN
elif brent_zscore_60d < 0.5 and overweight_energy:
    exit_overweight()

if usdrub_zscore_60d > 1.0:  # RUB weakening
    overweight_exporters()    # GMKN, LKOH
elif usdrub_zscore_60d < -1.0:  # RUB strengthening
    overweight_importers()    # YNDX
```

**Rebalance:** Bi-weekly. Hold period: 2-4 weeks.

---

## Risk Management Per Layer

### Per-Layer Isolation

Each layer operates as a virtual sub-account:
- **Own cash allocation** — hard reservation at initialization
- **Own position limits** — not shared with other layers
- **Own circuit breaker** — independent peak-to-trough DD monitoring
- **Own broker instance** — `SimulatedBroker` (equities) or `BondSimulatedBroker` (bonds)

At periodic rebalance points (monthly), unused cash is redistributed toward target allocations.

### Layer Risk Parameters

| Layer | Max DD | Max Positions | Stop-Loss | Circuit Breaker | Sizing |
|-------|--------|--------------|-----------|-----------------|--------|
| Core | 3% (MTM), exempt from L3 liquidation | 4 bonds | None (hold to maturity) | L3 only (portfolio) | Equal face-value weight |
| Strategic | 5% | 5 bonds | Yield +50bps from entry YTM | L2 at -3%, L3 at -5% | DV01 budget |
| Tactical | 5% | 5 (bonds + stocks) | Yield +30bps (bonds), 2.5 ATR (stocks) | L1 at -2%, L2 at -3% | DV01 (bonds) / Kelly (stocks) |
| Short | 5% | 6 stocks | 2.5 ATR | L1 at -1.5%, L2 at -3% | Half-Kelly |
| **Portfolio** | **10%** | — | — | L3 at -10%: liquidate Strategic+Tactical+Short | — |

### Core Layer MTM Exemption

Core (OFZ-PK floaters, hold-to-maturity intent) uses **amortized cost accounting** for DD
calculation, similar to banking book treatment. This means temporary MTM drops (like the 2022
crisis where floaters lost 15-20% on MTM) do not trigger circuit breakers for the Core layer.

However, portfolio-level L3 (-10%) still triggers liquidation of Strategic+Tactical+Short.
Core positions are NOT force-liquidated at L3, because they have committed cash flows.

### Duration Limits

| Regime | Max Portfolio Duration | Max Single Position Duration |
|--------|----------------------|----------------------------|
| Dovish | 5.0Y | 5.5Y |
| Neutral | 3.5Y | 4.0Y |
| Hawkish | 1.5Y | 2.0Y |

**Stress test requirement:** Before going live, validate that CBR +300bps, +500bps, +1000bps
scenarios do not breach 10% portfolio DD. If any breach occurs, tighten duration limits.

### DV01 Budgeting (replaces Kelly for bonds)

```python
# DV01 = modified_duration * dirty_price * 0.0001
# Budget: max portfolio DV01 = equity * max_dd_pct / expected_max_rate_move
# Example: 1.5M RUB equity, 5% DD limit, expect max 200bps move:
# max_dv01 = 1_500_000 * 0.05 / 200 = 375 RUB per basis point

def compute_position_size(dv01_budget_remaining, bond_dv01_per_unit):
    return min(
        dv01_budget_remaining / bond_dv01_per_unit,
        max_position_face_value,
    )
```

### Correlation Risk

MOEX bonds and equities correlate during stress (both driven by CBR/sanctions/oil).
- Compute trailing 60-day bond-equity correlation (IMOEX returns vs OFZ price returns)
- If correlation > 0.5: reduce combined position limit, increase cash buffer
- Add bonds to correlation position limit (an OFZ-PD portfolio = 1-2 correlated positions)
- Model the 2022 crisis explicitly in stress tests

### Stress Scenarios (Required Before Live)

| Scenario | CBR Action | MOEX | OFZ Impact | Portfolio Est. |
|----------|-----------|------|------------|---------------|
| Continued easing | Cut to 13% | +10-15% | OFZ-PD +10%, floaters yield drops | +5-8% excess |
| Rates flat | Hold 15.5% | ±5% | OFZ-PD ±1%, floaters ~15% | +1-3% excess |
| Moderate hawkish | Hike to 17% | -5-10% | OFZ-PD -7%, floaters +17% | -1% to +2% |
| **Severe crisis** | Hike to 20% | -30-40% | OFZ-PD -15-20%, market frozen | **-8% to -12%** |
| **2022 replay** | +1050bps to 25% | Closed 33 days | OFZ-PD -25-30% MTM | **-12% to -18%** |

*The plan acknowledges that a severe crisis / 2022-replay scenario can breach the 10% DD limit.
Mitigation: Core layer MTM exemption limits realized loss. Strategic/Tactical/Short get
liquidated at -10% portfolio DD. Recovery depends on holding Core to maturity.*

---

## Bond Universe

### OFZ-PD (Fixed Coupon) — Duration Rotation Candidates

| Series | Ticker | Coupon | Maturity | Est. Duration | Layer |
|--------|--------|--------|----------|--------------|-------|
| 26244 | SU26244RMFS2 | 11.25% | 15.03.2034 | ~5.0Y | Strategic |
| 26246 | SU26246RMFS7 | 12.00% | 12.03.2036 | ~5.8Y | Strategic (dovish only) |
| 26252 | SU26252RMFS5 | 12.50% | 12.10.2033 | ~4.5Y | Strategic |
| 26253 | SU26253RMFS3 | 13.00% | 06.10.2038 | ~6.2Y | Strategic (dovish only, near 5Y cap) |
| 26241 | SU26241RMFS8 | 9.50% | 17.11.2032 | ~4.2Y | Strategic/Tactical |
| 26239 | SU26239RMFS2 | 6.90% | 23.07.2031 | ~3.5Y | Tactical |
| 26243 | SU26243RMFS4 | 9.80% | 19.05.2038 | ~6.0Y | Strategic (dovish only) |
| 26238 | SU26238RMFS4 | 7.10% | 15.05.2041 | ~7.5Y | **Excluded** (exceeds 5Y cap, deep discount, convexity issues) |
| 26247 | SU26247RMFS5 | 12.25% | 11.05.2039 | ~6.5Y | **Excluded** (exceeds 5Y cap) |
| 26248 | SU26248RMFS3 | 12.25% | 16.05.2040 | ~6.8Y | **Excluded** (exceeds 5Y cap) |

*Post-review: excluded 3 bonds exceeding 5Y duration cap. Added 26243 (was missing in v1).*

### OFZ-PK (Floating Coupon) — Core Holdings

| Series | Ticker | Spread/RUONIA | Maturity | Notes |
|--------|--------|---------------|----------|-------|
| 29007 | SU29007RMFS0 | +1.30% | 03.03.2027 | Matures in <12mo, will rotate out |
| 29008 | SU29008RMFS8 | +1.40% | 03.10.2029 | Core |
| 29009 | SU29009RMFS6 | +1.50% | 05.05.2032 | Core |
| 29010 | SU29010RMFS4 | +1.60% | 06.12.2034 | Core |

*Dynamic registry: if Minfin issues new OFZ-PK (29024, 29025), add automatically.*

### FIGIs

Query T-Bank API at startup: `services.instruments.bonds(instrument_status=BASE)`,
filter by `class_code == "TQOB"`. Static fallback in `instruments.py` if API unavailable.
Cache dynamic result to `.cache/ofz_registry.json` for offline restarts.

### CBR Meeting Calendar 2026

| # | Date | Type | Status |
|---|------|------|--------|
| 1 | 13 Feb 2026 | Core | PAST (cut to 15.50%) |
| 2 | **20 Mar 2026** | Interim | NEXT |
| 3 | 24 Apr 2026 | Core | |
| 4 | 19 Jun 2026 | Interim | |
| 5 | 24 Jul 2026 | Core | |
| 6 | 11 Sep 2026 | Interim | |
| 7 | 23 Oct 2026 | Core | |
| 8 | 18 Dec 2026 | Interim | |

---

## Multi-Timeframe Feature Alignment Protocol

### Completed-Period-Only Rule (CRITICAL for look-ahead prevention)

```python
def align_higher_tf_to_daily(higher_tf_candles: list[Candle], daily_timestamp: date) -> Candle | None:
    """Return features from the most recent COMPLETED higher-TF bar.

    Weekly: use the bar whose period ended on the PREVIOUS Friday (or earlier).
    Monthly: use the bar for the PREVIOUS completed calendar month.

    Never use the in-progress partial period.
    """
    completed = [c for c in higher_tf_candles if c.period_end_date < daily_timestamp]
    if not completed:
        return None
    # Apply additional 2-bar lag for safety (matches _EXTERNAL_DATA_LAG_BARS)
    if len(completed) >= 3:
        return completed[-3]  # 2-bar lag from most recent completed
    return completed[0]
```

### MultiTimeframeContext

```python
@dataclass(frozen=True)
class MultiTimeframeContext:
    weekly_completed: Candle | None    # last COMPLETED weekly bar (with lag)
    monthly_completed: Candle | None   # last COMPLETED monthly bar (with lag)
    # Derived features
    weekly_rsi_14: float | None
    weekly_sma_50_ratio: float | None
    monthly_trend_direction: int | None  # +1, 0, -1
```

Strategies receive this as a parameter. Bond strategies access it for higher-TF trend context.
Equity strategies use it for regime confirmation.

---

## Implementation Phases

### Phase 0: Data Validation (Days 1-3) — NEW

**Blocker for all subsequent phases. Must confirm before starting.**

| # | Task | Validation |
|---|------|-----------|
| 0.1 | Confirm T-Bank API provides OFZ-PD candle data back to 2022-01-01 | `fetch_candles(figi, start=2022-01-01)` returns data |
| 0.2 | Confirm T-Bank API provides OFZ coupon schedules | `get_bond_coupons(figi)` returns future + past coupons |
| 0.3 | Confirm CBR key rate history available (CBRFetcher already built) | Back to 2022 |
| 0.4 | Cross-reference bond math: compute YTM/duration for 3 OFZ series, compare to published MOEX values | Match within 0.05% |
| 0.5 | Assess MOEX closure gaps in data (Feb-Mar 2022) | Document gap handling strategy |

**If any validation fails:** Identify alternative data sources (MOEX ISS for historical candles,
cbr.ru for coupon schedules) before proceeding.

### Phase 1: Bond Infrastructure (Weeks 1-2)

**Week 1: Data Layer**

| # | Task | Files | Tests | Agent |
|---|------|-------|-------|-------|
| 1.1 | Move InstrumentType to core/schemas.py (Layer 0) | core/schemas.py, markets/instruments.py | Update imports | core-agent |
| 1.2 | Bond schemas: BondInfo, CouponPayment, AccruedInterest | core/schemas.py | unit/test_bond_schemas.py | core-agent |
| 1.3 | Extend Instrument with bond fields | markets/instruments.py | unit/test_instruments.py | markets-agent |
| 1.4 | TinkoffFetcher: fetch_bonds(), fetch_bond_coupons(), fetch_accrued_interest() | data/fetchers/tinkoff_data.py | unit/test_tinkoff_bond.py | data-agent |
| 1.5 | Dynamic OFZ registry with static fallback | markets/instruments.py | unit/test_ofz_registry.py | markets-agent |
| 1.6 | CBR meeting calendar + CPI publication dates | data/fetchers/cbr.py | unit/test_cbr_calendar.py | data-agent |
| 1.7 | Bond segment configs: ru_ofz_pd, ru_ofz_pk + instrument_type on SegmentConfig | config/segments.py | unit/test_segments.py | config-agent |

**Week 2: Bond Math + Engine**

| # | Task | Files | Tests | Agent |
|---|------|-------|-------|-------|
| 2.1 | Bond math library (YTM Newton-Raphson, modified duration, convexity, DV01, NKD) | core/bond_math.py | unit/test_bond_math.py (validate vs published MOEX values) | core-agent |
| 2.2 | TradeResult: add coupon_income, instrument_type fields | core/schemas.py | update existing tests | core-agent |
| 2.3 | BondSimulatedBroker subclass: NKD-adjusted buy/sell, process_coupons() | execution/bond_simulated_broker.py (new) | unit/test_bond_broker.py | execution-agent |
| 2.4 | BondBacktestEngine: separate engine for bonds, dirty-price PnL, coupon tracking | backtest/bond_engine.py (new) | unit/test_bond_engine.py | backtest-agent |
| 2.5 | MOEX_BOND_COSTS + MOEX_BOND_EVENT_COSTS | backtest/costs.py | unit/test_costs.py | backtest-agent |
| 2.6 | DV01BudgetStep for bond position sizing | risk/position_sizing_pipeline.py | unit/test_dv01_sizing.py | risk-agent |
| 2.7 | Yield-based stop-loss function | risk/stop_loss.py | unit/test_yield_stop.py | risk-agent |

**Bond math test requirements:**
- YTM convergence edge cases: at-par, deep discount (<70%), near-maturity (1 coupon left)
- Validate against published MOEX/CBR values for 3 specific OFZ series on specific dates
- Leap year NKD calculation
- Bond maturity forced-close in mid-backtest
- Coupon on record-date boundary (T-2 business days before payment)

### Phase 2: Bond Strategies (Weeks 3-4)

| # | Task | Files | Tests | Agent |
|---|------|-------|-------|-------|
| 3.1 | BondCarryStrategy | strategies/bond_carry.py (new) | unit/test_bond_carry.py | strategies-agent |
| 3.2 | BondDurationRotationStrategy with explicit rule-based regime classifier | strategies/bond_duration_rotation.py (new) | unit/test_duration_rotation.py | strategies-agent |
| 3.3 | CBREventStrategy with look-ahead guards | strategies/cbr_event.py (new) | unit/test_cbr_event.py | strategies-agent |
| 3.4 | Bond strategy preset YAMLs | strategies/presets/ | integration | strategies-agent |
| 3.5 | Bond backtest validation on **2022-2026** (full cycle: hike+pause+cut) | scripts/ | backtest results | backtest-agent |
| 3.6 | Bond performance metrics: excess return over RUONIA, duration-adj Sharpe | backtest/performance.py | unit | backtest-agent |
| 3.7 | SectorRotationStrategy (rule-based with hysteresis) | strategies/sector_rotation.py (new) | unit | strategies-agent |

**Backtest window:** Extended to 2022-2026 (4 years, covers full rate cycle:
9.5% → 20% → 21% → 15.5%). Walk-forward with bond-appropriate parameters:
24-month train, 12-month test, 6-month step. Minimum trades per fold: 5 (not 30 — event strategies).

### Phase 3: Multi-Timeframe + Equity + Integration (Weeks 5-7)

*Can start some tasks in parallel with Phase 2 (4.1, 4.5 are independent of bond infra).*

| # | Task | Files | Tests | Agent |
|---|------|-------|-------|-------|
| 4.1 | MultiTimeframeContext with completed-period-only protocol | core/schemas.py, ml/features/ | unit (verify no partial bars) | ml-agent |
| 4.2 | PortfolioLayer enum + LayerLedger (persistent, DB-backed) | core/schemas.py, core/layer_ledger.py | unit | core-agent |
| 4.3 | Per-layer broker instances in run_iteration.py | scripts/run_iteration.py | integration | backtest-agent |
| 4.4 | Portfolio Aggregator: combines per-layer PnL, computes portfolio metrics | backtest/portfolio_aggregator.py (new) | unit | backtest-agent |
| 4.5 | MOEX equity rehab: disable momentum, focus on MR + dividend_gap | strategies/presets/ru_*.yaml | backtest | strategies-agent |
| 4.6 | Retrain MOEX ML models (MI selection capped at 10 features) | scripts/train_models.py | training | ml-agent |
| 4.7 | Per-layer circuit breakers (peak-to-trough DD, not intraday) | risk/circuit_breaker.py | unit | risk-agent |
| 4.8 | Portfolio-level DrawdownMonitor at 10% threshold | risk/ | unit | risk-agent |

### Phase 4: Portfolio Validation (Weeks 8-9)

| # | Task | Agent |
|---|------|-------|
| 5.1 | Combined portfolio backtest: all layers, bonds + equities (2022-2026) | backtest-agent |
| 5.2 | Cross-layer DD cascading tests | risk-agent |
| 5.3 | Walk-forward validation (bond-appropriate parameters) | evaluation-agent |
| 5.4 | Stress tests: CBR +300bps, +500bps, +1000bps, 2022 replay | risk-officer |
| 5.5 | Bond-equity correlation analysis (trailing 60-day) | quant-analyst |
| 5.6 | Verify excess Sharpe > 0.3 over RUONIA | quant-analyst |

**Exit criteria:**
- [ ] Excess Sharpe over RUONIA > 0.3
- [ ] All stress scenarios: portfolio DD < 10% (excluding Core MTM exemption)
- [ ] No look-ahead bias detected (CBR decisions, CPI dates, multi-TF)
- [ ] Bond math validated against published values (3 OFZ series)
- [ ] Backtest reproducible (same inputs → same outputs)

### Phase 5: Paper Trading (Weeks 10-13) — Extended to 4 Weeks

Must cover at least 1 CBR meeting (next: March 20, April 24).

| # | Task | Agent |
|---|------|-------|
| 6.1 | Deploy to VPS (Tinkoff Sandbox mode) | infra-agent |
| 6.2 | Smoke test all layers (bond orders, coupon tracking, equity signals) | execution-agent |
| 6.3 | Monitor 28 days including ≥1 CBR meeting | manual |
| 6.4 | Fix issues, validate CBREventStrategy on live data | appropriate agent |
| 6.5 | Telegram alerts for bond events (coupons, CBR decisions) | core-agent |

**Rollback criteria:** If any layer underperforms backtest Sharpe by >1 std dev in first
4 weeks of paper, disable that layer and investigate.

### Phase 6: Live Ramp-Up (Week 14+)

| # | Step | Capital | Duration | Pass Criteria |
|---|------|---------|----------|---------------|
| 6.1 | Core only (OFZ-PK floaters) | 40% of capital | 1 week | Coupons received, no errors |
| 6.2 | Add Strategic (OFZ-PD duration rotation) | +25% | 2 weeks | CBR meeting navigated correctly |
| 6.3 | Add Tactical (event + sector rotation) | +20% | 2 weeks | Event trade executed, no anomalies |
| 6.4 | Add Short (equities) **ONLY IF** equity Sharpe positive | +15% | ongoing | Positive contribution |

*If MOEX equity Sharpe remains negative after Phase 4, the Short layer is NOT deployed.
Capital is redistributed: Core 55%, Strategic 30%, Tactical 15%.*

---

## Expected Portfolio Returns (1.5M RUB, Excess Over RUONIA)

### Scenario: Continued Easing (CBR 15.5% → 13%)

| Layer | Capital | Gross Return | Risk-Free | Excess | Excess P&L |
|-------|---------|-------------|-----------|--------|-----------|
| Core (floaters) | 675K | 14% (drops with rates) | 14% | 0% | 0K |
| Strategic (dur) | 375K | 18% | 14% | 4% | +15K |
| Tactical | 300K | 12% | 14% | -2% | -6K |
| Short | 150K | 5% | 0% (equity) | 5% | +7.5K |
| **Total** | **1.5M** | — | — | **1.1%** | **+16.5K** |

*Note: Core layer excess return = 0 in easing because floater yield tracks rates down.*

### Scenario: Rates Flat

| Layer | Capital | Excess Return | Excess P&L |
|-------|---------|--------------|-----------|
| Core | 675K | 0% (tracks RUONIA) | 0K |
| Strategic | 375K | 1% (coupon > MTM drag) | +3.75K |
| Tactical | 300K | 0% | 0K |
| Short | 150K | 2% | +3K |
| **Total** | **1.5M** | **0.45%** | **+6.75K** |

### Scenario: Moderate Hawkish (CBR hikes to 17%)

| Layer | Capital | Excess Return | Excess P&L |
|-------|---------|--------------|-----------|
| Core | 675K | +2% (higher RUONIA) | +13.5K |
| Strategic | 375K | -5% (if classifier late) | -18.75K |
| Tactical | 300K | -3% | -9K |
| Short | 150K | -5% | -7.5K |
| **Total** | **1.5M** | **-1.45%** | **-21.75K** |

### Scenario: Severe Crisis (2022 replay)

| Layer | Capital | Impact | Notes |
|-------|---------|--------|-------|
| Core | 675K | MTM -20%, but hold to maturity | No realized loss if held |
| Strategic | 375K | -10% (liquidated at DD limit) | -37.5K realized |
| Tactical | 300K | -8% (liquidated) | -24K realized |
| Short | 150K | -10% (liquidated) | -15K realized |
| **Total realized** | — | **-5.1% of portfolio** | **-76.5K** |
| **Including Core MTM** | — | **-15% of portfolio** | **But Core recovers at maturity** |

*Honest assessment: a severe crisis causes real pain. The system limits realized losses
to ~5% by liquidating non-Core layers. Core floaters recover at maturity.*

---

## Agent Team Updates Needed

No new agents required. Update specs for existing agents:

| Agent | Update |
|-------|--------|
| **strategies-agent** | Add bond strategy patterns (carry, duration rotation, CBR event) |
| **backtest-agent** | Add BondBacktestEngine scope, dirty-price PnL, coupon tracking |
| **risk-agent** | Add DV01 budgeting, yield-based stops, duration limits |
| **data-agent** | Add T-Bank bond metadata API, coupon schedule fetching |
| **execution-agent** | Add BondSimulatedBroker subclass scope |
| **core-agent** | Add bond_math.py, LayerLedger scope |

---

## Dependencies & Prerequisites

| Dependency | Status | Action |
|-----------|--------|--------|
| T-Bank API access | Have token | Ready |
| CBRFetcher (FX + key rate) | Merged | Ready |
| MoexISSFetcher (IMOEX, turnover) | Merged | Ready |
| MOEX ML features (4 z-scores) | Merged | Ready |
| Historical OFZ candle data | **VALIDATED** (Phase 0, 2026-03-11) | 12/12 bonds OK |
| OFZ coupon schedules via API | **VALIDATED** (Phase 0, 2026-03-11) | 12/12 bonds OK |
| Accrued interest (NKD) via API | **VALIDATED** (Phase 0, 2026-03-11) | 4/4 sampled OK |
| Bond math reference values | **Not collected** | Phase 1 (task 2.1) |
| MOEX closure gap handling | **Assessed** | 8 bars available during closure for long-lived OFZ |

---

## Success Criteria

### Phase 0 Exit (Day 3) — PASSED (2026-03-11)
- [x] T-Bank API returns OFZ data back to 2022 (12/12 bonds, 75-1047 daily bars)
- [x] Coupon schedules available (11-28 coupons per bond, paid+future)
- [x] NKD (accrued interest) available daily
- [x] MOEX closure gap assessed (8 bars available for pre-2022 bonds)
- [ ] Bond math matches published values for 3 OFZ series (deferred to Phase 1 task 2.1)

**Phase 0 findings:**
- 26252 (95 bars, from Oct 2025) and 26253 (75 bars, from Nov 2025) have insufficient history for WF
- All OFZ-PD bonds exceed 5Y maturity-to-today, but modified duration at ~15% yields is ~60-70% of maturity → OK
- OFZ-PK coupon amounts vary as expected (RUONIA-linked): 29007 shows 32→45→69 RUB progression
- Bond FIGIs obtained: 26238=BBG011FJ4HS6, 26239=BBG011FHF1F7, 26241=BBG01BJBR2W0, etc.

### Phase 1-2 Exit (Week 4)
- [ ] BondBacktestEngine runs end-to-end (separate from equity engine)
- [ ] OFZ-PD backtest on 2022-2026 shows positive excess Sharpe over RUONIA
- [ ] Coupon income correctly tracked (after 13% NDFL tax)
- [ ] Bond math validated (YTM to 2dp, duration to 2dp)
- [ ] Backtest is deterministically reproducible

### Phase 3-4 Exit (Week 9)
- [ ] Combined portfolio excess Sharpe over RUONIA > 0.3
- [ ] All stress scenarios: realized portfolio DD < 10%
- [ ] Walk-forward validation: OOS excess Sharpe > 0.0
- [ ] No look-ahead bias found
- [ ] Layer isolation verified (per-layer PnL independent)

### Phase 5-6 Exit (Week 14)
- [ ] 28+ days paper trading including ≥1 CBR meeting
- [ ] Paper P&L positive or > -2%
- [ ] Core + Strategic layers live
- [ ] Short layer live ONLY IF equity Sharpe positive

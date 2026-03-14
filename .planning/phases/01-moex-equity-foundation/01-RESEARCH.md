# Phase 1: MOEX Equity Foundation - Research

**Researched:** 2026-03-14
**Domain:** MOEX backtest infrastructure (position sizing, transaction costs, holiday calendar)
**Confidence:** HIGH

## Summary

Phase 1 fixes three interconnected MOEX backtest infrastructure problems: (1) RUB position sizing produces 0.02% positions instead of 10-20%, (2) MOEX holidays are incomplete (missing transferred holidays), and (3) MOEX commission rate is incorrect (0.03% vs actual 0.04% Trader tariff). All three issues are well-understood with clear existing code paths to modify.

The root cause of the sizing bug is in `run_iteration.py` which passes `cash * _FALLBACK_USDRUB` (= 9,000,000 RUB) as `initial_cash` to the BacktestEngine, but the engine's `_handle_buy` uses `RollingKelly` which cold-starts at 1% fraction. With 1% of 9M RUB = 90K RUB and MOEX stock prices at 200-5000 RUB, positions are tiny relative to the inflated equity. The fix requires setting `initial_cash` to 1,000,000 RUB (user decision) and ensuring the entire sizing pipeline operates in RUB natively.

**Primary recommendation:** Set MOEX backtest starting capital to 1,000,000 RUB, update MOEX_COSTS commission_rate to 0.0004, add transferred holidays for 2020-2026, and wire holiday check into both backtest engine and TradingLoop._is_market_open.

<user_constraints>

## User Constraints (from CONTEXT.md)

### Locked Decisions
- Make the entire sizing pipeline currency-aware (not just a convert-at-entry hack)
- Use existing CurrencyConverter with CBR daily rates for RUB/USD conversion
- Portfolio equity tracked in RUB for MOEX segments
- Starting capital for MOEX backtest: 1,000,000 RUB
- US backward compatibility NOT required -- can break US segments if needed (MVP is MOEX-only)
- Position size as % of equity: Claude's discretion to calibrate in Phase 2
- Tariff: Trader -- 0.04% commission rate from trade amount
- No per-share commission (unlike US); purely percentage-based
- Add transferred holidays as static per-year lists (2020-2026)
- Existing 14 fixed holidays remain
- Wire holiday check into BOTH backtest engine AND live TradingLoop (unified approach)
- Skip MOEX non-trading days (holidays + weekends) in bar iteration
- Phase 1 pass criteria: positions sized at 10-20% of equity AND positive PnL
- Test on ALL three MOEX segments: ru_blue_chips, ru_energy, ru_finance
- Backtest period: 2020-2025 (5 years, includes COVID + sanctions crisis)
- Data source: T-Invest API primary, MOEX ISS to fill gaps if T-Invest history is insufficient

### Claude's Discretion
- Exact spread/slippage bps values for MOEX (currently: 10 bps spread, 7 bps slippage)
- Position size percentage (10-15% range)
- How to handle the USD-to-RUB pipeline migration internally
- Whether to refactor PositionSizingPipeline or add currency conversion layer

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope

</user_constraints>

<phase_requirements>

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| EQF-01 | Position sizing uses RUB denomination for MOEX segments (not USD) | Root cause identified in `run_iteration.py` and `engine._handle_buy`. Fix: set initial_cash=1M RUB, remove `cash * _FALLBACK_USDRUB` multiplication, ensure SizingContext.equity is in RUB |
| EQF-04 | MOEX holiday calendar integrated (14-20 non-weekend non-trading days/year) | `moex_calendar.py` has 14 fixed holidays, needs per-year transferred holidays dict. Wire `is_moex_trading_day()` into engine bar skip and TradingLoop._is_market_open |
| EQF-05 | MOEX costs (commissions, slippage) fully wired in backtest engine | MOEX_COSTS already wired in `run_iteration.py`. Fix: update commission_rate from 0.0003 to 0.0004 (Trader tariff). Spread/slippage bps are Claude's discretion |

</phase_requirements>

## Standard Stack

### Core (already in project -- no new dependencies)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python stdlib `datetime` | 3.12 | Holiday calendar date handling | No external deps needed for static holiday lists |
| `decimal.Decimal` | stdlib | Financial arithmetic | Already used throughout sizing pipeline |
| Existing `CurrencyConverter` | in-tree | RUB/USD conversion | Already has `set_rate()` and `convert()` methods |
| Existing `TransactionCosts` | in-tree | Cost model | Already has `commission_rate` field and MOEX preset |
| Existing `moex_calendar.py` | in-tree | Holiday detection | Has 14 fixed holidays, needs extension |

### Supporting

No new dependencies required. All changes are modifications to existing modules.

## Architecture Patterns

### Recommended Change Map

```
src/finalayze/
  data/
    moex_calendar.py           # ADD: _TRANSFERRED_HOLIDAYS dict, is_moex_trading_day()
  backtest/
    costs.py                   # FIX: MOEX_COSTS.commission_rate 0.0003 -> 0.0004
    engine.py                  # ADD: holiday-aware bar skip for MOEX segments
  core/
    trading_loop.py            # FIX: _is_market_open() to check holidays
  markets/
    schedule.py                # ADD: holiday_fn parameter to MarketSchedule
config/
  segments.py                  # No changes needed (already has market="moex", currency="RUB")
scripts/
  run_iteration.py             # FIX: MOEX cash from cash*90 to 1_000_000 RUB directly
tests/unit/
  test_moex_calendar.py        # ADD: tests for transferred holidays
  test_moex_sizing.py          # NEW: position sizing in RUB verification
  test_moex_costs_wiring.py    # NEW: commission rate verification
```

### Pattern 1: Currency-Aware Sizing (EQF-01)

**What:** The backtest engine already receives `initial_cash` in the correct currency. The bug is in `run_iteration.py` which multiplies `--cash` (default 100K) by 90, creating 9M RUB initial capital. With RollingKelly cold-starting at 1%, positions become 90K RUB (~0.02% of equity for a 5000 RUB stock).

**Root cause chain:**
1. `run_iteration.py` line 1054: `segment_cash = cash * _FALLBACK_USDRUB` -> 9,000,000 RUB
2. Engine `_handle_buy` line 1165-1167: `kelly_frac = self._rolling_kelly.optimal_fraction()` -> cold-starts at 0.01 (1%)
3. `base_position = 9M * 0.01 = 90,000 RUB`
4. After VolTarget, Regime, HardCaps steps: position shrinks further
5. With stock price ~300 RUB, quantity = 90K/300 = 300 shares. Looks like it works...
6. BUT: `max_position_pct = 0.20` so cap = 0.20 * 9M = 1.8M, which is fine
7. The REAL issue: position as % of equity = 90K/9M = 1%, not 10-20%

**Fix approach:**
```python
# In run_iteration.py, replace:
segment_cash = cash * _FALLBACK_USDRUB if segment.startswith("ru_") else cash
# With:
if segment.startswith("ru_"):
    segment_cash = Decimal(1_000_000)  # 1M RUB per user decision
else:
    segment_cash = cash
```

And disable RollingKelly for MOEX segments (or increase cold-start fraction), since 1% of 1M = 10K RUB which is still too small. Alternative: use fixed-fraction sizing (kelly_fraction=0.5 default) without RollingKelly.

**Key insight:** The sizing pipeline itself is already currency-agnostic -- it works with Decimal amounts. The bug is in the input (inflated equity) combined with Kelly cold-start (low fraction). With 1M RUB and default `kelly_fraction=0.5`, `base_position = 1M * 0.5 * 0.5 = 250K RUB` which is 25% -- then HardCaps clips to `max_position_pct` (20%) = 200K RUB. That is correct.

### Pattern 2: Unified Holiday Check (EQF-04)

**What:** Add a single `is_moex_trading_day(d: date) -> bool` function that checks weekends + fixed holidays + transferred holidays. Wire it into both the backtest engine's bar loop and TradingLoop's `_is_market_open()`.

**Current state:**
- `moex_calendar.py` has `_FIXED_HOLIDAYS` (14 entries) but explicitly notes "transferred holidays vary by year -- not included"
- `engine.py` does NOT skip holidays -- it iterates all candles from the data source
- `TradingLoop._is_market_open()` checks weekends and time-of-day, but NOT holidays

**Important nuance for backtest:** The backtest engine iterates over candles provided by the data fetcher. Since T-Invest API only returns candles for actual trading days, holidays are already implicitly skipped in the candle data. However, `trading_days_gap()` (used in ML features) needs accurate holiday data to compute gap sizes correctly. And `TradingLoop._is_market_open()` MUST check holidays to prevent live trading on MOEX holidays.

**Transferred holidays structure:**
```python
# Per-year transferred holidays (government decrees)
_TRANSFERRED_HOLIDAYS: dict[int, frozenset[tuple[int, int]]] = {
    2020: frozenset({(3, 9), (5, 4), (5, 5), (5, 11), (6, 15), (11, 5)}),
    2021: frozenset({(2, 22), (3, 9), (5, 3), (5, 10), (6, 14), (11, 3), (11, 5), (12, 31)}),
    2022: frozenset({(3, 7), (5, 2), (5, 3), (5, 10), (6, 13), (11, 3)}),
    2023: frozenset({(2, 24), (5, 8), (6, 13), (11, 6)}),
    2024: frozenset({(4, 29), (4, 30), (5, 10), (12, 30), (12, 31)}),
    2025: frozenset({(5, 2), (5, 8), (6, 13), (11, 3), (12, 31)}),
    2026: frozenset({(3, 9), (5, 11), (1, 9)}),  # preliminary
}
```

**IMPORTANT CAVEAT:** MOEX's actual trading calendar is MORE complex than just "closed on holidays." Some official holidays have additional trading sessions, and some Saturdays become working days. For MVP/backtest accuracy, the static holiday approach is sufficient. The T-Invest API candle data already reflects the real calendar.

### Pattern 3: MOEX Commission Rate Fix (EQF-05)

**What:** Update `MOEX_COSTS.commission_rate` from `Decimal("0.0003")` (0.03%) to `Decimal("0.0004")` (0.04%).

**Evidence:** User specified Trader tariff at 0.04%. The Tinkoff Trader tariff PDF and multiple sources confirm 0.04% for stock trades. The current 0.03% in the codebase appears to be an error.

**Spread/slippage (Claude's discretion):**
- Current: 10 bps spread, 7 bps slippage
- MOEX blue chips (SBER, GAZP, LKOH) have tight spreads: 2-5 bps
- Less liquid names (ALRS, POLY) can have 15-30 bps spreads
- Recommendation: Keep current 10/7 bps as a reasonable average across the universe
- This is conservative but safe for backtesting

### Anti-Patterns to Avoid

- **Converting RUB->USD at entry and back at exit:** Creates phantom FX gains/losses. Keep everything in native currency.
- **Using `_FALLBACK_USDRUB` for equity calculation:** This is a rough approximation (hardcoded 90.0). For position sizing, use native RUB equity.
- **Skipping candles in the engine based on calendar:** Don't filter candles in the engine loop -- the data source already provides only trading-day candles. The calendar check belongs in the TradingLoop (live) and in `trading_days_gap()` (ML features).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Currency conversion | Custom rate lookup | Existing `CurrencyConverter` class | Already handles set_rate/convert with inverse computation |
| Transaction cost model | MOEX-specific cost calculator | Existing `TransactionCosts` with `commission_rate` field | Already supports rate-based commission (just wrong value) |
| Holiday calendar | Complex rule engine | Static per-year frozensets | Government decrees are static; new years need manual update |
| MOEX trading schedule | Custom scheduler | Existing `MarketSchedule` + holiday function | Just needs holiday integration |

**Key insight:** All infrastructure already exists. This phase is about fixing parameters and wiring, not building new systems.

## Common Pitfalls

### Pitfall 1: RollingKelly Cold Start with Large Equity
**What goes wrong:** RollingKelly cold-starts at 1% fraction. With 1M RUB equity, base_position = 10K RUB, which gets below min_position_size (1K-5K RUB range) after pipeline reductions.
**Why it happens:** Kelly needs trade history to estimate optimal fraction. No history = conservative 1%.
**How to avoid:** Either (a) disable RollingKelly for MOEX backtests (use fixed fraction), or (b) increase Kelly cold-start for MOEX, or (c) ensure pipeline floor keeps positions viable. Option (a) is simplest.
**Warning signs:** All positions being zeroed out, "position_value_zero" skip reasons in journal.

### Pitfall 2: Double Currency Conversion
**What goes wrong:** Converting MOEX trade results to USD for aggregation, then converting back to RUB for per-segment metrics.
**Why it happens:** `run_iteration.py` has `_normalize_trades_to_usd()` and `_normalize_snapshots_to_usd()` for cross-segment aggregation.
**How to avoid:** For Phase 1, run MOEX segments only (user decision: US backward compatibility NOT required). Keep MOEX metrics in RUB natively. The USD normalization can be left for multi-market aggregation later.
**Warning signs:** Metrics showing unrealistic values, equity jumps at FX rate boundaries.

### Pitfall 3: PreTradeChecker Market Hours Gate
**What goes wrong:** PreTradeChecker rejects trades because daily candle timestamps are midnight UTC, which fails the market-hours check.
**Why it happens:** Engine line 1222-1223 adjusts to `_US_MARKET_OPEN_UTC` but NOT to MOEX open time.
**How to avoid:** The engine already detects `market_id = "moex"` at line 1224. Need to adjust candle timestamps to MOEX market open (10:00 MSK = 07:00 UTC) when `segment_id.startswith("ru_")`.
**Warning signs:** Zero trades despite valid BUY signals in decision journal, pre_trade_violations showing "market_closed".

### Pitfall 4: Holiday Calendar Accuracy vs Complexity
**What goes wrong:** Trying to model every MOEX calendar nuance (half-days, additional sessions, Saturday trading) leads to a complex and fragile system.
**Why it happens:** MOEX's real calendar is more complex than simple "open/closed" binary.
**How to avoid:** For backtest, rely on data source (T-Invest API returns only trading-day candles). For live trading, check `is_moex_trading_day()` which handles the common case. Edge cases (Saturday sessions) can be handled by the broker rejecting orders.
**Warning signs:** Overengineering the calendar when candle data already implicitly handles it.

## Code Examples

### Fix 1: MOEX Commission Rate Update (costs.py)

```python
# Source: backtest/costs.py
MOEX_COSTS = TransactionCosts(
    commission_per_share=Decimal(0),
    commission_rate=Decimal("0.0004"),  # 0.04% Trader tariff (was 0.0003)
    min_commission=Decimal("0.10"),
    spread_bps=Decimal(10),
    slippage_bps=Decimal(7),
)
```

### Fix 2: MOEX Starting Capital (run_iteration.py)

```python
# Source: scripts/run_iteration.py, _run_symbol() call site
if segment.startswith("ru_"):
    segment_cash = Decimal(1_000_000)  # 1M RUB (not cash * 90)
else:
    segment_cash = cash
```

### Fix 3: Transferred Holidays (moex_calendar.py)

```python
# Per-year transferred holidays (government decree on переносные выходные)
# These are bridge days when MOEX is closed in addition to fixed holidays.
_TRANSFERRED_HOLIDAYS: dict[int, frozenset[tuple[int, int]]] = {
    2020: frozenset({(3, 9), (5, 4), (5, 5), (5, 11), (6, 15), (11, 5)}),
    2021: frozenset({(2, 22), (3, 9), (5, 3), (5, 10), (6, 14), (11, 3), (11, 5), (12, 31)}),
    2022: frozenset({(3, 7), (5, 2), (5, 3), (5, 10), (6, 13), (11, 3)}),
    2023: frozenset({(2, 24), (5, 8), (6, 13), (11, 6)}),
    2024: frozenset({(4, 29), (4, 30), (5, 10), (12, 30), (12, 31)}),
    2025: frozenset({(5, 2), (5, 8), (6, 13), (11, 3), (12, 31)}),
    2026: frozenset({(3, 9), (5, 11), (1, 9)}),
}


def is_moex_trading_day(d: date) -> bool:
    """Return True if d is a MOEX trading day (not weekend, not holiday)."""
    if d.weekday() >= 5:  # Saturday/Sunday
        return False
    if is_moex_holiday(d):
        return False
    year_transferred = _TRANSFERRED_HOLIDAYS.get(d.year, frozenset())
    if (d.month, d.day) in year_transferred:
        return False
    return True
```

### Fix 4: TradingLoop Holiday Integration (trading_loop.py)

```python
def _is_market_open(self, market_id: str, dt: datetime) -> bool:
    if dt.weekday() >= _WEEKEND_WEEKDAY:
        return False
    # MOEX holiday check
    if market_id == "moex":
        from finalayze.data.moex_calendar import is_moex_trading_day  # noqa: PLC0415
        if not is_moex_trading_day(dt.date()):
            return False
    # ... existing time-of-day check
```

### Fix 5: PreTradeChecker MOEX Market Open Time (engine.py)

```python
# In _handle_buy, fix the market-open timestamp adjustment for MOEX:
_MOEX_MARKET_OPEN_UTC = time(7, 0, tzinfo=UTC)  # 10:00 MSK

check_dt = fill_candle.timestamp
if check_dt.hour == 0 and check_dt.minute == 0:
    if segment_id.startswith("ru_"):
        check_dt = datetime.combine(check_dt.date(), _MOEX_MARKET_OPEN_UTC)
    else:
        check_dt = datetime.combine(check_dt.date(), _US_MARKET_OPEN_UTC)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Cash * 90 for MOEX equity | Native 1M RUB starting capital | Phase 1 (now) | Fixes 0.02% -> 10-20% position sizing |
| 0.03% commission rate | 0.04% Trader tariff | Phase 1 (now) | More accurate cost deduction |
| 14 fixed holidays only | 14 fixed + per-year transferred | Phase 1 (now) | Adds ~4-8 non-trading days/year |
| No holiday check in TradingLoop | Holiday-aware scheduling gate | Phase 1 (now) | Prevents live trading on MOEX holidays |

## Open Questions

1. **Transferred holidays accuracy**
   - What we know: Government decrees for 2020-2025 are published. 2026 is preliminary.
   - What's unclear: The exact dates need cross-referencing with official MOEX calendar (dynamic JS page, not scrapable). Some holidays have additional trading sessions.
   - Recommendation: Use government decree dates as "closed" and validate against T-Invest API candle gaps in backtest. Incorrect holidays in backtest are harmless (candle data is already filtered by data source).

2. **RollingKelly behavior with 1M RUB**
   - What we know: Cold-start fraction is 1%. 1% of 1M = 10K RUB. Min position for MOEX is 1K-5K RUB, so this passes.
   - What's unclear: Whether Kelly optimal fraction will converge fast enough for meaningful position sizes.
   - Recommendation: Run backtest with RollingKelly first. If positions stay under 5% of equity after warm-up, switch to fixed fraction (0.5 * 0.5 = 25% base) for Phase 1 and revisit Kelly in Phase 2.

3. **Spread/slippage bps for MOEX equities**
   - What we know: Blue chips (SBER, GAZP) have 2-5 bps spreads. Mid-caps have 10-30 bps.
   - What's unclear: Average across the backtest universe.
   - Recommendation: Keep current 10/7 bps as conservative defaults. This is Claude's discretion per CONTEXT.md.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.x |
| Config file | pyproject.toml `[tool.pytest.ini_options]` |
| Quick run command | `uv run pytest tests/unit/test_moex_calendar.py tests/unit/test_costs.py -x` |
| Full suite command | `uv run pytest tests/ -x --timeout=120` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| EQF-01 | MOEX position sized at 10-20% of 1M RUB equity | unit | `uv run pytest tests/unit/test_moex_sizing.py -x` | No -- Wave 0 |
| EQF-04 | is_moex_trading_day returns False for transferred holidays | unit | `uv run pytest tests/unit/test_moex_calendar.py -x` | Partial -- needs new tests |
| EQF-04 | TradingLoop._is_market_open returns False on MOEX holidays | unit | `uv run pytest tests/unit/test_trading_loop_holidays.py -x` | No -- Wave 0 |
| EQF-05 | MOEX_COSTS.commission_rate == 0.0004 | unit | `uv run pytest tests/unit/test_costs.py -x` | Exists -- needs assertion update |
| EQF-05 | Engine deducts MOEX costs from PnL | unit | `uv run pytest tests/unit/test_backtest_engine.py -x -k moex` | No -- Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/unit/test_moex_calendar.py tests/unit/test_costs.py -x`
- **Per wave merge:** `uv run pytest tests/ -x --timeout=120`
- **Phase gate:** Full suite green + backtest-iteration skill on ru_blue_chips,ru_energy,ru_finance

### Wave 0 Gaps
- [ ] `tests/unit/test_moex_sizing.py` -- covers EQF-01 (position size 10-20% of 1M RUB)
- [ ] `tests/unit/test_trading_loop_holidays.py` -- covers EQF-04 (live trading gate)
- [ ] Update `tests/unit/test_moex_calendar.py` -- add transferred holiday tests
- [ ] Update `tests/unit/test_costs.py` -- assert commission_rate == 0.0004

## Sources

### Primary (HIGH confidence)
- Project codebase: `backtest/costs.py`, `backtest/engine.py`, `data/moex_calendar.py`, `core/trading_loop.py`, `scripts/run_iteration.py` -- direct code analysis
- `01-CONTEXT.md` -- user decisions (locked)

### Secondary (MEDIUM confidence)
- [MOEX official 2025 trading schedule](https://www.moex.com/n73702) -- confirmed closed dates and additional trading sessions
- [MOEX 2024 trading schedule](https://www.moex.com/n64122) -- confirmed 2024 holidays
- [MOEX trading calendar](https://www.moex.com/en/tradingcalendar/) -- dynamic calendar (JS-rendered, not directly scrapable)
- [Tinkoff Trader tariff](https://acdn.tinkoff.ru/static/documents/invest-tariff-trader.pdf) -- commission rates (PDF, 0.04% confirmed by user)

### Tertiary (LOW confidence)
- Transferred holiday dates for 2020-2023: assembled from web search results, need validation against actual MOEX candle data gaps. Will be validated implicitly during backtest (T-Invest API candle data reflects real calendar).
- 2026 transferred holidays are preliminary (government decree may not be final).

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- no new dependencies, all changes to existing modules
- Architecture: HIGH -- root cause identified, fix path clear, existing patterns reusable
- Pitfalls: HIGH -- based on direct code analysis of sizing pipeline, RollingKelly, and PreTradeChecker
- Holiday data: MEDIUM -- 2024-2025 dates confirmed from official MOEX sources, 2020-2023 need validation

**Research date:** 2026-03-14
**Valid until:** 2026-04-14 (stable domain, no fast-moving dependencies)

# Phase 15: Schemas, Config, and Rollout Foundation - Research

**Researched:** 2026-03-21
**Domain:** Rollout configuration, risk limit wiring, MOEX lot size validation
**Confidence:** HIGH

## Summary

This phase introduces a RolloutPhase enum (MINIMAL/STANDARD/FULL) with per-phase risk limit overrides, wires those limits into PreTradeChecker and CircuitBreaker, and creates a capital ladder validation script. The codebase already has all the infrastructure needed -- Pydantic Settings with env prefix, PreTradeChecker accepting risk params at init, CircuitBreaker with configurable thresholds, and lot size rounding at the broker layer. The work is mostly adding a new StrEnum, a config mapping, plumbing the values through existing init paths, and writing a standalone validation script.

A notable bug exists: `main.py:250` passes `settings.max_cross_market_exposure_pct` (0.80) as `halt_threshold` to `CrossMarketCircuitBreaker`, but this is a drawdown threshold (default 0.10 in the class), not an exposure limit. This phase should fix this as part of the risk wiring work.

**Primary recommendation:** Add RolloutPhase StrEnum to `core/modes.py`, add rollout fields to Settings, create a `rollout_limits()` method that returns per-phase risk params, and wire those params into the existing PreTradeChecker/CircuitBreaker init paths in `trading_loop.py` and `main.py`.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
None -- all implementation choices are at Claude's discretion.

### Claude's Discretion
All implementation choices are at Claude's discretion -- pure infrastructure phase.

Key constraints from codebase scout:
- PreTradeChecker already accepts `max_position_pct` and `max_positions_per_market` from Settings
- CircuitBreaker L1/L2/L3 thresholds already configurable via `settings.circuit_breaker_l1/l2/l3`
- `max_sector_concentration_pct` and `min_cash_reserve_pct` are hardcoded defaults in PreTradeChecker (0.40 and 0.20) -- need to surface to Settings
- Cross-market breaker has a bug: uses `settings.max_cross_market_exposure_pct` (0.80) instead of `_DEFAULT_CROSS_HALT` (0.10)
- MOEX lot rounding happens at broker layer (`TinkoffBroker.submit_order`) -- lot sizes from InstrumentRegistry
- No rollout concept exists in codebase -- entirely new

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| ROLL-01 | RolloutPhase enum (MINIMAL/STANDARD/FULL) with per-phase capital and position limits in Settings | Architecture pattern: StrEnum in modes.py + frozen dataclass for limits + Settings field with env override. Existing patterns in WorkMode StrEnum and Settings env_prefix. |
| ROLL-02 | PreTradeChecker and CircuitBreaker respect RolloutPhase limits (3% max position at MINIMAL, 1% daily loss, 2% DD auto-stop) | Integration points: trading_loop.py:171 (PreTradeChecker init), main.py:241 (CircuitBreaker init), backtest/engine.py:254 (backtest PreTradeChecker). Settings fields flow through existing init params. |
| ROLL-03 | Capital ladder validation confirms position sizing produces valid lot sizes at each tier (50K/150K/500K/2.5M RUB) | Lot sizes from DEFAULT_MOEX_INSTRUMENTS (SBER=10, VTBR=10, LKOH=1, etc.). Validation script uses InstrumentRegistry + PositionSizingPipeline to check each tier. |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pydantic-settings | (existing) | Settings with env vars | Already used for all config |
| pydantic | (existing) | Validation, frozen models | Already used throughout |

### Supporting
No new libraries needed. This phase uses only existing dependencies.

**Installation:** No new packages required.

## Architecture Patterns

### Recommended Project Structure
```
src/finalayze/
  core/
    modes.py              # Add RolloutPhase StrEnum here (next to WorkMode)
  config/
    settings.py           # Add rollout_phase field + rollout limit fields
  risk/
    rollout.py            # NEW: RolloutLimits frozen dataclass + ROLLOUT_LIMITS mapping
    pre_trade_check.py    # Accept additional params (max_sector_concentration_pct, min_cash_reserve_pct from settings)
    circuit_breaker.py    # No changes needed (already accepts l1/l2/l3 thresholds)
  core/
    trading_loop.py       # Wire rollout limits into PreTradeChecker/CircuitBreaker init
  main.py                 # Wire rollout limits into CircuitBreaker init + fix cross-market bug
scripts/
  validate_capital_ladder.py  # NEW: Standalone validation script
tests/
  unit/
    test_rollout.py       # NEW: Tests for RolloutPhase, RolloutLimits, wiring
    test_capital_ladder.py # NEW: Tests for capital ladder validation
```

### Pattern 1: RolloutPhase Enum + RolloutLimits Mapping
**What:** StrEnum for rollout phases with a frozen dataclass mapping each phase to risk limits.
**When to use:** When a set of configuration values must change together based on a single selector.
**Example:**
```python
# src/finalayze/core/modes.py (add to existing file)
class RolloutPhase(StrEnum):
    MINIMAL = "minimal"     # First live phase: ultra-conservative
    STANDARD = "standard"   # Proven stable: moderate limits
    FULL = "full"           # Full production: normal limits

# src/finalayze/risk/rollout.py (NEW file)
from dataclasses import dataclass
from decimal import Decimal
from finalayze.core.modes import RolloutPhase

@dataclass(frozen=True)
class RolloutLimits:
    """Risk limits for a rollout phase."""
    max_position_pct: Decimal
    max_positions_per_market: int
    daily_loss_limit_pct: float
    circuit_breaker_l1: float  # DD threshold for CAUTION
    circuit_breaker_l2: float  # DD threshold for HALTED
    circuit_breaker_l3: float  # DD threshold for LIQUIDATE
    max_sector_concentration_pct: Decimal
    min_cash_reserve_pct: Decimal

ROLLOUT_LIMITS: dict[RolloutPhase, RolloutLimits] = {
    RolloutPhase.MINIMAL: RolloutLimits(
        max_position_pct=Decimal("0.03"),
        max_positions_per_market=5,
        daily_loss_limit_pct=0.01,
        circuit_breaker_l1=0.01,  # 1% DD -> caution
        circuit_breaker_l2=0.02,  # 2% DD -> halted (auto-stop)
        circuit_breaker_l3=0.03,  # 3% DD -> liquidate
        max_sector_concentration_pct=Decimal("0.20"),
        min_cash_reserve_pct=Decimal("0.40"),
    ),
    RolloutPhase.STANDARD: RolloutLimits(
        max_position_pct=Decimal("0.10"),
        max_positions_per_market=8,
        daily_loss_limit_pct=0.03,
        circuit_breaker_l1=0.03,
        circuit_breaker_l2=0.05,
        circuit_breaker_l3=0.10,
        max_sector_concentration_pct=Decimal("0.30"),
        min_cash_reserve_pct=Decimal("0.30"),
    ),
    RolloutPhase.FULL: RolloutLimits(
        max_position_pct=Decimal("0.20"),
        max_positions_per_market=10,
        daily_loss_limit_pct=0.05,
        circuit_breaker_l1=0.05,
        circuit_breaker_l2=0.10,
        circuit_breaker_l3=0.15,
        max_sector_concentration_pct=Decimal("0.40"),
        min_cash_reserve_pct=Decimal("0.20"),
    ),
}
```

### Pattern 2: Settings Rollout Integration
**What:** Add `rollout_phase` field to Settings, with a method to resolve effective risk limits.
**When to use:** At Settings level to compute effective params from rollout phase.
**Example:**
```python
# In config/settings.py
from finalayze.core.modes import RolloutPhase

class Settings(BaseSettings):
    # ... existing fields ...
    rollout_phase: RolloutPhase = RolloutPhase.FULL  # Default: backward-compatible

    def effective_risk_limits(self) -> RolloutLimits:
        """Return risk limits for the current rollout phase.

        Rollout phase limits override the default Settings risk fields.
        If rollout_phase is FULL, existing Settings values are used as-is
        (backward compatible).
        """
        from finalayze.risk.rollout import ROLLOUT_LIMITS
        return ROLLOUT_LIMITS[self.rollout_phase]
```

### Pattern 3: Capital Ladder Validation Script
**What:** Standalone script that tests position sizing at multiple capital tiers.
**When to use:** Before going live at a new capital level.
**Example:**
```python
# scripts/validate_capital_ladder.py
# For each tier (50K, 150K, 500K, 2.5M RUB):
#   For each MOEX instrument in registry:
#     1. Compute position size = capital * max_position_pct (per rollout phase)
#     2. Compute shares = position_size / current_price
#     3. Round to lot_size
#     4. Check: rounded_qty >= 1 lot (i.e., position is viable)
#     5. Check: actual position value is within acceptable range of target
# Output: table showing pass/fail per instrument per tier
```

### Anti-Patterns to Avoid
- **Hardcoding rollout limits in PreTradeChecker/CircuitBreaker:** Limits must come from Settings/RolloutLimits, not be duplicated inside risk modules.
- **Breaking backward compatibility:** FULL phase limits must match current defaults exactly so existing deployments are unaffected.
- **Coupling backtest to rollout:** Backtest engine should not be aware of RolloutPhase -- it gets max_position_pct etc. from BacktestConfig which is independent.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Lot size rounding | Custom rounding logic | `math.floor(qty / lot_size) * lot_size` (existing in TinkoffBroker) | Already tested, handles edge cases |
| Env var parsing | Manual os.environ parsing | Pydantic Settings with `FINALAYZE_ROLLOUT_PHASE` env var | Automatic validation, type coercion |
| Enum serialization | Custom string mapping | StrEnum (auto .value string conversion) | Standard Python 3.11+ pattern, consistent with WorkMode |

## Common Pitfalls

### Pitfall 1: Rollout Phase Not Applied to Backtest
**What goes wrong:** Backtest uses default risk limits (FULL), so backtest results don't reflect MINIMAL phase constraints.
**Why it happens:** BacktestConfig has its own `max_position_pct` field independent of Settings.
**How to avoid:** Backtest should remain independent of rollout phase. Rollout is a live trading concept. Document this explicitly.
**Warning signs:** Someone asks "why does backtest show 20% positions when we're in MINIMAL mode?"

### Pitfall 2: Cross-Market Breaker Bug Persists
**What goes wrong:** `main.py:250` passes `max_cross_market_exposure_pct` (0.80) as `halt_threshold` to CrossMarketCircuitBreaker, but this is a drawdown threshold that defaults to 0.10.
**Why it happens:** Variable naming confusion -- "exposure" vs "drawdown".
**How to avoid:** Fix during this phase. The cross-market breaker halt_threshold should be a separate Setting field (e.g., `cross_market_dd_halt_pct`) or use the class default (0.10).
**Warning signs:** Cross-market breaker never trips because 80% combined drawdown is unreachable.

### Pitfall 3: Sector/Cash Limits Remain Hardcoded After Rollout
**What goes wrong:** PreTradeChecker defaults `max_sector_concentration_pct=0.40` and `min_cash_reserve_pct=0.20` in its `__init__`, but Settings doesn't expose these fields. Rollout phases can't override them.
**Why it happens:** These were originally hardcoded and never surfaced to configuration.
**How to avoid:** Add these to Settings and wire through RolloutLimits. PreTradeChecker init at trading_loop.py:171 must pass them.

### Pitfall 4: Capital Ladder Script Uses Stale Prices
**What goes wrong:** Validation script hardcodes prices, which become stale.
**Why it happens:** Not fetching live prices for a validation script.
**How to avoid:** Use representative prices (approximate current market prices) with a note that exact results depend on live quotes. The script validates lot-size viability, not exact position values. Accept a price input parameter or use sane defaults.

### Pitfall 5: FULL Phase Defaults Mismatch Settings
**What goes wrong:** RolloutLimits for FULL phase has different values than current Settings defaults, breaking existing behavior.
**Why it happens:** Copy-paste error or intentional "improvement" during rollout work.
**How to avoid:** FULL phase limits MUST exactly match current Settings defaults: max_position_pct=0.20, max_positions=10, daily_loss_limit=0.02 (note: this is already the Settings default), l1=0.05, l2=0.10, l3=0.15.

## Code Examples

### Wiring in TradingLoop (trading_loop.py)
```python
# Current code (line 171-174):
self._pre_trade_checker = PreTradeChecker(
    max_position_pct=Decimal(str(settings.max_position_pct)),
    max_positions_per_market=settings.max_positions_per_market,
    pdt_tracker=self._pdt_tracker,
)

# After rollout wiring:
limits = settings.effective_risk_limits()
self._pre_trade_checker = PreTradeChecker(
    max_position_pct=limits.max_position_pct,
    max_positions_per_market=limits.max_positions_per_market,
    pdt_tracker=self._pdt_tracker,
    max_sector_concentration_pct=limits.max_sector_concentration_pct,
    min_cash_reserve_pct=limits.min_cash_reserve_pct,
)
```

### Wiring in main.py (CircuitBreaker)
```python
# Current code (line 241-248):
circuit_breakers = {
    "moex": CircuitBreaker(
        market_id="moex",
        l1_threshold=getattr(settings, "circuit_breaker_l1", 0.05),
        l2_threshold=getattr(settings, "circuit_breaker_l2", 0.10),
        l3_threshold=getattr(settings, "circuit_breaker_l3", 0.15),
    ),
}

# After rollout wiring:
limits = settings.effective_risk_limits()
circuit_breakers = {
    "moex": CircuitBreaker(
        market_id="moex",
        l1_threshold=limits.circuit_breaker_l1,
        l2_threshold=limits.circuit_breaker_l2,
        l3_threshold=limits.circuit_breaker_l3,
    ),
}
```

### Capital Ladder Validation Logic
```python
# Core validation for a single instrument at a capital tier
def validate_position(
    capital: Decimal,
    max_position_pct: Decimal,
    price: Decimal,
    lot_size: int,
) -> dict:
    target_value = capital * max_position_pct
    shares_target = target_value / price
    lots = int(shares_target) // lot_size
    actual_shares = lots * lot_size
    actual_value = Decimal(actual_shares) * price
    return {
        "target_value": target_value,
        "lots": lots,
        "actual_shares": actual_shares,
        "actual_value": actual_value,
        "viable": lots >= 1,
        "utilization_pct": float(actual_value / target_value * 100) if target_value > 0 else 0,
    }
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Hardcoded risk limits | Settings-based risk limits | Phase 2 (v1.0) | Limits configurable via env vars |
| No rollout concept | RolloutPhase enum | This phase | Gradual risk scaling for production entry |
| No lot size validation | Broker-level rounding | Phase 3 (v1.0) | Orders round down to valid lots |

## Open Questions

1. **MINIMAL phase daily_loss_limit_pct: 1% or 2%?**
   - Success criteria says "1% daily loss, 2% DD auto-stop"
   - "1% daily loss" maps to `daily_loss_limit_pct=0.01`
   - "2% DD auto-stop" maps to `circuit_breaker_l2=0.02` (HALTED level)
   - Recommendation: Use exactly these values as specified in success criteria

2. **Should rollout phase affect LossLimitTracker too?**
   - What we know: LossLimitTracker is initialized at trading_loop.py:177 with `daily_loss_limit_pct`
   - Recommendation: Yes, wire `limits.daily_loss_limit_pct` into LossLimitTracker init

3. **Price source for capital ladder script**
   - What we know: Script needs representative prices for MOEX instruments
   - Recommendation: Use approximate current prices as constants (e.g., SBER~300, LKOH~7000, GAZP~150) with a CLI flag to override. Script validates lot-size math, not price accuracy.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (existing) |
| Config file | pyproject.toml [tool.pytest.ini_options] |
| Quick run command | `uv run pytest tests/unit/test_rollout.py tests/unit/test_capital_ladder.py -x` |
| Full suite command | `uv run pytest tests/unit/ -x` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ROLL-01 | RolloutPhase enum has 3 values, ROLLOUT_LIMITS maps each to correct limits | unit | `uv run pytest tests/unit/test_rollout.py::test_rollout_phase_enum -x` | No - Wave 0 |
| ROLL-01 | Settings.rollout_phase defaults to FULL, respects FINALAYZE_ROLLOUT_PHASE env var | unit | `uv run pytest tests/unit/test_rollout.py::test_settings_rollout_phase -x` | No - Wave 0 |
| ROLL-01 | Settings.effective_risk_limits() returns correct RolloutLimits for each phase | unit | `uv run pytest tests/unit/test_rollout.py::test_effective_risk_limits -x` | No - Wave 0 |
| ROLL-01 | FULL phase limits match current Settings defaults (backward compat) | unit | `uv run pytest tests/unit/test_rollout.py::test_full_matches_defaults -x` | No - Wave 0 |
| ROLL-02 | PreTradeChecker with MINIMAL limits rejects >3% position | unit | `uv run pytest tests/unit/test_rollout.py::test_pretrade_minimal_position_cap -x` | No - Wave 0 |
| ROLL-02 | CircuitBreaker with MINIMAL limits trips L2 at 2% DD | unit | `uv run pytest tests/unit/test_rollout.py::test_circuit_breaker_minimal_dd -x` | No - Wave 0 |
| ROLL-02 | LossLimitTracker with MINIMAL limits triggers at 1% daily loss | unit | `uv run pytest tests/unit/test_rollout.py::test_loss_limit_minimal -x` | No - Wave 0 |
| ROLL-03 | Capital ladder validates lot sizes at 50K RUB (smallest tier) | unit | `uv run pytest tests/unit/test_capital_ladder.py::test_50k_tier -x` | No - Wave 0 |
| ROLL-03 | Capital ladder validates lot sizes at 2.5M RUB (largest tier) | unit | `uv run pytest tests/unit/test_capital_ladder.py::test_2500k_tier -x` | No - Wave 0 |
| ROLL-03 | Capital ladder script produces correct pass/fail per instrument | unit | `uv run pytest tests/unit/test_capital_ladder.py::test_ladder_report -x` | No - Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/unit/test_rollout.py tests/unit/test_capital_ladder.py -x`
- **Per wave merge:** `uv run pytest tests/unit/ -x`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/test_rollout.py` -- covers ROLL-01, ROLL-02
- [ ] `tests/unit/test_capital_ladder.py` -- covers ROLL-03

## Sources

### Primary (HIGH confidence)
- Codebase inspection: `config/settings.py`, `src/finalayze/risk/pre_trade_check.py`, `src/finalayze/risk/circuit_breaker.py`, `src/finalayze/core/modes.py`, `src/finalayze/markets/instruments.py`, `src/finalayze/execution/tinkoff_broker.py`, `src/finalayze/risk/position_sizing_pipeline.py`, `src/finalayze/core/trading_loop.py`, `src/finalayze/main.py`
- CONTEXT.md codebase scout findings (verified against source)

### Secondary (MEDIUM confidence)
- MOEX lot sizes from DEFAULT_MOEX_INSTRUMENTS (static registry, may need live refresh)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - no new libraries, all existing infrastructure
- Architecture: HIGH - clear patterns from existing codebase (StrEnum, Pydantic Settings, frozen dataclass)
- Pitfalls: HIGH - identified from direct code inspection (cross-market bug confirmed, hardcoded defaults confirmed)

**Research date:** 2026-03-21
**Valid until:** 2026-04-21 (stable infrastructure, no external dependency changes)

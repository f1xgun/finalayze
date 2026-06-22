"""Phase 76 (v11.2 W1): RGBITR duration swap + look-ahead-safe regime tilt.

RED first (TDD). Pins the redesign contract before it exists:

- ``rate_regime_as_of`` (cbr.py): look-ahead-safe regime selector -- ``high_rate``
  until the most recent CBR decision on/before ``as_of`` is a ``cut``, then
  ``easing``. The 2025-06-06 first cut is the flip point; 2025-06-05 is still
  ``high_rate`` (the cut is one day in the future -> no look-ahead).
- ``AllocationProfile.regime_weights`` + loader: each (profile x regime) tilt
  vector is the locked deposit-anchored table, non-negative, sums to 1.0, all 3
  classes; a vector not summing to 1.0 fails closed (ConfigurationError).
- ``AllocationOrchestrator``: applies the regime tilt PER quarterly boundary
  (high_rate weights before the cut, easing weights after) as a deterministic
  table lookup -- zero trainable params, two runs byte-identical.
- ``run_allocation_gate``: the OFZ leg secid is RGBITR (not RUFLBITR) and the
  snapshot OFZ leg key is ``ofz_rgbitr_net``.

All fixtures are tiny in-memory ``(date, Decimal)`` curves -- no live engine/API.
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

import pytest

from finalayze.config.allocation_profiles import load_allocation_profiles
from finalayze.core.exceptions import ConfigurationError
from finalayze.core.schemas import AssetClass, RiskProfile
from finalayze.data.fetchers.cbr import rate_regime_as_of
from finalayze.orchestration.allocation import AllocationOrchestrator

# -- Constants (named -- no magic numbers, ruff PLR2004) ----------------------

_HIGH_RATE = "high_rate"
_EASING = "easing"

# CBR calendar anchors (cbr.py CBR_MEETINGS): 2025-04-25 hold @ 21.00, then the
# FIRST cut 2025-06-06 @ 20.00.
_PRE_CUT = date(2025, 6, 5)  # last decision is the 2025-04-25 hold -> high_rate
_FIRST_CUT = date(2025, 6, 6)  # the cut itself -> easing
_POST_CUT = date(2025, 6, 7)  # after the cut -> easing
_DEEP_EASING = date(2025, 9, 30)  # several cuts in -> easing

# Locked Phase-76 regime tilt table (deposit / ofz_pk / equity), each Sigma=1.0.
_D = AssetClass.DEPOSIT
_O = AssetClass.OFZ_PK
_E = AssetClass.EQUITY
_CONS_HIGH = {_D: Decimal("0.75"), _O: Decimal("0.10"), _E: Decimal("0.15")}
_CONS_EASE = {_D: Decimal("0.45"), _O: Decimal("0.35"), _E: Decimal("0.20")}
_BAL_HIGH = {_D: Decimal("0.60"), _O: Decimal("0.10"), _E: Decimal("0.30")}
_BAL_EASE = {_D: Decimal("0.25"), _O: Decimal("0.40"), _E: Decimal("0.35")}
_GROW_HIGH = {_D: Decimal("0.40"), _O: Decimal("0.10"), _E: Decimal("0.50")}
_GROW_EASE = {_D: Decimal("0.10"), _O: Decimal("0.40"), _E: Decimal("0.50")}

_EXPECTED_TILT = {
    RiskProfile.CONSERVATIVE: {_HIGH_RATE: _CONS_HIGH, _EASING: _CONS_EASE},
    RiskProfile.BALANCED: {_HIGH_RATE: _BAL_HIGH, _EASING: _BAL_EASE},
    RiskProfile.GROWTH: {_HIGH_RATE: _GROW_HIGH, _EASING: _GROW_EASE},
}

_VECTOR_SUM = Decimal("1.0")
_FLAT_LEVEL = Decimal(100)


def _daily(start: date, days: int) -> list[date]:
    return [start + timedelta(days=i) for i in range(days)]


def _flat(dates: list[date], level: Decimal = _FLAT_LEVEL) -> list[tuple[date, Decimal]]:
    return [(d, level) for d in dates]


# -- rate_regime_as_of: look-ahead-safe selector ------------------------------


def test_regime_high_rate_before_first_cut() -> None:
    """2025-06-05 is still high_rate -- the 2025-06-06 cut is one day in the future."""
    assert rate_regime_as_of(_PRE_CUT) == _HIGH_RATE


def test_regime_easing_on_and_after_first_cut() -> None:
    """On/after the 2025-06-06 first cut the regime is easing (look-ahead-safe)."""
    assert rate_regime_as_of(_FIRST_CUT) == _EASING
    assert rate_regime_as_of(_POST_CUT) == _EASING
    assert rate_regime_as_of(_DEEP_EASING) == _EASING


def test_regime_selector_is_lookahead_safe() -> None:
    """The selector never reads a meeting after as_of (removing future meetings is a no-op)."""
    # Same as_of, evaluated twice: deterministic, depends only on past meetings.
    assert rate_regime_as_of(_PRE_CUT) == rate_regime_as_of(_PRE_CUT)
    assert rate_regime_as_of(_PRE_CUT) != rate_regime_as_of(_POST_CUT)


# -- AllocationProfile.regime_weights from the loader -------------------------


@pytest.mark.parametrize("profile", list(RiskProfile))
def test_regime_weights_loaded_exact(profile: RiskProfile) -> None:
    """Each profile carries the locked high_rate + easing tilt vectors (Decimal-exact)."""
    loaded = load_allocation_profiles()[profile]
    assert loaded.regime_weights is not None
    assert loaded.regime_weights[_HIGH_RATE] == _EXPECTED_TILT[profile][_HIGH_RATE]
    assert loaded.regime_weights[_EASING] == _EXPECTED_TILT[profile][_EASING]


@pytest.mark.parametrize("profile", list(RiskProfile))
def test_every_regime_vector_sums_to_one(profile: RiskProfile) -> None:
    """Every (profile x regime) tilt vector sums to exactly 1.0 (V5, Decimal-exact)."""
    loaded = load_allocation_profiles()[profile]
    assert loaded.regime_weights is not None
    for vec in loaded.regime_weights.values():
        assert sum(vec.values()) == _VECTOR_SUM


def test_loader_fail_closed_on_bad_regime_sum(tmp_path: object) -> None:
    """A regime vector not summing to 1.0 raises ConfigurationError (no renormalization)."""
    bad = """
conservative:
  weights: {deposit: 0.60, ofz_pk: 0.25, equity: 0.15}
  max_drawdown_pct: 0.08
  regime_weights:
    high_rate: {deposit: 0.75, ofz_pk: 0.10, equity: 0.15}
    easing: {deposit: 0.50, ofz_pk: 0.35, equity: 0.20}
balanced:
  weights: {deposit: 0.45, ofz_pk: 0.25, equity: 0.30}
  max_drawdown_pct: 0.15
growth:
  weights: {deposit: 0.25, ofz_pk: 0.25, equity: 0.50}
  max_drawdown_pct: 0.25
"""  # conservative easing sums to 1.05 -> must fail closed
    from pathlib import Path  # noqa: PLC0415

    p = Path(str(tmp_path)) / "bad_regime.yaml"
    p.write_text(bad, encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_allocation_profiles(path=p)


# -- Orchestrator applies the tilt per quarterly boundary ---------------------


def test_orchestrator_applies_regime_tilt_per_boundary() -> None:
    """Balanced run over 2025 holds high_rate weights pre-cut, easing weights post-cut.

    Flat equal curves -> at each rebalance bar the realized weight share equals the
    target weight EXACTLY (Decimal). The Q2 (2025-04-01) boundary is high_rate (last
    decision 2025-03-21 hold); the Q3 (2025-07-01) boundary is easing (last decision
    2025-06-06 cut). RED today: the orchestrator applies one static vector at all
    boundaries.
    """
    dates = _daily(date(2025, 1, 1), 365)
    orch = AllocationOrchestrator(risk_profile=RiskProfile.BALANCED)
    result = orch.run(
        deposit_curve=_flat(dates),
        ofz_pk_curve=_flat(dates),
        equity_curve=_flat(dates),
    )
    pre = [d for d in result.rebalance_dates if d < _FIRST_CUT]
    post = [d for d in result.rebalance_dates if d >= _FIRST_CUT]
    assert pre, "expected a pre-cut rebalance boundary"
    assert post, "expected a post-cut rebalance boundary"

    i_pre = result.dates.index(pre[-1])
    i_post = result.dates.index(post[0])
    ws = result.weight_series

    assert ws[_D][i_pre] == _BAL_HIGH[_D]
    assert ws[_O][i_pre] == _BAL_HIGH[_O]
    assert ws[_E][i_pre] == _BAL_HIGH[_E]

    assert ws[_D][i_post] == _BAL_EASE[_D]
    assert ws[_O][i_post] == _BAL_EASE[_O]
    assert ws[_E][i_post] == _BAL_EASE[_E]


def test_regime_tilt_is_deterministic_table_lookup() -> None:
    """Two identical runs are byte-identical -- the tilt has zero trainable params (D-03)."""
    dates = _daily(date(2025, 1, 1), 365)
    orch_a = AllocationOrchestrator(risk_profile=RiskProfile.GROWTH)
    orch_b = AllocationOrchestrator(risk_profile=RiskProfile.GROWTH)
    a = orch_a.run(_flat(dates), _flat(dates), _flat(dates))
    b = orch_b.run(_flat(dates), _flat(dates), _flat(dates))
    assert a.merged_equity_curve == b.merged_equity_curve
    assert a.weight_series == b.weight_series


# -- RGBITR secid swap + snapshot key rename ----------------------------------


def test_ofz_secid_is_rgbitr() -> None:
    """The live OFZ leg fetches the fixed-coupon RGBITR index (not the RUFLBITR floater)."""
    import scripts.run_allocation_gate as gate_cli  # noqa: PLC0415

    assert gate_cli._OFZ_SECID == "RGBITR"
    assert gate_cli._SNAP_LEG_OFZ == "ofz_rgbitr_net"


def test_snapshot_ofz_leg_key_renamed() -> None:
    """The committed-snapshot OFZ leg key is ofz_rgbitr_net (3-leg shape preserved)."""
    from finalayze.backtest import allocation_gate as gate  # noqa: PLC0415

    assert gate._SNAPSHOT_LEG_KEYS[1] == "ofz_rgbitr_net"
    assert len(gate._SNAPSHOT_LEG_KEYS) == 3  # noqa: PLR2004

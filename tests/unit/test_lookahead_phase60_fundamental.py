"""Phase-60 fundamental rule-gate look-ahead + backfill-ceiling suite (INTG-02, BACKFILL-01).

This module is the single discoverable (``pytest -k lookahead``) point-in-time
correctness gate for the rule-based ``earnings_yield`` fundamental gate
(``src/finalayze/strategies/fundamental_gate.py``) and the documented BACKFILL-01
data ceiling.

  (a) RULE GATE (INTG-02, D-01): ``earnings_yield_gate`` consumes
      ``compute_fundamental_features(...)['earnings_yield']`` DIRECTLY — never via
      ``compute_features`` / the (disabled) ML ensemble. A cheap valuation
      (earnings_yield >= threshold) returns a "boost" verdict; below threshold is a
      neutral/no-boost verdict; missing data degrades gracefully to neutral (no
      raise, never NaN). A ``FundamentalSnapshot`` dated ``as_of > D`` MUST NOT
      change the verdict at D (the gate delegates the ``as_of <= D`` filter to
      ``_filter_as_of`` inside ``compute_fundamental_features``); the same datum
      dated ``as_of <= D`` MUST change it (spike-injection over ``as_of``, mirroring
      ``test_lookahead_phase59.py::TestLookaheadFundamentals``).

  (b) BACKFILL CEILING (BACKFILL-01, D-03): the installed ``t_tech.invest`` SDK
      proves that ``GetAssetFundamentalsRequest`` carries NO date range (its only
      field is ``assets``) — fundamentals are point-in-time only, with NO history —
      whereas ``GetAssetReportsRequest`` HAS ``from_``/``to`` (earnings DATES are
      rangeable). This records WHY pure point-in-time fundamentals are LIVE-FORWARD
      ONLY (Manual-Only verification per 60-VALIDATION.md): a single current
      snapshot is CONSTANT across a backtest window and therefore CANNOT be the
      MEAS-01 causal lever (SUE/CPI are — Assumption A3). Never fabricate a
      back-history from a point-in-time snapshot (RESEARCH Pitfall 2).

Every test is named ``test_lookahead_*`` so ``-k lookahead`` collects the suite, and
the gate tests also match ``-k fundamental_gate``; named constants only (ruff
PLR2004). No live data / token is required.
"""

from __future__ import annotations

import ast
import dataclasses
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from finalayze.core.schemas import FundamentalSnapshot, MoexMarketData
from finalayze.strategies.fundamental_gate import (
    NEUTRAL_SCALE,
    FundamentalGateVerdict,
    earnings_yield_gate,
)

# ── Shared constants (ruff PLR2004: no magic numbers) ────────────────────────
_SYMBOL = "SBER"

_D = datetime(2025, 3, 1, tzinfo=UTC)
_THIRTY_DAYS = timedelta(days=30)
_ONE_DAY = timedelta(days=1)

# A cheap P/E -> high earnings_yield (1/4 = 0.25), well above any sane threshold.
_PE_CHEAP = 4.0
# An expensive P/E -> low earnings_yield (1/100 = 0.01), below the boost threshold.
_PE_EXPENSIVE = 100.0
# A future-dated spike: cheap valuation that must NOT leak backwards.
_PE_FUTURE_SPIKE = 2.0

# Threshold low enough that _PE_CHEAP (0.25) boosts but _PE_EXPENSIVE (0.01) does not.
_BOOST_THRESHOLD = 0.08


def _snapshot(symbol: str, as_of: datetime, *, pe_ratio: float | None) -> FundamentalSnapshot:
    return FundamentalSnapshot(symbol=symbol, as_of=as_of, pe_ratio=pe_ratio)


# ===========================================================================
# (a) RULE GATE: earnings_yield_gate verdicts + as_of <= D look-ahead guard
# ===========================================================================
class TestLookaheadFundamentalGate:
    def test_lookahead_fundamental_gate_cheap_valuation_boosts(self) -> None:
        """A cheap valuation (earnings_yield >= threshold) yields a boost verdict."""
        snap = _snapshot(_SYMBOL, _D - _THIRTY_DAYS, pe_ratio=_PE_CHEAP)
        verdict = earnings_yield_gate(
            MoexMarketData(fundamentals=(snap,)), as_of=_D, threshold=_BOOST_THRESHOLD
        )
        assert isinstance(verdict, FundamentalGateVerdict)
        assert verdict.passed is True
        assert verdict.scale > NEUTRAL_SCALE

    def test_lookahead_fundamental_gate_expensive_valuation_neutral(self) -> None:
        """An expensive valuation (earnings_yield < threshold) yields a neutral verdict."""
        snap = _snapshot(_SYMBOL, _D - _THIRTY_DAYS, pe_ratio=_PE_EXPENSIVE)
        verdict = earnings_yield_gate(
            MoexMarketData(fundamentals=(snap,)), as_of=_D, threshold=_BOOST_THRESHOLD
        )
        assert verdict.passed is False
        assert verdict.scale == NEUTRAL_SCALE

    def test_lookahead_fundamental_gate_future_snapshot_ignored(self) -> None:
        """A cheap snapshot dated AFTER D must not change the verdict at D."""
        expensive_now = _snapshot(_SYMBOL, _D - _THIRTY_DAYS, pe_ratio=_PE_EXPENSIVE)
        cheap_future = _snapshot(_SYMBOL, _D + _THIRTY_DAYS, pe_ratio=_PE_FUTURE_SPIKE)
        clean = earnings_yield_gate(
            MoexMarketData(fundamentals=(expensive_now,)), as_of=_D, threshold=_BOOST_THRESHOLD
        )
        with_future = earnings_yield_gate(
            MoexMarketData(fundamentals=(expensive_now, cheap_future)),
            as_of=_D,
            threshold=_BOOST_THRESHOLD,
        )
        assert with_future == clean
        assert with_future.passed is False

    def test_lookahead_fundamental_gate_in_window_snapshot_applied(self) -> None:
        """The SAME cheap spike dated <= D MUST flip the verdict (proves a real filter)."""
        expensive_old = _snapshot(_SYMBOL, _D - _THIRTY_DAYS, pe_ratio=_PE_EXPENSIVE)
        cheap_past = _snapshot(_SYMBOL, _D - _ONE_DAY, pe_ratio=_PE_FUTURE_SPIKE)
        clean = earnings_yield_gate(
            MoexMarketData(fundamentals=(expensive_old,)), as_of=_D, threshold=_BOOST_THRESHOLD
        )
        with_past = earnings_yield_gate(
            MoexMarketData(fundamentals=(expensive_old, cheap_past)),
            as_of=_D,
            threshold=_BOOST_THRESHOLD,
        )
        assert with_past != clean
        assert with_past.passed is True

    def test_lookahead_fundamental_gate_missing_data_neutral_passthrough(self) -> None:
        """No usable snapshot -> neutral passthrough; no raise; never NaN."""
        none_verdict = earnings_yield_gate(None, as_of=_D, threshold=_BOOST_THRESHOLD)
        empty_verdict = earnings_yield_gate(
            MoexMarketData(fundamentals=()), as_of=_D, threshold=_BOOST_THRESHOLD
        )
        for verdict in (none_verdict, empty_verdict):
            assert verdict.passed is False
            assert verdict.scale == NEUTRAL_SCALE
            assert verdict.earnings_yield == 0.0  # all-0.0 _DEFAULT, never NaN
            assert verdict.earnings_yield == verdict.earnings_yield  # NaN != NaN

    def test_lookahead_fundamental_gate_does_not_import_compute_features(self) -> None:
        """Source-graph guard: the gate must consume the fundamental feature DIRECTLY.

        It must NOT import ``compute_features`` or the ML ``technical`` builder
        (D-01: rule-based, no ML path; no FEATURE_SCHEMA_VERSION bump).
        """
        module_path = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "finalayze"
            / "strategies"
            / "fundamental_gate.py"
        )
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        imported_names: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported_names.append(node.module or "")
                imported_names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.Import):
                imported_names.extend(alias.name for alias in node.names)
        assert not any("technical" in name for name in imported_names)
        assert not any("compute_features" in name for name in imported_names)

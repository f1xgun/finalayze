"""Unit tests for StrategyCombiner hooks and forward-compatibility methods.

APPLY-04: invalidate_segment_cache() is a no-op but callable without error.
UNIV-03 (68-03/68-04): the combiner must build a NON-EMPTY strategies allow-list
for the activated liquid + thin sectors (the no_signals -> alive preset fix).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from finalayze.strategies.base import BaseStrategy
from finalayze.strategies.combiner import StrategyCombiner

# Sectors activated via per-segment preset YAMLs in Phase 68 Waves 3 (liquid)
# and 4 (thin). Each must expose a non-empty strategies allow-list keying at
# least momentum + mean_reversion ON (the activation mechanism — RESEARCH).
_LIQUID_ACTIVATED_SEGMENTS = ("ru_metals", "ru_consumer", "ru_construction")
_THIN_ACTIVATED_SEGMENTS = ("ru_telecom", "ru_transport", "ru_chemicals")
_ACTIVATED_SEGMENTS = _LIQUID_ACTIVATED_SEGMENTS + _THIN_ACTIVATED_SEGMENTS


class _MinimalStrategy(BaseStrategy):
    """Minimal mock strategy for combiner instantiation."""

    @property
    def name(self) -> str:
        return "mock"

    def supported_segments(self) -> list[str]:
        return ["us_tech"]

    def get_parameters(self, segment_id: str) -> dict[str, object]:
        return {}

    def generate_signal(
        self,
        symbol: str,
        candles: object,
        segment_id: str,
        sentiment_score: float = 0.0,
        has_open_position: bool = False,
    ) -> None:
        return None


@pytest.fixture
def combiner() -> StrategyCombiner:
    """Return a minimal StrategyCombiner instance."""
    return StrategyCombiner([_MinimalStrategy()])


class TestInvalidateSegmentCache:
    """APPLY-04: invalidate_segment_cache() is a forward-compatibility no-op hook."""

    def test_invalidate_segment_cache_is_callable(self, combiner: StrategyCombiner) -> None:
        """APPLY-04: invalidate_segment_cache() is a no-op but callable without error."""
        # Should not raise
        combiner.invalidate_segment_cache("us_tech")
        combiner.invalidate_segment_cache("nonexistent_segment")

    def test_invalidate_segment_cache_multiple_calls(self, combiner: StrategyCombiner) -> None:
        """Multiple calls with the same segment_id are safe."""
        combiner.invalidate_segment_cache("us_tech")
        combiner.invalidate_segment_cache("us_tech")
        combiner.invalidate_segment_cache("us_tech")

    def test_invalidate_segment_cache_returns_none(self, combiner: StrategyCombiner) -> None:
        """invalidate_segment_cache() returns None (no meaningful return value)."""
        result = combiner.invalidate_segment_cache("us_broad")
        assert result is None


class TestActivatedSegmentStrategiesCfg:
    """UNIV-03: activated sectors must build a non-empty strategies allow-list.

    The combiner reads ONLY the preset ``strategies:`` block (combiner.py:394);
    an empty block makes ``generate_signal`` return ``None`` on every bar (the
    ``no_signals`` root cause). Authoring a per-segment preset that keys
    momentum + mean_reversion ON is the entire activation fix.
    """

    @pytest.mark.parametrize("segment_id", _ACTIVATED_SEGMENTS)
    def test_liquid_segment_strategies_cfg_is_non_empty(
        self, combiner: StrategyCombiner, segment_id: str
    ) -> None:
        """The parsed strategies block keys momentum + mean_reversion enabled."""
        config = combiner._load_config(segment_id)
        strategies = config.get("strategies")
        assert isinstance(strategies, dict)
        assert strategies, f"{segment_id}: empty strategies allow-list (no_signals)"
        for name in ("momentum", "mean_reversion"):
            block = strategies.get(name)
            assert isinstance(block, dict), f"{segment_id}: missing {name} block"
            assert block.get("enabled") is True, f"{segment_id}: {name} not enabled"


# Phase 68-06 retry: the verbatim-from-ru_finance presets 0-traded for
# ru_consumer + ru_chemicals. DIAGNOSIS (combiner-level instrumentation,
# results/iterations/phase68-activate-*/<seg>/decision_journal.jsonl +
# /tmp diag): signals CLEAR the 0.38 confidence gate (consumer 18, chemicals 5)
# -- the gate is NOT the killer. Entries die downstream:
#   - consumer: 8 BUYs (7 from dividend_gap, 1 momentum) get quantity_zero'd
#     (expensive staples; low-conf BUYs size below one whole share);
#   - chemicals: ZERO BUY signals at all (5 cleared signals ALL momentum SELL),
#     so no position ever opens -- the OMITTED dividend_gap was the missing
#     BUY-entry source (PHOR 7 + AKRN 1 dividend records justify it).
# mean_reversion fires ZERO times in BOTH (structurally dead on these names).
# PRINCIPLED (not curve-fit) fix: emphasize the strategies that DO generate
# entries -- dividend_gap (justified by dividends) at an elevated weight, and
# momentum -- so BUY-entry confidence sizes above the one-share floor. No
# threshold change (signals already clear 0.38), no per-symbol tuning, weights
# stay in [0.10, 0.55].
_RETRY_SEGMENTS = ("ru_consumer", "ru_chemicals")
_RETRY_DIVIDEND_GAP_WEIGHT_FLOOR = 0.25  # elevated above the original 0.10 dilution
_WEIGHT_BAND_LO = 0.10
_WEIGHT_BAND_HI = 0.55
_MIN_COMBINED_CONFIDENCE_FLOOR = 0.35


class TestRetrySectorEntrySourceWeighting:
    """Phase 68-06: principled re-weighting toward the firing BUY-entry source.

    The diagnosis showed dividend_gap is the only reliable BUY-entry generator
    for these expensive, mean-reversion-dead sectors. The retry preset MUST key
    dividend_gap ON at an elevated weight (the BUY-entry lever) while keeping
    every weight inside [0.10, 0.55] and min_combined_confidence >= 0.35.
    """

    @pytest.mark.parametrize("segment_id", _RETRY_SEGMENTS)
    def test_dividend_gap_keyed_on_as_buy_entry_source(
        self, combiner: StrategyCombiner, segment_id: str
    ) -> None:
        """dividend_gap is enabled with an elevated weight (the BUY-entry fix)."""
        config = combiner._load_config(segment_id)
        strategies = config.get("strategies")
        assert isinstance(strategies, dict)
        dg = strategies.get("dividend_gap")
        assert isinstance(dg, dict), f"{segment_id}: dividend_gap omitted (no BUY entries)"
        assert dg.get("enabled") is True, f"{segment_id}: dividend_gap not enabled"
        weight = float(dg.get("weight", 0.0))
        assert weight >= _RETRY_DIVIDEND_GAP_WEIGHT_FLOOR, (
            f"{segment_id}: dividend_gap weight {weight} too low to drive entries"
        )

    @pytest.mark.parametrize("segment_id", _RETRY_SEGMENTS)
    def test_weights_inside_anti_curve_fit_band(
        self, combiner: StrategyCombiner, segment_id: str
    ) -> None:
        """Every enabled strategy weight stays in [0.10, 0.55] (D-05 bound)."""
        config = combiner._load_config(segment_id)
        strategies = config.get("strategies")
        assert isinstance(strategies, dict)
        for name, block in strategies.items():
            if not isinstance(block, dict) or not block.get("enabled"):
                continue
            weight = float(block.get("weight", 0.0))
            assert _WEIGHT_BAND_LO <= weight <= _WEIGHT_BAND_HI, (
                f"{segment_id}.{name}: weight {weight} outside [0.10, 0.55]"
            )

    @pytest.mark.parametrize("segment_id", _RETRY_SEGMENTS)
    def test_min_combined_confidence_not_below_floor(
        self, combiner: StrategyCombiner, segment_id: str
    ) -> None:
        """min_combined_confidence is NOT lowered below the 0.35 floor (D-05)."""
        config = combiner._load_config(segment_id)
        mcc = float(config.get("min_combined_confidence", 0.0))
        assert mcc >= _MIN_COMBINED_CONFIDENCE_FLOOR, (
            f"{segment_id}: min_combined_confidence {mcc} below 0.35 floor (curve-fit ban)"
        )

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

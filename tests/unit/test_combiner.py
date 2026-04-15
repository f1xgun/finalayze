"""Unit tests for StrategyCombiner hooks and forward-compatibility methods.

APPLY-04: invalidate_segment_cache() is a no-op but callable without error.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from finalayze.strategies.base import BaseStrategy
from finalayze.strategies.combiner import StrategyCombiner


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

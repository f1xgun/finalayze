"""Unit tests for DualMomentum YAML key alignment (Bug B1).

Verifies that YAML presets define the keys the strategy code actually reads:
lookback_1m, lookback_3m, lookback_6m -- NOT lookback_fast / lookback_slow.
"""

from __future__ import annotations

from finalayze.strategies.dual_momentum import DualMomentumStrategy

# ---- Constants (no magic numbers -- ruff PLR2004) ----------------------------

EXPECTED_LOOKBACK_1M = 21
EXPECTED_LOOKBACK_3M = 63
EXPECTED_LOOKBACK_6M = 126
EXPECTED_MIN_CONFIDENCE = 0.65

# Segments that define a dual_momentum section in their YAML preset
# us_industrial removed in S1.2 (orphan preset; no corresponding segment).
US_SEGMENTS_WITH_DUAL_MOMENTUM = [
    "us_tech",
    "us_broad",
    "us_finance",
    "us_healthcare",
]


class TestDualMomentumYAMLKeys:
    """YAML presets must define lookback_1m / lookback_3m / lookback_6m."""

    def test_us_tech_has_correct_lookback_keys(self) -> None:
        """us_tech preset returns params with lookback_1m/3m/6m keys."""
        strategy = DualMomentumStrategy()
        params = strategy.get_parameters("us_tech")

        assert "lookback_1m" in params, "us_tech preset missing lookback_1m key"
        assert "lookback_3m" in params, "us_tech preset missing lookback_3m key"
        assert "lookback_6m" in params, "us_tech preset missing lookback_6m key"

    def test_us_tech_lookback_values(self) -> None:
        """us_tech preset lookback values match expected 21/63/126."""
        strategy = DualMomentumStrategy()
        params = strategy.get_parameters("us_tech")

        assert int(params["lookback_1m"]) == EXPECTED_LOOKBACK_1M  # type: ignore[call-overload]
        assert int(params["lookback_3m"]) == EXPECTED_LOOKBACK_3M  # type: ignore[call-overload]
        assert int(params["lookback_6m"]) == EXPECTED_LOOKBACK_6M  # type: ignore[call-overload]

    def test_no_stale_lookback_fast_slow_keys(self) -> None:
        """YAML presets must NOT contain the old lookback_fast/lookback_slow keys."""
        strategy = DualMomentumStrategy()
        for segment in US_SEGMENTS_WITH_DUAL_MOMENTUM:
            # Clear cache to force reload
            strategy._params_cache.clear()
            params = strategy.get_parameters(segment)
            assert "lookback_fast" not in params, (
                f"{segment} preset still has stale lookback_fast key"
            )
            assert "lookback_slow" not in params, (
                f"{segment} preset still has stale lookback_slow key"
            )

    def test_all_us_segments_have_correct_lookback_keys(self) -> None:
        """All US segments with dual_momentum define lookback_1m/3m/6m."""
        strategy = DualMomentumStrategy()
        for segment in US_SEGMENTS_WITH_DUAL_MOMENTUM:
            strategy._params_cache.clear()
            params = strategy.get_parameters(segment)
            assert "lookback_1m" in params, f"{segment} preset missing lookback_1m"
            assert "lookback_3m" in params, f"{segment} preset missing lookback_3m"
            assert "lookback_6m" in params, f"{segment} preset missing lookback_6m"

    def test_min_confidence_present(self) -> None:
        """us_tech preset has min_confidence = 0.65."""
        strategy = DualMomentumStrategy()
        params = strategy.get_parameters("us_tech")

        assert "min_confidence" in params
        assert float(params["min_confidence"]) == EXPECTED_MIN_CONFIDENCE  # type: ignore[arg-type]

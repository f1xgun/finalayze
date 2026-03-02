"""Tests for MOEX-specific training pipeline (B.6).

Validates that RU segments use different parameters than US segments.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

# Import after path setup
from scripts.train_models import (  # noqa: E402
    _LOOKBACK_DAYS,
    _MOEX_LOOKBACK_DAYS,
    _MOEX_MAX_DEPTH,
    _SEGMENT_SYMBOLS,
    _US_MAX_DEPTH,
    _get_lookback_days,
    _get_xgboost_max_depth,
    _is_moex_segment,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_EXPECTED_US_LOOKBACK = 1825  # 5 years
_EXPECTED_MOEX_LOOKBACK = 730  # 2 years
_EXPECTED_US_MAX_DEPTH = 5
_EXPECTED_MOEX_MAX_DEPTH = 3


class TestMoexSegmentDetection:
    """_is_moex_segment correctly identifies MOEX segments."""

    def test_ru_blue_chips_is_moex(self) -> None:
        assert _is_moex_segment("ru_blue_chips") is True

    def test_ru_energy_is_moex(self) -> None:
        assert _is_moex_segment("ru_energy") is True

    def test_ru_tech_is_moex(self) -> None:
        assert _is_moex_segment("ru_tech") is True

    def test_ru_finance_is_moex(self) -> None:
        assert _is_moex_segment("ru_finance") is True

    def test_us_tech_is_not_moex(self) -> None:
        assert _is_moex_segment("us_tech") is False

    def test_us_broad_is_not_moex(self) -> None:
        assert _is_moex_segment("us_broad") is False


class TestMoexShorterLookback:
    """RU segments use a 2-year lookback instead of 5-year."""

    def test_moex_lookback_is_2_years(self) -> None:
        assert _get_lookback_days("ru_blue_chips") == _EXPECTED_MOEX_LOOKBACK

    def test_moex_lookback_constant(self) -> None:
        assert _MOEX_LOOKBACK_DAYS == _EXPECTED_MOEX_LOOKBACK

    def test_us_lookback_is_5_years(self) -> None:
        assert _get_lookback_days("us_tech") == _EXPECTED_US_LOOKBACK

    def test_us_lookback_constant(self) -> None:
        assert _LOOKBACK_DAYS == _EXPECTED_US_LOOKBACK

    def test_all_ru_segments_use_short_lookback(self) -> None:
        for seg_id in _SEGMENT_SYMBOLS:
            if seg_id.startswith("ru_"):
                assert _get_lookback_days(seg_id) == _EXPECTED_MOEX_LOOKBACK


class TestMoexShallowerXGBoost:
    """RU segments use max_depth=3 for XGBoost."""

    def test_moex_max_depth_is_3(self) -> None:
        assert _get_xgboost_max_depth("ru_blue_chips") == _EXPECTED_MOEX_MAX_DEPTH

    def test_moex_max_depth_constant(self) -> None:
        assert _MOEX_MAX_DEPTH == _EXPECTED_MOEX_MAX_DEPTH

    def test_us_max_depth_is_5(self) -> None:
        assert _get_xgboost_max_depth("us_tech") == _EXPECTED_US_MAX_DEPTH

    def test_us_max_depth_constant(self) -> None:
        assert _US_MAX_DEPTH == _EXPECTED_US_MAX_DEPTH

    def test_all_ru_segments_use_shallow_depth(self) -> None:
        for seg_id in _SEGMENT_SYMBOLS:
            if seg_id.startswith("ru_"):
                assert _get_xgboost_max_depth(seg_id) == _EXPECTED_MOEX_MAX_DEPTH


class TestUSSegmentsUnchanged:
    """US segments continue to use existing parameters."""

    def test_us_tech_lookback(self) -> None:
        assert _get_lookback_days("us_tech") == _EXPECTED_US_LOOKBACK

    def test_us_healthcare_lookback(self) -> None:
        assert _get_lookback_days("us_healthcare") == _EXPECTED_US_LOOKBACK

    def test_us_finance_lookback(self) -> None:
        assert _get_lookback_days("us_finance") == _EXPECTED_US_LOOKBACK

    def test_us_broad_lookback(self) -> None:
        assert _get_lookback_days("us_broad") == _EXPECTED_US_LOOKBACK

    def test_us_tech_max_depth(self) -> None:
        assert _get_xgboost_max_depth("us_tech") == _EXPECTED_US_MAX_DEPTH

    def test_us_healthcare_max_depth(self) -> None:
        assert _get_xgboost_max_depth("us_healthcare") == _EXPECTED_US_MAX_DEPTH

    def test_us_finance_max_depth(self) -> None:
        assert _get_xgboost_max_depth("us_finance") == _EXPECTED_US_MAX_DEPTH

    def test_us_broad_max_depth(self) -> None:
        assert _get_xgboost_max_depth("us_broad") == _EXPECTED_US_MAX_DEPTH

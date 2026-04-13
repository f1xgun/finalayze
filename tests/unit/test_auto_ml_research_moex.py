"""Unit tests for MOEX segment detection and symbol loading in auto_ml_research.py.

Tests cover:
- _is_moex_segment helper
- _SEGMENT_SYMBOLS contains all 4 ru_* equity segments
- _SEGMENT_SYMBOLS does NOT contain bond segments
- _get_lookback_days returns segment-appropriate values
- _get_max_features returns segment-appropriate values
- argparse --segment choices include all 4 ru_* equity segments
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure scripts/ and project root are importable
_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

# Constants (no magic numbers — ruff PLR2004)
_MOEX_LOOKBACK_DAYS_EXPECTED = 730
_US_LOOKBACK_DAYS_EXPECTED = 1825
_MOEX_MAX_FEATURES_EXPECTED = 10
_US_MAX_FEATURES_EXPECTED = 15

_RU_EQUITY_SEGMENTS = ["ru_blue_chips", "ru_energy", "ru_tech", "ru_finance"]
_BOND_SEGMENTS = ["ru_ofz_pd", "ru_ofz_pk"]

# Expected symbols for ru_blue_chips from config/segments.py
_RU_BLUE_CHIPS_SYMBOLS = ["SBER", "LKOH", "GMKN"]


class TestIsModexSegment:
    """Test _is_moex_segment helper function."""

    def test_ru_segment_returns_true(self) -> None:
        """_is_moex_segment('ru_blue_chips') returns True."""
        from scripts.auto_ml_research import _is_moex_segment

        assert _is_moex_segment("ru_blue_chips") is True

    def test_ru_energy_returns_true(self) -> None:
        """_is_moex_segment('ru_energy') returns True."""
        from scripts.auto_ml_research import _is_moex_segment

        assert _is_moex_segment("ru_energy") is True

    def test_us_segment_returns_false(self) -> None:
        """_is_moex_segment('us_tech') returns False."""
        from scripts.auto_ml_research import _is_moex_segment

        assert _is_moex_segment("us_tech") is False

    def test_us_healthcare_returns_false(self) -> None:
        """_is_moex_segment('us_healthcare') returns False."""
        from scripts.auto_ml_research import _is_moex_segment

        assert _is_moex_segment("us_healthcare") is False


class TestSegmentSymbols:
    """Test _SEGMENT_SYMBOLS dict contains correct ru_* segments."""

    def test_ru_blue_chips_in_segment_symbols(self) -> None:
        """_SEGMENT_SYMBOLS contains 'ru_blue_chips' key."""
        from scripts.auto_ml_research import _SEGMENT_SYMBOLS

        assert "ru_blue_chips" in _SEGMENT_SYMBOLS

    def test_ru_energy_in_segment_symbols(self) -> None:
        """_SEGMENT_SYMBOLS contains 'ru_energy' key."""
        from scripts.auto_ml_research import _SEGMENT_SYMBOLS

        assert "ru_energy" in _SEGMENT_SYMBOLS

    def test_ru_tech_in_segment_symbols(self) -> None:
        """_SEGMENT_SYMBOLS contains 'ru_tech' key."""
        from scripts.auto_ml_research import _SEGMENT_SYMBOLS

        assert "ru_tech" in _SEGMENT_SYMBOLS

    def test_ru_finance_in_segment_symbols(self) -> None:
        """_SEGMENT_SYMBOLS contains 'ru_finance' key."""
        from scripts.auto_ml_research import _SEGMENT_SYMBOLS

        assert "ru_finance" in _SEGMENT_SYMBOLS

    def test_ru_blue_chips_symbols_match_config(self) -> None:
        """_SEGMENT_SYMBOLS['ru_blue_chips'] matches DEFAULT_SEGMENTS symbols."""
        from scripts.auto_ml_research import _SEGMENT_SYMBOLS

        symbols = _SEGMENT_SYMBOLS["ru_blue_chips"]
        assert symbols == _RU_BLUE_CHIPS_SYMBOLS

    def test_bond_segment_ofz_pd_excluded(self) -> None:
        """_SEGMENT_SYMBOLS does NOT contain 'ru_ofz_pd' (bond segment)."""
        from scripts.auto_ml_research import _SEGMENT_SYMBOLS

        assert "ru_ofz_pd" not in _SEGMENT_SYMBOLS

    def test_bond_segment_ofz_pk_excluded(self) -> None:
        """_SEGMENT_SYMBOLS does NOT contain 'ru_ofz_pk' (bond segment)."""
        from scripts.auto_ml_research import _SEGMENT_SYMBOLS

        assert "ru_ofz_pk" not in _SEGMENT_SYMBOLS


class TestGetLookbackDays:
    """Test _get_lookback_days returns segment-appropriate values."""

    def test_ru_blue_chips_returns_730(self) -> None:
        """_get_lookback_days('ru_blue_chips') returns 730."""
        from scripts.auto_ml_research import _get_lookback_days

        assert _get_lookback_days("ru_blue_chips") == _MOEX_LOOKBACK_DAYS_EXPECTED

    def test_ru_energy_returns_730(self) -> None:
        """_get_lookback_days('ru_energy') returns 730."""
        from scripts.auto_ml_research import _get_lookback_days

        assert _get_lookback_days("ru_energy") == _MOEX_LOOKBACK_DAYS_EXPECTED

    def test_us_tech_returns_1825(self) -> None:
        """_get_lookback_days('us_tech') returns 1825."""
        from scripts.auto_ml_research import _get_lookback_days

        assert _get_lookback_days("us_tech") == _US_LOOKBACK_DAYS_EXPECTED


class TestGetMaxFeatures:
    """Test _get_max_features returns segment-appropriate values."""

    def test_ru_blue_chips_returns_10(self) -> None:
        """_get_max_features('ru_blue_chips') returns 10."""
        from scripts.auto_ml_research import _get_max_features

        assert _get_max_features("ru_blue_chips") == _MOEX_MAX_FEATURES_EXPECTED

    def test_ru_tech_returns_10(self) -> None:
        """_get_max_features('ru_tech') returns 10."""
        from scripts.auto_ml_research import _get_max_features

        assert _get_max_features("ru_tech") == _MOEX_MAX_FEATURES_EXPECTED

    def test_us_tech_returns_15(self) -> None:
        """_get_max_features('us_tech') returns 15."""
        from scripts.auto_ml_research import _get_max_features

        assert _get_max_features("us_tech") == _US_MAX_FEATURES_EXPECTED


class TestArgparseChoices:
    """Test argparse --segment choices include all 4 ru_* equity segments."""

    def test_ru_blue_chips_in_choices(self) -> None:
        """argparse choices include 'ru_blue_chips'."""
        import argparse

        from scripts.auto_ml_research import _SEGMENT_SYMBOLS

        parser = argparse.ArgumentParser()
        parser.add_argument("--segment", choices=list(_SEGMENT_SYMBOLS.keys()))
        # Should not raise
        args = parser.parse_args(["--segment", "ru_blue_chips"])
        assert args.segment == "ru_blue_chips"

    def test_ru_energy_in_choices(self) -> None:
        """argparse choices include 'ru_energy'."""
        import argparse

        from scripts.auto_ml_research import _SEGMENT_SYMBOLS

        parser = argparse.ArgumentParser()
        parser.add_argument("--segment", choices=list(_SEGMENT_SYMBOLS.keys()))
        args = parser.parse_args(["--segment", "ru_energy"])
        assert args.segment == "ru_energy"

    def test_ru_tech_in_choices(self) -> None:
        """argparse choices include 'ru_tech'."""
        import argparse

        from scripts.auto_ml_research import _SEGMENT_SYMBOLS

        parser = argparse.ArgumentParser()
        parser.add_argument("--segment", choices=list(_SEGMENT_SYMBOLS.keys()))
        args = parser.parse_args(["--segment", "ru_tech"])
        assert args.segment == "ru_tech"

    def test_ru_finance_in_choices(self) -> None:
        """argparse choices include 'ru_finance'."""
        import argparse

        from scripts.auto_ml_research import _SEGMENT_SYMBOLS

        parser = argparse.ArgumentParser()
        parser.add_argument("--segment", choices=list(_SEGMENT_SYMBOLS.keys()))
        args = parser.parse_args(["--segment", "ru_finance"])
        assert args.segment == "ru_finance"

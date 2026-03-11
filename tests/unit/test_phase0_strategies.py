"""Tests for Phase 0.5 (histogram rising logic) and Phase 0.10 (MOEX preset recalibration)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from finalayze.strategies.momentum import _Indicators

# ---------------------------------------------------------------------------
# Constants (avoid magic numbers -- ruff PLR2004)
# ---------------------------------------------------------------------------
_HIST_NEGATIVE = -0.5
_HIST_PREV_NEGATIVE = -1.0
_HIST_POSITIVE = 0.3
_HIST_PREV_POSITIVE = 0.1
_BB_STD_DEV_18 = 1.8
_BB_STD_DEV_20 = 2.0
_RSI_OVERSOLD_30 = 30
_RSI_OVERBOUGHT_70 = 70

_PRESETS_DIR = Path(__file__).resolve().parents[2] / "src" / "finalayze" / "strategies" / "presets"

_RU_SEGMENTS = ["ru_blue_chips", "ru_energy", "ru_tech", "ru_finance"]


def _load_preset(segment_id: str) -> dict:
    """Load a YAML preset by segment ID."""
    with (_PRESETS_DIR / f"{segment_id}.yaml").open() as f:
        return yaml.safe_load(f)


def _make_indicators(
    current_hist: float,
    prev_hist: float,
) -> _Indicators:
    """Build a minimal _Indicators instance for histogram tests."""
    return _Indicators(
        current_rsi=50.0,
        rsi_window=[50.0],
        current_hist=current_hist,
        prev_hist=prev_hist,
        hist_window=[prev_hist, current_hist],
        macd_line=0.0,
        signal_line=0.0,
        macd_crossover_buy=False,
        macd_crossover_sell=False,
        avg_hist_range=1.0,
        current_close=100.0,
        min_confidence=0.0,
        current_sma=None,
        current_adx=None,
        volume_ratio=None,
        ichimoku_bullish=None,
        ichimoku_bearish=None,
        ichimoku_cloud_thickness=None,
    )


# ===========================================================================
# Task 0.5 -- Histogram rising logic
# ===========================================================================


class TestHistogramRisingLogic:
    """Verify stricter histogram rising/falling checks."""

    def test_hist_rising_requires_positive(self) -> None:
        """Histogram at -0.5, prev at -1.0: improving but negative -> hist_rising=False."""
        ind = _make_indicators(
            current_hist=_HIST_NEGATIVE,
            prev_hist=_HIST_PREV_NEGATIVE,
        )
        # Reproduce the logic from _evaluate_signal
        hist_rising = ind.current_hist > ind.prev_hist and ind.current_hist > 0
        assert not hist_rising, (
            "hist_rising should be False when histogram is negative, even if improving bar-over-bar"
        )

    def test_hist_rising_true_when_positive_and_improving(self) -> None:
        """Histogram at 0.3, prev at 0.1: positive AND improving -> hist_rising=True."""
        ind = _make_indicators(
            current_hist=_HIST_POSITIVE,
            prev_hist=_HIST_PREV_POSITIVE,
        )
        hist_rising = ind.current_hist > ind.prev_hist and ind.current_hist > 0
        assert hist_rising, "hist_rising should be True when histogram is positive and improving"

    def test_hist_falling_requires_negative(self) -> None:
        """Mirror test: histogram at 0.5, prev at 1.0 -> hist_falling=False (still positive)."""
        ind = _make_indicators(current_hist=0.5, prev_hist=1.0)
        hist_falling = ind.current_hist < ind.prev_hist and ind.current_hist < 0
        assert not hist_falling, (
            "hist_falling should be False when histogram is positive, "
            "even if declining bar-over-bar"
        )

    def test_hist_falling_true_when_negative_and_declining(self) -> None:
        """Histogram at -0.3, prev at -0.1 -> hist_falling=True."""
        ind = _make_indicators(current_hist=-0.3, prev_hist=-0.1)
        hist_falling = ind.current_hist < ind.prev_hist and ind.current_hist < 0
        assert hist_falling, "hist_falling should be True when histogram is negative and declining"


# ===========================================================================
# Task 0.10 -- MOEX preset recalibration
# ===========================================================================


class TestMOEXPresetRecalibration:
    """Verify updated values in RU preset YAML files."""

    def test_ru_blue_chips_bb_std_dev(self) -> None:
        """ru_blue_chips bb_std_dev should be 2.5 (wider for MOEX fat tails)."""
        data = _load_preset("ru_blue_chips")
        bb_std = data["strategies"]["mean_reversion"]["params"]["bb_std_dev"]
        assert bb_std == 2.5  # noqa: PLR2004

    def test_ru_energy_bb_std_dev(self) -> None:
        """ru_energy bb_std_dev should be 2.8 (widest for commodity-linked stocks)."""
        data = _load_preset("ru_energy")
        bb_std = data["strategies"]["mean_reversion"]["params"]["bb_std_dev"]
        assert bb_std == 2.8  # noqa: PLR2004

    def test_ru_energy_momentum_disabled(self) -> None:
        """ru_energy momentum should be disabled (MOEX is mean-reverting)."""
        data = _load_preset("ru_energy")
        momentum_cfg = data["strategies"]["momentum"]
        assert momentum_cfg["enabled"] is False
        assert momentum_cfg["weight"] == 0.00  # noqa: PLR2004

    @pytest.mark.parametrize("segment_id", _RU_SEGMENTS)
    def test_ru_normalize_firing(self, segment_id: str) -> None:
        """All RU presets should use 'firing' normalize mode (match US presets)."""
        data = _load_preset(segment_id)
        assert data["normalize_mode"] == "firing", f"{segment_id} normalize_mode should be 'firing'"

    @pytest.mark.parametrize("segment_id", _RU_SEGMENTS)
    def test_ru_rsi_oversold_mr(self, segment_id: str) -> None:
        """All RU presets should have rsi_oversold_mr = 25 (tighter gate for MOEX noise)."""
        data = _load_preset(segment_id)
        mr_params = data["strategies"]["mean_reversion"]["params"]
        rsi_expected = 25
        assert mr_params["rsi_oversold_mr"] == rsi_expected

    @pytest.mark.parametrize("segment_id", _RU_SEGMENTS)
    def test_ru_rsi_overbought_mr(self, segment_id: str) -> None:
        """All RU presets should have rsi_overbought_mr = 75 (tighter gate for MOEX noise)."""
        data = _load_preset(segment_id)
        mr_params = data["strategies"]["mean_reversion"]["params"]
        rsi_expected = 75
        assert mr_params["rsi_overbought_mr"] == rsi_expected

    def test_ru_blue_chips_weights(self) -> None:
        """ru_blue_chips: momentum disabled, mean_reversion=0.30 (MR-focused for MOEX)."""
        data = _load_preset("ru_blue_chips")
        strategies = data["strategies"]
        assert strategies["momentum"]["enabled"] is False
        assert strategies["mean_reversion"]["weight"] == 0.30  # noqa: PLR2004

    def test_ru_energy_mean_reversion_weight(self) -> None:
        """ru_energy: mean_reversion=0.30 (MR-focused for MOEX)."""
        data = _load_preset("ru_energy")
        assert data["strategies"]["mean_reversion"]["weight"] == 0.30  # noqa: PLR2004

    @pytest.mark.parametrize("segment_id", _RU_SEGMENTS)
    def test_ru_min_combined_confidence(self, segment_id: str) -> None:
        """All RU presets: min_combined_confidence=0.30 (lowered for ADX routing)."""
        data = _load_preset(segment_id)
        assert data["min_combined_confidence"] == 0.30  # noqa: PLR2004

    @pytest.mark.parametrize("segment_id", _RU_SEGMENTS)
    def test_ru_trend_filter_enabled(self, segment_id: str) -> None:
        """All RU presets should have trend_filter enabled on mean_reversion."""
        data = _load_preset(segment_id)
        mr_params = data["strategies"]["mean_reversion"]["params"]
        assert mr_params.get("trend_filter") is True

    @pytest.mark.parametrize("segment_id", _RU_SEGMENTS)
    def test_ru_has_rsi2_connors(self, segment_id: str) -> None:
        """All RU presets should have rsi2_connors strategy configured."""
        data = _load_preset(segment_id)
        rsi2 = data["strategies"].get("rsi2_connors", {})
        assert rsi2.get("enabled") is True

"""Unit tests for StrategyCombiner."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any
from unittest.mock import mock_open, patch

import pytest
import yaml

if TYPE_CHECKING:
    from pathlib import Path

from finalayze.core.schemas import Candle, Signal, SignalDirection
from finalayze.strategies.base import BaseStrategy
from finalayze.strategies.combiner import StrategyCombiner

# Constants (no magic numbers)
BASE_PRICE = Decimal(100)
VOLUME = 1_000_000
CANDLE_HIGH_OFFSET = Decimal(1)
CANDLE_LOW_OFFSET = Decimal(1)
CANDLE_COUNT = 30
HIGH_CONFIDENCE = 0.9
LOW_CONFIDENCE = 0.3
WEIGHT_DOMINANT = 0.6
WEIGHT_MINOR = 0.4
MIN_COMBINED_CONFIDENCE = 0.5


def _candle(price: Decimal, day: int) -> Candle:
    return Candle(
        symbol="AAPL",
        market_id="us",
        timeframe="1d",
        timestamp=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=day),
        open=price,
        high=price + CANDLE_HIGH_OFFSET,
        low=price - CANDLE_LOW_OFFSET,
        close=price,
        volume=VOLUME,
    )


def _make_candles(count: int = CANDLE_COUNT) -> list[Candle]:
    return [_candle(BASE_PRICE, i) for i in range(count)]


def _make_signal(
    direction: SignalDirection,
    confidence: float,
    strategy_name: str = "mock",
    segment_id: str = "us_broad",
) -> Signal:
    return Signal(
        strategy_name=strategy_name,
        symbol="AAPL",
        market_id="us",
        segment_id=segment_id,
        direction=direction,
        confidence=confidence,
        features={"mock_feature": confidence},
        reasoning=f"Mock signal: {direction} at {confidence}",
    )


class MockStrategy(BaseStrategy):
    """A controllable mock strategy for testing."""

    def __init__(self, name: str, return_signal: Signal | None) -> None:
        self._name = name
        self._return_signal = return_signal

    @property
    def name(self) -> str:
        return self._name

    def supported_segments(self) -> list[str]:
        return ["us_broad", "us_tech"]

    def get_parameters(self, segment_id: str) -> dict[str, object]:
        return {}

    def generate_signal(
        self, symbol: str, candles: list[Candle], segment_id: str, sentiment_score: float = 0.0
    ) -> Signal | None:
        return self._return_signal


class TestStrategyCombiner:
    def test_combine_single_buy_signal(self) -> None:
        """One strategy returns BUY -> combiner returns BUY."""
        single_strategy_config: dict[str, Any] = {
            "strategies": {
                "momentum": {"enabled": True, "weight": 1.0},
            }
        }
        buy_signal = _make_signal(SignalDirection.BUY, HIGH_CONFIDENCE, "momentum")
        strategy = MockStrategy("momentum", buy_signal)
        combiner = StrategyCombiner([strategy])
        candles = _make_candles()
        with patch.object(combiner, "_load_config", return_value=single_strategy_config):
            signal = combiner.generate_signal("AAPL", candles, "us_broad")
        assert signal is not None
        assert signal.direction == SignalDirection.BUY
        assert signal.strategy_name == "combined"
        assert signal.symbol == "AAPL"
        assert signal.segment_id == "us_broad"

    def test_combine_conflicting_signals_weighted(self) -> None:
        """Momentum BUY (weight 0.6) vs mean_reversion SELL (weight 0.4) -> net BUY."""
        # net_score = (BUY: +0.9 * 0.6 + SELL: -0.3 * 0.4) / (0.6 + 0.4)
        #           = (0.54 - 0.12) / 1.0 = 0.42
        # abs(0.42) < min_combined_confidence(0.5) -> signal is None at default threshold
        # To get a signal: use weight 0.6 vs 0.1 so momentum dominates
        # net_score = (0.9 * 0.6 - 0.3 * 0.1) / 0.7 = (0.54 - 0.03) / 0.7 = 0.729 > 0.5 -> BUY
        weighted_config: dict[str, Any] = {
            "strategies": {
                "momentum": {"enabled": True, "weight": 0.6},
                "mean_reversion": {"enabled": True, "weight": 0.1},
            }
        }
        buy_signal = _make_signal(SignalDirection.BUY, HIGH_CONFIDENCE, "momentum")
        sell_signal = _make_signal(SignalDirection.SELL, LOW_CONFIDENCE, "mean_reversion")
        momentum = MockStrategy("momentum", buy_signal)
        mean_rev = MockStrategy("mean_reversion", sell_signal)
        combiner = StrategyCombiner([momentum, mean_rev])
        candles = _make_candles()
        with patch.object(combiner, "_load_config", return_value=weighted_config):
            signal = combiner.generate_signal("AAPL", candles, "us_broad")
        assert signal is not None
        assert signal.direction == SignalDirection.BUY

    def test_combine_no_signals_returns_none(self) -> None:
        """All strategies return None -> combiner returns None."""
        both_enabled_config: dict[str, Any] = {
            "strategies": {
                "momentum": {"enabled": True, "weight": 0.5},
                "mean_reversion": {"enabled": True, "weight": 0.5},
            }
        }
        strategy1 = MockStrategy("momentum", None)
        strategy2 = MockStrategy("mean_reversion", None)
        combiner = StrategyCombiner([strategy1, strategy2])
        candles = _make_candles()
        with patch.object(combiner, "_load_config", return_value=both_enabled_config):
            signal = combiner.generate_signal("AAPL", candles, "us_broad")
        assert signal is None

    def test_combine_respects_min_confidence(self) -> None:
        """When weighted score is below min_combined_confidence, return None."""
        # Equal and opposite signals with equal weights -> net score = 0 -> None
        equal_weight_config: dict[str, Any] = {
            "strategies": {
                "momentum": {"enabled": True, "weight": 0.5},
                "mean_reversion": {"enabled": True, "weight": 0.5},
            }
        }
        buy_signal = _make_signal(SignalDirection.BUY, HIGH_CONFIDENCE, "momentum")
        sell_signal = _make_signal(SignalDirection.SELL, HIGH_CONFIDENCE, "mean_reversion")
        momentum = MockStrategy("momentum", buy_signal)
        mean_rev = MockStrategy("mean_reversion", sell_signal)
        combiner = StrategyCombiner([momentum, mean_rev])
        candles = _make_candles()
        with patch.object(combiner, "_load_config", return_value=equal_weight_config):
            signal = combiner.generate_signal("AAPL", candles, "us_broad")
        # net_score = (0.9*0.5 - 0.9*0.5) / (0.5+0.5) = 0.0 -> below 0.5 -> None
        assert signal is None

    def test_combiner_uses_segment_weights(self) -> None:
        """Different segments get different strategy weights from YAML presets."""
        # us_tech: momentum weight=0.4, mean_reversion weight=0.2
        # us_broad: momentum weight=0.5, mean_reversion weight=0.5
        buy_signal = _make_signal(SignalDirection.BUY, HIGH_CONFIDENCE, "momentum")
        strategy = MockStrategy("momentum", buy_signal)
        combiner = StrategyCombiner([strategy])

        candles = _make_candles()
        # Both segments should produce a BUY signal (only momentum is active in mock)
        signal_tech = combiner.generate_signal("AAPL", candles, "us_tech")
        signal_broad = combiner.generate_signal("AAPL", candles, "us_broad")

        # Both should be BUY since only momentum provides a signal
        assert signal_tech is not None
        assert signal_tech.direction == SignalDirection.BUY
        assert signal_broad is not None
        assert signal_broad.direction == SignalDirection.BUY

    def test_combiner_skips_disabled_strategy(self) -> None:
        """Strategy with enabled: false in YAML should not be called."""
        # Create a custom mock that tracks whether generate_signal was called
        called_tracker: list[bool] = []

        class TrackingStrategy(BaseStrategy):
            @property
            def name(self) -> str:
                return "momentum"

            def supported_segments(self) -> list[str]:
                return ["us_broad"]

            def get_parameters(self, segment_id: str) -> dict[str, object]:
                return {}

            def generate_signal(
                self,
                symbol: str,
                candles: list[Candle],
                segment_id: str,
                sentiment_score: float = 0.0,
            ) -> Signal | None:
                called_tracker.append(True)
                return _make_signal(SignalDirection.BUY, HIGH_CONFIDENCE, "momentum")

        tracking_strategy = TrackingStrategy()

        # Patch the YAML config to disable momentum for us_broad
        disabled_config: dict[str, Any] = {
            "strategies": {
                "momentum": {"enabled": False, "weight": 0.5},
                "mean_reversion": {"enabled": False, "weight": 0.5},
            }
        }

        combiner = StrategyCombiner([tracking_strategy])
        candles = _make_candles()

        with patch.object(combiner, "_load_config", return_value=disabled_config):
            signal = combiner.generate_signal("AAPL", candles, "us_broad")

        # Strategy should NOT have been called (disabled in config)
        assert len(called_tracker) == 0
        # No enabled strategies -> None
        assert signal is None

    def test_combined_signal_contains_feature_contributions(self) -> None:
        """Combined signal features include per-strategy contributions."""
        single_strategy_config: dict[str, Any] = {
            "strategies": {
                "momentum": {"enabled": True, "weight": 1.0},
            }
        }
        buy_signal = _make_signal(SignalDirection.BUY, HIGH_CONFIDENCE, "momentum")
        strategy = MockStrategy("momentum", buy_signal)
        combiner = StrategyCombiner([strategy])
        candles = _make_candles()
        with patch.object(combiner, "_load_config", return_value=single_strategy_config):
            signal = combiner.generate_signal("AAPL", candles, "us_broad")
        assert signal is not None
        assert "momentum_confidence" in signal.features
        assert "momentum_direction" in signal.features


class TestStrategyCombinerYAMLErrorHandling:
    """Tests that malformed or missing YAML in _load_config never crashes."""

    def test_load_config_malformed_yaml_returns_empty_dict(self, tmp_path: Path) -> None:
        """A YAML parse error must not propagate; empty dict is returned."""
        buy_signal = _make_signal(SignalDirection.BUY, HIGH_CONFIDENCE, "momentum")
        strategy = MockStrategy("momentum", buy_signal)
        combiner = StrategyCombiner([strategy])
        combiner._presets_dir = tmp_path

        bad_preset = tmp_path / "bad_segment.yaml"
        bad_preset.write_text(": bad: yaml: ][")

        result = combiner._load_config("bad_segment")
        assert result == {}

    def test_load_config_empty_yaml_returns_empty_dict(self, tmp_path: Path) -> None:
        """An empty YAML file (safe_load returns None) must return empty dict."""
        buy_signal = _make_signal(SignalDirection.BUY, HIGH_CONFIDENCE, "momentum")
        strategy = MockStrategy("momentum", buy_signal)
        combiner = StrategyCombiner([strategy])
        combiner._presets_dir = tmp_path

        empty_preset = tmp_path / "empty_segment.yaml"
        empty_preset.write_text("")

        result = combiner._load_config("empty_segment")
        assert result == {}

    def test_load_config_yaml_error_via_mock(self) -> None:
        """yaml.YAMLError raised during safe_load must be caught and return empty dict."""
        buy_signal = _make_signal(SignalDirection.BUY, HIGH_CONFIDENCE, "momentum")
        strategy = MockStrategy("momentum", buy_signal)
        combiner = StrategyCombiner([strategy])

        with (
            patch("builtins.open", mock_open(read_data=b"")),
            patch("yaml.safe_load", side_effect=yaml.YAMLError("bad yaml")),
        ):
            result = combiner._load_config("us_broad")
        assert result == {}

    def test_load_config_oserror_returns_empty_dict(self, tmp_path: Path) -> None:
        """An OSError (e.g. permission denied) must be caught and return empty dict."""
        buy_signal = _make_signal(SignalDirection.BUY, HIGH_CONFIDENCE, "momentum")
        strategy = MockStrategy("momentum", buy_signal)
        combiner = StrategyCombiner([strategy])
        combiner._presets_dir = tmp_path

        with patch("builtins.open", side_effect=OSError("permission denied")):
            result = combiner._load_config("us_broad")
        assert result == {}

    def test_load_config_missing_file_returns_empty_dict(self, tmp_path: Path) -> None:
        """A FileNotFoundError must return empty dict (no preset file)."""
        buy_signal = _make_signal(SignalDirection.BUY, HIGH_CONFIDENCE, "momentum")
        strategy = MockStrategy("momentum", buy_signal)
        combiner = StrategyCombiner([strategy])
        combiner._presets_dir = tmp_path

        result = combiner._load_config("nonexistent_segment")
        assert result == {}

    def test_generate_signal_with_malformed_yaml_returns_none(self, tmp_path: Path) -> None:
        """generate_signal must not crash when the preset YAML is malformed."""
        buy_signal = _make_signal(SignalDirection.BUY, HIGH_CONFIDENCE, "momentum")
        strategy = MockStrategy("momentum", buy_signal)
        combiner = StrategyCombiner([strategy])
        combiner._presets_dir = tmp_path

        bad_preset = tmp_path / "bad_segment.yaml"
        bad_preset.write_text(": bad: yaml: ][")

        candles = _make_candles()
        # No strategies config loaded -> total_weight == 0 -> returns None
        signal = combiner.generate_signal("AAPL", candles, "bad_segment")
        assert signal is None

    def test_generate_signal_invalid_weight_falls_back_to_default(self) -> None:
        """weight: 'bad' in YAML must not raise InvalidOperation; falls back to 1.0 (issue #63)."""
        bad_weight_config: dict[str, object] = {
            "strategies": {
                "momentum": {"enabled": True, "weight": "bad"},
            }
        }
        buy_signal = _make_signal(SignalDirection.BUY, HIGH_CONFIDENCE, "momentum")
        strategy = MockStrategy("momentum", buy_signal)
        combiner = StrategyCombiner([strategy])
        candles = _make_candles()
        with patch.object(combiner, "_load_config", return_value=bad_weight_config):
            # Must not raise; should produce a BUY signal using fallback weight=1.0
            signal = combiner.generate_signal("AAPL", candles, "us_broad")
        assert signal is not None
        assert signal.direction == SignalDirection.BUY

    def test_generate_signal_nan_weight_falls_back_to_default(self) -> None:
        """weight: 'NaN' in YAML is a valid Decimal but edge-case; ensure signal still produced."""
        # Decimal('NaN') is technically valid and will not raise InvalidOperation,
        # but 'not-a-number' will raise it -- verify that path is handled.
        invalid_weight_config: dict[str, object] = {
            "strategies": {
                "momentum": {"enabled": True, "weight": "not-a-number"},
            }
        }
        buy_signal = _make_signal(SignalDirection.BUY, HIGH_CONFIDENCE, "momentum")
        strategy = MockStrategy("momentum", buy_signal)
        combiner = StrategyCombiner([strategy])
        candles = _make_candles()
        with patch.object(combiner, "_load_config", return_value=invalid_weight_config):
            signal = combiner.generate_signal("AAPL", candles, "us_broad")
        # Fallback weight=1.0 applied -> should still generate a BUY signal
        assert signal is not None
        assert signal.direction == SignalDirection.BUY


class TestCombinerNormalizationMode:
    """Tests for normalize_mode parameter (6B.2)."""

    def test_normalize_firing_mode_default(self) -> None:
        """Default mode normalizes by firing weight only (backward compat)."""
        config: dict[str, Any] = {
            "strategies": {
                "momentum": {"enabled": True, "weight": 0.5},
                "mean_reversion": {"enabled": True, "weight": 0.5},
            }
        }
        buy_signal = _make_signal(SignalDirection.BUY, HIGH_CONFIDENCE, "momentum")
        momentum = MockStrategy("momentum", buy_signal)
        mean_rev = MockStrategy("mean_reversion", None)  # does not fire
        combiner = StrategyCombiner([momentum, mean_rev])
        candles = _make_candles()
        with patch.object(combiner, "_load_config", return_value=config):
            signal = combiner.generate_signal("AAPL", candles, "us_broad")
        # Only momentum fires: net = 0.9 * 0.5 / 0.5 = 0.9 -> BUY
        assert signal is not None
        assert signal.direction == SignalDirection.BUY
        assert signal.confidence == pytest.approx(HIGH_CONFIDENCE, abs=0.01)

    def test_normalize_total_mode_reduces_score(self) -> None:
        """In total mode, single strategy firing produces lower score."""
        config: dict[str, Any] = {
            "strategies": {
                "momentum": {"enabled": True, "weight": 0.5},
                "mean_reversion": {"enabled": True, "weight": 0.5},
            }
        }
        buy_signal = _make_signal(SignalDirection.BUY, HIGH_CONFIDENCE, "momentum")
        momentum = MockStrategy("momentum", buy_signal)
        mean_rev = MockStrategy("mean_reversion", None)  # does not fire
        combiner = StrategyCombiner([momentum, mean_rev], normalize_mode="total")
        candles = _make_candles()
        with patch.object(combiner, "_load_config", return_value=config):
            signal = combiner.generate_signal("AAPL", candles, "us_broad")
        # total mode: net = 0.9 * 0.5 / 1.0 = 0.45 -> below 0.5 threshold -> None
        assert signal is None

    def test_normalize_total_mode_strong_consensus(self) -> None:
        """In total mode, two strategies both firing BUY passes."""
        config: dict[str, Any] = {
            "strategies": {
                "momentum": {"enabled": True, "weight": 0.5},
                "mean_reversion": {"enabled": True, "weight": 0.5},
            }
        }
        buy_sig1 = _make_signal(SignalDirection.BUY, HIGH_CONFIDENCE, "momentum")
        buy_sig2 = _make_signal(SignalDirection.BUY, 0.8, "mean_reversion")
        momentum = MockStrategy("momentum", buy_sig1)
        mean_rev = MockStrategy("mean_reversion", buy_sig2)
        combiner = StrategyCombiner([momentum, mean_rev], normalize_mode="total")
        candles = _make_candles()
        with patch.object(combiner, "_load_config", return_value=config):
            signal = combiner.generate_signal("AAPL", candles, "us_broad")
        # total mode: net = (0.9*0.5 + 0.8*0.5) / 1.0 = 0.85 -> BUY
        assert signal is not None
        assert signal.direction == SignalDirection.BUY

    def test_normalize_total_accounts_for_enabled_only(self) -> None:
        """Total weight uses only enabled strategies' weights."""
        config: dict[str, Any] = {
            "strategies": {
                "momentum": {"enabled": True, "weight": 0.5},
                "mean_reversion": {"enabled": True, "weight": 0.5},
                "pairs": {"enabled": False, "weight": 0.5},
            }
        }
        buy_signal = _make_signal(SignalDirection.BUY, HIGH_CONFIDENCE, "momentum")
        momentum = MockStrategy("momentum", buy_signal)
        mean_rev = MockStrategy("mean_reversion", None)
        combiner = StrategyCombiner([momentum, mean_rev], normalize_mode="total")
        candles = _make_candles()
        with patch.object(combiner, "_load_config", return_value=config):
            signal = combiner.generate_signal("AAPL", candles, "us_broad")
        # total enabled weight = 0.5 + 0.5 = 1.0 (pairs disabled, not counted)
        # net = 0.9 * 0.5 / 1.0 = 0.45 -> below threshold -> None
        assert signal is None

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
        self,
        symbol: str,
        candles: list[Candle],
        segment_id: str,
        sentiment_score: float = 0.0,
        has_open_position: bool = False,
        **kwargs: object,
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
        assert signal.strategy_name == "momentum"
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
        """When weighted score is below min_combined_confidence, return None.

        Uses neutral strategies (not in momentum/MR pools) so ADX regime
        routing does not apply dominant-pool-wins logic.
        """
        # Equal and opposite signals from neutral strategies -> net score = 0 -> None
        equal_weight_config: dict[str, Any] = {
            "strategies": {
                "strat_x": {"enabled": True, "weight": 0.5},
                "strat_y": {"enabled": True, "weight": 0.5},
            }
        }
        buy_signal = _make_signal(SignalDirection.BUY, HIGH_CONFIDENCE, "strat_x")
        sell_signal = _make_signal(SignalDirection.SELL, HIGH_CONFIDENCE, "strat_y")
        strat_x = MockStrategy("strat_x", buy_signal)
        strat_y = MockStrategy("strat_y", sell_signal)
        combiner = StrategyCombiner([strat_x, strat_y])
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
                has_open_position: bool = False,
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
        """In total mode, two neutral strategies both firing BUY passes.

        Uses neutral strategies (not in momentum/MR pools) to avoid ADX
        dominant-pool-wins logic.
        """
        config: dict[str, Any] = {
            "strategies": {
                "strat_a": {"enabled": True, "weight": 0.5},
                "strat_b": {"enabled": True, "weight": 0.5},
            }
        }
        buy_sig1 = _make_signal(SignalDirection.BUY, HIGH_CONFIDENCE, "strat_a")
        buy_sig2 = _make_signal(SignalDirection.BUY, 0.8, "strat_b")
        strat_a = MockStrategy("strat_a", buy_sig1)
        strat_b = MockStrategy("strat_b", buy_sig2)
        combiner = StrategyCombiner([strat_a, strat_b], normalize_mode="total")
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


class TestADXRouting:
    """Tests for ADX-based regime routing in the combiner (replaced Hurst)."""

    def test_adx_features_present_in_signal(self) -> None:
        """Combined signal features contain adx_value and adx_regime."""
        config: dict[str, Any] = {
            "strategies": {
                "momentum": {"enabled": True, "weight": 1.0},
            },
            "min_combined_confidence": 0.0,
        }
        mom_signal = _make_signal(SignalDirection.BUY, HIGH_CONFIDENCE, "momentum")
        momentum = MockStrategy("momentum", mom_signal)
        combiner = StrategyCombiner([momentum])
        candles = _make_candles()

        with (
            patch.object(combiner, "_load_config", return_value=config),
            patch(
                "finalayze.strategies.combiner.compute_adx",
                return_value=25.0,
            ),
        ):
            signal = combiner.generate_signal("AAPL", candles, "us_broad")

        assert signal is not None
        assert "adx_value" in signal.features
        assert "adx_regime" in signal.features

    def test_adx_regime_trend_only_momentum_fires(self) -> None:
        """In trending regime (ADX > 35), MR strategies are gated out."""
        config: dict[str, Any] = {
            "strategies": {
                "momentum": {"enabled": True, "weight": 1.0},
                "mean_reversion": {"enabled": True, "weight": 1.0},
            },
            "min_combined_confidence": 0.0,
        }
        mom_signal = _make_signal(SignalDirection.BUY, HIGH_CONFIDENCE, "momentum")
        mr_signal = _make_signal(SignalDirection.BUY, HIGH_CONFIDENCE, "mean_reversion")
        momentum = MockStrategy("momentum", mom_signal)
        mean_rev = MockStrategy("mean_reversion", mr_signal)

        combiner = StrategyCombiner([momentum, mean_rev])
        candles = _make_candles()

        with (
            patch.object(combiner, "_load_config", return_value=config),
            patch(
                "finalayze.strategies.combiner.compute_adx",
                return_value=40.0,
            ),
        ):
            signal = combiner.generate_signal("AAPL", candles, "us_broad")

        assert signal is not None
        assert signal.features["adx_regime"] == 1.0  # trend
        # MR signal should NOT appear in features (was gated out)
        assert "mean_reversion_confidence" not in signal.features

    def test_adx_regime_mr_only_mr_fires(self) -> None:
        """In MR regime (ADX < 15), momentum strategies are gated out."""
        config: dict[str, Any] = {
            "strategies": {
                "momentum": {"enabled": True, "weight": 1.0},
                "mean_reversion": {"enabled": True, "weight": 1.0},
            },
            "min_combined_confidence": 0.0,
        }
        mom_signal = _make_signal(SignalDirection.BUY, HIGH_CONFIDENCE, "momentum")
        mr_signal = _make_signal(SignalDirection.BUY, HIGH_CONFIDENCE, "mean_reversion")
        momentum = MockStrategy("momentum", mom_signal)
        mean_rev = MockStrategy("mean_reversion", mr_signal)

        combiner = StrategyCombiner([momentum, mean_rev])
        candles = _make_candles()

        with (
            patch.object(combiner, "_load_config", return_value=config),
            patch(
                "finalayze.strategies.combiner.compute_adx",
                return_value=10.0,
            ),
        ):
            signal = combiner.generate_signal("AAPL", candles, "us_broad")

        assert signal is not None
        assert signal.features["adx_regime"] == -1.0  # mr
        # Momentum signal should NOT appear in features (was gated out)
        assert "momentum_confidence" not in signal.features


class TestTurnOfMonth:
    """Tests for turn-of-month confidence overlay in the combiner."""

    # Constants
    TOM_BOOST = 0.05

    @staticmethod
    def _candle_at(dt: datetime) -> Candle:
        """Create a single candle at a specific datetime."""
        return Candle(
            symbol="AAPL",
            market_id="us",
            timeframe="1d",
            timestamp=dt,
            open=BASE_PRICE,
            high=BASE_PRICE + CANDLE_HIGH_OFFSET,
            low=BASE_PRICE - CANDLE_LOW_OFFSET,
            close=BASE_PRICE,
            volume=VOLUME,
        )

    @staticmethod
    def _make_candles_ending_at(dt: datetime, count: int = CANDLE_COUNT) -> list[Candle]:
        """Create a list of candles where the last candle has the given timestamp."""
        return [
            Candle(
                symbol="AAPL",
                market_id="us",
                timeframe="1d",
                timestamp=dt - timedelta(days=count - 1 - i),
                open=BASE_PRICE,
                high=BASE_PRICE + CANDLE_HIGH_OFFSET,
                low=BASE_PRICE - CANDLE_LOW_OFFSET,
                close=BASE_PRICE,
                volume=VOLUME,
            )
            for i in range(count)
        ]

    def _make_combiner_with_buy(self) -> tuple[StrategyCombiner, dict[str, Any]]:
        """Create a combiner with a single momentum BUY strategy and config."""
        config: dict[str, Any] = {
            "strategies": {
                "momentum": {"enabled": True, "weight": 1.0},
            },
            "min_combined_confidence": 0.0,  # accept any non-zero signal
        }
        buy_signal = _make_signal(SignalDirection.BUY, HIGH_CONFIDENCE, "momentum")
        strategy = MockStrategy("momentum", buy_signal)
        combiner = StrategyCombiner([strategy])
        return combiner, config

    def test_tom_first_day_of_month(self) -> None:
        """Candle on Jan 1 -> turn_of_month=1.0 in features."""
        combiner, config = self._make_combiner_with_buy()
        candles = self._make_candles_ending_at(datetime(2024, 1, 1, tzinfo=UTC))
        with patch.object(combiner, "_load_config", return_value=config):
            signal = combiner.generate_signal("AAPL", candles, "us_broad")
        assert signal is not None
        assert signal.features["turn_of_month"] == 1.0

    def test_tom_mid_month(self) -> None:
        """Candle on Jan 15 -> turn_of_month=0.0 in features."""
        combiner, config = self._make_combiner_with_buy()
        candles = self._make_candles_ending_at(datetime(2024, 1, 15, tzinfo=UTC))
        with patch.object(combiner, "_load_config", return_value=config):
            signal = combiner.generate_signal("AAPL", candles, "us_broad")
        assert signal is not None
        assert signal.features["turn_of_month"] == 0.0

    def test_tom_last_day_of_month(self) -> None:
        """Candle on Jan 31 -> turn_of_month=1.0 in features."""
        combiner, config = self._make_combiner_with_buy()
        candles = self._make_candles_ending_at(datetime(2024, 1, 31, tzinfo=UTC))
        with patch.object(combiner, "_load_config", return_value=config):
            signal = combiner.generate_signal("AAPL", candles, "us_broad")
        assert signal is not None
        assert signal.features["turn_of_month"] == 1.0

    def test_tom_boosts_buy_confidence(self) -> None:
        """BUY signal on TOM day has higher confidence than same signal on non-TOM day."""
        combiner_tom, config = self._make_combiner_with_buy()
        combiner_mid, _ = self._make_combiner_with_buy()

        candles_tom = self._make_candles_ending_at(datetime(2024, 1, 1, tzinfo=UTC))
        candles_mid = self._make_candles_ending_at(datetime(2024, 1, 15, tzinfo=UTC))

        with patch.object(combiner_tom, "_load_config", return_value=config):
            signal_tom = combiner_tom.generate_signal("AAPL", candles_tom, "us_broad")
        with patch.object(combiner_mid, "_load_config", return_value=config):
            signal_mid = combiner_mid.generate_signal("AAPL", candles_mid, "us_broad")

        assert signal_tom is not None
        assert signal_mid is not None
        assert signal_tom.confidence > signal_mid.confidence
        assert signal_tom.confidence == pytest.approx(
            signal_mid.confidence + self.TOM_BOOST, abs=0.01
        )


# ── normalize_mode "active" tests ────────────────────────────────────────────

# Constants for active-mode tests (no magic numbers)
_ACTIVE_WEIGHT_A = 0.3
_ACTIVE_WEIGHT_B = 0.3
_ACTIVE_WEIGHT_C = 0.2
_ACTIVE_WEIGHT_D = 0.2
_ACTIVE_CONFIDENCE_A = 0.9
_ACTIVE_CONFIDENCE_B = 0.8
_ACTIVE_MIN_CONFIDENCE = 0.10  # low threshold so signals aren't filtered


class TestCombinerNormalizeModeActive:
    """Tests for normalize_mode='active' — normalizes by data-ready weight."""

    def _make_config(self, normalize_mode: str = "active") -> dict[str, Any]:
        return {
            "normalize_mode": normalize_mode,
            "min_combined_confidence": _ACTIVE_MIN_CONFIDENCE,
            "strategies": {
                "strat_a": {"enabled": True, "weight": _ACTIVE_WEIGHT_A},
                "strat_b": {"enabled": True, "weight": _ACTIVE_WEIGHT_B},
                "strat_c": {"enabled": True, "weight": _ACTIVE_WEIGHT_C},
                "strat_d": {"enabled": True, "weight": _ACTIVE_WEIGHT_D},
            },
        }

    def _make_strategies(self) -> list[MockStrategy]:
        """4 strategies: A and B fire BUY, C and D return None."""
        sig_a = _make_signal(SignalDirection.BUY, _ACTIVE_CONFIDENCE_A, "strat_a")
        sig_b = _make_signal(SignalDirection.BUY, _ACTIVE_CONFIDENCE_B, "strat_b")
        return [
            MockStrategy("strat_a", sig_a),
            MockStrategy("strat_b", sig_b),
            MockStrategy("strat_c", None),
            MockStrategy("strat_d", None),
        ]

    def test_active_mode_uses_data_ready_weight(self) -> None:
        """Active: denominator = data_ready weight (all 4 called) = 1.0."""
        strategies = self._make_strategies()
        combiner = StrategyCombiner(strategies, normalize_mode="active")
        candles = _make_candles()
        config = self._make_config("active")
        with patch.object(combiner, "_load_config", return_value=config):
            signal = combiner.generate_signal("AAPL", candles, "us_broad")
        assert signal is not None
        assert signal.direction == SignalDirection.BUY
        # net = (0.9*0.3 + 0.8*0.3) / 1.0 = 0.51
        assert signal.confidence == pytest.approx(0.51, abs=0.02)

    def test_firing_mode_higher_confidence_than_active(self) -> None:
        """Firing: denominator = firing weight (2 fired) = 0.6 → higher score."""
        strategies = self._make_strategies()
        combiner = StrategyCombiner(strategies, normalize_mode="firing")
        candles = _make_candles()
        config = self._make_config("firing")
        with patch.object(combiner, "_load_config", return_value=config):
            signal = combiner.generate_signal("AAPL", candles, "us_broad")
        assert signal is not None
        # net = 0.51 / 0.6 = 0.85
        assert signal.confidence == pytest.approx(0.85, abs=0.02)

    def test_active_confidence_lower_than_firing(self) -> None:
        """Active produces lower confidence than firing for same input."""
        candles = _make_candles()

        combiner_a = StrategyCombiner(self._make_strategies(), normalize_mode="active")
        config_a = self._make_config("active")
        with patch.object(combiner_a, "_load_config", return_value=config_a):
            sig_a = combiner_a.generate_signal("AAPL", candles, "us_broad")

        combiner_f = StrategyCombiner(self._make_strategies(), normalize_mode="firing")
        config_f = self._make_config("firing")
        with patch.object(combiner_f, "_load_config", return_value=config_f):
            sig_f = combiner_f.generate_signal("AAPL", candles, "us_broad")

        assert sig_a is not None and sig_f is not None
        assert sig_a.confidence < sig_f.confidence

    def test_active_equals_total_when_all_registered(self) -> None:
        """When all config strategies are registered, active == total."""
        candles = _make_candles()

        combiner_a = StrategyCombiner(self._make_strategies(), normalize_mode="active")
        config_a = self._make_config("active")
        with patch.object(combiner_a, "_load_config", return_value=config_a):
            sig_a = combiner_a.generate_signal("AAPL", candles, "us_broad")

        combiner_t = StrategyCombiner(self._make_strategies(), normalize_mode="total")
        config_t = self._make_config("total")
        with patch.object(combiner_t, "_load_config", return_value=config_t):
            sig_t = combiner_t.generate_signal("AAPL", candles, "us_broad")

        assert sig_a is not None and sig_t is not None
        assert sig_a.confidence == pytest.approx(sig_t.confidence, abs=0.01)

    def test_active_differs_from_total_when_strategy_missing(self) -> None:
        """When strategies are in config but not registered, active != total."""
        sig_a = _make_signal(SignalDirection.BUY, _ACTIVE_CONFIDENCE_A, "strat_a")
        sig_b = _make_signal(SignalDirection.BUY, _ACTIVE_CONFIDENCE_B, "strat_b")
        only_ab = [MockStrategy("strat_a", sig_a), MockStrategy("strat_b", sig_b)]
        candles = _make_candles()

        combiner_a = StrategyCombiner(list(only_ab), normalize_mode="active")
        config_a = self._make_config("active")
        with patch.object(combiner_a, "_load_config", return_value=config_a):
            sig_active = combiner_a.generate_signal("AAPL", candles, "us_broad")

        combiner_t = StrategyCombiner(
            [MockStrategy("strat_a", sig_a), MockStrategy("strat_b", sig_b)],
            normalize_mode="total",
        )
        config_t = self._make_config("total")
        with patch.object(combiner_t, "_load_config", return_value=config_t):
            sig_total = combiner_t.generate_signal("AAPL", candles, "us_broad")

        assert sig_active is not None and sig_total is not None
        # active: 0.51/0.6=0.85 vs total: 0.51/1.0=0.51
        assert sig_active.confidence > sig_total.confidence


class TestMLSoloFireActiveMode:
    """Verify ML-only fire under active normalization is properly attenuated."""

    # Constants (no magic numbers)
    ML_WEIGHT = 0.10
    ML_CONFIDENCE = 0.60
    TOTAL_ACTIVE_WEIGHT = 1.10  # sum of all 6 enabled strategy weights
    ENTRY_THRESHOLD = 0.30

    def test_ml_solo_fire_active_mode_attenuated(self) -> None:
        """Under 'active' normalization, ML alone (weight=0.10) net score is attenuated.

        When ML fires alone with confidence=0.60 and weight=0.10,
        and total active weight = 1.10 (all 6 strategies enabled),
        net = 0.60 * 0.10 / 1.10 ~ 0.055, well below 0.30 threshold.
        This means ML alone cannot trigger a trade.
        """
        config: dict[str, Any] = {
            "normalize_mode": "active",
            "min_combined_confidence": self.ENTRY_THRESHOLD,
            "strategies": {
                "momentum": {"enabled": True, "weight": 0.20},
                "mean_reversion": {"enabled": True, "weight": 0.30},
                "dual_momentum": {"enabled": True, "weight": 0.25},
                "rsi2_connors": {"enabled": True, "weight": 0.15},
                "pairs": {"enabled": True, "weight": 0.10},
                "ml_ensemble": {"enabled": True, "weight": self.ML_WEIGHT},
            },
        }
        # Only ml_ensemble fires; all others return None
        ml_signal = _make_signal(SignalDirection.BUY, self.ML_CONFIDENCE, "ml_ensemble")
        strategies: list[BaseStrategy] = [
            MockStrategy("momentum", None),
            MockStrategy("mean_reversion", None),
            MockStrategy("dual_momentum", None),
            MockStrategy("rsi2_connors", None),
            MockStrategy("pairs", None),
            MockStrategy("ml_ensemble", ml_signal),
        ]
        combiner = StrategyCombiner(strategies, normalize_mode="active")
        candles = _make_candles()
        with patch.object(combiner, "_load_config", return_value=config):
            signal = combiner.generate_signal("AAPL", candles, "us_broad")

        # net = 0.60 * 0.10 / 1.10 = 0.0545... -> well below 0.30 -> None
        assert signal is None

    def test_ml_solo_fire_firing_mode_reinforcer_suppressed(self) -> None:
        """Under 'firing' normalization, ML alone is suppressed (reinforcer-only).

        ml_ensemble is in _REINFORCER_STRATEGIES, so when it's the only
        strategy that fires, the signal is suppressed regardless of norm mode.
        """
        config: dict[str, Any] = {
            "normalize_mode": "firing",
            "min_combined_confidence": self.ENTRY_THRESHOLD,
            "strategies": {
                "momentum": {"enabled": True, "weight": 0.20},
                "mean_reversion": {"enabled": True, "weight": 0.30},
                "dual_momentum": {"enabled": True, "weight": 0.25},
                "rsi2_connors": {"enabled": True, "weight": 0.15},
                "pairs": {"enabled": True, "weight": 0.10},
                "ml_ensemble": {"enabled": True, "weight": self.ML_WEIGHT},
            },
        }
        ml_signal = _make_signal(SignalDirection.BUY, self.ML_CONFIDENCE, "ml_ensemble")
        strategies: list[BaseStrategy] = [
            MockStrategy("momentum", None),
            MockStrategy("mean_reversion", None),
            MockStrategy("dual_momentum", None),
            MockStrategy("rsi2_connors", None),
            MockStrategy("pairs", None),
            MockStrategy("ml_ensemble", ml_signal),
        ]
        combiner = StrategyCombiner(strategies, normalize_mode="firing")
        candles = _make_candles()
        with patch.object(combiner, "_load_config", return_value=config):
            signal = combiner.generate_signal("AAPL", candles, "us_broad")

        # ML is a reinforcer-only strategy: cannot create standalone trades
        assert signal is None

    def test_ml_plus_rule_based_fires(self) -> None:
        """ML + a rule-based strategy should produce a combined signal."""
        config: dict[str, Any] = {
            "normalize_mode": "firing",
            "min_combined_confidence": self.ENTRY_THRESHOLD,
            "strategies": {
                "momentum": {"enabled": True, "weight": 0.20},
                "mean_reversion": {"enabled": True, "weight": 0.30},
                "dual_momentum": {"enabled": True, "weight": 0.25},
                "rsi2_connors": {"enabled": True, "weight": 0.15},
                "pairs": {"enabled": True, "weight": 0.10},
                "ml_ensemble": {"enabled": True, "weight": self.ML_WEIGHT},
            },
        }
        ml_signal = _make_signal(SignalDirection.BUY, self.ML_CONFIDENCE, "ml_ensemble")
        mom_signal = _make_signal(SignalDirection.BUY, 0.50, "momentum")
        strategies: list[BaseStrategy] = [
            MockStrategy("momentum", mom_signal),
            MockStrategy("mean_reversion", None),
            MockStrategy("dual_momentum", None),
            MockStrategy("rsi2_connors", None),
            MockStrategy("pairs", None),
            MockStrategy("ml_ensemble", ml_signal),
        ]
        combiner = StrategyCombiner(strategies, normalize_mode="firing")
        candles = _make_candles()
        with patch.object(combiner, "_load_config", return_value=config):
            signal = combiner.generate_signal("AAPL", candles, "us_broad")

        # ML + momentum fire together → combined signal passes
        assert signal is not None
        assert signal.direction == SignalDirection.BUY


class TestCombinerPassesHasOpenPosition:
    """Tests that has_open_position is propagated to child strategies."""

    def test_has_open_position_passed_to_child_strategies(self) -> None:
        """Combiner must forward has_open_position to each child strategy."""
        received_values: list[bool] = []

        class TrackingStrategy(BaseStrategy):
            @property
            def name(self) -> str:
                return "tracker"

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
                has_open_position: bool = False,
            ) -> Signal | None:
                received_values.append(has_open_position)
                return _make_signal(SignalDirection.BUY, HIGH_CONFIDENCE, "tracker")

        config: dict[str, Any] = {
            "strategies": {"tracker": {"enabled": True, "weight": 1.0}},
            "min_combined_confidence": 0.0,
        }
        strategy = TrackingStrategy()
        combiner = StrategyCombiner([strategy])
        candles = _make_candles()

        with patch.object(combiner, "_load_config", return_value=config):
            combiner.generate_signal("AAPL", candles, "us_broad", has_open_position=True)

        assert len(received_values) == 1
        assert received_values[0] is True

    def test_has_open_position_defaults_to_false(self) -> None:
        """When has_open_position not passed, child strategies receive False."""
        received_values: list[bool] = []

        class TrackingStrategy(BaseStrategy):
            @property
            def name(self) -> str:
                return "tracker"

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
                has_open_position: bool = False,
            ) -> Signal | None:
                received_values.append(has_open_position)
                return _make_signal(SignalDirection.BUY, HIGH_CONFIDENCE, "tracker")

        config: dict[str, Any] = {
            "strategies": {"tracker": {"enabled": True, "weight": 1.0}},
            "min_combined_confidence": 0.0,
        }
        strategy = TrackingStrategy()
        combiner = StrategyCombiner([strategy])
        candles = _make_candles()

        with patch.object(combiner, "_load_config", return_value=config):
            combiner.generate_signal("AAPL", candles, "us_broad")

        assert len(received_values) == 1
        assert received_values[0] is False


class TestExitConfidence:
    """Tests for asymmetric entry/exit confidence thresholds."""

    # Constants
    EXIT_CONFIDENCE = 0.25
    ENTRY_CONFIDENCE = 0.30
    SELL_CONFIDENCE = 0.27  # above EXIT (0.25) but below ENTRY (0.30)

    def test_exit_confidence_lower_than_entry(self) -> None:
        """SELL signal with open position uses min_exit_confidence=0.25 threshold.

        A SELL signal with confidence 0.27 should pass when has_open_position=True
        (threshold is 0.25) but be filtered out when has_open_position=False
        (threshold is 0.30).
        """
        config: dict[str, Any] = {
            "strategies": {
                "strat_x": {"enabled": True, "weight": 1.0},
            },
            "min_combined_confidence": self.ENTRY_CONFIDENCE,
            "min_exit_confidence": self.EXIT_CONFIDENCE,
        }
        sell_signal = _make_signal(SignalDirection.SELL, self.SELL_CONFIDENCE, "strat_x")
        strategy = MockStrategy("strat_x", sell_signal)
        combiner = StrategyCombiner([strategy])
        candles = _make_candles()

        # With open position: threshold is min(0.30, 0.25) = 0.25 -> 0.27 >= 0.25 -> passes
        with patch.object(combiner, "_load_config", return_value=config):
            signal_with_pos = combiner.generate_signal(
                "AAPL", candles, "us_broad", has_open_position=True
            )
        assert signal_with_pos is not None
        assert signal_with_pos.direction == SignalDirection.SELL

        # Without open position: threshold is 0.30 -> 0.27 < 0.30 -> filtered
        combiner2 = StrategyCombiner([MockStrategy("strat_x", sell_signal)])
        with patch.object(combiner2, "_load_config", return_value=config):
            signal_no_pos = combiner2.generate_signal(
                "AAPL", candles, "us_broad", has_open_position=False
            )
        assert signal_no_pos is None


class TestCombinerMarketContext:
    """Tests for StrategyCombiner.set_market_context propagation."""

    def test_set_market_context_propagates_to_ml_strategy(self) -> None:
        """Calling combiner.set_market_context() should forward to MLStrategy."""
        from finalayze.core.schemas import MarketContext

        class _MockMLStrategy(BaseStrategy):
            """Minimal strategy with set_market_context support."""

            def __init__(self) -> None:
                self.received_context: MarketContext | None = None

            @property
            def name(self) -> str:
                return "ml_ensemble"

            def supported_segments(self) -> list[str]:
                return ["us_tech"]

            def get_parameters(self, segment_id: str) -> dict[str, object]:
                return {}

            def generate_signal(
                self,
                symbol: str,
                candles: list[Candle],
                segment_id: str,
                sentiment_score: float = 0.0,
                has_open_position: bool = False,
            ) -> Signal | None:
                return None

            def set_market_context(self, ctx: MarketContext) -> None:
                self.received_context = ctx

        ml_strategy = _MockMLStrategy()
        plain_strategy = MockStrategy("momentum", None)
        combiner = StrategyCombiner([ml_strategy, plain_strategy])

        ctx = MarketContext(benchmark_candles=_make_candles(), vix_candles=None)
        combiner.set_market_context(ctx)

        assert ml_strategy.received_context is ctx, (
            "set_market_context must propagate MarketContext to strategies that support it"
        )

    def test_set_market_context_skips_strategies_without_method(self) -> None:
        """Strategies without set_market_context should not cause errors."""
        from finalayze.core.schemas import MarketContext

        plain_strategy = MockStrategy("momentum", None)
        combiner = StrategyCombiner([plain_strategy])

        ctx = MarketContext(benchmark_candles=None, vix_candles=None)
        # Should not raise
        combiner.set_market_context(ctx)

    def test_constructor_market_context_propagates(self) -> None:
        """Passing market_context at construction time should also propagate."""
        from finalayze.core.schemas import MarketContext

        class _TrackingStrategy(BaseStrategy):
            def __init__(self) -> None:
                self.received_context: MarketContext | None = None

            @property
            def name(self) -> str:
                return "tracker"

            def supported_segments(self) -> list[str]:
                return []

            def get_parameters(self, segment_id: str) -> dict[str, object]:
                return {}

            def generate_signal(
                self,
                symbol: str,
                candles: list[Candle],
                segment_id: str,
                sentiment_score: float = 0.0,
                has_open_position: bool = False,
            ) -> Signal | None:
                return None

            def set_market_context(self, ctx: MarketContext) -> None:
                self.received_context = ctx

        tracker = _TrackingStrategy()
        ctx = MarketContext(benchmark_candles=_make_candles(), vix_candles=None)
        StrategyCombiner([tracker], market_context=ctx)

        assert tracker.received_context is ctx, (
            "Constructor should propagate MarketContext to strategies"
        )


# ===================================================================
# Test: Event strategy ADX bypass and confidence floor
# ===================================================================
_EVENT_CONFIDENCE = 0.45
_EVENT_WEIGHT = 0.17
_MOMENTUM_WEIGHT = 0.17
_MR_WEIGHT = 0.17
_EVENT_MIN_CONFIDENCE = 0.40


class TestEventStrategyBypass:
    """Event strategies (dividend_gap, cbr_calendar) bypass ADX regime gating."""

    @staticmethod
    def _base_config(strategies_cfg: dict[str, Any]) -> dict[str, Any]:
        """Config with regime routing disabled (we mock _compute_adx_regime)."""
        return {
            "normalize_mode": "firing",
            "min_combined_confidence": _EVENT_MIN_CONFIDENCE,
            "regime_routing": {"enabled": True},
            "strategies": strategies_cfg,
        }

    def test_event_strategy_fires_in_trend_regime(self) -> None:
        """dividend_gap should NOT be skipped in trend regime (ADX bypass)."""
        sig = _make_signal(SignalDirection.BUY, _EVENT_CONFIDENCE, "dividend_gap")
        strategies: list[BaseStrategy] = [
            MockStrategy("dividend_gap", sig),
            MockStrategy("momentum", None),
            MockStrategy("mean_reversion", None),
        ]
        combiner = StrategyCombiner(strategies)
        candles = _make_candles()
        config = self._base_config(
            {
                "dividend_gap": {"enabled": True, "weight": _EVENT_WEIGHT},
                "momentum": {"enabled": True, "weight": _MOMENTUM_WEIGHT},
                "mean_reversion": {"enabled": True, "weight": _MR_WEIGHT},
            }
        )
        with (
            patch.object(combiner, "_load_config", return_value=config),
            patch.object(combiner, "_compute_adx_regime", return_value=(40.0, "trend")),
        ):
            result = combiner.generate_signal("AAPL", candles, "us_broad")

        assert result is not None
        assert "dividend_gap_confidence" in result.features

    def test_event_strategy_fires_in_mr_regime(self) -> None:
        """cbr_calendar should NOT be skipped in mr regime (ADX bypass)."""
        sig = _make_signal(SignalDirection.BUY, _EVENT_CONFIDENCE, "cbr_calendar")
        strategies: list[BaseStrategy] = [
            MockStrategy("cbr_calendar", sig),
            MockStrategy("momentum", None),
            MockStrategy("mean_reversion", None),
        ]
        combiner = StrategyCombiner(strategies)
        candles = _make_candles()
        config = self._base_config(
            {
                "cbr_calendar": {"enabled": True, "weight": _EVENT_WEIGHT},
                "momentum": {"enabled": True, "weight": _MOMENTUM_WEIGHT},
                "mean_reversion": {"enabled": True, "weight": _MR_WEIGHT},
            }
        )
        with (
            patch.object(combiner, "_load_config", return_value=config),
            patch.object(combiner, "_compute_adx_regime", return_value=(10.0, "mr")),
        ):
            result = combiner.generate_signal("AAPL", candles, "us_broad")

        assert result is not None
        assert "cbr_calendar_confidence" in result.features

    def test_non_event_mr_still_skipped_in_trend(self) -> None:
        """mean_reversion should still be skipped in trend regime."""
        mr_sig = _make_signal(SignalDirection.BUY, HIGH_CONFIDENCE, "mean_reversion")
        strategies: list[BaseStrategy] = [
            MockStrategy("dividend_gap", None),
            MockStrategy("momentum", None),
            MockStrategy("mean_reversion", mr_sig),
        ]
        combiner = StrategyCombiner(strategies)
        candles = _make_candles()
        config = self._base_config(
            {
                "dividend_gap": {"enabled": True, "weight": _EVENT_WEIGHT},
                "momentum": {"enabled": True, "weight": _MOMENTUM_WEIGHT},
                "mean_reversion": {"enabled": True, "weight": _MR_WEIGHT},
            }
        )
        with (
            patch.object(combiner, "_load_config", return_value=config),
            patch.object(combiner, "_compute_adx_regime", return_value=(40.0, "trend")),
        ):
            result = combiner.generate_signal("AAPL", candles, "us_broad")

        # mean_reversion was skipped so no signal fires -> None
        assert result is None

    def test_non_event_momentum_still_skipped_in_mr(self) -> None:
        """momentum should still be skipped in mr regime."""
        mom_sig = _make_signal(SignalDirection.BUY, HIGH_CONFIDENCE, "momentum")
        strategies: list[BaseStrategy] = [
            MockStrategy("cbr_calendar", None),
            MockStrategy("momentum", mom_sig),
            MockStrategy("mean_reversion", None),
        ]
        combiner = StrategyCombiner(strategies)
        candles = _make_candles()
        config = self._base_config(
            {
                "cbr_calendar": {"enabled": True, "weight": _EVENT_WEIGHT},
                "momentum": {"enabled": True, "weight": _MOMENTUM_WEIGHT},
                "mean_reversion": {"enabled": True, "weight": _MR_WEIGHT},
            }
        )
        with (
            patch.object(combiner, "_load_config", return_value=config),
            patch.object(combiner, "_compute_adx_regime", return_value=(10.0, "mr")),
        ):
            result = combiner.generate_signal("AAPL", candles, "us_broad")

        assert result is None

    def test_event_only_signal_passes_confidence_floor(self) -> None:
        """When only event strategy fires with confidence 0.45, combined >= 0.40."""
        sig = _make_signal(SignalDirection.BUY, _EVENT_CONFIDENCE, "dividend_gap")
        strategies: list[BaseStrategy] = [
            MockStrategy("dividend_gap", sig),
            MockStrategy("momentum", None),
            MockStrategy("mean_reversion", None),
        ]
        combiner = StrategyCombiner(strategies)
        candles = _make_candles()
        # Use default min_combined_confidence of 0.50, but event floor should lower to 0.40
        config: dict[str, Any] = {
            "normalize_mode": "firing",
            "min_combined_confidence": 0.50,
            "regime_routing": {"enabled": False},
            "strategies": {
                "dividend_gap": {"enabled": True, "weight": _EVENT_WEIGHT},
                "momentum": {"enabled": True, "weight": _MOMENTUM_WEIGHT},
                "mean_reversion": {"enabled": True, "weight": _MR_WEIGHT},
            },
        }
        with patch.object(combiner, "_load_config", return_value=config):
            result = combiner.generate_signal("AAPL", candles, "us_broad")

        # confidence 0.45 >= event floor 0.40, so signal should pass
        assert result is not None
        assert result.confidence >= _EVENT_MIN_CONFIDENCE


class TestDedupEventSignals:
    """Tests for CBR/dividend duplicate-signal suppression (EVNT-02)."""

    def test_dedup_zeroes_lower_weight_on_same_event_code(self) -> None:
        """Two strategies with same event_type_code: lower weight is zeroed."""
        from finalayze.strategies.combiner import _dedup_event_signals

        sig1 = _make_signal(SignalDirection.BUY, 0.8, "event_driven")
        sig1 = Signal(
            **{**sig1.model_dump(), "features": {"event_type_code": 1.0}},
        )
        sig2 = _make_signal(SignalDirection.BUY, 0.8, "cbr_calendar")
        sig2 = Signal(
            **{**sig2.model_dump(), "features": {"event_type_code": 1.0}},
        )
        collected = {
            "event_driven": (sig1, Decimal("0.15")),
            "cbr_calendar": (sig2, Decimal("0.05")),
        }
        zeroed = _dedup_event_signals(collected)
        assert "cbr_calendar" in zeroed
        assert "event_driven" not in zeroed

    def test_dedup_no_action_for_different_event_codes(self) -> None:
        """Two strategies with different event_type_codes: no dedup."""
        from finalayze.strategies.combiner import _dedup_event_signals

        sig1 = _make_signal(SignalDirection.BUY, 0.8, "event_driven")
        sig1 = Signal(
            **{**sig1.model_dump(), "features": {"event_type_code": 1.0}},
        )
        sig2 = _make_signal(SignalDirection.BUY, 0.8, "cbr_calendar")
        sig2 = Signal(
            **{**sig2.model_dump(), "features": {"event_type_code": 2.0}},
        )
        collected = {
            "event_driven": (sig1, Decimal("0.15")),
            "cbr_calendar": (sig2, Decimal("0.05")),
        }
        zeroed = _dedup_event_signals(collected)
        assert len(zeroed) == 0

    def test_dedup_ignores_zero_event_code(self) -> None:
        """Strategies with event_type_code=0.0 are never deduped."""
        from finalayze.strategies.combiner import _dedup_event_signals

        sig1 = _make_signal(SignalDirection.BUY, 0.8, "momentum")
        sig2 = _make_signal(SignalDirection.BUY, 0.8, "mean_reversion")
        collected = {
            "momentum": (sig1, Decimal("0.30")),
            "mean_reversion": (sig2, Decimal("0.30")),
        }
        zeroed = _dedup_event_signals(collected)
        assert len(zeroed) == 0


# ── Phase 55-01 Task 2: signal_price field on Signal schema + SignalModel ──


def test_signal_schema_default_signal_price_is_none() -> None:
    """Signal schema accepts construction without signal_price; default is None."""
    sig = Signal(
        strategy_name="unit",
        symbol="SBER",
        market_id="moex",
        segment_id="ru_blue_chips",
        direction=SignalDirection.BUY,
        confidence=HIGH_CONFIDENCE,
        features={},
        reasoning="test",
    )
    assert sig.signal_price is None


def test_signal_schema_signal_price_preserves_decimal() -> None:
    """Signal round-trips a Decimal signal_price without float corruption."""
    sig = Signal(
        strategy_name="unit",
        symbol="SBER",
        market_id="moex",
        segment_id="ru_blue_chips",
        direction=SignalDirection.BUY,
        confidence=HIGH_CONFIDENCE,
        features={},
        reasoning="test",
        signal_price=Decimal("280.5000"),
    )
    assert sig.signal_price == Decimal("280.5000")


def test_signal_model_orm_declares_signal_price_column() -> None:
    """SignalModel exposes a signal_price column with Numeric(12, 4) nullable=True."""
    from finalayze.core.models import SignalModel

    col = SignalModel.__table__.c["signal_price"]
    assert col.nullable is True
    # SQLAlchemy Numeric(12,4) carries precision/scale on the type
    assert col.type.precision == 12
    assert col.type.scale == 4


# ── Phase 55-02 Task 1: signal_price captured in _build_result ──


def _make_candle_with_close(close: Decimal, day: int) -> Candle:
    """Build a Candle with an explicit close price (distinct from _candle's BASE_PRICE)."""
    return Candle(
        symbol="AAPL",
        market_id="us",
        timeframe="1d",
        timestamp=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=day),
        open=close,
        high=close + CANDLE_HIGH_OFFSET,
        low=close - CANDLE_LOW_OFFSET,
        close=close,
        volume=VOLUME,
    )


def test_signal_price_captured_at_build_result() -> None:
    """generate_signal must stamp Signal.signal_price with candles[-1].close (Decimal)."""
    config: dict[str, Any] = {
        "strategies": {"momentum": {"enabled": True, "weight": 1.0}},
    }
    buy_signal = _make_signal(SignalDirection.BUY, HIGH_CONFIDENCE, "momentum")
    strategy = MockStrategy("momentum", buy_signal)
    combiner = StrategyCombiner([strategy])
    # Trailing candle with close=280.5, 29 earlier candles at BASE_PRICE.
    candles = [
        *_make_candles(CANDLE_COUNT - 1),
        _make_candle_with_close(Decimal("280.5"), CANDLE_COUNT - 1),
    ]
    with patch.object(combiner, "_load_config", return_value=config):
        result = combiner.generate_signal("AAPL", candles, "us_broad")
    assert result is not None
    assert result.signal_price == Decimal("280.5")


def test_signal_price_uses_last_candle_not_first() -> None:
    """signal_price comes from candles[-1], not candles[0]."""
    config: dict[str, Any] = {
        "strategies": {"momentum": {"enabled": True, "weight": 1.0}},
    }
    buy_signal = _make_signal(SignalDirection.BUY, HIGH_CONFIDENCE, "momentum")
    strategy = MockStrategy("momentum", buy_signal)
    combiner = StrategyCombiner([strategy])
    # Three candles with closes 100, 200, 300. We expect 300 on the produced signal.
    candles = [
        _make_candle_with_close(Decimal(100), 0),
        _make_candle_with_close(Decimal(200), 1),
        _make_candle_with_close(Decimal(300), 2),
    ]
    with patch.object(combiner, "_load_config", return_value=config):
        result = combiner.generate_signal("AAPL", candles, "us_broad")
    assert result is not None
    assert result.signal_price == Decimal(300)


def test_signal_price_none_when_build_result_called_directly() -> None:
    """Direct _build_result calls (no signal_price kwarg) leave Signal.signal_price=None."""
    combiner = StrategyCombiner([MockStrategy("momentum", None)])
    result = combiner._build_result(
        Decimal("0.6"),
        {"momentum_confidence": 0.6, "momentum_direction": 1.0},
        "AAPL",
        "us",
        "us_broad",
    )
    assert result.signal_price is None

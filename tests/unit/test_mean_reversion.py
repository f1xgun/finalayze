"""Unit tests for MeanReversionStrategy (Bollinger Bands)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING
from unittest.mock import mock_open, patch

import yaml

if TYPE_CHECKING:
    from pathlib import Path

from finalayze.core.schemas import Candle, SignalDirection
from finalayze.strategies.mean_reversion import MeanReversionStrategy

# Constants (no magic numbers)
BASE_PRICE = Decimal(100)
BAND_PERIOD = 20
VOLUME = 1_000_000
STABLE_HIGH_PRICE = Decimal(200)
CRASH_PRICE = Decimal(50)
SPIKE_PRICE = Decimal(300)
CANDLE_HIGH_OFFSET = Decimal(1)
CANDLE_LOW_OFFSET = Decimal(1)
STABLE_COUNT = 25
EXTRA_CANDLE_COUNT = 5
MIN_CANDLES_INSUFFICIENT = 5


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


def _make_stable_candles(price: Decimal, count: int, start_day: int = 0) -> list[Candle]:
    return [_candle(price, start_day + i) for i in range(count)]


class TestMeanReversionStrategy:
    def test_name_returns_string(self) -> None:
        strategy = MeanReversionStrategy()
        assert strategy.name == "mean_reversion"
        assert isinstance(strategy.name, str)

    def test_supported_segments_includes_us_tech(self) -> None:
        strategy = MeanReversionStrategy()
        segments = strategy.supported_segments()
        assert "us_tech" in segments
        assert isinstance(segments, list)

    def test_get_parameters_returns_dict(self) -> None:
        strategy = MeanReversionStrategy()
        params = strategy.get_parameters("us_tech")
        assert isinstance(params, dict)
        assert "bb_period" in params
        assert "bb_std_dev" in params
        assert "min_confidence" in params

    def test_get_parameters_unknown_segment_returns_defaults(self) -> None:
        strategy = MeanReversionStrategy()
        params = strategy.get_parameters("nonexistent_segment_xyz")
        # Unknown segment returns empty dict (no params)
        assert isinstance(params, dict)
        assert len(params) == 0

    def test_no_signal_with_insufficient_candles(self) -> None:
        strategy = MeanReversionStrategy()
        # Less than bb_period + 1 candles -> return None
        candles = [_candle(BASE_PRICE, i) for i in range(MIN_CANDLES_INSUFFICIENT)]
        signal = strategy.generate_signal("AAPL", candles, "us_tech")
        assert signal is None

    def test_buy_signal_when_price_below_lower_band(self) -> None:
        strategy = MeanReversionStrategy()
        # Stable price at high level to establish BB, then sudden crash below lower band
        candles = _make_stable_candles(STABLE_HIGH_PRICE, STABLE_COUNT)
        # Add a final candle that crashes far below the lower band
        crash_candle = _candle(CRASH_PRICE, STABLE_COUNT)
        candles.append(crash_candle)
        signal = strategy.generate_signal("AAPL", candles, "us_tech")
        assert signal is not None, "Expected BUY signal: price crashed below lower BB"
        assert signal.direction == SignalDirection.BUY

    def test_no_signal_when_price_near_midline(self) -> None:
        strategy = MeanReversionStrategy()
        # Flat price exactly at middle band -> no signal
        candles = _make_stable_candles(BASE_PRICE, STABLE_COUNT + EXTRA_CANDLE_COUNT)
        signal = strategy.generate_signal("AAPL", candles, "us_tech")
        assert signal is None

    def test_sell_signal_when_price_above_upper_band(self) -> None:
        strategy = MeanReversionStrategy()
        # Stable price at low level to establish BB, then sudden spike above upper band
        candles = _make_stable_candles(BASE_PRICE, STABLE_COUNT)
        # Add a final candle that spikes far above the upper band
        spike_candle = _candle(SPIKE_PRICE, STABLE_COUNT)
        candles.append(spike_candle)
        signal = strategy.generate_signal("AAPL", candles, "us_tech")
        assert signal is not None, "Expected SELL signal: price spiked above upper BB"
        assert signal.direction == SignalDirection.SELL

    def test_generate_signal_returns_correct_metadata(self) -> None:
        strategy = MeanReversionStrategy()
        # Stable price then crash to trigger BUY
        candles = _make_stable_candles(STABLE_HIGH_PRICE, STABLE_COUNT)
        candles.append(_candle(CRASH_PRICE, STABLE_COUNT))
        signal = strategy.generate_signal("AAPL", candles, "us_tech")
        assert signal is not None
        assert signal.symbol == "AAPL"
        assert signal.market_id == "us"
        assert signal.segment_id == "us_tech"
        assert signal.strategy_name == "mean_reversion"
        assert 0.0 <= signal.confidence <= 1.0


class TestMeanReversionYAMLErrorHandling:
    """Tests that malformed or missing YAML never crashes the strategy."""

    def test_get_parameters_malformed_yaml_returns_empty_dict(self, tmp_path: Path) -> None:
        """A YAML parse error must not propagate; empty dict is returned."""
        strategy = MeanReversionStrategy()
        bad_yaml_content = ": this is: not: valid: yaml: ]["
        preset = tmp_path / "bad_segment.yaml"
        preset.write_text(bad_yaml_content)
        with patch("finalayze.strategies.mean_reversion._PRESETS_DIR", tmp_path):
            result = strategy.get_parameters("bad_segment")
        assert result == {}

    def test_get_parameters_empty_yaml_returns_empty_dict(self, tmp_path: Path) -> None:
        """An empty YAML file (safe_load returns None) must return empty dict."""
        strategy = MeanReversionStrategy()
        preset = tmp_path / "empty_segment.yaml"
        preset.write_text("")
        with patch("finalayze.strategies.mean_reversion._PRESETS_DIR", tmp_path):
            result = strategy.get_parameters("empty_segment")
        assert result == {}

    def test_get_parameters_yaml_error_via_mock(self) -> None:
        """yaml.YAMLError raised during open must be caught and return empty dict."""
        strategy = MeanReversionStrategy()
        with (
            patch("builtins.open", mock_open(read_data=b"")),
            patch("yaml.safe_load", side_effect=yaml.YAMLError("bad yaml")),
        ):
            result = strategy.get_parameters("us_tech")
        assert result == {}

    def test_supported_segments_malformed_yaml_skips_file(self, tmp_path: Path) -> None:
        """A YAML parse error in one preset must skip that file, not crash."""
        strategy = MeanReversionStrategy()
        # Create one valid preset and one malformed one
        valid_preset = tmp_path / "seg_a.yaml"
        valid_preset.write_text(
            "segment_id: seg_a\nstrategies:\n  mean_reversion:\n    enabled: true\n"
        )
        bad_preset = tmp_path / "seg_b.yaml"
        bad_preset.write_text(": bad: yaml: ][")
        with patch("finalayze.strategies.mean_reversion._PRESETS_DIR", tmp_path):
            segments = strategy.supported_segments()
        assert segments == ["seg_a"]

    def test_supported_segments_missing_segment_id_skips_file(self, tmp_path: Path) -> None:
        """A preset without a segment_id key must be silently skipped."""
        strategy = MeanReversionStrategy()
        preset = tmp_path / "no_id.yaml"
        preset.write_text("strategies:\n  mean_reversion:\n    enabled: true\n")
        with patch("finalayze.strategies.mean_reversion._PRESETS_DIR", tmp_path):
            segments = strategy.supported_segments()
        assert segments == []

    def test_supported_segments_empty_yaml_skips_file(self, tmp_path: Path) -> None:
        """An empty YAML file must be silently skipped."""
        strategy = MeanReversionStrategy()
        preset = tmp_path / "empty.yaml"
        preset.write_text("")
        with patch("finalayze.strategies.mean_reversion._PRESETS_DIR", tmp_path):
            segments = strategy.supported_segments()
        assert segments == []

    def test_get_parameters_non_dict_strategies_returns_empty(self, tmp_path: Path) -> None:
        """If data['strategies'] is not a dict (e.g. a list), return empty dict (issue #62)."""
        strategy = MeanReversionStrategy()
        # YAML where 'strategies' is a list, not a dict
        preset = tmp_path / "bad_strategies.yaml"
        preset.write_text("segment_id: bad_strategies\nstrategies:\n  - item_one\n  - item_two\n")
        with patch("finalayze.strategies.mean_reversion._PRESETS_DIR", tmp_path):
            result = strategy.get_parameters("bad_strategies")
        assert result == {}

    def test_get_parameters_non_dict_mr_cfg_returns_empty(self, tmp_path: Path) -> None:
        """If data['strategies']['mean_reversion'] is not a dict, return empty dict (issue #62)."""
        strategy = MeanReversionStrategy()
        # YAML where 'mean_reversion' is a scalar string
        preset = tmp_path / "bad_mr.yaml"
        preset.write_text("segment_id: bad_mr\nstrategies:\n  mean_reversion: disabled\n")
        with patch("finalayze.strategies.mean_reversion._PRESETS_DIR", tmp_path):
            result = strategy.get_parameters("bad_mr")
        assert result == {}

    def test_get_parameters_non_dict_params_returns_empty(self, tmp_path: Path) -> None:
        """If data['strategies']['mean_reversion']['params'] is not a dict, return empty dict."""
        strategy = MeanReversionStrategy()
        # YAML where 'params' is a list instead of a mapping
        preset = tmp_path / "bad_params.yaml"
        yaml_content = (
            "segment_id: bad_params\n"
            "strategies:\n"
            "  mean_reversion:\n"
            "    enabled: true\n"
            "    params:\n"
            "      - bb_period\n"
            "      - 20\n"
        )
        preset.write_text(yaml_content)
        with patch("finalayze.strategies.mean_reversion._PRESETS_DIR", tmp_path):
            result = strategy.get_parameters("bad_params")
        assert result == {}


class TestMeanReversionFilters:
    """Tests for squeeze filter, RSI confirmation, and minimum band distance."""

    def test_squeeze_filter_blocks_narrow_bands(self, tmp_path: Path) -> None:
        """Very narrow bands (squeeze) should suppress signals."""
        strategy = MeanReversionStrategy()
        # Use params with high squeeze threshold to force block
        preset = tmp_path / "squeeze_test.yaml"
        yaml_content = (
            "segment_id: squeeze_test\n"
            "strategies:\n"
            "  mean_reversion:\n"
            "    enabled: true\n"
            "    params:\n"
            "      bb_period: 20\n"
            "      bb_std_dev: 2.0\n"
            "      min_confidence: 0.1\n"
            "      squeeze_threshold: 0.99\n"  # Unreasonably high -> always blocks
        )
        preset.write_text(yaml_content)
        with patch("finalayze.strategies.mean_reversion._PRESETS_DIR", tmp_path):
            # Add crash to try to trigger signal
            candles_with_crash = _make_stable_candles(STABLE_HIGH_PRICE, STABLE_COUNT)
            candles_with_crash.append(_candle(CRASH_PRICE, STABLE_COUNT))
            signal = strategy.generate_signal("AAPL", candles_with_crash, "squeeze_test")
        assert signal is None, "Expected squeeze filter to block signal"

    def test_squeeze_filter_allows_wide_bands(self) -> None:
        """Wide bands (high volatility) should not block signals."""
        strategy = MeanReversionStrategy()
        # Stable at high price then crash -> wide bands relative to mean
        candles = _make_stable_candles(STABLE_HIGH_PRICE, STABLE_COUNT)
        candles.append(_candle(CRASH_PRICE, STABLE_COUNT))
        # Default squeeze_threshold is 0.02, RSI confirmation default 40
        # Crash gives RSI ~0 which is < 40 so RSI filter passes too
        result = strategy.generate_signal("AAPL", candles, "us_tech")
        assert result is not None, "Expected signal to pass squeeze filter with wide bands"

    def test_rsi_confirmation_blocks_buy_with_high_rsi(self) -> None:
        """BUY signal blocked when RSI > rsi_oversold_mr threshold."""
        strategy = MeanReversionStrategy()
        candles = _make_stable_candles(STABLE_HIGH_PRICE, STABLE_COUNT)
        candles.append(_candle(CRASH_PRICE, STABLE_COUNT))

        # Mock _compute_rsi to return 45 (above default threshold of 40) → should block
        with patch("finalayze.strategies.mean_reversion._compute_rsi", return_value=45.0):
            signal = strategy.generate_signal("AAPL", candles, "us_tech")
        assert signal is None, "Expected RSI confirmation to block BUY (RSI=45 > 40)"

    def test_rsi_confirmation_allows_buy_with_low_rsi(self) -> None:
        """BUY signal allowed when RSI < 40 (oversold enough)."""
        strategy = MeanReversionStrategy()
        # Stable high price, then extreme crash -> RSI deeply oversold
        candles = _make_stable_candles(STABLE_HIGH_PRICE, STABLE_COUNT)
        candles.append(_candle(CRASH_PRICE, STABLE_COUNT))
        signal = strategy.generate_signal("AAPL", candles, "us_tech")
        assert signal is not None, "Expected BUY: extreme crash should have RSI < 40"
        assert signal.direction == SignalDirection.BUY

    def test_min_band_distance_blocks_marginal_breach(self, tmp_path: Path) -> None:
        """Price just barely touching the band should be filtered out."""
        strategy = MeanReversionStrategy()
        preset = tmp_path / "dist_test.yaml"
        yaml_content = (
            "segment_id: dist_test\n"
            "strategies:\n"
            "  mean_reversion:\n"
            "    enabled: true\n"
            "    params:\n"
            "      bb_period: 20\n"
            "      bb_std_dev: 2.0\n"
            "      min_confidence: 0.1\n"
            "      squeeze_threshold: 0.0\n"
            "      min_band_distance_pct: 0.99\n"  # 99% distance required -> blocks everything
            "      rsi_oversold_mr: 100\n"  # Disable RSI filter
            "      rsi_overbought_mr: 0\n"
        )
        preset.write_text(yaml_content)
        with patch("finalayze.strategies.mean_reversion._PRESETS_DIR", tmp_path):
            candles = _make_stable_candles(STABLE_HIGH_PRICE, STABLE_COUNT)
            candles.append(_candle(CRASH_PRICE, STABLE_COUNT))
            signal = strategy.generate_signal("AAPL", candles, "dist_test")
        assert signal is None, "Expected min_band_distance to block marginal breach"

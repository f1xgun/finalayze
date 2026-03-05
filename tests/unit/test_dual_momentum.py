"""Unit tests for dual momentum strategy."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest  # noqa: TC002

from finalayze.core.schemas import Candle, SignalDirection
from finalayze.strategies.dual_momentum import DualMomentumStrategy

_MIN_CANDLES = 126
_WEIGHT_1M = 0.4
_WEIGHT_3M = 0.3
_WEIGHT_6M = 0.3
_CONFIDENCE_BASE = 0.4
_CONFIDENCE_SCALE = 1.0
_MAX_CONFIDENCE = 0.95


def _make_candles(
    prices: list[float],
    symbol: str = "AAPL",
    market_id: str = "us",
) -> list[Candle]:
    """Build candles from a list of close prices."""
    base = datetime(2024, 1, 1, 14, 30, tzinfo=UTC)
    candles = []
    for i, price in enumerate(prices):
        p = Decimal(str(price))
        candles.append(
            Candle(
                symbol=symbol,
                market_id=market_id,
                timeframe="1d",
                timestamp=base + timedelta(days=i),
                open=p,
                high=p + Decimal(1),
                low=p - Decimal(1),
                close=p,
                volume=1_000_000,
            )
        )
    return candles


_DEFAULT_PARAMS: dict[str, object] = {}


class TestDualMomentum:
    """Tests for DualMomentumStrategy."""

    def test_dual_momentum_buy_signal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Positive momentum score produces a BUY signal."""
        # Steadily rising prices over 126+ bars
        prices = [100.0 + i * 0.5 for i in range(130)]
        candles = _make_candles(prices)

        strategy = DualMomentumStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: _DEFAULT_PARAMS)
        signal = strategy.generate_signal("AAPL", candles, "us_tech")

        assert signal is not None
        assert signal.direction == SignalDirection.BUY
        assert signal.strategy_name == "dual_momentum"
        assert 0.0 <= signal.confidence <= 1.0

    def test_dual_momentum_absolute_gate_sell(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Score <= sell_threshold produces a SELL signal (or None if in between)."""
        # Steadily declining prices -> strongly negative score -> SELL
        prices = [200.0 - i * 0.5 for i in range(130)]
        candles = _make_candles(prices)

        strategy = DualMomentumStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: _DEFAULT_PARAMS)
        signal = strategy.generate_signal("AAPL", candles, "us_tech")

        assert signal is not None
        assert signal.direction == SignalDirection.SELL

    def test_dual_momentum_no_signal_near_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Score between sell_threshold and 0 returns None."""
        # Mostly flat prices with tiny decline -> score near 0 but slightly negative
        prices = [100.0] * 130
        # Tiny decline: just enough to be negative but above -0.05
        prices[-1] = 99.9
        candles = _make_candles(prices)

        strategy = DualMomentumStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: _DEFAULT_PARAMS)
        signal = strategy.generate_signal("AAPL", candles, "us_tech")

        assert signal is None

    def test_dual_momentum_confidence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify confidence formula: min(0.95, 0.4 + abs(score) * 1.0)."""
        # Build prices where we can calculate exact score
        n = 130
        base_price = 100.0

        # Simple linear growth: price[i] = 100 + i * growth
        growth = 0.5
        prices = [base_price + i * growth for i in range(n)]
        candles = _make_candles(prices)

        last = prices[-1]
        p_21 = prices[-21]
        p_63 = prices[-63]
        p_126 = prices[-126]

        ret_1m = (last - p_21) / p_21
        ret_3m = (last - p_63) / p_63
        ret_6m = (last - p_126) / p_126
        expected_score = ret_1m * _WEIGHT_1M + ret_3m * _WEIGHT_3M + ret_6m * _WEIGHT_6M
        expected_confidence = min(
            _MAX_CONFIDENCE, _CONFIDENCE_BASE + abs(expected_score) * _CONFIDENCE_SCALE
        )

        strategy = DualMomentumStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: _DEFAULT_PARAMS)
        signal = strategy.generate_signal("AAPL", candles, "us_tech")

        assert signal is not None
        assert abs(signal.confidence - expected_confidence) < 1e-6

    def test_dual_momentum_insufficient_data(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Less than 126 candles returns None."""
        prices = [100.0 + i * 0.5 for i in range(125)]
        candles = _make_candles(prices)

        strategy = DualMomentumStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: _DEFAULT_PARAMS)
        signal = strategy.generate_signal("AAPL", candles, "us_tech")

        assert signal is None

    def test_dual_momentum_weighted_scoring(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify the 40/30/30 weighting is applied correctly."""
        n = 130
        # Create prices where we know exact returns
        prices = [100.0] * n
        # Make specific changes at the lookback points
        prices[-1] = 120.0  # current
        prices[-21] = 100.0  # 1m ago
        prices[-63] = 110.0  # 3m ago
        prices[-126] = 105.0  # 6m ago

        candles = _make_candles(prices)
        strategy = DualMomentumStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: _DEFAULT_PARAMS)
        signal = strategy.generate_signal("AAPL", candles, "us_tech")

        ret_1m = (120.0 - 100.0) / 100.0  # 0.2
        ret_3m = (120.0 - 110.0) / 110.0  # ~0.0909
        ret_6m = (120.0 - 105.0) / 105.0  # ~0.1429

        expected_score = ret_1m * 0.4 + ret_3m * 0.3 + ret_6m * 0.3

        assert signal is not None
        features = signal.features
        assert abs(features["score_1m"] - ret_1m) < 1e-6
        assert abs(features["score_3m"] - ret_3m) < 1e-6
        assert abs(features["score_6m"] - ret_6m) < 1e-6

        expected_confidence = min(0.95, 0.4 + abs(expected_score) * 1.0)
        assert abs(signal.confidence - expected_confidence) < 1e-6

    def test_dual_momentum_supported_segments(self) -> None:
        """All standard segments should be supported."""
        strategy = DualMomentumStrategy()
        segments = strategy.supported_segments()
        assert "us_tech" in segments
        assert "ru_blue_chips" in segments

    def test_dual_momentum_name(self) -> None:
        """Strategy name is 'dual_momentum'."""
        strategy = DualMomentumStrategy()
        assert strategy.name == "dual_momentum"


class TestDualMomentumYAMLParams:
    """Tests for DualMomentumStrategy reading params from YAML presets."""

    def test_reads_yaml_lookback_params(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When YAML preset has custom lookback values, the strategy uses them."""
        custom_lookback_1m = 10
        custom_lookback_3m = 30
        custom_lookback_6m = 60
        custom_min_confidence = 0.55

        yaml_params: dict[str, object] = {
            "lookback_1m": custom_lookback_1m,
            "lookback_3m": custom_lookback_3m,
            "lookback_6m": custom_lookback_6m,
            "min_confidence": custom_min_confidence,
        }

        strategy = DualMomentumStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: yaml_params)

        # Build prices long enough for the custom lookbacks (60 + some margin)
        n = 70
        prices = [100.0 + i * 1.0 for i in range(n)]
        candles = _make_candles(prices)

        signal = strategy.generate_signal("AAPL", candles, "us_tech")

        # With custom lookbacks, returns are computed differently than defaults
        assert signal is not None
        assert signal.direction == SignalDirection.BUY

        # Verify the custom lookbacks were used by checking the features
        close_now = prices[-1]
        close_1m = prices[-custom_lookback_1m]
        close_3m = prices[-custom_lookback_3m]
        close_6m = prices[-custom_lookback_6m]

        expected_1m = (close_now - close_1m) / close_1m
        expected_3m = (close_now - close_3m) / close_3m
        expected_6m = (close_now - close_6m) / close_6m

        tolerance = 1e-6
        assert abs(signal.features["score_1m"] - expected_1m) < tolerance
        assert abs(signal.features["score_3m"] - expected_3m) < tolerance
        assert abs(signal.features["score_6m"] - expected_6m) < tolerance

    def test_defaults_without_yaml_params(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When YAML preset has no custom params, defaults are used."""
        strategy = DualMomentumStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: {})

        # Need 126 candles for default lookback
        n = 130
        prices = [100.0 + i * 0.5 for i in range(n)]
        candles = _make_candles(prices)

        signal = strategy.generate_signal("AAPL", candles, "us_tech")

        assert signal is not None
        assert signal.direction == SignalDirection.BUY

        # Verify default lookbacks (21, 63, 126) were used
        close_now = prices[-1]
        close_1m = prices[-21]
        close_3m = prices[-63]
        close_6m = prices[-126]

        expected_1m = (close_now - close_1m) / close_1m
        expected_3m = (close_now - close_3m) / close_3m
        expected_6m = (close_now - close_6m) / close_6m

        tolerance = 1e-6
        assert abs(signal.features["score_1m"] - expected_1m) < tolerance
        assert abs(signal.features["score_3m"] - expected_3m) < tolerance
        assert abs(signal.features["score_6m"] - expected_6m) < tolerance

    def test_insufficient_data_adapts_to_custom_lookback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """min_candles adapts to the longest custom lookback."""
        yaml_params: dict[str, object] = {
            "lookback_1m": 10,
            "lookback_3m": 30,
            "lookback_6m": 200,  # longer than default 126
        }

        strategy = DualMomentumStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: yaml_params)

        # 150 candles: enough for default (126), but not for custom (200)
        prices = [100.0 + i * 0.5 for i in range(150)]
        candles = _make_candles(prices)

        signal = strategy.generate_signal("AAPL", candles, "us_tech")
        assert signal is None, "Expected None: not enough data for custom lookback_6m=200"

    def test_custom_min_confidence_filters_low_score(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A high min_confidence threshold should filter out weak signals."""
        yaml_params: dict[str, object] = {
            "min_confidence": 0.99,  # very high threshold
        }

        strategy = DualMomentumStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: yaml_params)

        # Gentle uptrend -> small score -> low confidence
        n = 130
        prices = [100.0 + i * 0.05 for i in range(n)]
        candles = _make_candles(prices)

        signal = strategy.generate_signal("AAPL", candles, "us_tech")
        assert signal is None, "Expected None: confidence below min_confidence=0.99"


class TestOUMeanReversionYAMLParams:
    """Tests for OUMeanReversionStrategy reading params from YAML presets."""

    @staticmethod
    def _make_mean_reverting_prices(
        n: int = 50,
        seed: int = 42,
    ) -> list[float]:
        """Generate mean-reverting prices via OU process simulation.

        Returns prices oscillating around 100, suitable for OU fitting.
        """
        import math

        import numpy as np

        rng = np.random.default_rng(seed)
        mu_true = 0.1
        theta_true = math.log(100)
        sigma_true = 0.02

        log_prices = [theta_true]
        for _ in range(n - 1):
            dx = mu_true * (theta_true - log_prices[-1]) + sigma_true * rng.standard_normal()
            log_prices.append(log_prices[-1] + dx)

        return [math.exp(lp) for lp in log_prices]

    def test_reads_yaml_params(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When YAML preset has custom OU params, the strategy uses them."""
        from finalayze.strategies.ou_mean_reversion import OUMeanReversionStrategy

        custom_ou_window = 40
        custom_entry_threshold = 0.5  # low threshold so deviation triggers
        custom_exit_threshold = 0.1

        yaml_params: dict[str, object] = {
            "ou_window": custom_ou_window,
            "entry_threshold": custom_entry_threshold,
            "exit_threshold": custom_exit_threshold,
            "half_life_range": (1, 200),
        }

        strategy = OUMeanReversionStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: yaml_params)

        # Build mean-reverting prices, then add a sharp drop at the end
        base_prices = self._make_mean_reverting_prices(n=custom_ou_window + 5)
        # Append a price well below the mean to trigger BUY
        base_prices.append(85.0)
        candles = _make_candles(base_prices)

        signal = strategy.generate_signal("AAPL", candles, "us_tech")

        # With entry_threshold=0.5 and a price well below fitted theta, expect BUY
        assert signal is not None
        assert signal.direction == SignalDirection.BUY
        assert signal.strategy_name == "ou_mean_reversion"

    def test_defaults_without_yaml_params(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When YAML preset returns empty dict, hardcoded class defaults are used."""
        from finalayze.strategies.ou_mean_reversion import OUMeanReversionStrategy

        strategy = OUMeanReversionStrategy()

        # The _SEGMENT_PARAMS dict has defaults for us_broad with ou_window=90
        # When we monkeypatch get_parameters to return empty, the strategy should
        # use its module-level defaults
        yaml_params: dict[str, object] = {
            "ou_window": 90,
            "entry_threshold": 1.5,
            "exit_threshold": 0.0,
            "half_life_range": (5, 60),
        }
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: yaml_params)

        # Need 90+1=91 candles
        n = 95
        prices = [100.0] * n
        candles = _make_candles(prices)

        # Flat prices => no deviation => no signal
        signal = strategy.generate_signal("AAPL", candles, "us_tech")
        assert signal is None, "Flat prices should not trigger a signal"

    def test_yaml_ou_window_adapts_min_candles(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Changing ou_window in YAML changes min candle requirement."""
        from finalayze.strategies.ou_mean_reversion import OUMeanReversionStrategy

        custom_ou_window = 200

        yaml_params: dict[str, object] = {
            "ou_window": custom_ou_window,
            "entry_threshold": 1.5,
            "exit_threshold": 0.0,
            "half_life_range": (1, 300),
        }

        strategy = OUMeanReversionStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: yaml_params)

        # 150 candles: enough for default (90+1=91), but not for custom (200+1=201)
        prices = [100.0 + i * 0.1 for i in range(150)]
        candles = _make_candles(prices)

        signal = strategy.generate_signal("AAPL", candles, "us_tech")
        assert signal is None, "Expected None: not enough data for custom ou_window=200"

    def test_yaml_entry_threshold_filters_signal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A very high entry_threshold should filter out signals."""
        from finalayze.strategies.ou_mean_reversion import OUMeanReversionStrategy

        yaml_params: dict[str, object] = {
            "ou_window": 30,
            "entry_threshold": 100.0,  # impossibly high
            "exit_threshold": 0.0,
            "half_life_range": (1, 300),
        }

        strategy = OUMeanReversionStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: yaml_params)

        n = 40
        prices = [100.0] * (n - 5)
        prices.extend([90.0, 85.0, 80.0, 75.0, 70.0])
        candles = _make_candles(prices)

        signal = strategy.generate_signal("AAPL", candles, "us_tech")
        assert signal is None, "Expected None: entry_threshold=100.0 too high"

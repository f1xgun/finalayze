"""Unit tests for PairsStrategy."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np
import pytest

from finalayze.core.schemas import Candle, SignalDirection

# Constants — no magic numbers
MIN_CANDLES = 60
N_CANDLES = 80
Z_ENTRY = 2.0
Z_EXIT = 0.5
BASE_PRICE = 100.0
SPREAD_STD = 1.0


def _make_candles(
    n: int,
    symbol: str,
    prices: list[float],
    market_id: str = "us",
) -> list[Candle]:
    base = datetime(2023, 1, 1, tzinfo=UTC)
    return [
        Candle(
            symbol=symbol,
            market_id=market_id,
            timeframe="1d",
            timestamp=base + timedelta(days=i),
            open=Decimal(str(round(prices[i] * 0.999, 4))),
            high=Decimal(str(round(prices[i] * 1.005, 4))),
            low=Decimal(str(round(prices[i] * 0.995, 4))),
            close=Decimal(str(round(prices[i], 4))),
            volume=1000,
        )
        for i in range(n)
    ]


def _cointegrated_pair(
    n: int = N_CANDLES,
    z_score: float = 0.0,
    rng_seed: int = 42,
) -> tuple[list[Candle], list[Candle]]:
    """Build two cointegrated price series (AAPL, MSFT).

    z_score shifts the final spread observation by z_score * std to simulate
    a z-score of approximately z_score.
    """
    rng = np.random.default_rng(rng_seed)
    common = rng.standard_normal(n).cumsum() + BASE_PRICE
    noise_a = rng.standard_normal(n) * 0.05
    noise_b = rng.standard_normal(n) * 0.05

    prices_a = common + noise_a
    prices_b = common * 0.5 + noise_b  # beta ≈ 0.5

    if z_score != 0.0:
        import numpy as np2  # noqa: PLC0415

        log_a = np2.log(prices_a)
        log_b = np2.log(prices_b)
        beta = float(np2.cov(log_a, log_b)[0, 1] / np2.var(log_b))
        spread = log_a - beta * log_b
        target_shift = z_score * float(spread.std())
        prices_a[-1] = float(np2.exp(log_a[-1] + target_shift))

    candles_a = _make_candles(n, "AAPL", prices_a.tolist())
    candles_b = _make_candles(n, "MSFT", prices_b.tolist())
    return candles_a, candles_b


def _non_cointegrated_pair(n: int = N_CANDLES) -> tuple[list[Candle], list[Candle]]:
    """Two independent random walks — not cointegrated."""
    rng = np.random.default_rng(7)
    prices_a = (BASE_PRICE + rng.standard_normal(n).cumsum()).tolist()
    rng2 = np.random.default_rng(99)
    prices_b = (BASE_PRICE * 2 + rng2.standard_normal(n).cumsum()).tolist()
    return (
        _make_candles(n, "AAPL", prices_a),
        _make_candles(n, "MSFT", prices_b),
    )


@pytest.fixture
def pairs_strategy() -> object:
    from finalayze.strategies.pairs import PairsStrategy

    return PairsStrategy()


@pytest.mark.unit
class TestPairsStrategyName:
    def test_name(self, pairs_strategy: object) -> None:
        assert pairs_strategy.name == "pairs"  # type: ignore[union-attr]


@pytest.mark.unit
class TestPairsStrategyInsufficientCandles:
    def test_returns_none_when_too_few_candles(self, pairs_strategy: object) -> None:
        from finalayze.strategies.pairs import PairsStrategy

        strategy: PairsStrategy = pairs_strategy  # type: ignore[assignment]
        candles_a, candles_b = _cointegrated_pair(n=30)
        strategy.set_peer_candles("MSFT", candles_b)
        result = strategy.generate_signal("AAPL", candles_a, "us_tech")
        assert result is None


@pytest.mark.unit
class TestPairsStrategyNonCointegrated:
    def test_non_cointegrated_returns_none(self, pairs_strategy: object) -> None:
        from finalayze.strategies.pairs import PairsStrategy

        strategy: PairsStrategy = pairs_strategy  # type: ignore[assignment]
        candles_a, candles_b = _non_cointegrated_pair()
        strategy.set_peer_candles("MSFT", candles_b)
        result = strategy.generate_signal("AAPL", candles_a, "us_tech")
        # p-value > 0.05 for random walks → should return None
        # Note: may occasionally pass if the random walk happens to pass the test;
        # seed 7/99 are chosen to reliably fail cointegration
        assert result is None


@pytest.mark.unit
class TestPairsStrategySignals:
    def test_z_below_negative_entry_returns_buy(self, pairs_strategy: object) -> None:
        from finalayze.strategies.pairs import PairsStrategy

        strategy: PairsStrategy = pairs_strategy  # type: ignore[assignment]
        candles_a, candles_b = _cointegrated_pair(z_score=-3.0)
        strategy.set_peer_candles("MSFT", candles_b)
        signal = strategy.generate_signal("AAPL", candles_a, "us_tech")
        assert signal is not None
        assert signal.direction == SignalDirection.BUY

    def test_z_above_positive_entry_returns_sell(self, pairs_strategy: object) -> None:
        from finalayze.strategies.pairs import PairsStrategy

        strategy: PairsStrategy = pairs_strategy  # type: ignore[assignment]
        candles_a, candles_b = _cointegrated_pair(z_score=3.0)
        strategy.set_peer_candles("MSFT", candles_b)
        signal = strategy.generate_signal("AAPL", candles_a, "us_tech")
        assert signal is not None
        assert signal.direction == SignalDirection.SELL

    def test_z_within_exit_band_returns_none(self, pairs_strategy: object) -> None:
        from finalayze.strategies.pairs import PairsStrategy

        strategy: PairsStrategy = pairs_strategy  # type: ignore[assignment]
        candles_a, candles_b = _cointegrated_pair(z_score=0.0)  # z near 0
        strategy.set_peer_candles("MSFT", candles_b)
        # For a cointegrated pair with z≈0, |z| < z_exit → None
        signal = strategy.generate_signal("AAPL", candles_a, "us_tech")
        # z=0 → |0| < 0.5 → return None
        assert signal is None


@pytest.mark.unit
class TestPairsStrategyConfidence:
    def test_confidence_bounded(self, pairs_strategy: object) -> None:
        from finalayze.strategies.pairs import PairsStrategy

        strategy: PairsStrategy = pairs_strategy  # type: ignore[assignment]
        candles_a, candles_b = _cointegrated_pair(z_score=-4.0)
        strategy.set_peer_candles("MSFT", candles_b)
        signal = strategy.generate_signal("AAPL", candles_a, "us_tech")
        if signal is not None:
            assert 0.0 <= signal.confidence <= 1.0

    def test_reasoning_contains_z_and_beta(self, pairs_strategy: object) -> None:
        from finalayze.strategies.pairs import PairsStrategy

        strategy: PairsStrategy = pairs_strategy  # type: ignore[assignment]
        candles_a, candles_b = _cointegrated_pair(z_score=-3.0)
        strategy.set_peer_candles("MSFT", candles_b)
        signal = strategy.generate_signal("AAPL", candles_a, "us_tech")
        if signal is not None:
            assert "z=" in signal.reasoning
            assert "beta=" in signal.reasoning


@pytest.mark.unit
class TestPairsStrategySupportedSegments:
    def test_supported_segments_returns_list(self, pairs_strategy: object) -> None:
        segments = pairs_strategy.supported_segments()  # type: ignore[union-attr]
        assert isinstance(segments, list)
        # us_tech and ru_blue_chips should be in list after YAML update in step 5.3
        assert "us_tech" in segments
        assert "ru_blue_chips" in segments

    def test_get_parameters_us_tech(self, pairs_strategy: object) -> None:
        params = pairs_strategy.get_parameters("us_tech")  # type: ignore[union-attr]
        assert "pairs" in params
        assert "z_entry" in params
        assert "z_exit" in params

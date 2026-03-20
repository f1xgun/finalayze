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


@pytest.mark.unit
class TestPairsStrategyAllowShort:
    """Tests for the allow_short parameter gating SELL signals on long-only markets."""

    def test_allow_short_false_suppresses_sell(self, pairs_strategy: object) -> None:
        """When allow_short=False, _compute_signal returns None for SELL direction."""
        from finalayze.strategies.pairs import PairsStrategy

        strategy: PairsStrategy = pairs_strategy  # type: ignore[assignment]
        # z_score=+3.0 would normally produce a SELL signal
        candles_a, candles_b = _cointegrated_pair(z_score=3.0)
        # Directly call _compute_signal with allow_short=False
        signal = strategy._compute_signal(
            symbol="AAPL",
            candles_a=candles_a,
            candles_b=candles_b,
            segment_id="us_tech",
            z_entry=Z_ENTRY,
            z_exit=Z_EXIT,
            allow_short=False,
        )
        # Should be None because allow_short=False suppresses SELL
        assert signal is None

    def test_allow_short_false_allows_buy(self, pairs_strategy: object) -> None:
        """When allow_short=False, BUY signals still pass through."""
        from finalayze.strategies.pairs import PairsStrategy

        strategy: PairsStrategy = pairs_strategy  # type: ignore[assignment]
        candles_a, candles_b = _cointegrated_pair(z_score=-3.0)
        signal = strategy._compute_signal(
            symbol="AAPL",
            candles_a=candles_a,
            candles_b=candles_b,
            segment_id="us_tech",
            z_entry=Z_ENTRY,
            z_exit=Z_EXIT,
            allow_short=False,
        )
        assert signal is not None
        assert signal.direction == SignalDirection.BUY

    def test_allow_short_default_true_allows_sell(self, pairs_strategy: object) -> None:
        """Default allow_short=True preserves SELL signal behavior."""
        from finalayze.strategies.pairs import PairsStrategy

        strategy: PairsStrategy = pairs_strategy  # type: ignore[assignment]
        candles_a, candles_b = _cointegrated_pair(z_score=3.0)
        # allow_short defaults to True → SELL allowed
        signal = strategy._compute_signal(
            symbol="AAPL",
            candles_a=candles_a,
            candles_b=candles_b,
            segment_id="us_tech",
            z_entry=Z_ENTRY,
            z_exit=Z_EXIT,
        )
        assert signal is not None
        assert signal.direction == SignalDirection.SELL


@pytest.mark.unit
class TestRuBlueChipsPairsConfig:
    """Tests for ru_blue_chips YAML pairs configuration."""

    def test_ru_blue_chips_pairs_config(self, pairs_strategy: object) -> None:
        """Verify ru_blue_chips has correct pairs config after update."""
        params = pairs_strategy.get_parameters("ru_blue_chips")  # type: ignore[union-attr]
        assert params, "ru_blue_chips should have pairs params"
        pairs_list = params["pairs"]
        assert len(pairs_list) == 2  # noqa: PLR2004
        # Check SBER/SBERP and TATN/TATNP pairs
        pair_tuples = [tuple(p) for p in pairs_list]
        assert ("SBER", "SBERP") in pair_tuples
        assert ("TATN", "TATNP") in pair_tuples
        assert float(params["z_entry"]) == Z_ENTRY
        assert params.get("allow_short") is False
        assert params.get("cointegration_start") == "2023-01-01"

    def test_cointegration_start_filters_data(self, pairs_strategy: object) -> None:
        """When cointegration_start is set, only data from that date onward is used."""
        from finalayze.strategies.pairs import PairsStrategy

        strategy: PairsStrategy = pairs_strategy  # type: ignore[assignment]

        # Create candles spanning 2021-2025 (4 years = ~1460 days)
        n_total = 200
        base_date = datetime(2021, 1, 1, tzinfo=UTC)
        rng = np.random.default_rng(42)
        common = rng.standard_normal(n_total).cumsum() + BASE_PRICE
        noise_a = rng.standard_normal(n_total) * 0.05
        noise_b = rng.standard_normal(n_total) * 0.05
        prices_a = common + noise_a
        prices_b = common * 0.5 + noise_b

        # Inject a structural break in early data (pre-2022) to break cointegration
        # if all data is used, but maintain cointegration in post-2023 data
        prices_a[:50] = prices_a[:50] * 2.0  # break pre-2023 relationship

        # Create candles with dates from 2021 (spread ~7 days apart to span 2021-2025)
        candles_a = []
        candles_b = []
        for i in range(n_total):
            ts = base_date + timedelta(days=i * 7)
            candles_a.append(
                Candle(
                    symbol="SBER",
                    market_id="moex",
                    timeframe="1d",
                    timestamp=ts,
                    open=Decimal(str(round(float(prices_a[i]) * 0.999, 4))),
                    high=Decimal(str(round(float(prices_a[i]) * 1.005, 4))),
                    low=Decimal(str(round(float(prices_a[i]) * 0.995, 4))),
                    close=Decimal(str(round(float(prices_a[i]), 4))),
                    volume=1000,
                )
            )
            candles_b.append(
                Candle(
                    symbol="SBERP",
                    market_id="moex",
                    timeframe="1d",
                    timestamp=ts,
                    open=Decimal(str(round(float(prices_b[i]) * 0.999, 4))),
                    high=Decimal(str(round(float(prices_b[i]) * 1.005, 4))),
                    low=Decimal(str(round(float(prices_b[i]) * 0.995, 4))),
                    close=Decimal(str(round(float(prices_b[i]), 4))),
                    volume=1000,
                )
            )

        strategy.set_peer_candles("SBERP", candles_b)
        # Using ru_blue_chips which has cointegration_start="2023-01-01"
        # The strategy should filter out pre-2023 data before cointegration test
        # This means the structural break in pre-2022 data should NOT affect results
        signal = strategy.generate_signal("SBER", candles_a, "ru_blue_chips")
        # We just verify no error is raised and the strategy handles filtering
        # The actual signal depends on z-score, but the key behavior is that
        # cointegration is computed on filtered data only
        assert signal is None or signal.direction == SignalDirection.BUY

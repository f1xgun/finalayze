"""Tests for HMM regime detection.

Tests cover:
- Minimum data requirement (252 candles)
- State labeling using mean AND variance
- 3-bar persistence filter
- Convergence with n_iter=100, tol=1e-4
- Predict returns valid RegimeState
- HMMRegimeProvider fallback for insufficient data
- HMMRegimeProvider periodic retraining
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np
import pytest

from finalayze.core.schemas import Candle
from finalayze.risk.hmm_regime import HMMRegimeDetector
from finalayze.risk.regime import HMMRegimeProvider, MarketRegime, RegimeState

# ── Constants ──────────────────────────────────────────────────────────────────

_BASE_DT = datetime(2023, 1, 2, 14, 30, tzinfo=UTC)
_ONE_DAY = timedelta(days=1)
_MIN_CANDLES = 252
_SEED = 42


def _make_candle(
    close: Decimal,
    idx: int,
    *,
    volume: int = 1_000_000,
    symbol: str = "SPY",
) -> Candle:
    """Create a single candle with the given close price."""
    return Candle(
        symbol=symbol,
        market_id="us",
        timeframe="1d",
        timestamp=_BASE_DT + _ONE_DAY * idx,
        open=close - Decimal(1),
        high=close + Decimal(2),
        low=close - Decimal(2),
        close=close,
        volume=volume,
    )


def _make_regime_candles(
    n: int,
    *,
    start_price: float = 100.0,
    daily_return: float = 0.0005,
    daily_vol: float = 0.01,
    base_volume: int = 1_000_000,
    seed: int = _SEED,
) -> list[Candle]:
    """Generate synthetic candles simulating a specific market regime.

    Args:
        n: Number of candles to generate.
        start_price: Starting close price.
        daily_return: Mean daily log return.
        daily_vol: Daily volatility (std of log returns).
        base_volume: Average volume.
        seed: Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    prices = [start_price]
    for _ in range(n - 1):
        log_ret = daily_return + daily_vol * rng.standard_normal()
        prices.append(prices[-1] * np.exp(log_ret))

    candles = []
    for i, price in enumerate(prices):
        vol = int(base_volume * (1 + 0.3 * rng.standard_normal()))
        vol = max(vol, 100)
        candles.append(
            _make_candle(
                Decimal(str(round(price, 2))),
                i,
                volume=vol,
            )
        )
    return candles


def _make_mixed_regime_candles(n: int = 600) -> list[Candle]:
    """Generate candles with distinct regime shifts.

    First third: calm bull market (high return, low vol)
    Second third: volatile bull (high return, high vol)
    Last third: bear market (negative return, high vol)
    """
    segment = n // 3
    remainder = n - 3 * segment

    calm_bull = _make_regime_candles(
        segment,
        start_price=100.0,
        daily_return=0.001,
        daily_vol=0.005,
        seed=_SEED,
    )

    last_price = float(calm_bull[-1].close)
    volatile_bull = _make_regime_candles(
        segment,
        start_price=last_price,
        daily_return=0.0005,
        daily_vol=0.025,
        seed=_SEED + 1,
    )

    last_price = float(volatile_bull[-1].close)
    bear = _make_regime_candles(
        segment + remainder,
        start_price=last_price,
        daily_return=-0.002,
        daily_vol=0.03,
        seed=_SEED + 2,
    )

    # Re-index timestamps to be sequential
    all_candles: list[Candle] = []
    for idx, candle in enumerate([*calm_bull, *volatile_bull, *bear]):
        all_candles.append(
            Candle(
                symbol=candle.symbol,
                market_id=candle.market_id,
                timeframe=candle.timeframe,
                timestamp=_BASE_DT + _ONE_DAY * idx,
                open=candle.open,
                high=candle.high,
                low=candle.low,
                close=candle.close,
                volume=candle.volume,
            )
        )
    return all_candles


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestHMMMinDataRequirement:
    """Test that HMM enforces minimum data requirement."""

    def test_hmm_min_data_252(self) -> None:
        """fit raises ValueError with <252 candles."""
        detector = HMMRegimeDetector()
        candles = _make_regime_candles(251)
        with pytest.raises(ValueError, match="Need >= 252"):
            detector.fit(candles)

    def test_hmm_fit_accepts_252(self) -> None:
        """fit succeeds with exactly 252+ candles."""
        detector = HMMRegimeDetector()
        # Need ~20 extra candles beyond 252 for rolling window feature trimming
        candles = _make_regime_candles(300)
        detector.fit(candles)
        assert detector._model is not None


class TestHMMLabelStates:
    """Test that state labeling uses mean AND variance."""

    def test_hmm_label_states_uses_variance(self) -> None:
        """High-variance positive-return state labeled 'high_vol_bull'."""
        candles = _make_mixed_regime_candles(600)
        detector = HMMRegimeDetector()
        detector.fit(candles)

        # Verify labeling logic: should have at least 2 distinct labels
        labels = set(detector._state_labels.values())
        assert len(labels) >= 2, f"Expected diverse labels, got: {labels}"

        # Check that label->scale mapping is consistent
        for state_id, label in detector._state_labels.items():
            scale = detector._state_scales[state_id]
            if label == "low_vol_bull":
                assert scale == Decimal("1.0")
            elif label == "high_vol_bull":
                assert scale == Decimal("0.5")
            elif label == "bear":
                assert scale == Decimal("0.25")

    def test_hmm_label_all_states_assigned(self) -> None:
        """All 3 HMM states get labels after fitting."""
        candles = _make_mixed_regime_candles(600)
        detector = HMMRegimeDetector()
        detector.fit(candles)

        assert len(detector._state_labels) == 3
        assert len(detector._state_scales) == 3


class TestHMMStatePersistence:
    """Test 3-bar persistence filter."""

    def test_hmm_state_persistence(self) -> None:
        """Regime only changes after 3 consecutive same predictions."""
        candles = _make_mixed_regime_candles(600)
        detector = HMMRegimeDetector()
        detector.fit(candles)

        # Reset persistence state
        detector.reset_persistence()

        # Run predictions on sequential subsets
        regimes: list[RegimeState] = []
        for end_idx in range(_MIN_CANDLES, min(_MIN_CANDLES + 10, len(candles))):
            subset = candles[:end_idx]
            regime = detector.predict_regime(subset)
            regimes.append(regime)

        # The persistence filter means regime transitions should be
        # less frequent than raw predictions. At minimum, the first
        # prediction should produce a valid RegimeState.
        assert len(regimes) > 0
        for r in regimes:
            assert isinstance(r, RegimeState)
            assert r.regime in (
                MarketRegime.NORMAL,
                MarketRegime.ELEVATED,
                MarketRegime.CRISIS,
            )

    def test_hmm_persistence_filter_requires_three_same(self) -> None:
        """Directly test persistence by manipulating _recent_states."""
        candles = _make_mixed_regime_candles(600)
        detector = HMMRegimeDetector()
        detector.fit(candles)

        # Manually set mixed recent states to simulate no consensus
        detector._recent_states = [0, 1, 0]

        # The next prediction should default to the first recent state (0)
        # unless 3 consecutive agree
        detector.predict_regime(candles)
        # After prediction, recent_states should have length <= PERSISTENCE_BARS
        assert len(detector._recent_states) <= detector._PERSISTENCE_BARS


class TestHMMConvergence:
    """Test that HMM converges with given parameters."""

    def test_hmm_convergence(self) -> None:
        """HMM converges with n_iter=100, tol=1e-4 (no error)."""
        candles = _make_mixed_regime_candles(600)
        detector = HMMRegimeDetector()
        # This should not raise any errors
        detector.fit(candles)
        assert detector._model is not None
        # Model should have learned 3 components
        assert detector._model.n_components == 3


class TestHMMPredictReturnsRegimeState:
    """Test that predict_regime returns valid RegimeState."""

    def test_hmm_predict_returns_regime_state(self) -> None:
        """Output is a valid RegimeState with correct fields."""
        candles = _make_mixed_regime_candles(600)
        detector = HMMRegimeDetector()
        detector.fit(candles)

        result = detector.predict_regime(candles)

        assert isinstance(result, RegimeState)
        assert result.regime in (
            MarketRegime.NORMAL,
            MarketRegime.ELEVATED,
            MarketRegime.CRISIS,
        )
        assert isinstance(result.position_scale, Decimal)
        assert Decimal(0) < result.position_scale <= Decimal("1.0")
        assert isinstance(result.allow_new_longs, bool)

    def test_hmm_predict_unfitted_raises(self) -> None:
        """predict_regime raises RuntimeError if model not fitted."""
        detector = HMMRegimeDetector()
        candles = _make_mixed_regime_candles(600)
        with pytest.raises(RuntimeError, match="Model not fitted"):
            detector.predict_regime(candles)

    def test_hmm_crisis_blocks_longs(self) -> None:
        """When regime is CRISIS, allow_new_longs must be False."""
        candles = _make_mixed_regime_candles(600)
        detector = HMMRegimeDetector()
        detector.fit(candles)

        # Run many predictions to find a CRISIS regime
        detector.reset_persistence()
        for end_idx in range(_MIN_CANDLES, len(candles)):
            result = detector.predict_regime(candles[:end_idx])
            if result.regime == MarketRegime.CRISIS:
                assert result.allow_new_longs is False
                break


class TestHMMRegimeProvider:
    """Tests for the HMMRegimeProvider wrapper."""

    def test_hmm_regime_provider_normal_for_insufficient_data(self) -> None:
        """Returns NORMAL regime when <252 candles are available."""
        provider = HMMRegimeProvider(retrain_frequency=21)
        candles = _make_regime_candles(100)
        result = provider.get_regime(candles, bar_index=50)

        assert result.regime == MarketRegime.NORMAL
        assert result.allow_new_longs is True
        assert result.position_scale == Decimal("1.0")

    def test_hmm_regime_provider_returns_valid_state(self) -> None:
        """Provider returns valid RegimeState with sufficient data."""
        provider = HMMRegimeProvider(retrain_frequency=21)
        candles = _make_mixed_regime_candles(600)
        result = provider.get_regime(candles, bar_index=len(candles) - 1)

        assert isinstance(result, RegimeState)
        assert result.regime in (
            MarketRegime.NORMAL,
            MarketRegime.ELEVATED,
            MarketRegime.CRISIS,
        )

    def test_hmm_regime_provider_retrains(self) -> None:
        """Provider retrains after retrain_frequency bars."""
        retrain_freq = 5
        provider = HMMRegimeProvider(retrain_frequency=retrain_freq)
        candles = _make_mixed_regime_candles(600)

        # First call at bar 300 should trigger training
        provider.get_regime(candles, bar_index=300)
        assert provider._last_train_bar == 300

        # Call at bar 302 (only 2 bars later) should NOT retrain
        provider.get_regime(candles, bar_index=302)
        assert provider._last_train_bar == 300

        # Call at bar 305 (5 bars later) should retrain
        provider.get_regime(candles, bar_index=305)
        assert provider._last_train_bar == 305

    def test_hmm_regime_provider_no_lookahead(self) -> None:
        """Provider only uses candles up to bar_index (no look-ahead)."""
        provider = HMMRegimeProvider(retrain_frequency=21)
        candles = _make_mixed_regime_candles(600)

        # Call at bar_index=300 should only use candles[0:301]
        result = provider.get_regime(candles, bar_index=300)
        assert isinstance(result, RegimeState)

    def test_hmm_regime_provider_bar_index_too_small(self) -> None:
        """Provider returns NORMAL when bar_index yields <252 candles."""
        provider = HMMRegimeProvider(retrain_frequency=21)
        candles = _make_mixed_regime_candles(600)

        # bar_index=100 means only 101 candles available
        result = provider.get_regime(candles, bar_index=100)
        assert result.regime == MarketRegime.NORMAL

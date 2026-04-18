"""Pairs trading strategy using cointegration-based spread z-scores (Layer 4)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
import yaml
from statsmodels.tsa.stattools import coint

from finalayze.core.exceptions import InsufficientDataError
from finalayze.core.schemas import Candle, Signal, SignalDirection
from finalayze.strategies.base import BaseStrategy

_PRESETS_DIR = Path(__file__).parent / "presets"
_MIN_CANDLES = 60
_MIN_CANDLES_FOR_HIST = 3  # need at least 2 historical bars after slicing
_COINT_P_THRESHOLD = 0.05
_PAIR_LENGTH = 2
_MIN_KALMAN_POINTS = 20

# Kalman filter hyperparameters
_KALMAN_R = 0.001  # observation noise (measurement variance)
_KALMAN_Q = 1e-5  # process noise (state transition variance)
_KALMAN_STATE_DIM = 2

# Cointegration / hedge-ratio calibration cache.
# Re-running coint() on every bar is O(N^2) per evaluation and produces a
# beta that jitters with each new observation. We calibrate periodically
# (monthly by default) and reuse (beta, spread_mean, spread_std) between
# recalibrations — only the z-score on the most recent bar is recomputed.
_DEFAULT_CALIBRATION_BARS = 21  # ~one trading month
_CALIBRATION_CACHE_MAX = 128


@dataclass(frozen=True)
class _PairCalibration:
    """Cached cointegration calibration for a single pair."""

    calibrated_at_timestamp: datetime
    hist_length: int
    p_value: float
    beta: float
    spread_mean: float
    spread_std: float
    hedge_method: str


def compute_kalman_hedge_ratio(
    y_prices: list[float],
    x_prices: list[float],
) -> tuple[float, float]:
    """Compute Kalman-filtered intercept and slope for y = alpha + beta * x.

    Uses a simple linear regression Kalman filter with state [alpha, beta].

    Args:
        y_prices: Dependent variable (price series).
        x_prices: Independent variable (price series).

    Returns:
        Tuple of (alpha, beta) -- the Kalman-filtered intercept and slope.

    Raises:
        InsufficientDataError: If fewer than 20 data points are provided.
    """
    n = len(y_prices)
    if n < _MIN_KALMAN_POINTS or len(x_prices) < _MIN_KALMAN_POINTS:
        got = min(n, len(x_prices))
        msg = f"Kalman filter requires >= {_MIN_KALMAN_POINTS} points, got {got}"
        raise InsufficientDataError(msg)

    # State: [alpha, beta]
    state = np.array([0.0, 1.0])
    p_cov = np.eye(_KALMAN_STATE_DIM) * 1.0  # state covariance
    r_noise = _KALMAN_R  # observation noise
    q_noise = np.eye(_KALMAN_STATE_DIM) * _KALMAN_Q  # process noise

    for i in range(n):
        y_t = y_prices[i]
        x_t = x_prices[i]

        # Observation model: y_t = H @ state + noise, where H = [1, x_t]
        h_vec = np.array([1.0, x_t])

        # Predict step (random walk model: state unchanged, covariance grows)
        p_cov = p_cov + q_noise

        # Innovation
        y_pred = h_vec @ state
        innovation = y_t - y_pred

        # Innovation covariance: S = H @ P @ H^T + R
        s_innov = float(h_vec @ p_cov @ h_vec) + r_noise

        # Kalman gain: K = P @ H^T / S
        k_gain = (p_cov @ h_vec) / s_innov

        # Update state
        state = state + k_gain * innovation

        # Update covariance: P = (I - K @ H) @ P
        p_cov = (np.eye(_KALMAN_STATE_DIM) - np.outer(k_gain, h_vec)) @ p_cov

    return float(state[0]), float(state[1])


class PairsStrategy(BaseStrategy):
    """Statistical arbitrage via Engle-Granger cointegration spread z-score.

    Usage:
        strategy = PairsStrategy()
        strategy.set_peer_candles("MSFT", msft_candles)
        signal = strategy.generate_signal("AAPL", aapl_candles, "us_tech")
    """

    def __init__(self) -> None:
        self._peer_candles: dict[str, list[Candle]] = {}
        # Keyed by (segment_id, sym_a, sym_b). Calibration is re-used until
        # ``_DEFAULT_CALIBRATION_BARS`` new bars arrive. Eviction keeps the
        # cache bounded under many-segment workloads.
        self._calibration_cache: dict[tuple[str, str, str], _PairCalibration] = {}

    def invalidate_calibration_cache(self) -> None:
        """Clear cached pair calibrations (e.g. before a fresh backtest run)."""
        self._calibration_cache.clear()

    @property
    def name(self) -> str:
        return "pairs"

    def set_peer_candles(self, symbol: str, candles: list[Candle]) -> None:
        """Cache candles for a peer symbol so generate_signal can find them."""
        self._peer_candles[symbol] = candles

    def supported_segments(self) -> list[str]:
        """Return segment IDs where pairs strategy is enabled in YAML presets."""
        segments: list[str] = []
        for preset_path in sorted(_PRESETS_DIR.glob("*.yaml")):
            with preset_path.open() as f:
                data = yaml.safe_load(f)
            strategies = data.get("strategies", {})
            pairs_cfg = strategies.get("pairs", {})
            if pairs_cfg.get("enabled", False):
                segments.append(data["segment_id"])
        return segments

    def get_parameters(self, segment_id: str) -> dict[str, object]:
        """Load pairs parameters from the YAML preset for the given segment."""
        try:
            preset_path = _PRESETS_DIR / f"{segment_id}.yaml"
            with preset_path.open() as f:
                data = yaml.safe_load(f)
            params: dict[str, object] = dict(data["strategies"]["pairs"]["params"])
            return params
        except (FileNotFoundError, KeyError):
            return {}

    def generate_signal(
        self,
        symbol: str,
        candles: list[Candle],
        segment_id: str,
        sentiment_score: float = 0.0,  # noqa: ARG002
        has_open_position: bool = False,  # noqa: ARG002
        **kwargs: object,  # noqa: ARG002
    ) -> Signal | None:
        """Generate a pairs trading signal for symbol.

        Requires peer candles to be set via set_peer_candles() for all symbols
        configured as pairs with this symbol.

        Args:
            symbol: The primary symbol to generate a signal for.
            candles: Recent candles for symbol (must have >= 60).
            segment_id: Segment ID used to load YAML parameters.

        Returns:
            Signal if spread is beyond z_entry threshold, None otherwise.
        """
        if len(candles) < _MIN_CANDLES:
            return None

        params = self.get_parameters(segment_id)
        if not params:
            return None

        raw_pairs = cast("list[list[str]]", params.get("pairs", []))
        configured_pairs: list[list[str]] = [[str(s) for s in p] for p in raw_pairs]
        z_entry = float(cast("float", params.get("z_entry", 2.0)))
        z_exit = float(cast("float", params.get("z_exit", 0.5)))
        use_kalman = bool(params.get("use_kalman", False))
        allow_short = bool(params.get("allow_short", True))
        cointegration_start = str(params.get("cointegration_start", "")) or None

        for pair in configured_pairs:
            if len(pair) != _PAIR_LENGTH:
                continue
            sym_a, sym_b = pair[0], pair[1]

            # Only process pairs involving this symbol
            if symbol not in (sym_a, sym_b):
                continue

            # Determine which symbol is the "other" one
            peer_sym = sym_b if symbol == sym_a else sym_a
            peer_candles = self._peer_candles.get(peer_sym)
            if peer_candles is None or len(peer_candles) < _MIN_CANDLES:
                continue

            # Use the same symbol_a / symbol_b ordering as configured
            if symbol == sym_a:
                candles_a, candles_b = candles, peer_candles
            else:
                candles_a, candles_b = peer_candles, candles

            signal = self._compute_signal(
                symbol=symbol,
                candles_a=candles_a,
                candles_b=candles_b,
                segment_id=segment_id,
                z_entry=z_entry,
                z_exit=z_exit,
                use_kalman=use_kalman,
                allow_short=allow_short,
                cointegration_start=cointegration_start,
            )
            if signal is not None:
                return signal

        return None

    def _compute_signal(  # noqa: PLR0911
        self,
        symbol: str,
        candles_a: list[Candle],
        candles_b: list[Candle],
        segment_id: str,
        z_entry: float,
        z_exit: float,
        *,
        use_kalman: bool = False,
        allow_short: bool = True,
        cointegration_start: str | None = None,
    ) -> Signal | None:
        """Compute spread z-score and return signal or None."""
        # Filter candles by cointegration_start date if specified
        if cointegration_start:
            cutoff = datetime.fromisoformat(cointegration_start)
            candles_a = [c for c in candles_a if c.timestamp.replace(tzinfo=None) >= cutoff]
            candles_b = [c for c in candles_b if c.timestamp.replace(tzinfo=None) >= cutoff]

        n = min(len(candles_a), len(candles_b))
        if n < _MIN_CANDLES_FOR_HIST:
            return None

        sorted_a = sorted(candles_a, key=lambda c: c.timestamp)[-n:]
        sorted_b = sorted(candles_b, key=lambda c: c.timestamp)[-n:]

        log_a = np.log([float(c.close) for c in sorted_a])
        log_b = np.log([float(c.close) for c in sorted_b])

        calibration = self._get_or_refresh_calibration(
            segment_id=segment_id,
            sym_a=sorted_a[0].symbol,
            sym_b=sorted_b[0].symbol,
            log_a=log_a,
            log_b=log_b,
            latest_timestamp=sorted_a[-1].timestamp,
            use_kalman=use_kalman,
        )
        if calibration is None:
            return None

        beta = calibration.beta
        spread_mean = calibration.spread_mean
        spread_std = calibration.spread_std
        hedge_method = calibration.hedge_method

        # Current spread uses latest bar with historically-fitted beta
        current_spread = float(log_a[-1]) - beta * float(log_b[-1])
        z = float((current_spread - spread_mean) / spread_std)

        # Entry/exit logic
        if abs(z) < z_exit:
            return None  # spread closed — no new entry

        if z < -z_entry:
            direction = SignalDirection.BUY
        elif z > z_entry:
            direction = SignalDirection.SELL
        else:
            return None  # between z_exit and z_entry — ambiguous zone

        # Long-only constraint: suppress SELL signals when shorting is disallowed
        if direction == SignalDirection.SELL and not allow_short:
            return None

        confidence = min(1.0, abs(z) / z_entry)

        # Gate on min_confidence parameter from YAML preset
        params = self.get_parameters(segment_id)
        min_conf = float(params.get("min_confidence", 0.6))  # type: ignore[arg-type]
        if confidence < min_conf:
            return None

        market_id = candles_a[0].market_id

        return Signal(
            strategy_name=self.name,
            symbol=symbol,
            market_id=market_id,
            segment_id=segment_id,
            direction=direction,
            confidence=confidence,
            features={
                "z_score": round(z, 4),
                "beta": round(beta, 4),
                "kalman": 1.0 if hedge_method == "kalman" else 0.0,
            },
            reasoning=f"pairs z={z:.2f} beta={beta:.3f} ({hedge_method})",
        )

    def _get_or_refresh_calibration(
        self,
        *,
        segment_id: str,
        sym_a: str,
        sym_b: str,
        log_a: np.ndarray,  # type: ignore[type-arg]
        log_b: np.ndarray,  # type: ignore[type-arg]
        latest_timestamp: datetime,
        use_kalman: bool,
    ) -> _PairCalibration | None:
        """Return cached calibration, recalibrating periodically.

        Recalibration runs when the bar-count since the last calibration
        exceeds ``_DEFAULT_CALIBRATION_BARS``, or when no prior calibration
        exists for this pair. This avoids rerunning the O(N^2) Engle-Granger
        test on every bar while still keeping beta/spread stats reasonably
        fresh.
        """
        log_a_hist = log_a[:-1]
        log_b_hist = log_b[:-1]
        hist_length = len(log_a_hist)

        if hist_length < _MIN_CANDLES_FOR_HIST:
            return None

        key = (segment_id, sym_a, sym_b)
        cached = self._calibration_cache.get(key)
        if cached is not None and hist_length - cached.hist_length < _DEFAULT_CALIBRATION_BARS:
            return cached

        _, p_value, _ = coint(log_a_hist, log_b_hist)
        p_value_f = float(p_value)
        if p_value_f > _COINT_P_THRESHOLD:
            # Drop any stale calibration once cointegration breaks down.
            self._calibration_cache.pop(key, None)
            return None

        hedge_method: str
        if use_kalman and hist_length >= _MIN_KALMAN_POINTS:
            _, beta = compute_kalman_hedge_ratio(log_a_hist.tolist(), log_b_hist.tolist())
            hedge_method = "kalman"
        else:
            cov_matrix = np.cov(log_a_hist, log_b_hist)
            beta = float(cov_matrix[0, 1] / cov_matrix[1, 1])
            hedge_method = "ols"

        spread_hist = log_a_hist - beta * log_b_hist
        spread_mean = float(spread_hist.mean())
        spread_std = float(spread_hist.std(ddof=1))

        if spread_std == 0.0:
            self._calibration_cache.pop(key, None)
            return None

        # Trim the cache before inserting a fresh calibration.
        if len(self._calibration_cache) >= _CALIBRATION_CACHE_MAX:
            self._calibration_cache.pop(next(iter(self._calibration_cache)))

        calibration = _PairCalibration(
            calibrated_at_timestamp=latest_timestamp,
            hist_length=hist_length,
            p_value=p_value_f,
            beta=beta,
            spread_mean=spread_mean,
            spread_std=spread_std,
            hedge_method=hedge_method,
        )
        self._calibration_cache[key] = calibration
        return calibration

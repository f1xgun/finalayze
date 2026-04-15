"""Statistical anomaly detection for candle data (Layer 3).

Detects >3-sigma price moves and volume spikes vs a 20-bar rolling window.
Pure computation -- no IO, no alerting, no LLM calls.

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

import statistics
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from finalayze.core.schemas import Candle

_ANOMALY_SIGMA_THRESHOLD = 3.0
_ROLLING_WINDOW = 20
_VOLUME_RATIO_THRESHOLD = 2.0


class AnomalyResult(BaseModel):
    """Statistical anomaly detected in candle data."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    market_id: str
    price_move_pct: float
    sigma: float
    volume_ratio: float
    anomaly_type: str  # "price" | "volume" | "both"


class AnomalyDetector:
    """Detects statistical anomalies in candle price/volume data.

    Uses rolling z-score for price changes and ratio threshold for volume.
    Configurable thresholds via constructor parameters.
    """

    def __init__(
        self,
        sigma_threshold: float = _ANOMALY_SIGMA_THRESHOLD,
        rolling_window: int = _ROLLING_WINDOW,
        volume_ratio_threshold: float = _VOLUME_RATIO_THRESHOLD,
    ) -> None:
        self._sigma_threshold = sigma_threshold
        self._rolling_window = rolling_window
        self._volume_ratio_threshold = volume_ratio_threshold

    def check(
        self,
        candles: list[Candle],
        symbol: str,
        market_id: str,
    ) -> AnomalyResult | None:
        """Return AnomalyResult if latest candle shows statistical anomaly, else None.

        Requires at least rolling_window + 1 candles (default 21).
        """
        min_candles = self._rolling_window + 1
        if len(candles) < min_candles:
            return None

        window = candles[-min_candles:]
        closes = [float(c.close) for c in window]
        volumes = [float(c.volume) for c in window]

        # Price change z-score
        price_changes = [
            (closes[i] - closes[i - 1]) / closes[i - 1]
            for i in range(1, len(closes) - 1)
            if closes[i - 1] != 0.0
        ]
        if not price_changes:
            return None

        latest_change = (closes[-1] - closes[-2]) / closes[-2] if closes[-2] != 0.0 else 0.0

        mean_chg = statistics.mean(price_changes)
        std_chg = statistics.stdev(price_changes) if len(price_changes) > 1 else 0.0
        price_sigma = abs(latest_change - mean_chg) / std_chg if std_chg > 0 else 0.0

        # Volume ratio
        rolling_volumes = volumes[:-1]
        avg_vol = statistics.mean(rolling_volumes) if rolling_volumes else 1.0
        vol_ratio = volumes[-1] / avg_vol if avg_vol > 0 else 1.0

        is_price_anomaly = price_sigma >= self._sigma_threshold
        is_vol_anomaly = vol_ratio >= self._volume_ratio_threshold

        if not (is_price_anomaly or is_vol_anomaly):
            return None

        if is_price_anomaly and is_vol_anomaly:
            anomaly_type = "both"
        elif is_price_anomaly:
            anomaly_type = "price"
        else:
            anomaly_type = "volume"

        return AnomalyResult(
            symbol=symbol,
            market_id=market_id,
            price_move_pct=latest_change * 100,
            sigma=price_sigma,
            volume_ratio=vol_ratio,
            anomaly_type=anomaly_type,
        )

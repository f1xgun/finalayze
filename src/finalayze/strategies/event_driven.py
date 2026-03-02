"""Event-driven trading strategy using news sentiment (Layer 4)."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from finalayze.core.schemas import Signal, SignalDirection
from finalayze.strategies.base import BaseStrategy

if TYPE_CHECKING:
    from finalayze.core.schemas import Candle

_PRESETS_DIR = Path(__file__).parent / "presets"
_DEFAULT_MIN_SENTIMENT = 0.5
_DEFAULT_WEIGHT = Decimal("0.4")
# Maximum price move (as a fraction) since last candle before signal is suppressed.
# If news is already fully priced in, trading on it is futile.
_DEFAULT_MAX_PRICE_MOVE = 0.05


class EventDrivenStrategy(BaseStrategy):
    """News sentiment-driven strategy.

    Generates BUY when sentiment > min_sentiment threshold,
    SELL when sentiment < -min_sentiment.
    Confidence = min(1.0, abs(sentiment) * credibility).
    Falls back gracefully to None when sentiment == 0.
    """

    @property
    def name(self) -> str:
        """Strategy name."""
        return "event_driven"

    def supported_segments(self) -> list[str]:
        """Return segment IDs where event_driven strategy is enabled."""
        segments: list[str] = []
        for preset_path in sorted(_PRESETS_DIR.glob("*.yaml")):
            try:
                with preset_path.open() as f:
                    data = yaml.safe_load(f)
                if not isinstance(data, dict):
                    continue
                strategies = data.get("strategies", {})
                if not isinstance(strategies, dict):
                    continue
                ed_cfg = strategies.get("event_driven", {})
                if isinstance(ed_cfg, dict) and ed_cfg.get("enabled", False):
                    seg_id = data.get("segment_id")
                    if seg_id:
                        segments.append(str(seg_id))
            except (OSError, yaml.YAMLError):
                continue
        return segments

    def get_parameters(self, segment_id: str) -> dict[str, object]:
        """Load event_driven parameters from the YAML preset."""
        try:
            preset_path = _PRESETS_DIR / f"{segment_id}.yaml"
            with preset_path.open() as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                return {}
            strategies = data.get("strategies", {})
            if not isinstance(strategies, dict):
                return {}
            ed_cfg = strategies.get("event_driven", {})
            if not isinstance(ed_cfg, dict):
                return {}
            params = ed_cfg.get("params", {})
            return dict(params) if isinstance(params, dict) else {}
        except (FileNotFoundError, OSError, yaml.YAMLError):
            return {}

    def generate_signal(
        self,
        symbol: str,
        candles: list[Candle],
        segment_id: str,
        sentiment_score: float = 0.0,
        has_open_position: bool = False,  # noqa: ARG002
        credibility: float = 1.0,
    ) -> Signal | None:
        """Generate a trading signal based on news sentiment score.

        Args:
            symbol: Ticker symbol.
            candles: Recent OHLCV candles (used for context, not indicators).
            segment_id: The segment this symbol belongs to.
            sentiment_score: Sentiment in [-1.0, 1.0]. 0.0 → no signal.
            credibility: Source credibility [0.0, 1.0], scales confidence.

        Returns:
            Signal or None if sentiment is within neutral range.
        """
        params = self.get_parameters(segment_id)
        raw_min = params.get("min_sentiment", _DEFAULT_MIN_SENTIMENT)
        min_sentiment: float = (
            float(raw_min) if isinstance(raw_min, (int, float)) else _DEFAULT_MIN_SENTIMENT
        )

        abs_sent = abs(sentiment_score)
        if abs_sent < min_sentiment:
            return None

        # Price-move guard: if price has already moved more than the threshold
        # since the previous candle, the news is likely already priced in.
        if len(candles) >= 2:  # noqa: PLR2004
            raw_max_move = params.get("max_price_move", _DEFAULT_MAX_PRICE_MOVE)
            max_price_move: float = (
                float(raw_max_move)
                if isinstance(raw_max_move, (int, float))
                else _DEFAULT_MAX_PRICE_MOVE
            )
            prev_close = float(candles[-2].close)
            current_close = float(candles[-1].close)
            if prev_close > 0:
                price_move = abs(current_close - prev_close) / prev_close
                if price_move > max_price_move:
                    return None

        direction = SignalDirection.BUY if sentiment_score > 0 else SignalDirection.SELL
        confidence = min(1.0, abs_sent * credibility)

        return Signal(
            strategy_name=self.name,
            symbol=symbol,
            market_id=candles[-1].market_id if candles else "us",
            segment_id=segment_id,
            direction=direction,
            confidence=confidence,
            features={"sentiment": sentiment_score, "credibility": credibility},
            reasoning=f"News sentiment {sentiment_score:+.2f} (credibility={credibility:.2f})",
        )

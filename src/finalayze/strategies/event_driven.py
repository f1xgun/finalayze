"""Event-driven trading strategy using news sentiment (Layer 4)."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from finalayze.core.schemas import EventType, Signal, SignalDirection, SignalMetadata
from finalayze.strategies.base import BaseStrategy

if TYPE_CHECKING:
    from finalayze.core.schemas import Candle
    from finalayze.strategies.pead import EarningsSurprise

_PRESETS_DIR = Path(__file__).parent / "presets"
_DEFAULT_MIN_SENTIMENT = 0.5
_DEFAULT_WEIGHT = Decimal("0.4")
# Maximum price move (as a fraction) since last candle before signal is suppressed.
# If news is already fully priced in, trading on it is futile.
_DEFAULT_MAX_PRICE_MOVE = 0.05

# ── Earnings SUE path (Phase 60, INTG-01, D-02) ──────────────────────────────
# A self-resolving earnings event type mirroring PEADStrategy's calendar: the
# strategy resolves the active surprise from its own registered calendar by the
# latest candle's date, so neither the engine nor the combiner signature
# changes. This path is INDEPENDENT of sentiment_score (the backtest engine
# passes 0.0, so the news path is dead in backtest — Pitfall 1).
#
# SUE gate (Claude's discretion per D-04): a |sue_score| at/above this clears
# the gate. 0.75 is below the seeded ru_energy surprise (|sue| ~ 2.0) so an
# in-window event fires, yet high enough to ignore near-zero proxy noise.
_DEFAULT_SUE_THRESHOLD = 0.75
# Post-announcement drift window (bars), mirroring PEADStrategy (60d ≈ 1 quarter
# of trading days). A surprise older than this is no longer actionable.
_DEFAULT_DRIFT_WINDOW_BARS = 60
# Confidence scaling for the earnings signal. Tuned so a seeded |sue| ~ 2.0 at
# weight 0.15 clears ru_energy's min_combined_confidence (0.38) under "firing"
# renormalisation: confidence = min(_MAX, _BASE + (|sue| - threshold) * _SCALE).
_SUE_CONFIDENCE_BASE = 0.55
_SUE_CONFIDENCE_SCALE = 0.20
_SUE_MAX_CONFIDENCE = 0.95

# Sanctions proximity scores for Russian-listed equities.
# Higher values indicate greater exposure to sanctions-related risk,
# which reduces confidence on event-driven signals.
_SANCTIONS_PROXIMITY: dict[str, float] = {
    "GAZP": 0.8,
    "LKOH": 0.7,
    "ROSN": 0.7,
    "NVTK": 0.6,
    "SBER": 0.3,
    "VTBR": 0.5,
    "ALRS": 0.4,
    "NLMK": 0.5,
    "MGNT": 0.2,
    "YNDX": 0.3,
    "POLY": 0.6,
    "PHOR": 0.4,
    "MTSS": 0.2,
    "OZON": 0.3,
    "FIVE": 0.2,
}

# Event types that trigger sanctions proximity scaling.
_SANCTIONS_EVENT_TYPES = {"sanctions", "geopolitical"}


class EventDrivenStrategy(BaseStrategy):
    """News sentiment-driven strategy.

    Generates BUY when sentiment > min_sentiment threshold,
    SELL when sentiment < -min_sentiment.
    Confidence = min(1.0, abs(sentiment) * credibility).
    Falls back gracefully to None when sentiment == 0.
    """

    def __init__(
        self,
        sue_threshold: float = _DEFAULT_SUE_THRESHOLD,
        drift_window_bars: int = _DEFAULT_DRIFT_WINDOW_BARS,
    ) -> None:
        # Earnings SUE calendar (symbol -> list of surprises), resolved
        # per-bar exactly like PEADStrategy. Empty until a loader registers
        # surprises via add_earnings_surprise (run_iteration, ru_-gated).
        self._surprises: dict[str, list[EarningsSurprise]] = {}
        self._sue_threshold = sue_threshold
        self._drift_window_bars = drift_window_bars

    @property
    def name(self) -> str:
        """Strategy name."""
        return "event_driven"

    def add_earnings_surprise(self, surprise: EarningsSurprise) -> None:
        """Register an earnings surprise event (self-resolving calendar)."""
        self._surprises.setdefault(surprise.symbol, []).append(surprise)

    def reset(self) -> None:
        """Clear earnings calendar state between backtest runs."""
        self._surprises.clear()

    def _resolve_earnings_signal(
        self,
        symbol: str,
        candles: list[Candle],
        segment_id: str,
    ) -> Signal | None:
        """Resolve the active earnings surprise as-of the latest candle.

        Mirrors PEADStrategy's drift-window discipline verbatim (look-ahead
        guard): a future announcement (announcement_date > bar date) is
        skipped, and an event older than the drift window is skipped. This path
        does NOT read sentiment_score (D-02 / Pitfall 1). The emitted Signal
        carries the Phase-59 D-01 ``is_proxy`` label forward.
        """
        if not candles:
            return None
        surprises = self._surprises.get(symbol)
        if not surprises:
            return None

        current_candle = candles[-1]
        current_date = current_candle.timestamp

        best: EarningsSurprise | None = None
        for surprise in surprises:
            # Future event: silent (look-ahead guard).
            if current_date.date() < surprise.announcement_date.date():
                continue
            bars_since = sum(
                1 for c in candles if c.timestamp.date() > surprise.announcement_date.date()
            )
            # Out-of-drift-window: silent.
            if bars_since > self._drift_window_bars:
                continue
            if best is None or surprise.announcement_date > best.announcement_date:
                best = surprise

        if best is None:
            return None

        # Gate on SUE magnitude (independent of sentiment_score).
        if abs(best.sue_score) < self._sue_threshold:
            return None

        direction = SignalDirection.BUY if best.sue_score > 0 else SignalDirection.SELL
        excess = abs(best.sue_score) - self._sue_threshold
        confidence = min(
            _SUE_MAX_CONFIDENCE,
            _SUE_CONFIDENCE_BASE + excess * _SUE_CONFIDENCE_SCALE,
        )

        return Signal(
            strategy_name=self.name,
            symbol=symbol,
            market_id=current_candle.market_id,
            segment_id=segment_id,
            direction=direction,
            confidence=confidence,
            metadata=SignalMetadata(event_type=EventType.EARNINGS),
            strategy_payload={
                "sue_score": best.sue_score,
                "is_proxy": float(best.is_proxy),
            },
            reasoning=(f"earnings SUE proxy={best.sue_score:+.2f} (is_proxy={best.is_proxy})"),
        )

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
        event_type_code: float = 0.0,
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
        # Earnings SUE path (Phase 60, D-02): self-resolving from the registered
        # calendar by the latest candle's date. Checked FIRST and independent of
        # sentiment_score (the engine passes 0.0, so the news path below is dead
        # in backtest). Returns None when no in-window surprise clears the gate.
        earnings_signal = self._resolve_earnings_signal(symbol, candles, segment_id)
        if earnings_signal is not None:
            return earnings_signal

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

        strategy_payload: dict[str, float] = {
            "sentiment": sentiment_score,
            "credibility": credibility,
        }

        # Apply sanctions proximity scaling for segments with sanctions/geopolitical events.
        event_types = params.get("event_types", [])
        event_types_set = set(event_types) if isinstance(event_types, list) else set()
        if event_types_set & _SANCTIONS_EVENT_TYPES:
            proximity = _SANCTIONS_PROXIMITY.get(symbol, 0.0)
            confidence *= 1.0 - proximity * 0.5
            strategy_payload["sanctions_proximity"] = proximity

        return Signal(
            strategy_name=self.name,
            symbol=symbol,
            market_id=candles[-1].market_id if candles else "us",
            segment_id=segment_id,
            direction=direction,
            confidence=confidence,
            metadata=SignalMetadata(event_type=EventType(int(event_type_code))),
            strategy_payload=strategy_payload,
            reasoning=f"News sentiment {sentiment_score:+.2f} (credibility={credibility:.2f})",
        )

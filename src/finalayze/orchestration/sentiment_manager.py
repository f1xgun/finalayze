"""Sentiment cache management for trading loop (Layer 5 -- orchestrator).

Manages:
  - Per-ticker sentiment scores with exponential time-decay
  - Thread-safe access via threading.Lock
  - Redis cache fallback (if available)
  - Event-driven preset detection

Moved from trading_loop.py in Phase 1.3 (sentiment extraction).
"""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from finalayze.data.cache import RedisCache
    from finalayze.markets.instruments import InstrumentRegistry

# ── Constants ──────────────────────────────────────────────────────────────
_DEFAULT_SENTIMENT = 0.0
_SENTIMENT_HALF_LIFE_HOURS = 4.0
_SENTIMENT_DECAY_LAMBDA = math.log(2) / _SENTIMENT_HALF_LIFE_HOURS  # ~0.1733


class SentimentManager:
    """Manages sentiment cache for trading loop.

    Provides:
      - get_sentiment(seg_id, ticker=None): read with decay + Redis fallback
      - update_sentiment(seg_id, ticker, score, ts): thread-safe cache write
      - get_segment_tickers(seg_id): list tickers in segment
      - collect_active_segments(): list all segment IDs
      - is_event_driven_active(): cached check for event_driven presets
    """

    def __init__(
        self,
        registry: InstrumentRegistry,
        market_ids: list[str],
        cache: RedisCache | None = None,
    ):
        """Initialize sentiment manager.

        Args:
            registry: InstrumentRegistry to list instruments by market.
            market_ids: List of market IDs (e.g., ['us', 'moex']).
            cache: Optional RedisCache for sentiment persistence.
        """
        self._registry = registry
        self._market_ids = market_ids
        self._cache = cache

        # Per-ticker sentiment: (segment_id, ticker) -> (score, monotonic_timestamp)
        self._sentiment_cache: dict[tuple[str, str], tuple[float, float]] = {}
        self._sentiment_lock = threading.Lock()

        # Cached check: any segment has event_driven strategy enabled?
        self._event_driven_active: bool | None = None

    def get_segment_tickers(self, seg_id: str) -> list[str]:
        """Get ticker symbols for instruments in this segment."""
        return [
            instr.symbol
            for market_id in self._market_ids
            for instr in self._registry.list_by_market(market_id)
            if hasattr(instr, "segment_id") and instr.segment_id == seg_id
        ]

    def collect_active_segments(self) -> list[str]:
        """Collect distinct segment IDs across all markets."""
        return list(
            {
                seg
                for market_id in self._market_ids
                for instr in self._registry.list_by_market(market_id)
                if hasattr(instr, "segment_id") and instr.segment_id
                for seg in [instr.segment_id]
            }
        )

    def read_decayed_sentiment_unlocked(self, seg_id: str, ticker: str | None = None) -> float:
        """Read sentiment with exponential time-decay applied (internal, unlocked).

        If ticker is provided, reads per-ticker score.
        Falls back to segment average if no per-ticker entry.
        Must be called while holding _sentiment_lock.

        Args:
            seg_id: Segment ID.
            ticker: Optional ticker symbol (if None, returns segment average).

        Returns:
            Decayed sentiment score, or _DEFAULT_SENTIMENT if not found.
        """
        if ticker is not None:
            entry = self._sentiment_cache.get((seg_id, ticker))
            if entry is not None:
                score, ts = entry
                hours_elapsed = (time.monotonic() - ts) / 3600.0
                return score * math.exp(-_SENTIMENT_DECAY_LAMBDA * hours_elapsed)
            # Fallback: average of all per-ticker scores for this segment
            seg_scores = []
            for (s, _t), (score, ts) in self._sentiment_cache.items():
                if s == seg_id:
                    hours_elapsed = (time.monotonic() - ts) / 3600.0
                    seg_scores.append(score * math.exp(-_SENTIMENT_DECAY_LAMBDA * hours_elapsed))
            if seg_scores:
                return sum(seg_scores) / len(seg_scores)
            return _DEFAULT_SENTIMENT
        # Legacy: no ticker -- average all scores for segment
        seg_scores = []
        for (s, _t), (score, ts) in self._sentiment_cache.items():
            if s == seg_id:
                hours_elapsed = (time.monotonic() - ts) / 3600.0
                seg_scores.append(score * math.exp(-_SENTIMENT_DECAY_LAMBDA * hours_elapsed))
        if seg_scores:
            return sum(seg_scores) / len(seg_scores)
        return _DEFAULT_SENTIMENT

    def read_decayed_sentiment(self, seg_id: str, ticker: str | None = None) -> float:
        """Read sentiment with exponential time-decay applied (thread-safe).

        If ticker is provided, reads per-ticker score.
        Falls back to segment average if no per-ticker entry.

        Args:
            seg_id: Segment ID.
            ticker: Optional ticker symbol (if None, returns segment average).

        Returns:
            Decayed sentiment score, or _DEFAULT_SENTIMENT if not found.
        """
        with self._sentiment_lock:
            return self.read_decayed_sentiment_unlocked(seg_id, ticker)

    def get_sentiment(self, seg_id: str, ticker: str | None = None) -> float:
        """Read sentiment from Redis cache (if available) or in-memory fallback.

        Args:
            seg_id: Segment ID.
            ticker: Optional ticker symbol.

        Returns:
            Sentiment score with decay applied.
        """
        # Note: Redis cache is accessed asynchronously by trading_loop via _run_async
        # This method returns in-memory sentiment with decay applied
        with self._sentiment_lock:
            return self.read_decayed_sentiment_unlocked(seg_id, ticker)

    def update_sentiment(
        self, seg_id: str, ticker: str, score: float, ts: float | None = None
    ) -> None:
        """Update sentiment cache with new score.

        Called by news pipeline to update sentiment cache after analysis.

        Args:
            seg_id: Segment ID.
            ticker: Ticker symbol.
            score: New sentiment score.
            ts: Monotonic timestamp (if None, uses current time).
        """
        if ts is None:
            ts = time.monotonic()
        with self._sentiment_lock:
            cache_key = (seg_id, ticker)
            self._sentiment_cache[cache_key] = (score, ts)

    def is_event_driven_active(self) -> bool:
        """Check if any segment preset has event_driven strategy enabled.

        Caches result to avoid re-reading YAML files on every news cycle.

        Returns:
            True if any preset has event_driven enabled, False otherwise.
        """
        if self._event_driven_active is not None:
            return self._event_driven_active

        import yaml  # noqa: PLC0415

        presets_dir = Path(__file__).parent.parent / "strategies" / "presets"
        result = False
        try:
            for path in presets_dir.glob("*.yaml"):
                try:
                    with path.open() as f:
                        config = yaml.safe_load(f)
                    if isinstance(config, dict) and config.get("strategies", {}).get(
                        "event_driven", {}
                    ).get("enabled", False):
                        result = True
                        break
                except (OSError, yaml.YAMLError):
                    pass  # Caller will log if needed
        except OSError:
            pass  # Caller will log if needed

        self._event_driven_active = result
        return result

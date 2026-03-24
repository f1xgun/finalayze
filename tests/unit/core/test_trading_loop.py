"""Tests for TradingLoop news cycle skip guard and sentiment time-decay.

NEWS-01: _news_cycle returns immediately when no segment has event_driven enabled.
NEWS-02: Sentiment scores decay exponentially with a 4-hour half-life.
"""

from __future__ import annotations

import math
import time
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from finalayze.core.schemas import NewsArticle


def _make_loop(
    *,
    rss_fetcher: object | None = None,
    telegram_reader: object | None = None,
    news_impact_analyzer: object | None = None,
    sector_ticker_mapper: object | None = None,
    news_fetcher: object | None = None,
    settings: object | None = None,
) -> object:
    """Build a TradingLoop with mocked dependencies."""
    from finalayze.core.trading_loop import TradingLoop

    mock_settings = settings or MagicMock()
    if settings is None:
        mock_settings.news_cycle_minutes = 15
        mock_settings.news_poll_interval_minutes = 5
        mock_settings.strategy_cycle_minutes = 30
        mock_settings.daily_reset_hour_utc = 0
        mock_settings.max_position_pct = 0.1
        mock_settings.max_positions_per_market = 10
        mock_settings.daily_loss_limit_pct = 0.05
        mock_settings.kelly_fraction = 0.5
        mock_settings.ml_enabled = False
        mock_settings.telegram_channels = ["@test_channel"]

    mock_news_fetcher = news_fetcher or MagicMock()

    loop = TradingLoop(
        settings=mock_settings,
        fetchers={},
        news_fetcher=mock_news_fetcher,
        news_analyzer=MagicMock(),
        event_classifier=MagicMock(),
        impact_estimator=MagicMock(),
        strategy=MagicMock(),
        broker_router=MagicMock(),
        circuit_breakers={},
        cross_market_breaker=MagicMock(),
        alerter=MagicMock(),
        instrument_registry=MagicMock(),
        rss_fetcher=rss_fetcher,
        telegram_reader=telegram_reader,
        news_impact_analyzer=news_impact_analyzer,
        sector_ticker_mapper=sector_ticker_mapper,
    )
    return loop


class TestNewsCycleSkipGuard:
    """NEWS-01: _news_cycle returns immediately when event_driven is disabled everywhere."""

    def test_news_cycle_skips_when_no_event_driven_enabled(self) -> None:
        """_news_cycle returns immediately (no fetchers called) when
        _any_event_driven_enabled() returns False."""
        rss = MagicMock()
        loop = _make_loop(rss_fetcher=rss)

        # Force _any_event_driven_enabled to return False
        loop._any_event_driven_enabled = MagicMock(return_value=False)  # type: ignore[attr-defined]

        loop._news_cycle()  # type: ignore[attr-defined]

        # RSS fetcher should NOT have been called
        rss.fetch_news.assert_not_called()

    def test_news_cycle_proceeds_when_event_driven_enabled(self) -> None:
        """_news_cycle proceeds normally when at least one segment has event_driven enabled."""
        rss = MagicMock()
        rss.fetch_news.return_value = []
        loop = _make_loop(rss_fetcher=rss)

        # Force _any_event_driven_enabled to return True
        loop._any_event_driven_enabled = MagicMock(return_value=True)  # type: ignore[attr-defined]

        loop._news_cycle()  # type: ignore[attr-defined]

        # RSS fetcher SHOULD have been called
        rss.fetch_news.assert_called_once()

    def test_any_event_driven_enabled_reads_presets(self, tmp_path) -> None:
        """_any_event_driven_enabled() reads all segment preset YAMLs and returns True
        only if any has strategies.event_driven.enabled=true."""
        import yaml

        # Create two presets: one with event_driven disabled, one enabled
        presets_dir = tmp_path / "presets"
        presets_dir.mkdir()

        (presets_dir / "seg_a.yaml").write_text(yaml.dump({
            "strategies": {"event_driven": {"enabled": False}},
        }))
        (presets_dir / "seg_b.yaml").write_text(yaml.dump({
            "strategies": {"event_driven": {"enabled": True}},
        }))

        loop = _make_loop()
        # Patch the presets directory
        with patch(
            "finalayze.orchestration.trading_loop.Path.__truediv__",
        ):
            # Direct approach: patch the method internals
            loop._event_driven_active = None  # type: ignore[attr-defined]
            original_method = type(loop)._any_event_driven_enabled  # type: ignore[attr-defined]

            # We test by calling the method with a patched presets_dir
            with patch.object(
                type(loop),
                "_any_event_driven_enabled",
                wraps=original_method,
            ):
                # Simpler: just set the presets_dir and call
                pass

        # Simpler test: use the actual method with tmp_path presets
        loop._event_driven_active = None  # type: ignore[attr-defined]
        with patch(
            "finalayze.orchestration.trading_loop.Path",
        ) as mock_path_cls:
            # Make Path(__file__).parent.parent / "strategies" / "presets" -> tmp_path/presets
            mock_path_cls.return_value.parent.parent.__truediv__.return_value.__truediv__.return_value = presets_dir
            result = loop._any_event_driven_enabled()  # type: ignore[attr-defined]

        assert result is True

    def test_any_event_driven_enabled_returns_false_when_all_disabled(self, tmp_path) -> None:
        """Returns False when all presets have event_driven disabled or missing."""
        import yaml

        presets_dir = tmp_path / "presets"
        presets_dir.mkdir()

        (presets_dir / "seg_a.yaml").write_text(yaml.dump({
            "strategies": {"event_driven": {"enabled": False}},
        }))
        (presets_dir / "seg_b.yaml").write_text(yaml.dump({
            "strategies": {"momentum": {"enabled": True}},
        }))

        loop = _make_loop()
        loop._event_driven_active = None  # type: ignore[attr-defined]

        with patch(
            "finalayze.orchestration.trading_loop.Path",
        ) as mock_path_cls:
            mock_path_cls.return_value.parent.parent.__truediv__.return_value.__truediv__.return_value = presets_dir
            result = loop._any_event_driven_enabled()  # type: ignore[attr-defined]

        assert result is False

    def test_any_event_driven_enabled_caches_result(self, tmp_path) -> None:
        """_any_event_driven_enabled() caches result (does not re-read YAML every cycle)."""
        loop = _make_loop()
        # Pre-set the cached value
        loop._event_driven_active = False  # type: ignore[attr-defined]

        # Should return cached value without reading any files
        result = loop._any_event_driven_enabled()  # type: ignore[attr-defined]
        assert result is False


class TestSentimentTimeDecay:
    """NEWS-02: Sentiment scores decay exponentially with 4-hour half-life."""

    def test_sentiment_decay_at_zero_hours(self) -> None:
        """Sentiment score stored at time T, read at T=0, returns 100% of original."""
        loop = _make_loop()
        now = time.monotonic()

        with loop._sentiment_lock:  # type: ignore[attr-defined]
            loop._sentiment_cache[("test_seg", "SBER")] = (0.8, now)  # type: ignore[attr-defined]

        result = loop._read_decayed_sentiment("test_seg", "SBER")  # type: ignore[attr-defined]
        assert abs(result - 0.8) < 0.01

    def test_sentiment_decay_at_four_hours(self) -> None:
        """Sentiment score stored at time T, read at T+4h, returns ~50% of original."""
        loop = _make_loop()
        four_hours_ago = time.monotonic() - 4 * 3600

        with loop._sentiment_lock:  # type: ignore[attr-defined]
            loop._sentiment_cache[("test_seg", "SBER")] = (0.8, four_hours_ago)  # type: ignore[attr-defined]

        result = loop._read_decayed_sentiment("test_seg", "SBER")  # type: ignore[attr-defined]
        # half-life = 4h -> at 4h, should be ~50% = 0.4
        assert abs(result - 0.4) < 0.05

    def test_sentiment_decay_at_eight_hours(self) -> None:
        """Sentiment score stored at time T, read at T+8h, returns ~25% of original."""
        loop = _make_loop()
        eight_hours_ago = time.monotonic() - 8 * 3600

        with loop._sentiment_lock:  # type: ignore[attr-defined]
            loop._sentiment_cache[("test_seg", "SBER")] = (0.8, eight_hours_ago)  # type: ignore[attr-defined]

        result = loop._read_decayed_sentiment("test_seg", "SBER")  # type: ignore[attr-defined]
        # 2 half-lives -> ~25% = 0.2
        assert abs(result - 0.2) < 0.05

    def test_get_sentiment_applies_decay(self) -> None:
        """_get_sentiment applies decay before returning."""
        loop = _make_loop()
        four_hours_ago = time.monotonic() - 4 * 3600

        # No Redis cache
        loop._cache = None  # type: ignore[attr-defined]

        with loop._sentiment_lock:  # type: ignore[attr-defined]
            loop._sentiment_cache[("test_seg", "SBER")] = (0.8, four_hours_ago)  # type: ignore[attr-defined]

        result = loop._get_sentiment("test_seg", "SBER")  # type: ignore[attr-defined]
        # Should return decayed value (~0.4), not raw 0.8
        assert abs(result - 0.4) < 0.05

    def test_sentiment_default_for_missing_segment(self) -> None:
        """_read_decayed_sentiment returns default for missing segment."""
        loop = _make_loop()
        result = loop._read_decayed_sentiment("nonexistent", "SBER")  # type: ignore[attr-defined]
        assert result == 0.0

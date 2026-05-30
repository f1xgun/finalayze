"""Tests for loop_services factory."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, Mock

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class MockTradingLoopDeps:
    """Minimal mock of TradingLoopDeps for testing."""

    settings: Any = None
    fetchers: dict[str, Any] | None = None
    news_fetcher: Any = None
    news_analyzer: Any = None
    event_classifier: Any = None
    impact_estimator: Any = None
    strategy: Any = None
    broker_router: Any = None
    circuit_breakers: dict[str, Any] | None = None
    cross_market_breaker: Any = None
    alerter: Any = None
    instrument_registry: Any = None
    cache: Any = None
    ml_registry: Any = None
    fx_service: Any = None
    bond_cycle_processor: Any = None
    macro_cache: Any = None
    rss_fetcher: Any = None
    telegram_reader: Any = None
    news_impact_analyzer: Any = None
    sector_ticker_mapper: Any = None
    sandbox_monitor: Any = None
    health_monitor: Any = None
    metrics_collector: Any = None
    grpc_loop: Any = None
    kill_switch: Any = None
    meta_agent_runner: Any = None


class TestLoopServices:
    """Test suite for build_loop_services factory."""

    def test_build_loop_services_returns_all_six_services(self) -> None:
        """Verify build_loop_services returns LoopServices with all 6 non-None."""
        from finalayze.orchestration.loop_services import build_loop_services

        # Minimal mocks for deps and prerequisites
        deps = MockTradingLoopDeps(
            settings=Mock(database_url="sqlite:///:memory:"),
            fetchers={"moex": Mock()},
            broker_router=Mock(),
            alerter=Mock(),
            circuit_breakers={},
            cross_market_breaker=Mock(),
            instrument_registry=Mock(),
            macro_cache=Mock(),
            health_monitor=Mock(),
            sandbox_monitor=Mock(),
            metrics_collector=Mock(),
            cache=Mock(),
            rss_fetcher=Mock(),
            telegram_reader=Mock(),
            news_fetcher=Mock(),
            news_impact_analyzer=Mock(),
            news_analyzer=Mock(),
            sector_ticker_mapper=Mock(),
            strategy=Mock(),
            event_classifier=Mock(),
            impact_estimator=Mock(),
        )

        sentiment_mgr = Mock(collect_active_segments=Mock(return_value=["seg1", "seg2"]))
        kelly_sizer = Mock()
        pre_trade_checker = Mock()
        loss_limit_tracker = Mock()

        def mock_now() -> datetime:
            return datetime.now(UTC)

        def mock_run_async(coro: Any, *, timeout: int = 30) -> Any:
            return None

        services = build_loop_services(
            deps,
            sentiment_mgr=sentiment_mgr,
            kelly_sizer=kelly_sizer,
            pre_trade_checker=pre_trade_checker,
            loss_limit_tracker=loss_limit_tracker,
            now_fn=mock_now,
            run_async_fn=mock_run_async,
        )

        # Verify all 6 services are returned and non-None
        assert services.persistence is not None
        assert services.position_tracker is not None
        assert services.signal_executor is not None
        assert services.ml_retrainer is not None
        assert services.daily_reporter is not None
        assert services.news_pipeline is not None

    def test_persistence_injected_into_dependents(self) -> None:
        """Verify persistence is injected into all dependent services."""
        from finalayze.orchestration.loop_services import build_loop_services

        deps = MockTradingLoopDeps(
            settings=Mock(
                database_url="sqlite:///:memory:",
                effective_risk_limits=Mock(
                    return_value=Mock(
                        max_position_pct=Decimal("0.10"),
                        max_positions_per_market=10,
                        max_sector_concentration_pct=Decimal("0.20"),
                        min_cash_reserve_pct=Decimal("0.10"),
                        daily_loss_limit_pct=Decimal("0.05"),
                    )
                ),
            ),
            fetchers={"moex": Mock()},
            broker_router=Mock(),
            alerter=Mock(),
            circuit_breakers={},
            cross_market_breaker=Mock(),
            instrument_registry=Mock(),
            macro_cache=Mock(),
            health_monitor=Mock(),
            sandbox_monitor=Mock(),
            metrics_collector=Mock(),
            cache=Mock(),
            rss_fetcher=Mock(),
            telegram_reader=Mock(),
            news_fetcher=Mock(),
            news_impact_analyzer=Mock(),
            news_analyzer=Mock(),
            sector_ticker_mapper=Mock(),
            strategy=Mock(),
            event_classifier=Mock(),
            impact_estimator=Mock(),
            ml_registry=Mock(),
            bond_cycle_processor=Mock(),
            fx_service=Mock(),
        )

        sentiment_mgr = Mock(collect_active_segments=Mock(return_value=["seg1"]))
        kelly_sizer = Mock()
        pre_trade_checker = Mock()
        loss_limit_tracker = Mock()

        def mock_now() -> datetime:
            return datetime.now(UTC)

        def mock_run_async(coro: Any, *, timeout: int = 30) -> Any:
            return None

        services = build_loop_services(
            deps,
            sentiment_mgr=sentiment_mgr,
            kelly_sizer=kelly_sizer,
            pre_trade_checker=pre_trade_checker,
            loss_limit_tracker=loss_limit_tracker,
            now_fn=mock_now,
            run_async_fn=mock_run_async,
        )

        # The same persistence instance should be in all 4 dependents
        persistence = services.persistence
        assert services.position_tracker._persistence is persistence
        assert services.signal_executor._persistence is persistence
        assert services.daily_reporter._persistence is persistence
        assert services.news_pipeline._persistence is persistence

    def test_collect_segments_fn_wired(self) -> None:
        """Verify collect_active_segments from sentiment_mgr is wired to ml_retrainer."""
        from finalayze.orchestration.loop_services import build_loop_services

        deps = MockTradingLoopDeps(
            settings=Mock(database_url="sqlite:///:memory:"),
            fetchers={"moex": Mock()},
            broker_router=Mock(),
            alerter=Mock(),
            circuit_breakers={},
            cross_market_breaker=Mock(),
            instrument_registry=Mock(),
            macro_cache=Mock(),
            health_monitor=Mock(),
            sandbox_monitor=Mock(),
            metrics_collector=Mock(),
            cache=Mock(),
            rss_fetcher=Mock(),
            telegram_reader=Mock(),
            news_fetcher=Mock(),
            news_impact_analyzer=Mock(),
            news_analyzer=Mock(),
            sector_ticker_mapper=Mock(),
            strategy=Mock(),
            event_classifier=Mock(),
            impact_estimator=Mock(),
            ml_registry=Mock(),
        )

        collect_segments_spy = Mock(return_value=["seg1", "seg2", "seg3"])
        sentiment_mgr = Mock(collect_active_segments=collect_segments_spy)
        kelly_sizer = Mock()
        pre_trade_checker = Mock()
        loss_limit_tracker = Mock()

        def mock_now() -> datetime:
            return datetime.now(UTC)

        def mock_run_async(coro: Any, *, timeout: int = 30) -> Any:
            return None

        services = build_loop_services(
            deps,
            sentiment_mgr=sentiment_mgr,
            kelly_sizer=kelly_sizer,
            pre_trade_checker=pre_trade_checker,
            loss_limit_tracker=loss_limit_tracker,
            now_fn=mock_now,
            run_async_fn=mock_run_async,
        )

        # Verify that ml_retrainer received the collect_segments_fn
        ml_retrainer = services.ml_retrainer
        assert hasattr(ml_retrainer, "_collect_active_segments")
        assert ml_retrainer._collect_active_segments is collect_segments_spy

    def test_now_fn_wired_to_daily_reporter_and_ml_retrainer(self) -> None:
        """Verify now_fn is wired to both daily_reporter and ml_retrainer."""
        from finalayze.orchestration.loop_services import build_loop_services

        deps = MockTradingLoopDeps(
            settings=Mock(database_url="sqlite:///:memory:"),
            fetchers={"moex": Mock()},
            broker_router=Mock(),
            alerter=Mock(),
            circuit_breakers={},
            cross_market_breaker=Mock(),
            instrument_registry=Mock(),
            macro_cache=Mock(),
            health_monitor=Mock(),
            sandbox_monitor=Mock(),
            metrics_collector=Mock(),
            cache=Mock(),
            rss_fetcher=Mock(),
            telegram_reader=Mock(),
            news_fetcher=Mock(),
            news_impact_analyzer=Mock(),
            news_analyzer=Mock(),
            sector_ticker_mapper=Mock(),
            strategy=Mock(),
            event_classifier=Mock(),
            impact_estimator=Mock(),
            ml_registry=Mock(),
        )

        sentiment_mgr = Mock(collect_active_segments=Mock(return_value=["seg1"]))
        kelly_sizer = Mock()
        pre_trade_checker = Mock()
        loss_limit_tracker = Mock()

        now_fn_spy: Callable[[], datetime] = Mock(
            return_value=datetime.now(UTC), side_effect=lambda: datetime.now(UTC)
        )

        def mock_run_async(coro: Any, *, timeout: int = 30) -> Any:
            return None

        services = build_loop_services(
            deps,
            sentiment_mgr=sentiment_mgr,
            kelly_sizer=kelly_sizer,
            pre_trade_checker=pre_trade_checker,
            loss_limit_tracker=loss_limit_tracker,
            now_fn=now_fn_spy,
            run_async_fn=mock_run_async,
        )

        # Verify daily_reporter and ml_retrainer have now_fn wired
        assert services.daily_reporter._now is now_fn_spy
        assert services.ml_retrainer._now is now_fn_spy

    def test_run_async_fn_wired_to_news_pipeline(self) -> None:
        """Verify run_async_fn is wired to news_pipeline."""
        from finalayze.orchestration.loop_services import build_loop_services

        deps = MockTradingLoopDeps(
            settings=Mock(database_url="sqlite:///:memory:"),
            fetchers={"moex": Mock()},
            broker_router=Mock(),
            alerter=Mock(),
            circuit_breakers={},
            cross_market_breaker=Mock(),
            instrument_registry=Mock(),
            macro_cache=Mock(),
            health_monitor=Mock(),
            sandbox_monitor=Mock(),
            metrics_collector=Mock(),
            cache=Mock(),
            rss_fetcher=Mock(),
            telegram_reader=Mock(),
            news_fetcher=Mock(),
            news_impact_analyzer=Mock(),
            news_analyzer=Mock(),
            sector_ticker_mapper=Mock(),
            strategy=Mock(),
            event_classifier=Mock(),
            impact_estimator=Mock(),
        )

        sentiment_mgr = Mock(collect_active_segments=Mock(return_value=["seg1"]))
        kelly_sizer = Mock()
        pre_trade_checker = Mock()
        loss_limit_tracker = Mock()

        def mock_now() -> datetime:
            return datetime.now(UTC)

        run_async_fn_spy = Mock(return_value=None)

        services = build_loop_services(
            deps,
            sentiment_mgr=sentiment_mgr,
            kelly_sizer=kelly_sizer,
            pre_trade_checker=pre_trade_checker,
            loss_limit_tracker=loss_limit_tracker,
            now_fn=mock_now,
            run_async_fn=run_async_fn_spy,
        )

        # Verify news_pipeline has run_async_fn wired
        assert services.news_pipeline._async_loop_fn is run_async_fn_spy

    def test_loop_services_is_frozen_dataclass(self) -> None:
        """Verify LoopServices is a frozen dataclass."""
        from dataclasses import FrozenInstanceError

        from finalayze.orchestration.loop_services import LoopServices

        # Create a minimal instance
        mock_service = Mock()
        ls = LoopServices(
            persistence=mock_service,
            position_tracker=mock_service,
            signal_executor=mock_service,
            ml_retrainer=mock_service,
            daily_reporter=mock_service,
            news_pipeline=mock_service,
        )

        # Verify it's frozen (immutable)
        with pytest.raises(FrozenInstanceError):
            ls.persistence = Mock()

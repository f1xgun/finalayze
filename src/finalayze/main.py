"""FastAPI application entry point.

Layer 6 -- API / Dashboard layer.
"""

from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import structlog
from config.logging import setup_logging
from config.settings import get_settings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from finalayze.api.v1.router import api_router
from finalayze.core.modes import WorkMode

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

_settings = get_settings()
setup_logging(_settings.mode)
log = structlog.get_logger()

# Module-level reference for graceful shutdown
_trading_loop_instance: object | None = None
_trading_loop_thread: threading.Thread | None = None


@asynccontextmanager
async def lifespan(_application: FastAPI) -> AsyncIterator[None]:
    """Start TradingLoop in background thread for sandbox/real modes, shut down on exit."""
    global _trading_loop_instance, _trading_loop_thread  # noqa: PLW0603

    log.info("finalayze started", mode=_settings.mode.value)

    if _settings.mode in (WorkMode.SANDBOX, WorkMode.REAL):
        try:
            _trading_loop_instance = _build_trading_loop(_settings)
            if _trading_loop_instance is not None:
                # Wire real health probes
                from finalayze.api.v1.system import set_tinkoff_broker  # noqa: PLC0415
                from finalayze.execution.tinkoff_broker import TinkoffBroker  # noqa: PLC0415

                broker_router = getattr(_trading_loop_instance, "_broker_router", None)
                if broker_router is not None:
                    try:
                        moex_broker = broker_router.route("moex")
                        if isinstance(moex_broker, TinkoffBroker):
                            set_tinkoff_broker(moex_broker)
                            log.info("tinkoff_broker_wired_to_health")
                    except Exception:
                        log.debug("tinkoff_broker_health_wire_skipped", exc_info=True)

                _trading_loop_thread = threading.Thread(
                    target=_trading_loop_instance.start,  # type: ignore[union-attr]
                    daemon=True,
                    name="trading-loop",
                )
                _trading_loop_thread.start()
                log.info("trading_loop_started_in_background")
        except Exception:
            log.exception("trading_loop_build_failed")

    yield

    # Shutdown
    if _trading_loop_instance is not None:
        try:
            _trading_loop_instance.stop()  # type: ignore[union-attr]
            log.info("trading_loop_stopped")
        except Exception:
            log.debug("trading_loop_stop_failed", exc_info=True)
    if _trading_loop_thread is not None and _trading_loop_thread.is_alive():
        _trading_loop_thread.join(timeout=10)


def _build_trading_loop(settings: object) -> object | None:
    """Build TradingLoop with all dependencies. Returns None on failure.

    Reuses the full wiring from scripts/run_sandbox.py via subprocess
    bootstrap module. All component construction is done here so Docker
    can run both API + TradingLoop in one process.
    """
    try:
        import asyncio  # noqa: PLC0415
        import os  # noqa: PLC0415

        from finalayze.analysis.event_classifier import EventClassifier  # noqa: PLC0415
        from finalayze.analysis.impact_estimator import ImpactEstimator  # noqa: PLC0415
        from finalayze.analysis.news_analyzer import NewsAnalyzer  # noqa: PLC0415
        from finalayze.core.alerts import TelegramAlerter  # noqa: PLC0415
        from finalayze.core.trading_loop import TradingLoop  # noqa: PLC0415
        from finalayze.data.fetchers.newsapi import NewsApiFetcher  # noqa: PLC0415
        from finalayze.data.fetchers.tinkoff_data import TinkoffFetcher  # noqa: PLC0415
        from finalayze.execution.broker_router import BrokerRouter  # noqa: PLC0415
        from finalayze.execution.retry import RetryPolicy  # noqa: PLC0415
        from finalayze.execution.tinkoff_broker import TinkoffBroker  # noqa: PLC0415
        from finalayze.markets.instruments import (  # noqa: PLC0415
            Instrument,
            InstrumentRegistry,
        )
        from finalayze.risk.circuit_breaker import (  # noqa: PLC0415
            CircuitBreaker,
            CrossMarketCircuitBreaker,
        )
        from finalayze.strategies.combiner import StrategyCombiner  # noqa: PLC0415
        from finalayze.strategies.dual_momentum import DualMomentumStrategy  # noqa: PLC0415
        from finalayze.strategies.mean_reversion import MeanReversionStrategy  # noqa: PLC0415
        from finalayze.strategies.momentum import MomentumStrategy  # noqa: PLC0415
        from finalayze.strategies.rsi2_connors import RSI2ConnorsStrategy  # noqa: PLC0415

        # Force native gRPC DNS resolver
        os.environ.setdefault("GRPC_DNS_RESOLVER", "native")

        tinkoff_token = getattr(settings, "tinkoff_token", "") or ""
        is_sandbox = getattr(settings, "tinkoff_sandbox", True)

        # ── Instrument Registry ──────────────────────────────────────────
        from config.segments import DEFAULT_SEGMENTS  # noqa: PLC0415

        registry = InstrumentRegistry()
        moex_segments = [s for s in DEFAULT_SEGMENTS if s.market == "moex"]
        for seg in moex_segments:
            for sym in seg.symbols:
                registry.register(
                    Instrument(
                        symbol=sym,
                        market_id="moex",
                        name=sym,
                        instrument_type=seg.instrument_type,
                        currency=seg.currency,
                        segment_id=seg.segment_id,
                    )
                )

        # Discover MOEX shares via T-Bank API for FIGI resolution
        if tinkoff_token:
            from t_tech.invest import AsyncClient  # noqa: PLC0415

            target = (
                "sandbox-invest-public-api.tbank.ru:443"
                if is_sandbox
                else "invest-public-api.tbank.ru:443"
            )

            async def _discover(token: str) -> list[dict[str, object]]:
                client = AsyncClient(token, target=target)
                discovered: list[dict[str, object]] = []
                async with client as services:
                    resp = await services.instruments.shares()
                    for share in resp.instruments:
                        if not getattr(share, "api_trade_available_flag", False):
                            continue
                        if getattr(share, "class_code", "") != "TQBR":
                            continue
                        discovered.append(
                            {
                                "ticker": share.ticker,
                                "figi": share.figi,
                                "name": share.name,
                                "lot": share.lot,
                            }
                        )
                return discovered

            # Can't use asyncio.run() inside uvicorn (event loop already running).
            # Use a dedicated thread with its own loop for the sync gRPC call.
            import concurrent.futures  # noqa: PLC0415

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                all_shares = pool.submit(asyncio.run, _discover(tinkoff_token)).result(timeout=30)
            share_by_ticker = {str(s["ticker"]): s for s in all_shares}
            configured_symbols: set[str] = set()
            for seg in moex_segments:
                if seg.instrument_type == "stock":
                    configured_symbols.update(seg.symbols)
            for sym in configured_symbols:
                if sym in share_by_ticker:
                    share = share_by_ticker[sym]
                    try:
                        existing = registry.get(sym, "moex")
                    except Exception:
                        continue
                    registry.register(
                        Instrument(
                            symbol=existing.symbol,
                            market_id="moex",
                            name=str(share["name"]),
                            instrument_type=existing.instrument_type,
                            figi=str(share["figi"]),
                            lot_size=int(share["lot"]),
                            currency=existing.currency,
                            segment_id=existing.segment_id,
                        )
                    )
            log.info("moex_shares_discovered", count=len(all_shares))

        # ── Data Fetcher ─────────────────────────────────────────────────
        fetchers: dict[str, object] = {}
        if tinkoff_token:
            tinkoff_fetcher = TinkoffFetcher(
                token=tinkoff_token,
                registry=registry,
                sandbox=is_sandbox,
            )
            fetchers["moex"] = tinkoff_fetcher

        # ── Broker ───────────────────────────────────────────────────────
        retry_policy = RetryPolicy(max_retries=3, base_delay=1.0)
        brokers: dict[str, TinkoffBroker] = {}
        if tinkoff_token:
            brokers["moex"] = TinkoffBroker(
                token=tinkoff_token,
                registry=registry,
                sandbox=is_sandbox,
                retry_policy=retry_policy,
            )
            brokers["moex_bonds"] = TinkoffBroker(
                token=tinkoff_token,
                registry=registry,
                sandbox=is_sandbox,
                retry_policy=retry_policy,
            )
        broker_router = BrokerRouter(brokers=brokers)

        # ── Strategies ───────────────────────────────────────────────────
        strategies_list = [
            MomentumStrategy(),
            DualMomentumStrategy(),
            MeanReversionStrategy(),
            RSI2ConnorsStrategy(),
        ]
        combiner = StrategyCombiner(strategies=strategies_list)

        # ── Risk ─────────────────────────────────────────────────────────
        _limits = settings.effective_risk_limits()
        circuit_breakers: dict[str, CircuitBreaker] = {
            "moex": CircuitBreaker(
                market_id="moex",
                l1_threshold=_limits.circuit_breaker_l1,
                l2_threshold=_limits.circuit_breaker_l2,
                l3_threshold=_limits.circuit_breaker_l3,
            ),
        }
        cross_market_breaker = CrossMarketCircuitBreaker()  # uses _DEFAULT_CROSS_HALT=0.10

        # ── Alerting ────────────────────────────────────────────────────
        alerter = TelegramAlerter(
            getattr(settings, "telegram_bot_token", "") or "",
            getattr(settings, "telegram_chat_id", "") or "",
        )

        # ── News Analysis ────────────────────────────────────────────────
        from finalayze.analysis.llm_client import LLMClient  # noqa: PLC0415

        class _StubLLMClient(LLMClient):
            async def complete(self, prompt: str, system: str) -> str:  # noqa: ARG002
                return '{"sentiment": 0.0, "confidence": 0.0, "reasoning": "stub"}'

        _has_llm = bool(
            getattr(settings, "llm_api_key", "") or getattr(settings, "anthropic_api_key", "")
        )
        if _has_llm:
            from finalayze.analysis.llm_client import create_llm_client  # noqa: PLC0415

            llm_client = create_llm_client(settings)
        else:
            llm_client = _StubLLMClient()

        news_analyzer = NewsAnalyzer(llm_client=llm_client)
        event_classifier = EventClassifier(llm_client=llm_client)
        from finalayze.analysis.combined_analyzer import CombinedNewsAnalyzer  # noqa: PLC0415

        combined_analyzer = CombinedNewsAnalyzer(llm_client=llm_client)
        impact_estimator = ImpactEstimator()
        _has_news = bool(getattr(settings, "newsapi_api_key", ""))
        news_fetcher = (
            NewsApiFetcher(api_key=settings.newsapi_api_key)  # type: ignore[union-attr]
            if _has_news
            else NewsApiFetcher(api_key="")
        )

        # ── Entity Extractor (LLM-based MOEX ticker extraction) ──────────
        from finalayze.analysis.entity_extractor import EntityExtractor  # noqa: PLC0415

        entity_extractor = (
            EntityExtractor(llm_client) if not isinstance(llm_client, _StubLLMClient) else None
        )

        # ── RSS + Telegram Fetchers ──────────────────────────────────────
        from finalayze.data.fetchers.rss_fetcher import RssNewsFetcher  # noqa: PLC0415
        from finalayze.data.fetchers.telegram_reader import TelegramChannelReader  # noqa: PLC0415

        _rss_urls = getattr(settings, "news_rss_urls", []) or []
        rss_fetcher = RssNewsFetcher(feed_urls=_rss_urls) if _rss_urls else None

        _tg_channels = getattr(settings, "telegram_channels", []) or []
        telegram_reader = TelegramChannelReader(channels=_tg_channels) if _tg_channels else None

        # ── Sandbox Monitoring ─────────────────────────────────────────
        sandbox_monitor = None
        if settings.mode == WorkMode.SANDBOX:
            from finalayze.monitoring.sandbox_monitor import SandboxMonitorService  # noqa: PLC0415

            sandbox_monitor = SandboxMonitorService(alerter=alerter, market_id="moex")

        # ── Build TradingLoop ────────────────────────────────────────────
        loop = TradingLoop(
            settings=settings,  # type: ignore[arg-type]
            fetchers=fetchers,
            news_fetcher=news_fetcher,
            news_analyzer=news_analyzer,
            event_classifier=event_classifier,
            impact_estimator=impact_estimator,
            strategy=combiner,
            broker_router=broker_router,
            circuit_breakers=circuit_breakers,
            cross_market_breaker=cross_market_breaker,
            alerter=alerter,
            instrument_registry=registry,
            rss_fetcher=rss_fetcher,
            telegram_reader=telegram_reader,
            entity_extractor=entity_extractor,
            combined_analyzer=combined_analyzer,
            sandbox_monitor=sandbox_monitor,
        )
        log.info(
            "trading_loop_built",
            markets=broker_router.registered_markets,
            instruments=len(registry.list_by_market("moex")),
            strategies=len(strategies_list),
        )
        return loop
    except Exception:
        log.exception("trading_loop_build_failed")
        return None


def create_app() -> FastAPI:
    """Construct and configure the FastAPI application."""
    application = FastAPI(title="Finalayze", version="0.1.0", lifespan=lifespan)
    settings = get_settings()
    allowed_origins = settings.cors_origins or []
    application.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Authorization", "X-API-Key"],
    )
    application.include_router(api_router, prefix="/api/v1")

    # Mount Telegram webhook router when bot token and webhook secret are configured
    if settings.telegram_bot_token and settings.telegram_webhook_secret:
        from finalayze.api.v1.telegram import create_telegram_router  # noqa: PLC0415
        from finalayze.core.alerts import TelegramAlerter  # noqa: PLC0415
        from finalayze.core.telegram_bot import TelegramBotHandler  # noqa: PLC0415

        alerter = TelegramAlerter(settings.telegram_bot_token, settings.telegram_chat_id)
        bot_handler = TelegramBotHandler(
            alerter=alerter,
            broker_router=None,  # type: ignore[arg-type] -- wired in TradingLoop startup
            circuit_breakers={},
            settings=settings,  # type: ignore[arg-type]
        )
        telegram_router = create_telegram_router(bot_handler, settings.telegram_webhook_secret)
        application.include_router(telegram_router)
        log.info("telegram_webhook_mounted", path="/api/telegram/webhook")

    # Prometheus HTTP metrics -- no auth (internal network only)
    Instrumentator().instrument(application).expose(
        application, endpoint="/metrics", include_in_schema=False
    )
    return application


app = create_app()

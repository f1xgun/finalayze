"""Bootstrap TradingLoop and all dependencies.

Layer 6 -- API / Dashboard layer. Extracted from main.py to enable
modular testing and cleaner separation of concerns.
"""

from __future__ import annotations

from typing import Any


def build_trading_loop(settings: Any) -> Any | None:  # noqa: PLR0912, PLR0915
    """Build TradingLoop with all dependencies. Returns None on failure.

    Reuses the full wiring from scripts/run_sandbox.py via subprocess
    bootstrap module. All component construction is done here so Docker
    can run both API + TradingLoop in one process.
    """
    try:
        import os  # noqa: PLC0415

        import structlog  # noqa: PLC0415

        from finalayze.analysis.event_classifier import EventClassifier  # noqa: PLC0415
        from finalayze.analysis.impact_estimator import ImpactEstimator  # noqa: PLC0415
        from finalayze.analysis.llm_client import LLMClient  # noqa: PLC0415
        from finalayze.analysis.news_analyzer import NewsAnalyzer  # noqa: PLC0415
        from finalayze.api.alerts import TelegramAlerter  # noqa: PLC0415
        from finalayze.core.modes import WorkMode  # noqa: PLC0415
        from finalayze.data.fetchers.caching import CachingFetcher  # noqa: PLC0415
        from finalayze.data.fetchers.newsapi import NewsApiFetcher  # noqa: PLC0415
        from finalayze.data.fetchers.tinkoff_data import TinkoffFetcher  # noqa: PLC0415
        from finalayze.data.rate_limiter import RateLimiter  # noqa: PLC0415
        from finalayze.execution.broker_router import BrokerRouter  # noqa: PLC0415
        from finalayze.execution.retry import RetryPolicy  # noqa: PLC0415
        from finalayze.execution.tinkoff_broker import TinkoffBroker  # noqa: PLC0415
        from finalayze.markets.instruments import (  # noqa: PLC0415
            Instrument,
            InstrumentRegistry,
        )
        from finalayze.orchestration.trading_loop import (  # noqa: PLC0415
            TradingLoop,
            TradingLoopDeps,
        )
        from finalayze.risk.circuit_breaker import (  # noqa: PLC0415
            CircuitBreaker,
            CrossMarketCircuitBreaker,
        )
        from finalayze.strategies.combiner import StrategyCombiner  # noqa: PLC0415
        from finalayze.strategies.dual_momentum import DualMomentumStrategy  # noqa: PLC0415
        from finalayze.strategies.mean_reversion import MeanReversionStrategy  # noqa: PLC0415
        from finalayze.strategies.momentum import MomentumStrategy  # noqa: PLC0415
        from finalayze.strategies.preset_validator import (  # noqa: PLC0415
            log_preset_issues,
            validate_presets,
        )
        from finalayze.strategies.rsi2_connors import RSI2ConnorsStrategy  # noqa: PLC0415

        log = structlog.get_logger()

        # Force native gRPC DNS resolver
        os.environ.setdefault("GRPC_DNS_RESOLVER", "native")

        tinkoff_token = getattr(settings, "tinkoff_token", "") or ""
        is_sandbox = getattr(settings, "tinkoff_sandbox", True)

        # ── Instrument Registry ──────────────────────────────────────────
        from config.segments import DEFAULT_SEGMENTS  # noqa: PLC0415

        registry = InstrumentRegistry()
        moex_segments = [s for s in DEFAULT_SEGMENTS if s.enabled and s.market == "moex"]
        for seg in moex_segments:
            for sym in seg.symbols:
                registry.register(
                    Instrument(
                        symbol=sym,
                        market_id="moex",
                        name=sym,
                        instrument_type=seg.instrument_type,  # type: ignore[arg-type]
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
            import asyncio  # noqa: PLC0415
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
                    except Exception:  # noqa: S112
                        continue
                    registry.register(
                        Instrument(
                            symbol=existing.symbol,
                            market_id="moex",
                            name=str(share["name"]),
                            instrument_type=existing.instrument_type,
                            figi=str(share["figi"]),
                            lot_size=int(share["lot"]),  # type: ignore[call-overload]
                            currency=existing.currency,
                            segment_id=existing.segment_id,
                        )
                    )
            log.info("moex_shares_discovered", count=len(all_shares))

        # ── Data Fetcher ─────────────────────────────────────────────────
        fetchers: dict[str, object] = {}
        if tinkoff_token:
            _tbank_rate_limiter = RateLimiter(name="tbank", rate=4.0)
            tinkoff_fetcher = TinkoffFetcher(
                token=tinkoff_token,
                registry=registry,
                sandbox=is_sandbox,
                rate_limiter=_tbank_rate_limiter,
            )
            caching_fetcher = CachingFetcher(delegate=tinkoff_fetcher)
            fetchers["moex"] = caching_fetcher

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
        broker_router = BrokerRouter(brokers=brokers)  # type: ignore[arg-type]

        # ── Strategies ───────────────────────────────────────────────────
        strategies_list = [
            MomentumStrategy(),
            DualMomentumStrategy(),
            MeanReversionStrategy(),
            RSI2ConnorsStrategy(),
        ]
        combiner = StrategyCombiner(strategies=strategies_list)

        # Surface silent preset schema drift (typo'd keys, bad Decimal values,
        # unknown normalize_mode) before any trades execute. Logs only;
        # per-segment fail-soft handling in the combiner remains the guardrail.
        log_preset_issues(validate_presets(combiner.presets_dir), log)

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
        from typing import TypeVar  # noqa: PLC0415

        from pydantic import BaseModel  # noqa: PLC0415

        T = TypeVar("T", bound=BaseModel)

        class _StubLLMClient(LLMClient):
            """Stub LLM client for when no API key is configured."""

            async def complete(
                self,
                prompt: str,  # noqa: ARG002
                system: str,  # noqa: ARG002
                *,
                json_mode: bool = False,  # noqa: ARG002
                max_tokens: int | None = None,  # noqa: ARG002
            ) -> str:
                return '{"sentiment": 0.0, "confidence": 0.0, "reasoning": "stub"}'

            async def parse_structured(
                self,
                prompt: str,  # noqa: ARG002
                system: str,  # noqa: ARG002
                response_model: type[T],
                *,
                max_tokens: int | None = None,  # noqa: ARG002
            ) -> T:
                """Stub implementation that returns default instance."""
                return response_model()

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
        impact_estimator = ImpactEstimator()
        _has_news = bool(getattr(settings, "newsapi_api_key", ""))
        news_fetcher = (
            NewsApiFetcher(api_key=settings.newsapi_api_key)
            if _has_news
            else NewsApiFetcher(api_key="")
        )

        # ── News Impact Analyzer (single-call LLM pipeline) ──────────
        from finalayze.analysis.news_impact_analyzer import NewsImpactAnalyzer  # noqa: PLC0415
        from finalayze.analysis.sector_ticker_mapper import SectorTickerMapper  # noqa: PLC0415

        news_impact_analyzer = (
            NewsImpactAnalyzer(llm_client) if not isinstance(llm_client, _StubLLMClient) else None
        )
        sector_ticker_mapper = SectorTickerMapper()

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

        # ── Kill Switch ───────────────────────────────────────────────
        from pathlib import Path  # noqa: PLC0415

        # Build TradingLoop first, then create KillSwitch that references it
        # ── Meta-Agent Runner ────────────────────────────────────────────
        from finalayze.api.metrics import MetricsCollector  # noqa: PLC0415
        from finalayze.core.kill_switch import KillSwitch  # noqa: PLC0415

        meta_agent_runner = None
        if getattr(settings, "meta_agent_enabled", False):
            from finalayze.meta_agent.runner import MetaAgentRunner  # noqa: PLC0415

            meta_agent_runner = MetaAgentRunner(
                settings=settings,
                persistence=None,  # wired after persistence is available in loop.start()
            )

        # ── Build TradingLoop ────────────────────────────────────────────
        loop = TradingLoop(
            TradingLoopDeps(
                settings=settings,
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
                news_impact_analyzer=news_impact_analyzer,
                sector_ticker_mapper=sector_ticker_mapper,
                sandbox_monitor=sandbox_monitor,
                metrics_collector=MetricsCollector,
                meta_agent_runner=meta_agent_runner,
            )
        )
        # Wire persistence + executor + approver into meta-agent runner.
        if meta_agent_runner is not None:
            from finalayze.meta_agent.approver import MetaAgentApprover  # noqa: PLC0415
            from finalayze.meta_agent.executor import ActionExecutor  # noqa: PLC0415

            meta_agent_runner._persistence = loop._persistence
            _executor = ActionExecutor(
                settings=settings,
                alerter=alerter,
                persistence=loop._persistence,
            )
            meta_agent_runner._executor = _executor
            meta_agent_runner._approver = MetaAgentApprover(
                executor=_executor,
                persistence=loop._persistence,
            )

        # ── Create KillSwitch (after loop exists) ────────────────────
        kill_switch = KillSwitch(
            broker_router=broker_router,
            trading_loop=loop,
            circuit_breakers=circuit_breakers,
            alerter=alerter,
            flag_path=Path(
                getattr(settings, "kill_switch_flag_path", "/tmp/finalayze_killed"),  # noqa: S108
            ),
        )

        if kill_switch.is_killed:
            log.critical(
                "system_previously_killed",
                msg="System was previously killed. Clear flag to restart.",
                flag_path=str(kill_switch._flag_path),
            )
            return None

        # Store kill_switch on loop for access from lifespan
        loop._kill_switch = kill_switch
        loop._circuit_breakers = circuit_breakers
        loop._alerter_ref = alerter
        if meta_agent_runner is not None:
            from finalayze.meta_agent.killswitch import (  # noqa: PLC0415
                Killswitch as MetaKillswitch,
            )

            meta_agent_runner.killswitch = MetaKillswitch(
                scheduler=loop._scheduler,
                settings_provider=lambda: settings,
            )

        log.info(
            "trading_loop_built",
            markets=broker_router.registered_markets,
            instruments=len(registry.list_by_market("moex")),
            strategies=len(strategies_list),
        )
        return loop
    except Exception:
        log = structlog.get_logger()
        log.exception("trading_loop_build_failed")
        return None

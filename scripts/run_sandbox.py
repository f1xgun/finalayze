#!/usr/bin/env python3
"""Bootstrap script — start the trading system in Tinkoff Sandbox mode.

Wires all components (data, analysis, strategies, risk, execution) and
launches the TradingLoop orchestrator with APScheduler.

Usage:
    FINALAYZE_MODE=sandbox FINALAYZE_TINKOFF_TOKEN=<token> uv run python scripts/run_sandbox.py

Required env vars (via .env or shell):
    FINALAYZE_TINKOFF_TOKEN   — T-Bank Invest sandbox API token
    FINALAYZE_MODE            — must be "sandbox"

Optional env vars:
    FINALAYZE_LLM_API_KEY         — for news sentiment (skipped if empty)
    FINALAYZE_TELEGRAM_BOT_TOKEN  — for Telegram alerts (no-op if empty)
    FINALAYZE_TELEGRAM_CHAT_ID    — Telegram chat to send alerts to
    FINALAYZE_STRATEGY_CYCLE_MINUTES  — strategy cycle interval (default 60)
    FINALAYZE_NEWS_CYCLE_MINUTES      — news cycle interval (default 30)
"""

from __future__ import annotations

import os
import signal
import sys
from pathlib import Path

# Ensure project root is on sys.path for config/ imports
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)

# gRPC env vars MUST be set before importing grpc (via t_tech.invest).
# C-ares DNS resolver may fail; force native (system) resolver.
os.environ.setdefault("GRPC_DNS_RESOLVER", "native")
# T-Bank uses Russian Trusted Root CA not in standard CA bundles.
_GRPC_ROOTS = Path(_PROJECT_ROOT) / "certs" / "grpc_roots.pem"
if _GRPC_ROOTS.exists():
    os.environ.setdefault("GRPC_DEFAULT_SSL_ROOTS_FILE_PATH", str(_GRPC_ROOTS))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv(Path(_PROJECT_ROOT) / ".env")

import structlog
from config.logging import setup_logging
from config.segments import DEFAULT_SEGMENTS
from config.settings import Settings

from finalayze.core.modes import WorkMode

# ── Settings & Logging ────────────────────────────────────────────────────

settings = Settings()
setup_logging(settings.mode)
_log = structlog.get_logger()

if settings.mode != WorkMode.SANDBOX:
    _log.error("run_sandbox requires FINALAYZE_MODE=sandbox", current_mode=settings.mode.value)
    sys.exit(1)

if not settings.tinkoff_token:
    _log.error("FINALAYZE_TINKOFF_TOKEN is required for sandbox mode")
    sys.exit(1)

_log.info(
    "sandbox_bootstrap_start",
    mode=settings.mode.value,
    strategy_cycle_min=settings.strategy_cycle_minutes,
    news_cycle_min=settings.news_cycle_minutes,
    ml_enabled=settings.ml_enabled,
)

# ── Instrument Registry ──────────────────────────────────────────────────

from finalayze.markets.instruments import Instrument, InstrumentRegistry

registry = InstrumentRegistry()

# Register MOEX segments from config
_moex_segments = [s for s in DEFAULT_SEGMENTS if s.market == "moex"]
for seg in _moex_segments:
    for sym in seg.symbols:
        instrument = Instrument(
            symbol=sym,
            market_id="moex",
            name=sym,
            instrument_type=seg.instrument_type,  # type: ignore[arg-type]
            currency=seg.currency,
            segment_id=seg.segment_id,
        )
        registry.register(instrument)

_log.info(
    "instruments_registered",
    moex_count=len(registry.list_by_market("moex")),
    segments=[s.segment_id for s in _moex_segments],
)

# ── MOEX Instrument Discovery ────────────────────────────────────────────
# Discover ALL MOEX shares via T-Bank API (replaces per-symbol FIGI resolution).
# Bonds use hardcoded FIGIs from DEFAULT_MOEX_OFZ_INSTRUMENTS.

import asyncio

from t_tech.invest import AsyncClient

from finalayze.core.exceptions import InstrumentNotFoundError as _InstNotFound
from finalayze.markets.instruments import DEFAULT_MOEX_OFZ_INSTRUMENTS

_TBANK_GRPC_SANDBOX_TARGET = "sandbox-invest-public-api.tbank.ru:443"

# Register OFZ bonds with hardcoded FIGIs and full metadata
_bond_segment_symbols: set[str] = set()
for _seg in _moex_segments:
    if _seg.instrument_type == "bond":
        _bond_segment_symbols.update(_seg.symbols)

for _bond in DEFAULT_MOEX_OFZ_INSTRUMENTS:
    if _bond.symbol in _bond_segment_symbols:
        try:
            _existing = registry.get(_bond.symbol, "moex")
            _seg_id = _existing.segment_id
        except _InstNotFound:
            _seg_id = ""
        registry.register(
            Instrument(
                symbol=_bond.symbol,
                market_id="moex",
                name=_bond.name,
                instrument_type="bond",
                figi=_bond.figi,
                lot_size=_bond.lot_size,
                currency="RUB",
                segment_id=_seg_id,
                face_value=_bond.face_value,
                coupon_rate=_bond.coupon_rate,
                coupon_frequency=_bond.coupon_frequency,
                maturity_date=_bond.maturity_date,
                floating_coupon=_bond.floating_coupon,
            )
        )


async def _discover_moex_shares(token: str) -> list[dict[str, object]]:
    """Discover all MOEX shares available for API trading via T-Bank.

    Queries the T-Bank Invest API for all shares on the TQBR board
    (main T+ equities) and returns their metadata including FIGIs.
    """
    client = AsyncClient(token, target=_TBANK_GRPC_SANDBOX_TARGET)
    discovered: list[dict[str, object]] = []
    async with client as services:
        resp = await services.instruments.shares()
        for share in resp.instruments:
            if not getattr(share, "api_trade_available_flag", False):
                continue
            class_code = getattr(share, "class_code", "")
            if class_code != "TQBR":
                continue
            discovered.append(
                {
                    "ticker": share.ticker,
                    "figi": share.figi,
                    "name": share.name,
                    "lot": share.lot,
                    "currency": getattr(share, "currency", "rub"),
                }
            )
    return discovered


_log.info("discovering_moex_shares")
_all_moex_shares = asyncio.run(_discover_moex_shares(settings.tinkoff_token))
_log.info("moex_shares_discovered", count=len(_all_moex_shares))

# Build ticker->share lookup
_share_by_ticker: dict[str, dict[str, object]] = {str(s["ticker"]): s for s in _all_moex_shares}

# Update segment-registered stock instruments with FIGI and lot_size from API
_configured_symbols: set[str] = set()
for _seg in _moex_segments:
    if _seg.instrument_type == "stock":
        _configured_symbols.update(_seg.symbols)

for _sym in sorted(_configured_symbols):
    if _sym in _share_by_ticker:
        _share = _share_by_ticker[_sym]
        try:
            _existing_inst = registry.get(_sym, "moex")
        except _InstNotFound:
            continue
        _updated = Instrument(
            symbol=_existing_inst.symbol,
            market_id="moex",
            name=str(_share["name"]),
            instrument_type=_existing_inst.instrument_type,
            figi=str(_share["figi"]),
            lot_size=int(_share["lot"]),  # type: ignore[arg-type]
            currency=_existing_inst.currency,
            is_active=_existing_inst.is_active,
            segment_id=_existing_inst.segment_id,
        )
        registry.register(_updated)
    else:
        _log.warning("moex_share_not_in_api", symbol=_sym)

# Register ALL remaining discovered shares not in any configured segment
_new_count = 0
for _share_item in _all_moex_shares:
    _ticker = str(_share_item["ticker"])
    if _ticker in _configured_symbols:
        continue
    _instrument = Instrument(
        symbol=_ticker,
        market_id="moex",
        name=str(_share_item["name"]),
        instrument_type="stock",
        figi=str(_share_item["figi"]),
        lot_size=int(_share_item["lot"]),  # type: ignore[arg-type]
        currency="RUB",
        segment_id="ru_discovered",
    )
    registry.register(_instrument)
    _new_count += 1

_log.info(
    "moex_instruments_final",
    configured_stocks=len(_configured_symbols),
    discovered_new=_new_count,
    bonds=len(_bond_segment_symbols),
    total=len(registry.list_by_market("moex")),
)

# ── Data Fetcher ─────────────────────────────────────────────────────────

from finalayze.data.fetchers.tinkoff_data import TinkoffFetcher

tinkoff_fetcher = TinkoffFetcher(
    token=settings.tinkoff_token,
    registry=registry,
    sandbox=True,
)

fetchers: dict[str, object] = {"moex": tinkoff_fetcher}

# ── Execution ────────────────────────────────────────────────────────────

from finalayze.execution.broker_router import BrokerRouter
from finalayze.execution.retry import RetryPolicy
from finalayze.execution.tinkoff_broker import TinkoffBroker

retry_policy = RetryPolicy(max_retries=3, base_delay=1.0)
tinkoff_broker = TinkoffBroker(
    token=settings.tinkoff_token,
    registry=registry,
    sandbox=True,
    retry_policy=retry_policy,
)
# ── Bond Broker (separate instance for thread safety) ─────────────────
tinkoff_broker_bonds = TinkoffBroker(
    token=settings.tinkoff_token,
    registry=registry,
    sandbox=True,
    retry_policy=retry_policy,
)
broker_router = BrokerRouter(
    {
        "moex": tinkoff_broker,
        "moex_bonds": tinkoff_broker_bonds,
    }
)

_log.info("broker_wired", markets=broker_router.registered_markets)

# ── Strategies ───────────────────────────────────────────────────────────

from finalayze.strategies.combiner import StrategyCombiner
from finalayze.strategies.dual_momentum import DualMomentumStrategy
from finalayze.strategies.mean_reversion import MeanReversionStrategy
from finalayze.strategies.momentum import MomentumStrategy
from finalayze.strategies.rsi2_connors import RSI2ConnorsStrategy

strategies = [
    MomentumStrategy(),
    DualMomentumStrategy(),
    MeanReversionStrategy(),
    RSI2ConnorsStrategy(),
]

strategy_combiner = StrategyCombiner(strategies=strategies)
_log.info("strategies_loaded", count=len(strategies), names=[s.name for s in strategies])

# ── Risk (Circuit Breakers) ──────────────────────────────────────────────

from finalayze.risk.circuit_breaker import (
    CircuitBreaker,
    CrossMarketCircuitBreaker,
)

circuit_breakers = {
    "moex": CircuitBreaker(
        market_id="moex",
        l1_threshold=settings.circuit_breaker_l1,
        l2_threshold=settings.circuit_breaker_l2,
        l3_threshold=settings.circuit_breaker_l3,
    ),
}
cross_market_breaker = CrossMarketCircuitBreaker(
    halt_threshold=settings.max_cross_market_exposure_pct,
)

# ── Alerting ─────────────────────────────────────────────────────────────

from finalayze.api.alerts import TelegramAlerter

alerter = TelegramAlerter(
    bot_token=settings.telegram_bot_token,
    chat_id=settings.telegram_chat_id,
)
if settings.telegram_bot_token:
    _log.info("telegram_alerter_enabled", chat_id=settings.telegram_chat_id)
else:
    _log.info("telegram_alerter_disabled", reason="no bot token")

# ── News Analysis (optional) ────────────────────────────────────────────

from finalayze.analysis.event_classifier import EventClassifier
from finalayze.analysis.impact_estimator import ImpactEstimator
from finalayze.analysis.llm_client import create_llm_client
from finalayze.analysis.news_analyzer import NewsAnalyzer
from finalayze.data.fetchers.newsapi import NewsApiFetcher

_has_llm = bool(settings.llm_api_key or settings.anthropic_api_key)
_has_news = bool(settings.newsapi_api_key)

if _has_llm:
    llm_client = create_llm_client(settings)
    news_analyzer = NewsAnalyzer(llm_client)
    event_classifier = EventClassifier(llm_client)
    _log.info("llm_client_created", provider=settings.llm_provider, model=settings.llm_model)
else:
    # Stub: create minimal LLM client that returns neutral sentiment
    from finalayze.analysis.llm_client import LLMClient

    class _StubLLMClient(LLMClient):
        async def complete(self, prompt: str, system: str) -> str:  # noqa: ARG002
            return '{"sentiment": 0.0, "confidence": 0.0, "reasoning": "stub"}'

    llm_client = _StubLLMClient()
    news_analyzer = NewsAnalyzer(llm_client)
    event_classifier = EventClassifier(llm_client)
    _log.warning("llm_stub_active", reason="no LLM API key — news analysis will return neutral")

if _has_news:
    news_fetcher = NewsApiFetcher(api_key=settings.newsapi_api_key)
    _log.info("news_fetcher_enabled")
else:
    # Stub news fetcher: returns empty list
    class _StubNewsFetcher:
        def fetch_news(self, **kwargs: object) -> list:  # noqa: ARG002
            return []

    news_fetcher = _StubNewsFetcher()  # type: ignore[assignment]
    _log.warning("news_fetcher_stub", reason="no NewsAPI key — news cycle will be no-op")

impact_estimator = ImpactEstimator()

# ── FX Service ───────────────────────────────────────────────────────────

from finalayze.markets.currency import CurrencyConverter
from finalayze.markets.fx_service import FXRateService

converter = CurrencyConverter(base_currency="RUB")
fx_service = FXRateService(converter)

# ── ML Registry (optional) ──────────────────────────────────────────────

ml_registry = None
if settings.ml_enabled:
    try:
        from finalayze.ml.registry import MLModelRegistry

        ml_registry = MLModelRegistry(model_dir=settings.ml_model_dir)
        _log.info("ml_registry_loaded", model_dir=settings.ml_model_dir)
    except Exception:
        _log.exception("ml_registry_init_failed")

# ── Bond Layer Setup ──────────────────────────────────────────────────
import datetime as _dt
from decimal import Decimal as _Decimal

from finalayze.orchestration.bond_cycle import BondCycleProcessor
from finalayze.core.layer_ledger import LayerLedger
from finalayze.core.schemas import DEFAULT_LAYER_CONFIGS, PortfolioLayer
from finalayze.data.fetchers.cbr import MacroContextProvider
from finalayze.data.macro_cache import MacroCacheService
from finalayze.risk.dv01_sizing import DV01BudgetStep, EqualWeightBondSizer
from finalayze.risk.layer_circuit_breaker import AggregateBondBreaker, BondLayerBreaker
from finalayze.risk.yield_stop import YieldStop
from finalayze.strategies.bond_carry import BondCarryStrategy
from finalayze.strategies.bond_duration_rotation import BondDurationRotationStrategy
from finalayze.strategies.cbr_event import CBREventStrategy

layer_configs = DEFAULT_LAYER_CONFIGS

total_bond_capital = settings.bond_capital
layer_ledgers = {
    layer: LayerLedger(
        layer_id=layer.value,
        cash=_Decimal(str(total_bond_capital * float(cfg.capital_pct))),
    )
    for layer, cfg in layer_configs.items()
}

macro_provider = MacroContextProvider()
macro_cache = MacroCacheService(macro_provider)

# Build bond metadata from instrument registry
ofz_all = registry.list_by_type("moex", "bond")
ofz_pd = [i for i in ofz_all if not i.floating_coupon]
ofz_pk = [i for i in ofz_all if i.floating_coupon]

ofz_pk_symbols = [i.symbol for i in ofz_pk]
maturity_dates = {i.symbol: i.maturity_date for i in ofz_all if i.maturity_date}

# BondDurationRotationStrategy requires bond_durations, bond_maturities, coupon_rates
# Estimate modified duration from maturity (rough: years_to_maturity * 0.9)
_today = _dt.datetime.now(tz=_dt.UTC).date()
bond_durations: dict[str, _Decimal] = {}
bond_maturities_pd: dict[str, _dt.date] = {}
coupon_rates: dict[str, _Decimal] = {}
for inst in ofz_pd:
    if inst.maturity_date:
        years = (inst.maturity_date - _today).days / 365.25
        bond_durations[inst.symbol] = _Decimal(str(round(years * 0.9, 2)))
        bond_maturities_pd[inst.symbol] = inst.maturity_date
    if inst.coupon_rate is not None:
        coupon_rates[inst.symbol] = inst.coupon_rate

bond_strategies = {
    PortfolioLayer.CORE: [
        BondDurationRotationStrategy(
            bond_durations=bond_durations,
            bond_maturities=bond_maturities_pd,
            coupon_rates=coupon_rates,
        ),
    ],
    PortfolioLayer.STRATEGIC: [
        BondDurationRotationStrategy(
            bond_durations=bond_durations,
            bond_maturities=bond_maturities_pd,
            coupon_rates=coupon_rates,
        ),
    ],
    PortfolioLayer.TACTICAL: [
        BondCarryStrategy(symbols=ofz_pk_symbols, maturity_dates=maturity_dates),
        CBREventStrategy(),
    ],
    PortfolioLayer.SHORT: [
        BondCarryStrategy(symbols=ofz_pk_symbols, maturity_dates=maturity_dates),
    ],
}

layer_breakers = {
    layer: BondLayerBreaker(cfg, layer_ledgers[layer]) for layer, cfg in layer_configs.items()
}
aggregate_breaker = AggregateBondBreaker(
    ledgers=layer_ledgers,
    max_total_drawdown_pct=_Decimal("0.03"),
)

# Per-layer yield stops (from LayerConfig.yield_stop_bps)
yield_stops = {
    layer: YieldStop(threshold_bps=int(cfg.yield_stop_bps)) for layer, cfg in layer_configs.items()
}

# EqualWeightBondSizer needs n_symbols (number of floaters in portfolio)
equal_weight_sizer = EqualWeightBondSizer(n_symbols=max(len(ofz_pk_symbols), 1))

bond_processor = BondCycleProcessor(
    layer_configs=layer_configs,
    layer_ledgers=layer_ledgers,
    layer_breakers=layer_breakers,
    aggregate_breaker=aggregate_breaker,
    strategies=bond_strategies,
    macro_cache=macro_cache,
    dv01_sizer=DV01BudgetStep(),
    equal_weight_sizer=equal_weight_sizer,
    yield_stops=yield_stops,
    broker_router=broker_router,
    instrument_registry=registry,
    fetcher=tinkoff_fetcher,
    alerter=alerter,
    ml_registry=ml_registry,
)

_log.info(
    "bond_layer_wired",
    layers=len(layer_configs),
    ofz_pd=len(ofz_pd),
    ofz_pk=len(ofz_pk),
    bond_capital=total_bond_capital,
)

# ── Trading Loop ─────────────────────────────────────────────────────────

from finalayze.orchestration.trading_loop import TradingLoop

loop = TradingLoop(
    settings=settings,
    fetchers=fetchers,
    news_fetcher=news_fetcher,
    news_analyzer=news_analyzer,
    event_classifier=event_classifier,
    impact_estimator=impact_estimator,
    strategy=strategy_combiner,
    broker_router=broker_router,
    circuit_breakers=circuit_breakers,
    cross_market_breaker=cross_market_breaker,
    alerter=alerter,
    instrument_registry=registry,
    ml_registry=ml_registry,
    fx_service=fx_service,
    bond_cycle_processor=bond_processor,
    macro_cache=macro_cache,
)

# ── Signal Handlers ──────────────────────────────────────────────────────


def _shutdown(sig: int, frame: object) -> None:  # noqa: ARG001
    _log.info("shutdown_signal_received", signal=sig)
    loop.stop()


signal.signal(signal.SIGINT, _shutdown)
signal.signal(signal.SIGTERM, _shutdown)

# ── Launch ───────────────────────────────────────────────────────────────

_log.info(
    "trading_loop_launching",
    mode=settings.mode.value,
    markets=broker_router.registered_markets,
    instruments=len(registry.list_by_market("moex")),
    strategy_cycle_min=settings.strategy_cycle_minutes,
    news_cycle_min=settings.news_cycle_minutes,
)

alerter.send_alert(
    f"Sandbox started: mode={settings.mode.value}, "
    f"markets={broker_router.registered_markets}, "
    f"instruments={len(registry.list_by_market('moex'))}"
)

try:
    loop.start()  # blocks until stop() is called
except KeyboardInterrupt:
    _log.info("keyboard_interrupt")
finally:
    alerter.send_alert("Sandbox shutdown")
    loop.stop()
    tinkoff_broker.close()
    tinkoff_broker_bonds.close()
    tinkoff_fetcher.close()
    _log.info("sandbox_shutdown_complete")

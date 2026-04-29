"""Daily and weekly reporting service for TradingLoop.

Extracted from trading_loop.py (Phase 1.4).
Handles equity snapshots, P&L calculations, circuit breaker resets, and alerts.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from collections.abc import Callable

    from finalayze.api.alerts import TelegramAlerter
    from finalayze.execution.broker_base import BrokerBase
    from finalayze.execution.broker_router import BrokerRouter
    from finalayze.orchestration.db_persistence import TradingPersistence
    from finalayze.risk.circuit_breaker import (
        CircuitBreaker,
        CrossMarketCircuitBreaker,
    )

# ── Constants ──────────────────────────────────────────────────────────────
_ZERO = Decimal(0)
_MARKET_CURRENCY: dict[str, str] = {"us": "USD", "moex": "RUB"}

_log = structlog.get_logger()


class DailyReportingService:
    """Manages daily resets, weekly digests, and circuit breaker responses.

    Baselines (dict[str, Decimal]) are passed as parameters and returned
    from methods. TradingLoop maintains self._baseline_equities and passes
    it to these methods, then updates it with the returned value.
    """

    def __init__(
        self,
        broker_router: BrokerRouter,
        circuit_breakers: dict[str, CircuitBreaker],
        cross_market_breaker: CrossMarketCircuitBreaker,
        loss_limit_tracker: Any,
        alerter: TelegramAlerter,
        persistence: TradingPersistence,
        bond_processor: Any = None,
        fx_service: Any = None,
        metrics_collector: Any = None,
        settings: Any = None,
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        """Initialize DailyReportingService.

        Args:
            broker_router: Routes market_id to broker instance
            circuit_breakers: dict[market_id] -> CircuitBreaker
            cross_market_breaker: Cross-market circuit breaker
            loss_limit_tracker: Tracks daily/weekly loss limits
            alerter: TelegramAlerter for sending alerts
            persistence: TradingPersistence for DB writes
            bond_processor: BondCycleProcessor (optional)
            fx_service: FXRateService (optional)
            metrics_collector: MetricsCollector (optional)
            settings: Settings object (optional)
            now_fn: Override for datetime.now(UTC) (testability)
        """
        self._broker_router = broker_router
        self._circuit_breakers = circuit_breakers
        self._cross_market_breaker = cross_market_breaker
        self._loss_limit_tracker = loss_limit_tracker
        self._alerter = alerter
        self._persistence = persistence
        self._bond_processor = bond_processor
        self._fx_service = fx_service
        self._metrics = metrics_collector
        self._settings = settings
        self._now = now_fn or (lambda: datetime.now(UTC))

    def daily_reset(self, baselines: dict[str, Decimal]) -> dict[str, Decimal]:
        """Reset circuit breakers and send daily P&L summary.

        Computes separate P&L for US equity, MOEX equity, and MOEX bonds.
        Persists equity snapshots to DB. Includes top 3 movers and dual
        currency totals.

        Args:
            baselines: Current baseline equities (market_id -> Decimal)

        Returns:
            Updated baselines (market_id -> new equity)
        """
        market_pnl: dict[str, Decimal] = {}
        new_baselines: dict[str, Decimal] = {}

        now = self._now()
        for market_id, cb in self._circuit_breakers.items():
            try:
                broker = self._broker_router.route(market_id)
                portfolio = broker.get_portfolio()
                equity = portfolio.equity
                new_baselines[market_id] = equity

                # Compute P&L BEFORE updating baseline
                baseline = baselines.get(market_id, equity)
                market_pnl[market_id] = equity - baseline

                # Reset breaker for next trading day
                cb.reset_daily(new_baseline=equity)
            except Exception:
                _log.exception(
                    "daily_reset: failed to reset for market %s",
                    market_id,
                )

        # Bond P&L from LayerLedger (not broker portfolio)
        if self._bond_processor is not None:
            try:
                bond_equity: Decimal = sum(
                    (
                        ledger.current_equity
                        for ledger in self._bond_processor._layer_ledgers.values()
                    ),
                    _ZERO,
                )
                bond_baseline = baselines.get(
                    "moex_bonds",
                    bond_equity,
                )
                market_pnl["moex_bonds"] = bond_equity - bond_baseline
                new_baselines["moex_bonds"] = bond_equity
            except Exception:
                _log.exception("daily_reset: failed to compute bond P&L")

        self._cross_market_breaker.reset_daily(new_baselines)
        total_equity = sum(new_baselines.values(), _ZERO)

        # Reset loss limit tracker daily baseline
        self._loss_limit_tracker.reset_day(now, total_equity)

        # 6A.10: Reset weekly baseline on Monday (weekday 0)
        monday = 0
        if now.weekday() == monday:
            self._loss_limit_tracker.reset_week(now, total_equity)

        # Update Prometheus metrics
        if self._metrics:
            for market_id, equity in new_baselines.items():
                pnl_val = market_pnl.get(market_id, _ZERO)
                self._metrics.set_daily_pnl(market_id, float(pnl_val))
                self._metrics.set_portfolio_equity(market_id, float(equity))

        # Top 3 movers by absolute P&L %
        top_movers = self._compute_top_movers(baselines)

        # Dual currency total
        total_equity_rub: Decimal | None = None
        if self._fx_service is not None:
            try:
                usdrub = self._fx_service._last_rate
                if usdrub and usdrub > _ZERO:
                    total_equity_rub = total_equity  # already mixed RUB+USD
            except Exception:
                _log.debug("daily_reset: FX unavailable for dual currency")

        # Persist equity snapshots to DB
        self._persistence.persist_equity_snapshots(new_baselines, now)

        self._alerter.on_daily_summary(
            market_pnl,
            total_equity,
            top_movers,
            total_equity_rub,
        )
        _log.info("Daily reset complete. Total equity: %s", total_equity)

        return new_baselines

    def persist_cycle_snapshot(self, now: datetime) -> None:
        """Per-cycle equity snapshot writer (EQTY-01 D-01, D-02, D-03).

        Mirrors ``daily_reset()`` lines 102-138 but writes WITHOUT resetting
        circuit breakers and WITHOUT sending the daily Telegram summary.
        Idempotent on failure: ``TradingPersistence.persist_equity_snapshots``
        is fire-and-forget under the PERSIST-05 envelope at
        ``db_persistence.py:115-127``.

        Called from ``TradingLoop._strategy_cycle_impl`` after the per-market
        loop completes (D-02 Route B -- see Phase 56 Plan 02 objective for the
        routing rationale: ``SignalExecutor`` is per-instrument, not per-cycle,
        so the cycle boundary lives in ``TradingLoop``). Halt paths (cross-market
        breaker trip, loss-limit halt) early-return BEFORE this call site, so
        snapshots only fire on cycles that actually completed.

        Args:
            now: Cycle timestamp (UTC-aware).
        """
        baselines: dict[str, Decimal] = {}

        # Per-market broker equity (mirror daily_reset lines 102-119, minus cb.reset_daily)
        for market_id in self._circuit_breakers:
            try:
                broker = self._broker_router.route(market_id)
                portfolio = broker.get_portfolio()
                baselines[market_id] = portfolio.equity
            except Exception:
                _log.exception(
                    "persist_cycle_snapshot: market %s broker fetch failed",
                    market_id,
                )

        # Bond ledger equity (mirror daily_reset lines 122-138)
        if self._bond_processor is not None:
            try:
                bond_equity: Decimal = sum(
                    (
                        ledger.current_equity
                        for ledger in self._bond_processor._layer_ledgers.values()
                    ),
                    _ZERO,
                )
                baselines["moex_bonds"] = bond_equity
            except Exception:
                _log.exception("persist_cycle_snapshot: bond equity sum failed")

        if not baselines:
            return

        # Reuse existing TradingPersistence wrapper -- same one daily_reset uses.
        # Currency derivation (moex/ru_ -> RUB else USD) happens inside
        # _persist_snapshots_async at db_persistence.py:285-287, so no new
        # currency logic is needed here.
        self._persistence.persist_equity_snapshots(baselines, now)

    def _compute_top_movers(self, baselines: dict[str, Decimal]) -> list[tuple[str, float]]:
        """Compute top 3 movers by absolute P&L % across all markets.

        Args:
            baselines: Current baseline equities

        Returns:
            List of (symbol, pnl_pct) tuples, sorted by absolute value, max 3
        """
        movers: list[tuple[str, float]] = []
        for market_id in self._circuit_breakers:
            try:
                broker = self._broker_router.route(market_id)
                portfolio = broker.get_portfolio()
                for sym, qty in portfolio.positions.items():
                    if qty > _ZERO:
                        baseline = baselines.get(market_id, _ZERO)
                        if baseline > _ZERO:
                            # Approximate % using position weight
                            pct = float(qty) * 0.01  # placeholder
                            movers.append((sym, pct))
            except Exception:
                _log.debug("_compute_top_movers: failed for %s", market_id)
                continue
        movers.sort(key=lambda x: abs(x[1]), reverse=True)
        return movers[:3]

    def load_baselines_from_db(self, fetchers_keys: list[str]) -> dict[str, Decimal]:
        """Load latest equity snapshots from DB on startup.

        If snapshots exist for today, use them as baselines.
        Otherwise current broker equity becomes the baseline and is
        persisted so subsequent restarts within the same day find it.

        Args:
            fetchers_keys: List of market IDs (e.g., ["us", "moex"])

        Returns:
            dict[market_id] -> Decimal equity
        """
        try:
            return self._load_baseline_async(fetchers_keys)
        except Exception:
            _log.info(
                "baseline_from_broker",
                reason="no DB snapshots for today, persisting current equity",
            )
            # Persist current broker equity so next restart finds it
            baselines: dict[str, Decimal] = {}
            for market_id in fetchers_keys:
                equity = self._get_market_equity(market_id)
                if equity is not None:
                    baselines[market_id] = equity
            if baselines:
                now = datetime.now(UTC)
                self._persistence.persist_equity_snapshots(baselines, now)
            return baselines

    def _load_baseline_async(self, fetchers_keys: list[str]) -> dict[str, Decimal]:  # noqa: ARG002
        """Query today's equity snapshots from TimescaleDB.

        Fetches all DailyEquitySnapshot rows for today, groups by market_id,
        and takes the latest equity per market.

        Args:
            fetchers_keys: List of market IDs (unused, for consistency)

        Returns:
            dict[market_id] -> Decimal equity

        Raises:
            ValueError: If no snapshots found for today
        """
        try:
            baselines = self._query_snapshots_sync(self._persistence)
        except Exception:
            _log.exception("_load_baseline_async: query failed")
            raise ValueError("no snapshots for today") from None

        if not baselines:
            msg = "no snapshots for today"
            raise ValueError(msg)

        _log.info("baselines_loaded_from_db", count=len(baselines))
        return baselines

    @staticmethod
    def _query_snapshots_sync(persistence: Any) -> dict[str, Decimal]:
        """Run async DB query in a fresh event loop from a sync thread.

        Args:
            persistence: TradingPersistence instance

        Returns:
            dict[market_id] -> Decimal equity
        """
        from sqlalchemy import func, select  # noqa: PLC0415

        from finalayze.core.models import DailyEquitySnapshot  # noqa: PLC0415

        async def _run() -> dict[str, Decimal]:
            factory = persistence._get_bg_session_factory()
            baselines: dict[str, Decimal] = {}
            today_start = datetime.now(tz=UTC).replace(hour=0, minute=0, second=0, microsecond=0)
            subq = (
                select(
                    DailyEquitySnapshot.market_id,
                    func.max(DailyEquitySnapshot.timestamp).label("max_ts"),
                )
                .where(DailyEquitySnapshot.timestamp >= today_start)
                .group_by(DailyEquitySnapshot.market_id)
                .subquery()
            )
            stmt = select(DailyEquitySnapshot.market_id, DailyEquitySnapshot.equity).join(
                subq,
                (DailyEquitySnapshot.market_id == subq.c.market_id)
                & (DailyEquitySnapshot.timestamp == subq.c.max_ts),
            )
            async with factory() as session:
                result = await session.execute(stmt)
                rows = result.all()
                for row in rows:
                    baselines[row.market_id] = row.equity
            return baselines

        return asyncio.run(_run())

    def _get_market_equity(self, market_id: str) -> Decimal | None:
        """Return current portfolio equity for market, or None on failure.

        Args:
            market_id: Market identifier (e.g., "us", "moex")

        Returns:
            Decimal equity or None if fetch fails
        """
        try:
            broker = self._broker_router.route(market_id)
            portfolio = broker.get_portfolio()
            return Decimal(str(portfolio.equity))
        except Exception:
            _log.exception("_get_market_equity: failed for %s", market_id)
            return None

    def weekly_digest(self, baselines: dict[str, Decimal]) -> None:
        """Send weekly performance digest on Sunday evening.

        Computes week P&L from current baselines. Includes trade
        count, best/worst positions, circuit breaker trip count.

        Args:
            baselines: Current baseline equities
        """
        from finalayze.api.alerts import AlertPriority  # noqa: PLC0415

        now = self._now()
        week_start = now - timedelta(days=7)

        # Compute week P&L from current baselines
        week_pnl: dict[str, Decimal] = {}
        total_equity = _ZERO
        for market_id in self._circuit_breakers:
            try:
                broker = self._broker_router.route(market_id)
                portfolio = broker.get_portfolio()
                equity = portfolio.equity
                baseline = baselines.get(market_id, equity)
                week_pnl[market_id] = equity - baseline
                total_equity += equity
            except Exception:
                _log.debug("weekly_digest: failed for %s", market_id)

        # Bond layer P&L
        if self._bond_processor is not None:
            try:
                bond_equity_w: Decimal = sum(
                    (
                        ledger.current_equity
                        for ledger in self._bond_processor._layer_ledgers.values()
                    ),
                    _ZERO,
                )
                bond_baseline = baselines.get("moex_bonds", bond_equity_w)
                week_pnl["moex_bonds"] = bond_equity_w - bond_baseline
                total_equity += bond_equity_w
            except Exception:
                _log.debug("weekly_digest: bond P&L failed")

        # Format message
        lines: list[str] = ["\U0001f4ca <b>Weekly Digest</b>\n"]
        ws = week_start.strftime("%Y-%m-%d")
        ne = now.strftime("%Y-%m-%d")
        lines.append(f"Period: {ws} \u2014 {ne}\n")

        total_week_pnl = sum(week_pnl.values(), _ZERO)
        sign = "+" if total_week_pnl >= _ZERO else ""
        lines.append(f"<b>Week P&L:</b> <code>{sign}{total_week_pnl:,.2f}</code>")

        for market_id, pnl in sorted(week_pnl.items()):
            ms = "+" if pnl >= _ZERO else ""
            label = market_id.upper().replace("MOEX_BONDS", "BONDS")
            lines.append(f"  {label}: <code>{ms}{pnl:,.2f}</code>")

        lines.append(f"\n<b>Total Equity:</b> <code>{total_equity:,.2f}</code>")

        # Top movers
        top_movers = self._compute_top_movers(baselines)
        if top_movers:
            movers_str = ", ".join(f"<b>{sym}</b> {pct:+.1f}%" for sym, pct in top_movers[:3])
            lines.append(f"\n<b>Top Movers:</b> {movers_str}")

        self._alerter.send_alert(
            "\n".join(lines),
            priority=AlertPriority.INFO,
        )
        _log.info("weekly_digest_sent", total_pnl=str(total_week_pnl))

    def liquidate_market(self, market_id: str, baselines: dict[str, Decimal]) -> None:
        """Close all open positions in a market (L3 circuit breaker response).

        Args:
            market_id: Market to liquidate
            baselines: Current baseline equities (for drawdown calculation)
        """
        try:
            broker = self._broker_router.route(market_id)
            positions = broker.get_positions()
            portfolio = broker.get_portfolio()
            equity = portfolio.equity

            # #174: Correct drawdown = (baseline - current) / baseline
            baseline = baselines.get(market_id, equity)
            drawdown = float((baseline - equity) / baseline if baseline > _ZERO else _ZERO)

            # #129: No look-ahead bias — submit market orders without fill_candle
            self._close_positions(broker, positions)

            # Import CircuitLevel at call time
            from finalayze.risk.circuit_breaker import CircuitLevel  # noqa: PLC0415

            self._alerter.on_circuit_breaker_trip(market_id, CircuitLevel.LIQUIDATE, drawdown)
        except Exception:
            _log.exception("liquidate_market: failed for market %s", market_id)
            self._alerter.on_error("DailyReporting", f"liquidation failed for {market_id}")

    def _close_positions(self, broker: BrokerBase, positions: dict[str, Decimal]) -> None:
        """Submit SELL orders for all non-zero positions.

        Uses market orders without fill_candle (#129: no look-ahead bias).

        Args:
            broker: BrokerBase instance to submit orders
            positions: dict[symbol] -> Decimal quantity
        """
        from finalayze.execution.broker_base import OrderRequest  # noqa: PLC0415

        for symbol, qty in positions.items():
            if qty <= _ZERO:
                continue
            # #129: Do NOT pass fill_candle — live market orders have no look-ahead
            order = OrderRequest(symbol=symbol, side="SELL", quantity=qty)
            try:
                broker.submit_order(order)
            except Exception as exc:
                _log.error("liquidation_order_failed", symbol=symbol, error=str(exc))
                self._alerter.on_error("DailyReporting", f"Liquidation failed for {symbol}: {exc}")

"""Broker gRPC reconnection and in-flight order reconciliation (Layer 5 -- orchestrator).

Provides module-level functions for:
  - attempt_grpc_reconnect: exponential-backoff reconnection with Telegram alerts
  - reconcile_inflight_orders: query and cancel stale orders on startup

These are extracted from TradingLoop to improve testability and reduce
god-object complexity. TradingLoop delegates to these via thin wrappers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    import threading
    from collections.abc import Mapping

    from finalayze.api.alerts import TelegramAlerter
    from finalayze.execution.broker_router import BrokerRouter

_log = structlog.get_logger()


def attempt_grpc_reconnect(
    *,
    broker_router: BrokerRouter,
    alerter: TelegramAlerter,
    stop_event: threading.Event,
    reconnect_delays: list[float],
    broker_name: str,
) -> bool:
    """Try to reconnect gRPC channel with exponential backoff.

    Attempts up to 5 reconnections with delays [30, 60, 120, 240, 300]s
    (jittered 0.8-1.2x). Sends Telegram alert on each attempt.

    Args:
        broker_router: BrokerRouter instance for routing by market.
        alerter: TelegramAlerter instance for notifications.
        stop_event: threading.Event to check for early halt.
        reconnect_delays: List of delays (seconds) between attempts.
        broker_name: Market identifier (e.g. "moex") for logging/alerts.

    Returns:
        True if reconnection succeeded, False if all attempts exhausted
        (sets stop_event to halt trading).
    """
    import random  # noqa: PLC0415

    from finalayze.execution.tinkoff_broker import TinkoffBroker  # noqa: PLC0415

    broker = broker_router.route(broker_name)
    if not isinstance(broker, TinkoffBroker):
        _log.warning("reconnect_not_tinkoff", broker_name=broker_name)
        return False

    for attempt, delay in enumerate(reconnect_delays, 1):
        jitter = random.uniform(0.8, 1.2)  # noqa: S311
        actual_delay = delay * jitter
        _log.warning(
            "grpc_reconnect_attempt",
            broker=broker_name,
            attempt=attempt,
            max_attempts=len(reconnect_delays),
            delay_s=round(actual_delay, 1),
        )
        alerter.on_error(
            "TradingLoop",
            f"gRPC reconnect attempt {attempt}/{len(reconnect_delays)} "
            f"for {broker_name} (delay {round(actual_delay)}s)",
        )

        if stop_event.wait(timeout=actual_delay):
            _log.info("grpc_reconnect_cancelled", broker=broker_name)
            return False

        if broker.reconnect_client():
            _log.info("grpc_reconnected", broker=broker_name, attempt=attempt)
            return True

    _log.error("grpc_reconnect_exhausted", broker=broker_name)
    alerter.on_error(
        "TradingLoop",
        f"gRPC reconnection exhausted for {broker_name} -- halting trading",
    )
    stop_event.set()
    return False


def reconcile_inflight_orders(
    *,
    broker_router: BrokerRouter,
    circuit_breakers: Mapping[str, object],
) -> None:
    """Query open orders from all TinkoffBrokers, cancel stale ones, log fills.

    Stale orders: non-terminal orders older than 2 minutes (fill timeout).
    Called on startup before scheduler begins.

    Args:
        broker_router: BrokerRouter instance for routing by market.
        circuit_breakers: Dict[market_id, CircuitBreaker] to iterate markets.
    """
    from finalayze.execution.tinkoff_broker import TinkoffBroker  # noqa: PLC0415

    _fill_timeout_seconds = 120  # 2 minutes

    for market_id in list(circuit_breakers.keys()):
        try:
            broker = broker_router.route(market_id)
        except Exception:  # noqa: S112
            continue
        if not isinstance(broker, TinkoffBroker):
            continue

        try:
            open_orders = broker.get_open_orders()
        except Exception:
            _log.warning("reconcile_get_orders_failed", market=market_id)
            continue

        if not open_orders:
            _log.info("reconcile_no_inflight", market=market_id)
            continue

        for order in open_orders:
            _log.info(
                "reconcile_inflight_order",
                market=market_id,
                order_id=order.order_id,
                status=order.execution_status,
                filled_qty=str(order.filled_quantity),
            )
            # Log any partial fills
            if order.filled_quantity > 0:
                _log.warning(
                    "reconcile_partial_fill_detected",
                    order_id=order.order_id,
                    filled_qty=str(order.filled_quantity),
                    filled_price=str(order.filled_price),
                )
            # Cancel stale orders (all open orders on startup are stale)
            cancelled = broker.cancel_order_safe(order.order_id)
            if cancelled:
                _log.info("reconcile_cancelled_stale", order_id=order.order_id)
            else:
                _log.warning("reconcile_cancel_failed", order_id=order.order_id)

"""Tests for stop-loss state restoration on container restart.

Covers:
  - TradingPersistence.load_stop_snapshots() no-db guard
  - TradingLoop._restore_stop_states_from_db() wiring
  - _restore_stop_states_from_db called in start() before _preflight_check
  - Retroactive stop for orphaned positions in process_instrument
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

# ── TradingPersistence.load_stop_snapshots ──────────────────────────────────


def test_load_stop_snapshots_returns_empty_when_no_db_url() -> None:
    """No DB URL → empty dict, no exception."""
    from finalayze.orchestration.db_persistence import TradingPersistence

    p = TradingPersistence(db_url=None, async_loop=None)
    result = p.load_stop_snapshots()
    assert result == {}


# ── TradingLoop._restore_stop_states_from_db ───────────────────────────────


def _make_loop_with_mocked_persistence() -> object:
    from finalayze.core.trading_loop import TradingLoop, TradingLoopDeps

    settings = MagicMock()
    settings.mode = MagicMock()
    settings.mode.value = "sandbox"
    settings.mode.can_submit_orders.return_value = True
    settings.news_cycle_minutes = 30
    settings.strategy_cycle_minutes = 15
    settings.daily_reset_hour_utc = 0
    settings.max_position_pct = 0.10
    settings.max_positions_per_market = 10
    settings.daily_loss_limit_pct = 0.05
    settings.kelly_fraction = 0.5
    settings.database_url = None  # disables real DB
    settings.ml_enabled = False
    settings.bond_cycle_enabled = False
    settings.weekly_digest_hour_utc = 16
    settings.meta_agent_enabled = False

    return TradingLoop(
        TradingLoopDeps(
            settings=settings,
            fetchers={"moex": MagicMock()},
            news_fetcher=MagicMock(),
            news_analyzer=MagicMock(),
            event_classifier=MagicMock(),
            impact_estimator=MagicMock(),
            strategy=MagicMock(),
            broker_router=MagicMock(),
            circuit_breakers={"moex": MagicMock()},
            cross_market_breaker=MagicMock(),
            alerter=MagicMock(),
            instrument_registry=MagicMock(),
        )
    )


def test_restore_stop_states_from_db_no_persistence_is_noop() -> None:
    """When persistence has no db_url, _restore_stop_states_from_db is a silent no-op."""
    loop = _make_loop_with_mocked_persistence()
    # persistence exists but db_url=None → load_stop_snapshots returns {}
    # Should not raise
    loop._restore_stop_states_from_db()  # type: ignore[union-attr]


def test_restore_stop_states_from_db_calls_restore_stops() -> None:
    """When load_stop_snapshots returns states, restore_stops is called with
    only symbols that have open broker positions."""
    from finalayze.execution.simulated_broker import StopLossState

    loop = _make_loop_with_mocked_persistence()

    # Build a snapshot for two symbols
    def _make_state(entry: float) -> StopLossState:
        d = Decimal(str(entry))
        return StopLossState(
            initial_stop=d - 5,
            current_stop=d - 5,
            highest_price=d,
            trail_activated=False,
            activation_atr=Decimal("1.0"),
            trail_atr=Decimal("1.5"),
            entry_price=d,
            atr_value=Decimal("2.5"),
        )

    snapshot = {
        "SBER": ("moex", _make_state(100.0)),
        "CBOM": ("moex", _make_state(7.3)),
    }

    # Broker has only SBER open (CBOM was already closed / not in broker)
    broker_mock = MagicMock()
    broker_mock.get_positions.return_value = {"SBER": Decimal(238)}
    loop._broker_router.route.return_value = broker_mock  # type: ignore[union-attr]

    with (
        patch.object(loop._persistence, "load_stop_snapshots", return_value=snapshot),  # type: ignore[union-attr]
        patch.object(loop._position_tracker, "restore_stops") as mock_restore,  # type: ignore[union-attr]
    ):
        loop._restore_stop_states_from_db()  # type: ignore[union-attr]

    mock_restore.assert_called_once()
    restored_arg = mock_restore.call_args[0][0]
    assert "SBER" in restored_arg
    assert "CBOM" not in restored_arg  # not in broker positions


def test_restore_stop_states_called_in_start() -> None:
    """_restore_stop_states_from_db is called in start() between
    _reconcile_inflight_orders and _preflight_check."""
    import inspect

    from finalayze.core.trading_loop import TradingLoop, TradingLoopDeps

    src = inspect.getsource(TradingLoop.start)
    reconcile_idx = src.find("_reconcile_inflight_orders")
    restore_idx = src.find("_restore_stop_states_from_db")
    preflight_idx = src.find("_preflight_check")

    assert restore_idx != -1, "_restore_stop_states_from_db not found in start()"
    assert reconcile_idx < restore_idx < preflight_idx, (
        "_restore_stop_states_from_db must be called after _reconcile_inflight_orders "
        "and before _preflight_check"
    )


# ── Retroactive stop for orphaned positions ─────────────────────────────────


def _make_candles(n: int = 30, base_price: float = 7.3) -> list:
    import datetime

    from finalayze.core.schemas import Candle

    now = datetime.datetime.now(tz=datetime.UTC)
    candles = []
    for i in range(n):
        p = Decimal(str(base_price + i * 0.01))
        ts = now - datetime.timedelta(days=n - i - 1)
        candles.append(
            Candle(
                symbol="CBOM",
                market_id="moex",
                timeframe="1d",
                timestamp=ts,
                open=p - Decimal("0.1"),
                high=p + Decimal("0.2"),
                low=p - Decimal("0.2"),
                close=p,
                volume=10000,
            )
        )
    return candles


def test_retroactive_stop_set_when_position_has_no_stop_state() -> None:
    """If broker reports open position but PositionTracker has no stop state,
    process_instrument must retroactively register a stop."""
    from unittest.mock import MagicMock, patch

    from finalayze.orchestration.position_manager import PositionTracker
    from finalayze.orchestration.signal_executor import SignalExecutor

    tracker = PositionTracker(kelly_sizer=MagicMock(), broker_router=MagicMock())

    # Real (uninitialised) SignalExecutor so stage methods dispatch to their
    # real implementations rather than MagicMock auto-attrs.
    executor = SignalExecutor.__new__(SignalExecutor)
    executor._position_tracker = tracker
    executor._settings = MagicMock()
    executor._sentiment_mgr = MagicMock()
    executor._sentiment_mgr.get_sentiment.return_value = 0.0
    executor._strategy = MagicMock()
    executor._strategy.generate_signal.return_value = None  # no signal needed
    executor._health_monitor = None
    executor._sandbox_monitor = None
    executor._metrics = None
    executor._persistence = None
    executor._alerter = MagicMock()
    executor._registry = MagicMock()
    executor._loss_limit_tracker = MagicMock()
    executor._loss_limit_tracker.check_daily_loss.return_value = None
    executor._pre_trade_checker = MagicMock()
    executor._last_prices = {}
    executor._broker_router = MagicMock()
    executor._segment_min_confidence = {}

    # Broker reports CBOM as open position (qty=170000)
    broker = MagicMock()
    broker.has_position.return_value = True
    broker.get_positions.return_value = {"CBOM": Decimal(170000)}
    executor._broker_router.route.return_value = broker

    candles = _make_candles(30, base_price=7.3)
    instrument = MagicMock()
    instrument.figi = "BBG000000001"
    instrument.symbol = "CBOM"
    instrument.segment_id = "ru_finance"

    from finalayze.risk.circuit_breaker import CircuitLevel

    with patch.object(tracker, "check_stop_losses"):  # suppress actual check
        SignalExecutor.process_instrument(
            executor,
            instrument=instrument,
            market_id="moex",
            level=CircuitLevel.NORMAL,
            fetcher=MagicMock(**{"fetch_candles.return_value": candles}),
            now=candles[-1].timestamp,
            equity=Decimal(500000),
            cash=Decimal(100000),
            portfolio=None,
        )

    # After the cycle, stop state must have been registered
    assert tracker.has_stop("CBOM"), "retroactive stop must be set for orphaned position"
    state = tracker.get_stop_state("CBOM")
    assert state is not None
    assert state.current_stop > _ZERO


_ZERO = Decimal(0)

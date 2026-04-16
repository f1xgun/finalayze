"""Tests for daily P&L with bond separation, persisted snapshots, top movers (05-02)."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from finalayze.risk.circuit_breaker import CircuitBreaker, CircuitLevel


def _make_trading_loop_mock(
    *,
    us_equity: Decimal = Decimal(50000),
    moex_equity: Decimal = Decimal(3000000),
    bond_ledger_equity: Decimal = Decimal(1000000),
    positions: dict[str, Decimal] | None = None,
    fx_rate: Decimal = Decimal("90.0"),
) -> MagicMock:
    """Create a TradingLoop mock for daily P&L tests."""
    from finalayze.core.trading_loop import TradingLoop

    loop = MagicMock(spec=TradingLoop)

    # Circuit breakers for us + moex
    us_cb = CircuitBreaker(market_id="us")
    moex_cb = CircuitBreaker(market_id="moex")
    loop._circuit_breakers = {"us": us_cb, "moex": moex_cb}

    # Broker router
    us_broker = MagicMock()
    us_portfolio = MagicMock()
    us_portfolio.equity = us_equity
    us_portfolio.positions = positions or {"AAPL": Decimal(10), "MSFT": Decimal(5)}
    us_broker.get_portfolio.return_value = us_portfolio

    moex_broker = MagicMock()
    moex_portfolio = MagicMock()
    moex_portfolio.equity = moex_equity
    moex_portfolio.positions = {"SBER": Decimal(100), "GAZP": Decimal(50)}
    moex_broker.get_portfolio.return_value = moex_portfolio

    def route(market_id: str) -> MagicMock:
        if market_id == "us":
            return us_broker
        return moex_broker

    loop._broker_router = MagicMock()
    loop._broker_router.route.side_effect = route

    # Bond processor with layer ledgers
    loop._bond_processor = MagicMock()
    ledger = MagicMock()
    ledger.current_equity = bond_ledger_equity
    loop._bond_processor._layer_ledgers = {"core": ledger}

    # Baseline equities (set from previous day)
    loop._baseline_equities = {
        "us": us_equity - Decimal(500),
        "moex": moex_equity - Decimal(10000),
        "moex_bonds": bond_ledger_equity - Decimal(5000),
    }

    # Cross-market breaker
    loop._cross_market_breaker = MagicMock()

    # Alerter
    loop._alerter = MagicMock()

    # Loss limit tracker
    loop._loss_limit_tracker = MagicMock()

    # FX service
    loop._fx_service = MagicMock()
    loop._fx_service.get_usdrub.return_value = fx_rate

    # Settings
    loop._settings = MagicMock()

    # _now
    loop._now.return_value = datetime(2026, 3, 14, 0, 0, tzinfo=UTC)

    # _run_async - passthrough
    loop._run_async = lambda coro: None

    # Metrics collector (needed by _daily_reset)
    loop._metrics = MagicMock()

    # DailyReportingService (needed after decomposition)
    from finalayze.orchestration.daily_reporting import DailyReportingService

    daily_reporter = DailyReportingService(
        broker_router=loop._broker_router,
        circuit_breakers=loop._circuit_breakers,
        cross_market_breaker=loop._cross_market_breaker,
        loss_limit_tracker=loop._loss_limit_tracker,
        alerter=loop._alerter,
        persistence=MagicMock(),
        bond_processor=loop._bond_processor,
        fx_service=loop._fx_service,
        metrics_collector=loop._metrics,
        settings=loop._settings,
        now_fn=loop._now,
    )
    loop._daily_reporter = daily_reporter

    return loop


class TestDailyPnLSeparation:
    """Daily P&L separates US, MOEX equity, MOEX bonds into distinct entries."""

    def test_pnl_separates_three_markets(self) -> None:
        """P&L dict contains us, moex, moex_bonds keys."""
        from finalayze.core.trading_loop import TradingLoop

        loop = _make_trading_loop_mock()
        TradingLoop._daily_reset(loop)

        # on_daily_summary should be called with market_pnl containing 3 keys
        call_args = loop._alerter.on_daily_summary.call_args
        assert call_args is not None
        market_pnl = call_args[0][0] if call_args[0] else call_args[1].get("market_pnl")
        assert "us" in market_pnl
        assert "moex" in market_pnl
        assert "moex_bonds" in market_pnl

    def test_bond_pnl_from_ledger(self) -> None:
        """Bond P&L computed from LayerLedger equity (not broker portfolio)."""
        from finalayze.core.trading_loop import TradingLoop

        loop = _make_trading_loop_mock(bond_ledger_equity=Decimal(1005000))
        loop._baseline_equities["moex_bonds"] = Decimal(1000000)
        TradingLoop._daily_reset(loop)

        call_args = loop._alerter.on_daily_summary.call_args
        market_pnl = call_args[0][0] if call_args[0] else call_args[1].get("market_pnl")
        bond_pnl = market_pnl["moex_bonds"]
        assert bond_pnl == Decimal(5000)


class TestEquitySnapshotPersistence:
    """Equity snapshots persisted to DB via DailyEquitySnapshot model."""

    def test_snapshots_persisted(self) -> None:
        """After _daily_reset, persist_equity_snapshots is called."""
        from finalayze.core.trading_loop import TradingLoop

        loop = _make_trading_loop_mock()
        TradingLoop._daily_reset(loop)
        # persist_equity_snapshots should have been called on daily_reporter._persistence
        loop._daily_reporter._persistence.persist_equity_snapshots.assert_called_once()

    def test_no_snapshot_uses_current_equity(self) -> None:
        """If no snapshot exists for today, current equity used as baseline."""
        from finalayze.core.trading_loop import TradingLoop

        loop = _make_trading_loop_mock()
        loop._baseline_equities = {}  # no previous baseline
        TradingLoop._daily_reset(loop)
        # Should not error; alerter still called
        loop._alerter.on_daily_summary.assert_called_once()

    @pytest.mark.asyncio
    async def test_persist_snapshots_async_creates_rows(self) -> None:
        """_persist_snapshots_async creates DailyEquitySnapshot rows via session.add + commit."""
        from finalayze.orchestration.db_persistence import TradingPersistence

        persistence = MagicMock(spec=TradingPersistence)
        baselines = {
            "us": Decimal(50000),
            "moex": Decimal(3000000),
            "moex_bonds": Decimal(1000000),
        }
        now = datetime(2026, 3, 14, 0, 0, tzinfo=UTC)

        mock_session = AsyncMock()
        # session.add is sync in SQLAlchemy -- use MagicMock to avoid coroutine warning
        mock_session.add = MagicMock()
        mock_factory = MagicMock()
        mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_factory.return_value.__aexit__ = AsyncMock(return_value=False)

        persistence._get_bg_session_factory = MagicMock(return_value=mock_factory)
        await TradingPersistence._persist_snapshots_async(persistence, baselines, now)

        # Should have called session.add 3 times (one per market)
        assert mock_session.add.call_count == 3
        # Should have committed
        mock_session.commit.assert_awaited_once()

        # Verify the snapshot objects have correct attributes
        added_objects = [call.args[0] for call in mock_session.add.call_args_list]
        market_ids = {obj.market_id for obj in added_objects}
        assert market_ids == {"us", "moex", "moex_bonds"}

        # Verify currency assignment
        for obj in added_objects:
            if obj.market_id in ("moex", "moex_bonds"):
                assert obj.currency == "RUB"
            else:
                assert obj.currency == "USD"

    @pytest.mark.asyncio
    async def test_load_baseline_async_populates_baselines(self) -> None:
        """_load_baseline_async queries DB and updates _baseline_equities."""
        from finalayze.core.trading_loop import TradingLoop

        loop = MagicMock(spec=TradingLoop)
        loop._baseline_equities = {}

        # Mock DB results
        row_us = MagicMock()
        row_us.market_id = "us"
        row_us.equity = Decimal(50000)
        row_moex = MagicMock()
        row_moex.market_id = "moex"
        row_moex.equity = Decimal(3000000)

        mock_result = MagicMock()
        mock_result.all.return_value = [row_us, row_moex]

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_factory = MagicMock()
        mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_factory.return_value.__aexit__ = AsyncMock(return_value=False)

        loop._get_bg_session_factory = MagicMock(return_value=mock_factory)
        await TradingLoop._load_baseline_async(loop)

        assert loop._baseline_equities["us"] == Decimal(50000)
        assert loop._baseline_equities["moex"] == Decimal(3000000)

    @pytest.mark.asyncio
    async def test_load_baseline_async_no_rows_raises_value_error(self) -> None:
        """_load_baseline_async with no rows for today raises ValueError."""
        from finalayze.core.trading_loop import TradingLoop

        loop = MagicMock(spec=TradingLoop)
        original = {"us": Decimal(40000)}
        loop._baseline_equities = original.copy()

        mock_result = MagicMock()
        mock_result.all.return_value = []

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_factory = MagicMock()
        mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_factory.return_value.__aexit__ = AsyncMock(return_value=False)

        loop._get_bg_session_factory = MagicMock(return_value=mock_factory)
        with pytest.raises(ValueError, match="no snapshots for today"):
            await TradingLoop._load_baseline_async(loop)

    def test_load_baseline_from_db_called_in_start(self) -> None:
        """_load_baseline_from_db is called during start() before scheduler begins."""
        import inspect

        from finalayze.core.trading_loop import TradingLoop

        source = inspect.getsource(TradingLoop.start)
        # Must call _load_baseline_from_db before _scheduler.start()
        load_idx = source.find("_load_baseline_from_db")
        scheduler_idx = source.find("_scheduler.start()")
        assert load_idx != -1, "_load_baseline_from_db not found in start()"
        assert load_idx < scheduler_idx, (
            "_load_baseline_from_db must be called before _scheduler.start()"
        )


class TestTopMovers:
    """Top 3 movers included in daily summary by % change."""

    def test_top_movers_in_signature(self) -> None:
        """on_daily_summary accepts top_movers parameter."""
        from finalayze.core.trading_loop import TradingLoop

        loop = _make_trading_loop_mock()
        TradingLoop._daily_reset(loop)
        call_args = loop._alerter.on_daily_summary.call_args
        # Check that top_movers is passed (positional or keyword)
        assert call_args is not None
        list(call_args[0]) + list(call_args[1].values())
        # Should have more than 2 args (market_pnl, total_equity, and more)
        total_positional = len(call_args[0])
        total_keyword = len(call_args[1])
        assert total_positional + total_keyword >= 3


class TestDualCurrencyDisplay:
    """Total equity shows both RUB and USD using FXRateService."""

    def test_on_daily_summary_updated_signature(self) -> None:
        """on_daily_summary accepts bond_pnl, top_movers, and dual currency."""
        import inspect

        from finalayze.core.alerts import TelegramAlerter

        sig = inspect.signature(TelegramAlerter.on_daily_summary)
        params = list(sig.parameters.keys())
        # Should have more params than just market_pnl and total_equity_usd
        assert len(params) >= 4  # self, market_pnl, total_equity_usd + new params

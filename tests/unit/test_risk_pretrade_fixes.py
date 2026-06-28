"""Live-risk pre-trade-check regression tests (audit 2026-06-28, MEDIUM).

* The pre-trade circuit-breaker level was hardcoded None -> the check never fired.
* Sector concentration summed ALL open positions (over-stated, over-blocked) instead
  of same-sector exposure.
* The KillSwitch flag lived under /tmp (not durable across reboots).
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import MagicMock

from finalayze.core import kill_switch as ks_mod
from finalayze.core.schemas import PortfolioState
from finalayze.orchestration import signal_executor as se_mod
from finalayze.orchestration.signal_executor import SignalExecutor
from finalayze.risk.circuit_breaker import CircuitLevel


def _portfolio(positions: dict[str, Decimal]) -> PortfolioState:
    return PortfolioState(
        cash=Decimal(1000),
        positions=positions,
        equity=Decimal(10000),
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
    )


# ── KillSwitch durable path ─────────────────────────────────────────────────


def test_default_flag_path_honors_env(monkeypatch) -> None:
    monkeypatch.setenv("FINALAYZE_KILL_FLAG_PATH", "/data/state/killed")
    assert str(ks_mod._default_flag_path()) == "/data/state/killed"


def test_default_flag_path_default_is_not_tmp(monkeypatch) -> None:
    monkeypatch.delenv("FINALAYZE_KILL_FLAG_PATH", raising=False)
    path = ks_mod._default_flag_path()
    assert "/tmp" not in str(path)
    assert path.name == "killed"


# ── Sector exposure: same-sector only ───────────────────────────────────────


def test_compute_sector_exposure_sums_same_sector_only(monkeypatch) -> None:
    monkeypatch.setattr(se_mod, "_symbol_segment_map", lambda: {"AAA": "segX", "BBB": "segY"})
    ex = object.__new__(SignalExecutor)
    ex._get_last_price = lambda _s: Decimal(100)  # type: ignore[attr-defined]

    portfolio = _portfolio({"AAA": Decimal(2), "BBB": Decimal(3)})
    # Only AAA is in segX -> 2 * 100 = 200 (BBB excluded; prior code summed both = 500).
    assert ex._compute_sector_exposure(portfolio, "segX") == Decimal(200)


def test_compute_sector_exposure_none_without_segment() -> None:
    ex = object.__new__(SignalExecutor)
    assert ex._compute_sector_exposure(_portfolio({"AAA": Decimal(1)}), "") is None


# ── Circuit-breaker level is threaded into the pre-trade context ─────────────


def test_run_pre_trade_check_threads_circuit_level() -> None:
    ex = object.__new__(SignalExecutor)
    captured: dict[str, object] = {}
    checker = MagicMock()
    checker.check.side_effect = lambda ctx: captured.update(level=ctx.circuit_breaker_level)
    ex._pre_trade_checker = checker  # type: ignore[attr-defined]
    ex._position_tracker = MagicMock(  # type: ignore[attr-defined]
        get_stop_loss_price=lambda _s: None, has_stop=lambda _s: False
    )
    ex._has_pending_order = lambda *_a, **_k: False  # type: ignore[attr-defined]
    ex._compute_sector_exposure = lambda *_a, **_k: None  # type: ignore[attr-defined]
    ex._get_regime_state = lambda *_a, **_k: None  # type: ignore[attr-defined]
    ex._get_correlations = lambda *_a, **_k: {}  # type: ignore[attr-defined]

    ex._run_pre_trade_check(
        signal=MagicMock(strategy_name="x"),
        order_value=Decimal(100),
        portfolio=_portfolio({}),
        open_position_count=0,
        market_id="moex",
        symbol="SBER",
        seg_id="ru_energy",
        now=datetime(2026, 1, 1, tzinfo=UTC),
        cross_exposure=Decimal(0),
        max_exposure=Decimal(1),
        is_day_trade=False,
        level=CircuitLevel.HALTED,
    )

    assert captured["level"] == CircuitLevel.HALTED

"""Phase 80 P80-08: run_rebalance CLI -- mode mapping + fail-loud env/confirm gates.

The real sandbox run is the operator checkpoint (needs a Tinkoff token); these tests cover the
token-free safety logic: the --mode -> (plan_mode, submit) mapping, env validation, and the
LIVE-requires-confirm hard-stop gate (which must return BEFORE any broker wiring).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_rebalance.py"


def _load_cli() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_rebalance_cli", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_CLI = _load_cli()


def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("FINALAYZE_DATABASE_URL", "DATABASE_URL", "FINALAYZE_TINKOFF_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    # neutralize .env loading so the test env is authoritative.
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)


def test_resolve_run_mode() -> None:
    assert _CLI.resolve_run_mode("preview") == ("DRY_RUN", False)
    assert _CLI.resolve_run_mode("sandbox") == ("SANDBOX", True)
    assert _CLI.resolve_run_mode("live") == ("LIVE", True)


def test_default_mode_is_preview() -> None:
    args = _CLI._build_parser().parse_args([])
    assert args.mode == "preview"
    assert args.confirm is False


def test_missing_env_no_db(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("FINALAYZE_TINKOFF_TOKEN", "tok")  # token present, DB missing
    assert "DATABASE_URL" in (_CLI.missing_env_error() or "")


def test_missing_env_no_token(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("FINALAYZE_DATABASE_URL", "postgresql+asyncpg://x/y")
    assert "TINKOFF_TOKEN" in (_CLI.missing_env_error() or "")


def test_env_ok_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("FINALAYZE_DATABASE_URL", "postgresql+asyncpg://x/y")
    monkeypatch.setenv("FINALAYZE_TINKOFF_TOKEN", "tok")
    assert _CLI.missing_env_error() is None


def test_main_env_missing_returns_1(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing env var fails loud (exit 1) before any wiring."""
    _clear_env(monkeypatch)
    assert _CLI.main(["--mode", "sandbox"]) == 1


def test_main_live_without_confirm_returns_1(monkeypatch: pytest.MonkeyPatch) -> None:
    """--mode live without --confirm is refused (exit 1) BEFORE any broker is constructed."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("FINALAYZE_DATABASE_URL", "postgresql+asyncpg://x/y")
    monkeypatch.setenv("FINALAYZE_TINKOFF_TOKEN", "tok")
    # Reaching _run would attempt a real gRPC connection; exit 1 proves the gate fires first.
    assert _CLI.main(["--mode", "live"]) == 1


def test_fetch_nkd_by_symbol_returns_latest_value() -> None:
    """The OFZ NKD helper returns the latest (today's) accrued-coupon value (P82-R8)."""
    from datetime import date
    from decimal import Decimal
    from types import SimpleNamespace

    class _Fetcher:
        def fetch_accrued_interest(self, _symbol: str, _start: object, _end: object) -> list:
            return [SimpleNamespace(value=Decimal("0.30")), SimpleNamespace(value=Decimal("0.55"))]

    out = _CLI.fetch_nkd_by_symbol(_Fetcher(), "SU29024RMFS5", date(2026, 6, 24))
    assert out == {"SU29024RMFS5": Decimal("0.55")}


def test_fetch_nkd_by_symbol_empty_on_failure() -> None:
    """A fetch failure -> {} (clean-only fallback, never raises)."""
    from datetime import date

    class _Fetcher:
        def fetch_accrued_interest(self, _symbol: str, _start: object, _end: object) -> list:
            msg = "gRPC down"
            raise RuntimeError(msg)

    assert _CLI.fetch_nkd_by_symbol(_Fetcher(), "SU29024RMFS5", date(2026, 6, 24)) == {}


def test_fetch_nkd_by_symbol_empty_on_no_records() -> None:
    """No accrued-interest records -> {} (clean-only fallback)."""
    from datetime import date

    class _Fetcher:
        def fetch_accrued_interest(self, _symbol: str, _start: object, _end: object) -> list:
            return []

    assert _CLI.fetch_nkd_by_symbol(_Fetcher(), "SU29024RMFS5", date(2026, 6, 24)) == {}


def test_equity_point_value_error_when_symbol_overridden(monkeypatch: object) -> None:
    """Overriding the equity symbol without its point value fails closed (WR-02)."""
    monkeypatch.setenv("FINALAYZE_SAA_EQUITY_SYMBOL", "MXU6")  # type: ignore[attr-defined]
    monkeypatch.delenv("FINALAYZE_SAA_EQUITY_POINT_VALUE", raising=False)  # type: ignore[attr-defined]
    assert _CLI.equity_point_value_error() is not None


def test_equity_point_value_error_none_when_both_set(monkeypatch: object) -> None:
    """Overriding both the symbol and its point value is allowed."""
    monkeypatch.setenv("FINALAYZE_SAA_EQUITY_SYMBOL", "MXU6")  # type: ignore[attr-defined]
    monkeypatch.setenv("FINALAYZE_SAA_EQUITY_POINT_VALUE", "10")  # type: ignore[attr-defined]
    assert _CLI.equity_point_value_error() is None


def test_equity_point_value_error_none_on_default(monkeypatch: object) -> None:
    """The default IMOEXF (symbol not overridden) needs no explicit point value."""
    monkeypatch.delenv("FINALAYZE_SAA_EQUITY_SYMBOL", raising=False)  # type: ignore[attr-defined]
    assert _CLI.equity_point_value_error() is None

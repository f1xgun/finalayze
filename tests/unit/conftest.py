"""Shared fixtures for unit tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from finalayze.core.modes import RolloutPhase
from finalayze.risk.rollout import ROLLOUT_LIMITS


def patch_settings_with_rollout(settings_mock: MagicMock) -> MagicMock:
    """Add effective_risk_limits() to a Settings mock so TradingLoop.__init__ works.

    Phase 15 added Settings.effective_risk_limits() which TradingLoop.__init__ calls.
    Any test that creates a MagicMock for Settings must have this method return
    a real RolloutLimits object, otherwise Decimal conversion fails.
    """
    if not hasattr(settings_mock, "effective_risk_limits") or isinstance(
        getattr(settings_mock, "effective_risk_limits", None), MagicMock
    ):
        settings_mock.effective_risk_limits = MagicMock(
            return_value=ROLLOUT_LIMITS[RolloutPhase.FULL],
        )
    return settings_mock


@pytest.fixture(autouse=True)
def _patch_trading_loop_init(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto-patch TradingLoop.__init__ to handle mock Settings without effective_risk_limits.

    This prevents Decimal conversion errors when tests pass MagicMock settings
    that don't have effective_risk_limits() returning real RolloutLimits.
    """
    from finalayze.orchestration.trading_loop import TradingLoop

    _original_init = TradingLoop.__init__

    def _patched_init(self: TradingLoop, *args: object, **kwargs: object) -> None:
        settings = args[0] if args else kwargs.get("settings")
        if isinstance(settings, MagicMock):
            patch_settings_with_rollout(settings)
        _original_init(self, *args, **kwargs)  # type: ignore[misc]

    monkeypatch.setattr(TradingLoop, "__init__", _patched_init)

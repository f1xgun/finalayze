"""Unit tests for the Finalayze mode manager.

Tests verify:
- WorkMode enum values
- WorkMode predicate methods
- ModeManager current_mode property
- ModeManager transition_to() behavior including safety guard for REAL mode
"""

from __future__ import annotations

import os

import pytest

from finalayze.core.exceptions import FinalayzeError, ModeError
from finalayze.core.modes import ModeManager, WorkMode

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXPECTED_MODE_COUNT = 4
ENV_VAR_REAL_CONFIRMED = "FINALAYZE_REAL_CONFIRMED"


# ---------------------------------------------------------------------------
# WorkMode enum values
# ---------------------------------------------------------------------------
class TestWorkModeValues:
    def test_debug_value(self) -> None:
        assert WorkMode.DEBUG == "debug"

    def test_sandbox_value(self) -> None:
        assert WorkMode.SANDBOX == "sandbox"

    def test_test_value(self) -> None:
        assert WorkMode.TEST == "test"

    def test_real_value(self) -> None:
        assert WorkMode.REAL == "real"

    def test_has_all_four_modes(self) -> None:
        assert len(WorkMode) == EXPECTED_MODE_COUNT


# ---------------------------------------------------------------------------
# WorkMode.can_submit_orders()
# ---------------------------------------------------------------------------
class TestCanSubmitOrders:
    def test_debug_cannot_submit(self) -> None:
        assert WorkMode.DEBUG.can_submit_orders() is False

    def test_sandbox_can_submit(self) -> None:
        assert WorkMode.SANDBOX.can_submit_orders() is True

    def test_test_can_submit(self) -> None:
        assert WorkMode.TEST.can_submit_orders() is True

    def test_real_can_submit(self) -> None:
        assert WorkMode.REAL.can_submit_orders() is True


# ---------------------------------------------------------------------------
# WorkMode.requires_confirmation()
# ---------------------------------------------------------------------------
class TestRequiresConfirmation:
    def test_debug_no_confirmation(self) -> None:
        assert WorkMode.DEBUG.requires_confirmation() is False

    def test_sandbox_no_confirmation(self) -> None:
        assert WorkMode.SANDBOX.requires_confirmation() is False

    def test_test_no_confirmation(self) -> None:
        assert WorkMode.TEST.requires_confirmation() is False

    def test_real_requires_confirmation(self) -> None:
        assert WorkMode.REAL.requires_confirmation() is True


# ---------------------------------------------------------------------------
# WorkMode.is_real_trading()
# ---------------------------------------------------------------------------
class TestIsRealTrading:
    def test_debug_not_real(self) -> None:
        assert WorkMode.DEBUG.is_real_trading() is False

    def test_sandbox_not_real(self) -> None:
        assert WorkMode.SANDBOX.is_real_trading() is False

    def test_test_not_real(self) -> None:
        assert WorkMode.TEST.is_real_trading() is False

    def test_real_is_real(self) -> None:
        assert WorkMode.REAL.is_real_trading() is True


# ---------------------------------------------------------------------------
# ModeManager
# ---------------------------------------------------------------------------
class TestModeManagerInit:
    def test_default_mode_is_debug(self) -> None:
        mgr = ModeManager()
        assert mgr.current_mode == WorkMode.DEBUG

    def test_custom_initial_mode(self) -> None:
        mgr = ModeManager(initial_mode=WorkMode.SANDBOX)
        assert mgr.current_mode == WorkMode.SANDBOX

    def test_current_mode_returns_work_mode(self) -> None:
        mgr = ModeManager()
        assert isinstance(mgr.current_mode, WorkMode)


class TestModeManagerTransition:
    def test_transition_to_sandbox(self) -> None:
        mgr = ModeManager()
        mgr.transition_to(WorkMode.SANDBOX)
        assert mgr.current_mode == WorkMode.SANDBOX

    def test_transition_to_test(self) -> None:
        mgr = ModeManager()
        mgr.transition_to(WorkMode.TEST)
        assert mgr.current_mode == WorkMode.TEST

    def test_transition_to_debug(self) -> None:
        mgr = ModeManager(initial_mode=WorkMode.SANDBOX)
        mgr.transition_to(WorkMode.DEBUG)
        assert mgr.current_mode == WorkMode.DEBUG

    def test_transition_to_real_without_env_raises_mode_error(self) -> None:
        mgr = ModeManager()
        # Ensure env var is absent
        os.environ.pop(ENV_VAR_REAL_CONFIRMED, None)
        with pytest.raises(ModeError):
            mgr.transition_to(WorkMode.REAL)

    def test_transition_to_real_with_false_env_raises_mode_error(self) -> None:
        mgr = ModeManager()
        os.environ[ENV_VAR_REAL_CONFIRMED] = "false"
        try:
            with pytest.raises(ModeError):
                mgr.transition_to(WorkMode.REAL)
        finally:
            os.environ.pop(ENV_VAR_REAL_CONFIRMED, None)

    def test_transition_to_real_with_true_env_succeeds(self) -> None:
        mgr = ModeManager()
        os.environ[ENV_VAR_REAL_CONFIRMED] = "true"
        try:
            mgr.transition_to(WorkMode.REAL)
            assert mgr.current_mode == WorkMode.REAL
        finally:
            os.environ.pop(ENV_VAR_REAL_CONFIRMED, None)

    def test_transition_to_real_mode_not_changed_on_failure(self) -> None:
        mgr = ModeManager(initial_mode=WorkMode.SANDBOX)
        os.environ.pop(ENV_VAR_REAL_CONFIRMED, None)
        with pytest.raises(ModeError):
            mgr.transition_to(WorkMode.REAL)
        # Mode must be unchanged
        assert mgr.current_mode == WorkMode.SANDBOX

    def test_mode_error_is_finalayze_error(self) -> None:
        mgr = ModeManager()
        os.environ.pop(ENV_VAR_REAL_CONFIRMED, None)
        with pytest.raises(FinalayzeError):
            mgr.transition_to(WorkMode.REAL)

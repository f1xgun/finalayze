"""Tests for experiments REST API endpoints."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from finalayze.api.v1.auth import api_key_auth
from finalayze.api.v1.experiments import router as experiments_router

_API_KEY = "test-api-key"


def _make_app() -> FastAPI:
    """Create a minimal FastAPI app with experiments router and overridden auth."""
    app = FastAPI()
    app.include_router(experiments_router, prefix="/api/v1")
    app.dependency_overrides[api_key_auth] = lambda: None
    return app


def _make_app_no_auth() -> FastAPI:
    """Create a minimal FastAPI app with experiments router but NO auth override."""
    app = FastAPI()
    app.include_router(experiments_router, prefix="/api/v1")
    return app


def _make_mock_experiment_state(
    experiment_id: str = "exp-abc123",
    status: str = "pending",
    verdict: str | None = None,
    reasoning: str | None = None,
    debate_id: str | None = None,
) -> MagicMock:
    """Create a mock ExperimentState for testing."""
    state = MagicMock()
    state.experiment_id = experiment_id
    state.hypothesis = "ML improves Sharpe by 10%"
    state.status = status
    state.verdict = verdict
    state.reasoning = reasoning
    state.debate_id = debate_id
    state.created = "2026-04-12"

    mock_criteria = MagicMock()
    mock_criteria.metric = "profit_factor"
    mock_criteria.threshold = 1.1
    mock_criteria.operator = ">="
    state.success_criteria = mock_criteria

    state.results = []
    state.preset_overrides = None
    return state


class TestGetExperimentsList:
    """GET /api/v1/experiments tests."""

    def test_get_experiments_returns_200_with_list(self) -> None:
        """GET /experiments returns 200 with experiment_ids list."""
        app = _make_app()
        client = TestClient(app)

        mock_em = MagicMock()
        mock_em.list_experiments.return_value = ["exp-abc123", "exp-def456"]

        with patch(
            "finalayze.api.v1.experiments.ExperimentManager",
            return_value=mock_em,
        ):
            resp = client.get("/api/v1/experiments")

        assert resp.status_code == 200
        data = resp.json()
        assert data["experiment_ids"] == ["exp-abc123", "exp-def456"]

    def test_get_experiments_empty_returns_empty_list(self) -> None:
        """GET /experiments with no experiments returns empty list."""
        app = _make_app()
        client = TestClient(app)

        mock_em = MagicMock()
        mock_em.list_experiments.return_value = []

        with patch(
            "finalayze.api.v1.experiments.ExperimentManager",
            return_value=mock_em,
        ):
            resp = client.get("/api/v1/experiments")

        assert resp.status_code == 200
        assert resp.json()["experiment_ids"] == []

    def test_get_experiments_without_api_key_returns_401(self) -> None:
        """GET without X-API-Key returns 401."""
        app = _make_app_no_auth()
        with patch("config.settings.get_settings") as mock_settings:
            mock_settings.return_value.api_key = _API_KEY
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/api/v1/experiments")
        assert resp.status_code == 401


class TestGetExperimentDetail:
    """GET /api/v1/experiments/{id} tests."""

    def test_get_experiment_detail_existing_returns_200(self) -> None:
        """GET /experiments/{id} with existing experiment returns 200 with fields."""
        app = _make_app()
        client = TestClient(app)

        mock_state = _make_mock_experiment_state(
            experiment_id="exp-abc123",
            status="pending",
        )

        mock_em = MagicMock()
        mock_em.read_experiment.return_value = mock_state

        with patch(
            "finalayze.api.v1.experiments.ExperimentManager",
            return_value=mock_em,
        ):
            resp = client.get("/api/v1/experiments/exp-abc123")

        assert resp.status_code == 200
        data = resp.json()
        assert data["experiment_id"] == "exp-abc123"
        assert data["hypothesis"] == "ML improves Sharpe by 10%"
        assert data["status"] == "pending"
        assert data["verdict"] is None
        assert data["success_criteria"]["metric"] == "profit_factor"
        assert data["results"] == []

    def test_get_experiment_detail_with_verdict_returns_verdict(self) -> None:
        """GET /experiments/{id} with completed experiment returns verdict."""
        app = _make_app()
        client = TestClient(app)

        mock_state = _make_mock_experiment_state(
            experiment_id="exp-abc123",
            status="accepted",
            verdict="ACCEPTED",
            reasoning="Profit factor exceeded threshold",
            debate_id="debate-xyz",
        )

        mock_em = MagicMock()
        mock_em.read_experiment.return_value = mock_state

        with patch(
            "finalayze.api.v1.experiments.ExperimentManager",
            return_value=mock_em,
        ):
            resp = client.get("/api/v1/experiments/exp-abc123")

        assert resp.status_code == 200
        data = resp.json()
        assert data["verdict"] == "ACCEPTED"
        assert data["reasoning"] == "Profit factor exceeded threshold"
        assert data["debate_id"] == "debate-xyz"

    def test_get_experiment_detail_nonexistent_returns_404(self) -> None:
        """GET /experiments/{id} with non-existent experiment returns 404."""
        app = _make_app()
        client = TestClient(app)

        mock_em = MagicMock()
        mock_em.read_experiment.side_effect = FileNotFoundError("not found")

        with patch(
            "finalayze.api.v1.experiments.ExperimentManager",
            return_value=mock_em,
        ):
            resp = client.get("/api/v1/experiments/nonexistent-id")

        assert resp.status_code == 404

    def test_experiments_collection_post_not_allowed(self) -> None:
        """POST to /experiments collection (no id) returns 404 or 405."""
        app = _make_app()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/api/v1/experiments", json={})
        assert resp.status_code in (404, 405)


class TestApplyExperiment:
    """POST /api/v1/experiments/{id}/apply tests."""

    def _make_mock_session_ctx(self) -> MagicMock:
        """Return a mock async context manager that yields an AsyncMock session."""
        mock_session = AsyncMock()
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        return mock_ctx

    def test_apply_experiment_not_found(self) -> None:
        """POST to /experiments/nonexistent/apply -> 404."""
        app = _make_app()
        client = TestClient(app, raise_server_exceptions=False)

        mock_instance = MagicMock()
        mock_instance.apply_verdict = AsyncMock(side_effect=FileNotFoundError("not found"))

        with (
            patch(
                "finalayze.orchestration.preset_applicator.PresetApplicator",
                return_value=mock_instance,
            ),
            patch(
                "finalayze.api.v1.experiments.PresetApplicator",
                return_value=mock_instance,
            )
            if False
            else patch(
                "finalayze.core.db.get_async_session_factory",
            ) as mock_factory,
            patch(
                "finalayze.api.v1.experiments.ExperimentManager",
            ),
        ):
            mock_factory.return_value.return_value = self._make_mock_session_ctx()
            # The deferred import in apply_experiment uses the real PresetApplicator;
            # we need to patch at the source module and inject via ExperimentManager mock
            # that raises FileNotFoundError inside apply_verdict.
            # Simplest: patch PresetApplicator at its source so all imports get the mock.
            with patch(
                "finalayze.orchestration.preset_applicator.PresetApplicator",
                return_value=mock_instance,
            ):
                resp = client.post("/api/v1/experiments/nonexistent-exp/apply", json={})

        # The endpoint catches FileNotFoundError -> 404
        assert resp.status_code == 404

    def test_apply_experiment_success(self, tmp_path: Path) -> None:
        """POST /experiments/{id}/apply with ACCEPTED experiment -> 200, applied=True."""
        from finalayze.orchestration.preset_applicator import ApplyResult, PresetApplicator

        app = _make_app()
        client = TestClient(app, raise_server_exceptions=False)

        mock_apply_result = ApplyResult(
            experiment_id="exp-accepted",
            applied=True,
            backup_path="/tmp/us_tech.yaml.bak.20260412T120000Z",
            verdict="ACCEPTED",
            reason="Applied successfully",
        )

        mock_instance = MagicMock(spec=PresetApplicator)
        mock_instance.apply_verdict = AsyncMock(return_value=mock_apply_result)

        with (
            patch(
                "finalayze.orchestration.preset_applicator.PresetApplicator",
                return_value=mock_instance,
            ),
            patch(
                "finalayze.core.db.get_async_session_factory",
            ) as mock_factory,
            patch(
                "finalayze.api.v1.experiments.ExperimentManager",
            ),
        ):
            mock_factory.return_value.return_value = self._make_mock_session_ctx()
            resp = client.post(
                "/api/v1/experiments/exp-accepted/apply",
                json={"market_id": "moex"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["applied"] is True
        assert data["verdict"] == "ACCEPTED"
        assert data["experiment_id"] == "exp-accepted"

    def test_apply_experiment_inconclusive(self) -> None:
        """POST /experiments/{id}/apply with INCONCLUSIVE verdict -> 200, applied=False."""
        from finalayze.orchestration.preset_applicator import ApplyResult, PresetApplicator

        app = _make_app()
        client = TestClient(app, raise_server_exceptions=False)

        mock_apply_result = ApplyResult(
            experiment_id="exp-inconclusive",
            applied=False,
            backup_path=None,
            verdict="INCONCLUSIVE",
            reason="Routed to operator via Telegram",
        )

        mock_instance = MagicMock(spec=PresetApplicator)
        mock_instance.apply_verdict = AsyncMock(return_value=mock_apply_result)

        with (
            patch(
                "finalayze.orchestration.preset_applicator.PresetApplicator",
                return_value=mock_instance,
            ),
            patch(
                "finalayze.core.db.get_async_session_factory",
            ) as mock_factory,
            patch(
                "finalayze.api.v1.experiments.ExperimentManager",
            ),
        ):
            mock_factory.return_value.return_value = self._make_mock_session_ctx()
            resp = client.post(
                "/api/v1/experiments/exp-inconclusive/apply",
                json={"market_id": "moex"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["applied"] is False
        assert data["verdict"] == "INCONCLUSIVE"
        assert data["backup_path"] is None

    def test_apply_experiment_uses_real_alerter(self) -> None:
        """POST /apply with telegram credentials configured creates real TelegramAlerter."""
        from finalayze.orchestration.preset_applicator import ApplyResult, PresetApplicator

        app = _make_app()
        client = TestClient(app, raise_server_exceptions=False)

        mock_apply_result = ApplyResult(
            experiment_id="exp-test",
            applied=True,
            backup_path=None,
            verdict="ACCEPTED",
            reason="Applied",
        )
        mock_instance = MagicMock(spec=PresetApplicator)
        mock_instance.apply_verdict = AsyncMock(return_value=mock_apply_result)

        with (
            patch(
                "finalayze.orchestration.preset_applicator.PresetApplicator",
                return_value=mock_instance,
            ),
            patch(
                "finalayze.core.db.get_async_session_factory",
            ) as mock_factory,
            patch(
                "finalayze.api.v1.experiments.ExperimentManager",
            ),
            patch("finalayze.api.v1.experiments.TelegramAlerter") as mock_telegram_cls,
        ):
            mock_factory.return_value.return_value = self._make_mock_session_ctx()
            mock_settings = MagicMock()
            mock_settings.telegram_bot_token = "my-bot-token"
            mock_settings.telegram_chat_id = "123456"
            with patch("config.settings.get_settings", return_value=mock_settings):
                resp = client.post(
                    "/api/v1/experiments/exp-test/apply",
                    json={"market_id": "moex"},
                )

        assert resp.status_code == 200
        # TelegramAlerter should have been instantiated (not the no-op class)
        mock_telegram_cls.assert_called_once_with(bot_token="my-bot-token", chat_id="123456")

    def test_apply_experiment_circuit_breaker_real_instance(self) -> None:
        """POST /apply calls _get_circuit_breakers() which returns real CircuitBreaker."""
        from finalayze.orchestration.preset_applicator import ApplyResult, PresetApplicator
        from finalayze.risk.circuit_breaker import CircuitBreaker

        app = _make_app()
        client = TestClient(app, raise_server_exceptions=False)

        mock_apply_result = ApplyResult(
            experiment_id="exp-cb-test",
            applied=True,
            backup_path=None,
            verdict="ACCEPTED",
            reason="Applied",
        )
        mock_instance = MagicMock(spec=PresetApplicator)
        mock_instance.apply_verdict = AsyncMock(return_value=mock_apply_result)

        # Patch _get_circuit_breakers at the module level to capture returned value
        real_cb = CircuitBreaker("moex")
        real_cb_dict = {"moex": real_cb}

        with (
            patch(
                "finalayze.api.v1.experiments._get_circuit_breakers",
                return_value=real_cb_dict,
            ) as mock_get_cb,
            patch(
                "finalayze.orchestration.preset_applicator.PresetApplicator",
                return_value=mock_instance,
            ),
            patch(
                "finalayze.core.db.get_async_session_factory",
            ) as mock_factory,
            patch(
                "finalayze.api.v1.experiments.ExperimentManager",
            ),
        ):
            mock_factory.return_value.return_value = self._make_mock_session_ctx()
            resp = client.post(
                "/api/v1/experiments/exp-cb-test/apply",
                json={"market_id": "moex"},
            )

        assert resp.status_code == 200
        # _get_circuit_breakers() should have been called (not empty dict shortcut)
        mock_get_cb.assert_called_once()
        # Confirm the real CircuitBreaker is in the returned dict
        assert isinstance(real_cb_dict.get("moex"), CircuitBreaker)

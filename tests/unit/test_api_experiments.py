"""Tests for experiments REST API endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

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

    def test_experiments_has_no_write_endpoints(self) -> None:
        """Experiments router is read-only — no POST/PUT/PATCH/DELETE routes."""
        app = _make_app()
        # POST should return 405 Method Not Allowed (route exists but not POST)
        # or 404 if no route at all
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/api/v1/experiments", json={})
        assert resp.status_code in (404, 405)

"""Tests for debates REST API endpoints."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from finalayze.api.v1.auth import api_key_auth
from finalayze.api.v1.debates import router as debates_router

_API_KEY = "test-api-key"


def _make_app() -> FastAPI:
    """Create a minimal FastAPI app with debates router and overridden auth."""
    app = FastAPI()
    app.include_router(debates_router, prefix="/api/v1")
    app.dependency_overrides[api_key_auth] = lambda: None
    return app


def _make_app_no_auth() -> FastAPI:
    """Create a minimal FastAPI app with debates router but NO auth override."""
    app = FastAPI()
    app.include_router(debates_router, prefix="/api/v1")
    return app


def _make_agent_output_json() -> dict:
    """Return a minimal valid AgentOutput JSON for testing."""
    return {
        "agent_name": "quant-analyst",
        "recommendation": "Enable ml_ensemble for us_tech",
        "claims": [
            {
                "statement": "ML improves Sharpe by 10%",
                "source": {
                    "kind": "metric",
                    "metric_name": "sharpe",
                    "value": 0.137,
                    "iteration": "2026-04-05-adx-routing",
                },
                "confidence": 0.8,
            }
        ],
        "timestamp": datetime.now(UTC).isoformat(),
    }


def _make_conflict_output_json() -> dict:
    """Return a second AgentOutput that conflicts with the first."""
    return {
        "agent_name": "risk-officer",
        "recommendation": "Disable ml_ensemble for us_tech",
        "claims": [
            {
                "statement": "ML decreases Sharpe by 5%",
                "source": {
                    "kind": "metric",
                    "metric_name": "sharpe",
                    "value": 0.05,
                    "iteration": "2026-04-05-adx-routing",
                },
                "confidence": 0.7,
            }
        ],
        "timestamp": datetime.now(UTC).isoformat(),
    }


class TestPostDebates:
    """POST /api/v1/debates tests."""

    def test_post_debates_with_conflicts_returns_201(self) -> None:
        """POST with conflicting outputs returns 201 with debate_ids list."""
        app = _make_app()
        client = TestClient(app)

        mock_orch = MagicMock()
        mock_orch.run.return_value = ["debate-abc123"]

        with patch(
            "finalayze.api.v1.debates.AgentOrchestrator",
            return_value=mock_orch,
        ):
            resp = client.post(
                "/api/v1/debates",
                json={
                    "outputs": [
                        _make_agent_output_json(),
                        _make_conflict_output_json(),
                    ]
                },
            )

        assert resp.status_code == 201
        data = resp.json()
        assert data["debate_ids"] == ["debate-abc123"]
        assert data["conflicts_found"] == 1

    def test_post_debates_no_conflicts_returns_200(self) -> None:
        """POST with non-conflicting outputs returns 200 with debate_ids=[]."""
        app = _make_app()
        client = TestClient(app)

        mock_orch = MagicMock()
        mock_orch.run.return_value = []

        with patch(
            "finalayze.api.v1.debates.AgentOrchestrator",
            return_value=mock_orch,
        ):
            resp = client.post(
                "/api/v1/debates",
                json={"outputs": [_make_agent_output_json()]},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["debate_ids"] == []
        assert data["conflicts_found"] == 0

    def test_post_debates_without_api_key_returns_401(self) -> None:
        """POST without X-API-Key returns 401."""
        app = _make_app_no_auth()
        with patch("config.settings.get_settings") as mock_settings:
            mock_settings.return_value.api_key = _API_KEY
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post(
                "/api/v1/debates",
                json={"outputs": [_make_agent_output_json()]},
            )
        assert resp.status_code == 401


class TestGetDebatesList:
    """GET /api/v1/debates tests."""

    def test_get_debates_returns_200_with_list(self) -> None:
        """GET /debates returns 200 with debate_ids list."""
        app = _make_app()
        client = TestClient(app)

        mock_dm = MagicMock()
        mock_dm.list_debates.return_value = ["debate-abc123", "debate-def456"]

        with patch(
            "finalayze.api.v1.debates.DebateManager",
            return_value=mock_dm,
        ):
            resp = client.get("/api/v1/debates")

        assert resp.status_code == 200
        data = resp.json()
        assert data["debate_ids"] == ["debate-abc123", "debate-def456"]

    def test_get_debates_empty_returns_empty_list(self) -> None:
        """GET /debates with no debates returns empty list."""
        app = _make_app()
        client = TestClient(app)

        mock_dm = MagicMock()
        mock_dm.list_debates.return_value = []

        with patch(
            "finalayze.api.v1.debates.DebateManager",
            return_value=mock_dm,
        ):
            resp = client.get("/api/v1/debates")

        assert resp.status_code == 200
        assert resp.json()["debate_ids"] == []

    def test_get_debates_without_api_key_returns_401(self) -> None:
        """GET without X-API-Key returns 401."""
        app = _make_app_no_auth()
        with patch("config.settings.get_settings") as mock_settings:
            mock_settings.return_value.api_key = _API_KEY
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/api/v1/debates")
        assert resp.status_code == 401


class TestGetDebateDetail:
    """GET /api/v1/debates/{id} tests."""

    def test_get_debate_detail_existing_returns_200(self) -> None:
        """GET /debates/{id} with existing debate returns 200 with DebateState fields."""
        app = _make_app()
        client = TestClient(app)

        mock_state = MagicMock()
        mock_state.debate_id = "debate-abc123"
        mock_state.topic = "ML Sharpe vs. baseline"
        mock_state.status = "open"
        mock_state.created = "2026-04-12"
        mock_state.agents = ["quant-analyst", "risk-officer"]
        mock_state.resolution = None
        mock_state.experiment_id = None
        mock_state.arbiter_report = None

        mock_dm = MagicMock()
        mock_dm.read_debate.return_value = mock_state

        with patch(
            "finalayze.api.v1.debates.DebateManager",
            return_value=mock_dm,
        ):
            resp = client.get("/api/v1/debates/debate-abc123")

        assert resp.status_code == 200
        data = resp.json()
        assert data["debate_id"] == "debate-abc123"
        assert data["topic"] == "ML Sharpe vs. baseline"
        assert data["agents"] == ["quant-analyst", "risk-officer"]
        assert data["has_arbiter_report"] is False

    def test_get_debate_detail_with_arbiter_report_returns_has_report_true(self) -> None:
        """GET /debates/{id} with arbiter report sets has_arbiter_report=True."""
        app = _make_app()
        client = TestClient(app)

        mock_state = MagicMock()
        mock_state.debate_id = "debate-abc123"
        mock_state.topic = "ML Sharpe vs. baseline"
        mock_state.status = "resolved"
        mock_state.created = "2026-04-12"
        mock_state.agents = ["quant-analyst"]
        mock_state.resolution = "Claims verified"
        mock_state.experiment_id = None
        mock_state.arbiter_report = MagicMock()  # non-None => has_arbiter_report=True

        mock_dm = MagicMock()
        mock_dm.read_debate.return_value = mock_state

        with patch(
            "finalayze.api.v1.debates.DebateManager",
            return_value=mock_dm,
        ):
            resp = client.get("/api/v1/debates/debate-abc123")

        assert resp.status_code == 200
        assert resp.json()["has_arbiter_report"] is True

    def test_get_debate_detail_nonexistent_returns_404(self) -> None:
        """GET /debates/{id} with non-existent debate returns 404."""
        app = _make_app()
        client = TestClient(app)

        mock_dm = MagicMock()
        mock_dm.read_debate.side_effect = FileNotFoundError("not found")

        with patch(
            "finalayze.api.v1.debates.DebateManager",
            return_value=mock_dm,
        ):
            resp = client.get("/api/v1/debates/nonexistent-id")

        assert resp.status_code == 404


class TestPostDebatesMultiDebate:
    """POST /api/v1/debates multi-debate response tests (ORCH-02 fix)."""

    def test_post_debates_with_multiple_conflicts_returns_all_ids(self) -> None:
        """POST with 3 conflicts returns debate_ids list with all 3 IDs."""
        app = _make_app()
        client = TestClient(app)

        mock_orch = MagicMock()
        mock_orch.run.return_value = ["debate-id1", "debate-id2", "debate-id3"]

        with patch(
            "finalayze.api.v1.debates.AgentOrchestrator",
            return_value=mock_orch,
        ):
            resp = client.post(
                "/api/v1/debates",
                json={
                    "outputs": [
                        _make_agent_output_json(),
                        _make_conflict_output_json(),
                    ]
                },
            )

        assert resp.status_code == 201
        data = resp.json()
        assert data["debate_ids"] == ["debate-id1", "debate-id2", "debate-id3"]
        assert data["conflicts_found"] == 3

    def test_post_debates_no_conflicts_returns_empty_debate_ids(self) -> None:
        """POST with no conflicts returns debate_ids=[] and conflicts_found=0."""
        app = _make_app()
        client = TestClient(app)

        mock_orch = MagicMock()
        mock_orch.run.return_value = []

        with patch(
            "finalayze.api.v1.debates.AgentOrchestrator",
            return_value=mock_orch,
        ):
            resp = client.post(
                "/api/v1/debates",
                json={"outputs": [_make_agent_output_json()]},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["debate_ids"] == []
        assert data["conflicts_found"] == 0


class TestFinalizeDebate:
    """POST /api/v1/debates/{id}/finalize tests (ORCH-01)."""

    def _make_fact_check_report_json(
        self, debate_id: str = "debate-abc123", with_contradiction: bool = True
    ) -> dict:
        """Build a minimal FactCheckReport JSON body."""
        return {
            "report": {
                "debate_id": debate_id,
                "arbiter_timestamp": datetime.now(UTC).isoformat(),
                "results": [
                    {
                        "claim": {
                            "statement": "ML improves Sharpe by 10%",
                            "source": {
                                "kind": "metric",
                                "metric_name": "sharpe",
                                "value": 0.137,
                                "iteration": "2026-04-05-adx-routing",
                            },
                            "confidence": 0.8,
                        },
                        "verdict": "contradicted" if with_contradiction else "verified",
                        "evidence": (
                            "Contradicted by risk-officer" if with_contradiction else "Verified"
                        ),
                    }
                ],
            }
        }

    def test_finalize_debate_with_contradictions_returns_experiment_id(self) -> None:
        """POST /debates/{id}/finalize with contradictions returns experiment_id."""
        app = _make_app()
        client = TestClient(app)

        mock_orch = MagicMock()
        mock_orch.finalize_debate.return_value = "exp-debate-abc12"

        with patch(
            "finalayze.api.v1.debates.AgentOrchestrator",
            return_value=mock_orch,
        ):
            resp = client.post(
                "/api/v1/debates/debate-abc123/finalize",
                json=self._make_fact_check_report_json(
                    debate_id="debate-abc123", with_contradiction=True
                ),
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["debate_id"] == "debate-abc123"
        assert data["experiment_id"] == "exp-debate-abc12"
        assert data["resolved"] is False

    def test_finalize_debate_no_contradictions_returns_resolved(self) -> None:
        """POST /debates/{id}/finalize with no contradictions returns resolved=True."""
        app = _make_app()
        client = TestClient(app)

        mock_orch = MagicMock()
        mock_orch.finalize_debate.return_value = None

        with patch(
            "finalayze.api.v1.debates.AgentOrchestrator",
            return_value=mock_orch,
        ):
            resp = client.post(
                "/api/v1/debates/debate-xyz/finalize",
                json=self._make_fact_check_report_json(
                    debate_id="debate-xyz", with_contradiction=False
                ),
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["debate_id"] == "debate-xyz"
        assert data["experiment_id"] is None
        assert data["resolved"] is True

    def test_finalize_debate_nonexistent_returns_404(self) -> None:
        """POST /debates/{id}/finalize with nonexistent debate returns 404."""
        app = _make_app()
        client = TestClient(app, raise_server_exceptions=False)

        mock_orch = MagicMock()
        mock_orch.finalize_debate.side_effect = FileNotFoundError("debate not found")

        with patch(
            "finalayze.api.v1.debates.AgentOrchestrator",
            return_value=mock_orch,
        ):
            resp = client.post(
                "/api/v1/debates/nonexistent-debate/finalize",
                json=self._make_fact_check_report_json(debate_id="nonexistent-debate"),
            )

        assert resp.status_code == 404

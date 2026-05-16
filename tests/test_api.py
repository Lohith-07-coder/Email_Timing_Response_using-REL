"""
Integration tests for the FastAPI inference service.

Run: pytest tests/test_api.py -v
(Requires the app to be importable; model need not be loaded — mocked below.)
"""

import os
import sys
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

# Make project root importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def client_with_mock_agent():
    """
    Import the app with the agent pre-patched so no model file is required.
    """
    import app.main as app_module  # noqa: F401

    mock_agent = MagicMock()
    # Policy net returns Q-values where action 0 has highest value
    mock_q = MagicMock()
    mock_q.argmax.return_value = MagicMock(item=lambda: 0)
    import torch

    mock_agent._policy_net.return_value = torch.tensor([[1.5, 0.2, -0.1, 0.3]])

    app_module._agent = mock_agent
    app_module._model_version = "test-v1"

    from app.main import app

    return TestClient(app)


@pytest.fixture
def valid_payload():
    return {
        "subject": "Meeting rescheduled",
        "sender": "manager@company.com",
        "priority": 3,
        "sender_importance": 3,
        "waiting_time": 10,
        "workload": 2,
        "time_of_day": 14,
    }


# ── Health endpoint ───────────────────────────────────────────────────────────


class TestHealthEndpoint:
    def test_health_ok(self, client_with_mock_agent):
        r = client_with_mock_agent.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "uptime_seconds" in data

    def test_health_model_loaded(self, client_with_mock_agent):
        r = client_with_mock_agent.get("/health")
        assert r.json()["model_loaded"] is True


# ── Predict endpoint ──────────────────────────────────────────────────────────


class TestPredictEndpoint:
    def test_predict_success(self, client_with_mock_agent, valid_payload):
        r = client_with_mock_agent.post("/predict", json=valid_payload)
        assert r.status_code == 200
        data = r.json()
        assert "action_id" in data
        assert "action_label" in data
        assert 0 <= data["action_id"] <= 3
        assert data["action_label"] in (
            "reply_now",
            "delay_reply",
            "mark_important",
            "archive",
        )

    def test_predict_returns_state_vector(self, client_with_mock_agent, valid_payload):
        r = client_with_mock_agent.post("/predict", json=valid_payload)
        data = r.json()
        assert "state_vector" in data
        assert len(data["state_vector"]) == 5
        for v in data["state_vector"]:
            assert 0.0 <= v <= 1.0, "State values should be normalized to [0, 1]"

    def test_predict_invalid_priority(self, client_with_mock_agent, valid_payload):
        payload = {**valid_payload, "priority": 5}  # out of range
        r = client_with_mock_agent.post("/predict", json=payload)
        assert r.status_code == 422  # validation error

    def test_predict_missing_field(self, client_with_mock_agent):
        r = client_with_mock_agent.post("/predict", json={"subject": "Test"})
        assert r.status_code == 422

    def test_predict_time_of_day_boundary(self, client_with_mock_agent, valid_payload):
        for hour in [0, 12, 23]:
            payload = {**valid_payload, "time_of_day": hour}
            r = client_with_mock_agent.post("/predict", json=payload)
            assert r.status_code == 200

    def test_predict_confidence_range(self, client_with_mock_agent, valid_payload):
        r = client_with_mock_agent.post("/predict", json=valid_payload)
        data = r.json()
        assert 0.0 <= data["confidence"] <= 1.0


# ── Model version endpoint ────────────────────────────────────────────────────


class TestModelVersionEndpoint:
    def test_model_version_returns_200(self, client_with_mock_agent):
        r = client_with_mock_agent.get("/model/version")
        assert r.status_code == 200
        data = r.json()
        assert "current_version" in data
        assert "available_versions" in data

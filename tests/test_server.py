"""HTTP contract: /v1/systemone, /health, and the dashboard route."""
import pytest
from fastapi.testclient import TestClient

from localjev import lmstudio, server
from tests.conftest import fake_backend, probs


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(lmstudio, "resolve_model", lambda: "mock-llm")
    monkeypatch.setattr(lmstudio, "list_models", lambda: ["mock-llm"])
    monkeypatch.setattr(
        lmstudio, "model_catalog",
        lambda: [{"id": "mock-llm", "state": "loaded", "type": "llm"}],
    )
    monkeypatch.setattr(
        lmstudio, "first_token_logprobs",
        fake_backend(probs(0.82, 0.13, 0.05)),
    )
    return TestClient(server.app)


def test_systemone_returns_jev_shape(client):
    body = {
        "state": "payouts failing 3 days, refund now",
        "questions": {
            "department": {"type": "choice", "instructions": "team?",
                           "criteria": {"billing": "pay", "technical": "bug", "sales": "price"}},
            "is_urgent": {"type": "noul", "instructions": "urgent?",
                          "criteria": {"true": "time-sensitive", "false": "no"}},
        },
    }
    r = client.post("/v1/systemone", json=body)
    assert r.status_code == 200
    data = r.json()
    assert data["model"] == "mock-llm"
    assert data["answers"]["department"]["choice"] == "billing"
    assert set(data.keys()) == {"model", "answers", "usage", "latency_ms"}


def test_unknown_question_type_is_400(client):
    r = client.post("/v1/systemone", json={"state": "x", "questions": {"q": {"type": "bogus"}}})
    assert r.status_code == 400
    assert "unknown type" in r.json()["detail"]


def test_missing_fields_is_422_or_400(client):
    # pydantic rejects a missing required field before our handler runs
    r = client.post("/v1/systemone", json={"state": "x"})
    assert r.status_code in (400, 422)


def test_health_reports_model_when_reachable(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model"] == "mock-llm"
    assert body["models"] == ["mock-llm"]
    assert body["catalog"][0]["state"] == "loaded"
    assert body["lmstudio_url"] == lmstudio.BASE_URL


def test_health_503_when_lmstudio_down(monkeypatch):
    def boom():
        raise lmstudio.LMStudioError("Could not reach LM Studio")
    monkeypatch.setattr(lmstudio, "model_catalog", boom)
    monkeypatch.setattr(lmstudio, "resolve_model", boom)
    r = TestClient(server.app).get("/health")
    assert r.status_code == 503
    assert r.json()["status"] == "unavailable"


def test_models_endpoint_lists_and_defaults(client, monkeypatch):
    monkeypatch.setattr(lmstudio, "list_models", lambda: ["bonsai-1.7b", "muse-glimmer-30b"])
    r = client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["default"] == "mock-llm"
    assert "bonsai-1.7b" in body["models"]
    assert body["catalog"][0]["id"] == "mock-llm"


def test_dashboard_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "LocalJev" in r.text

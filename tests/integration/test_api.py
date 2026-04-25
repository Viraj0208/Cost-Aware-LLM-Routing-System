"""Integration tests for the FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.dependencies import reset_singletons


@pytest.fixture(autouse=True)
def clean_singletons():
    reset_singletons()
    yield
    reset_singletons()


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


def test_root_endpoint(client):
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "Cost-Aware LLM Routing System"
    assert data["mode"] == "simulation"


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "models" in data


def test_models_endpoint(client):
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["models"]) >= 2


def test_completion_simple_prompt(client):
    resp = client.post("/v1/completions", json={
        "prompt": "What is the capital of France?",
        "max_tokens": 128,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["text"]
    assert data["model_tier"] == "small"
    assert data["routing"]["complexity_score"] < 0.5
    assert data["cost"]["cost_usd"] > 0
    assert data["usage"]["total_tokens"] > 0


def test_completion_complex_prompt(client):
    resp = client.post("/v1/completions", json={
        "prompt": (
            "Implement a distributed hash table in Python with consistent hashing. "
            "Explain the algorithm step by step and analyze time complexity."
        ),
        "max_tokens": 256,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_tier"] == "large"


def test_completion_forced_model(client):
    resp = client.post("/v1/completions", json={
        "prompt": "Hello",
        "force_model": "llama-2-70b",
    })
    assert resp.status_code == 200
    assert resp.json()["model_used"] == "llama-2-70b"


def test_route_only_endpoint(client):
    resp = client.post("/v1/route", json={
        "prompt": "What is Python?",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "target_model" in data
    assert "complexity_score" in data
    assert "reasoning" in data
    assert len(data["reasoning"]) >= 1


def test_cost_analytics(client):
    # Make a few requests first
    for prompt in ["Hi", "What is AI?", "Define machine learning"]:
        client.post("/v1/completions", json={"prompt": prompt})

    resp = client.get("/v1/analytics/costs")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_requests"] == 3
    assert data["total_cost_usd"] > 0


def test_routing_analytics(client):
    client.post("/v1/completions", json={"prompt": "Hello"})
    resp = client.get("/v1/analytics/routing")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_requests"] == 1


def test_reset_analytics(client):
    client.post("/v1/completions", json={"prompt": "Test"})
    resp = client.post("/v1/analytics/reset")
    assert resp.status_code == 200

    resp = client.get("/v1/analytics/costs")
    assert resp.json()["total_requests"] == 0


def test_metrics_endpoint(client):
    client.post("/v1/completions", json={"prompt": "Test for metrics"})
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "llm_requests_total" in resp.text
    assert "llm_routing_latency_seconds" in resp.text


def test_invalid_request(client):
    resp = client.post("/v1/completions", json={"prompt": ""})
    assert resp.status_code == 422  # Validation error


def test_invalid_forced_model_returns_400(client):
    resp = client.post("/v1/completions", json={
        "prompt": "Hello",
        "force_model": "not-a-model",
    })
    assert resp.status_code == 400
    assert "Unknown force_model" in resp.json()["detail"]


def test_oversized_prompt_returns_400(client):
    resp = client.post("/v1/completions", json={
        "prompt": "word " * 500,
    })
    assert resp.status_code == 400
    assert "Prompt is too long" in resp.json()["detail"]


def test_response_has_timing_header(client):
    resp = client.get("/health")
    assert "X-Response-Time" in resp.headers

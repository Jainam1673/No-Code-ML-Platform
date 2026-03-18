from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_livez_endpoint() -> None:
    response = client.get("/livez")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"


def test_health_endpoint_contract() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"ok", "degraded"}
    assert "dependencies" in payload
    assert "database" in payload["dependencies"]
    assert "redis" in payload["dependencies"]


def test_system_capabilities_endpoint_contract() -> None:
    response = client.get("/v1/system/capabilities")
    assert response.status_code == 200
    payload = response.json()

    assert "runtime" in payload
    assert "accelerators" in payload
    assert payload["accelerators"]["hint"] in {"cpu", "gpu", "tpu"}

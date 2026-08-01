from fastapi.testclient import TestClient

from omni.router import GpuRef, Router
from omni.router_http import build_router_app


def _client(cap=1):
    return TestClient(build_router_app(Router([GpuRef("wss://g1", cap)])))


def test_allocate_returns_gpu_then_409_when_full():
    c = _client(1)
    r1 = c.post("/allocate", json={"camera_id": "cam1"})
    assert r1.status_code == 200
    assert r1.json()["gpu_ws_url"] == "wss://g1"
    assert "session_token" in r1.json()
    r2 = c.post("/allocate", json={"camera_id": "cam2"})
    assert r2.status_code == 409
    assert r2.json()["error"] == "capacity_exhausted"


def test_release_frees_a_slot():
    c = _client(1)
    tok = c.post("/allocate", json={"camera_id": "c"}).json()["session_token"]
    c.post("/release", json={"session_token": tok})
    assert c.post("/allocate", json={"camera_id": "c"}).status_code == 200


def test_health_reports_capacity():
    c = _client(2)
    body = c.get("/health").json()
    assert body["capacity"] == 2
    assert body["in_use"] == 0

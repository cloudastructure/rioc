"""HTTP wrapper around the Session Router core (omni/router.py).

Deployed in front of the GPU pool. Control-plane only: allocates a slot and hands
back the chosen GPU's direct ws_url + token. Rioc connects to the GPU directly.
"""
import os

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from omni.router import GpuRef, Router


def build_router_app(router: Router) -> FastAPI:
    app = FastAPI()

    @app.post("/allocate")
    async def allocate(body: dict):
        result = router.allocate(body.get("camera_id"))
        if result is None:
            return JSONResponse({"error": "capacity_exhausted"}, status_code=409)
        return result

    @app.post("/release")
    async def release(body: dict):
        router.release(body.get("session_token"))
        return {"ok": True}

    @app.get("/health")
    async def health():
        return {"in_use": router.fleet_in_use, "capacity": router.fleet_capacity}

    return app


def router_from_env() -> Router:
    """Build a Router from OMNI_GPUS ("wss://g1|4,wss://g2|4"); per-GPU cap defaults to LIVE_MAX_SESSIONS."""
    default_cap = int(os.environ.get("LIVE_MAX_SESSIONS", "4"))
    spec = os.environ.get("OMNI_GPUS", "")
    gpus = []
    for part in (p.strip() for p in spec.split(",")):
        if not part:
            continue
        url, _, cap = part.partition("|")
        gpus.append(GpuRef(url, int(cap) if cap else default_cap))
    return Router(gpus)

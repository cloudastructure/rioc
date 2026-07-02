"""Pick a GuardBackend based on the GUARD_BACKEND env var.

GUARD_BACKEND=vllm       (default) — VllmBackend, request/response HTTP.
GUARD_BACKEND=realtime            — RealtimeBackend (not implemented in this
                                    plan; see follow-up plan). Requires
                                    OpenBMB's realtime WebSocket server to
                                    be running on the MiniCPM-o host.
"""
import os

from guard_backend import GuardBackend
from vllm_backend import VllmBackend


def get_backend() -> GuardBackend:
    name = (os.environ.get("GUARD_BACKEND") or "vllm").strip().lower()
    if name == "vllm":
        return VllmBackend()
    if name == "realtime":
        raise NotImplementedError(
            "GUARD_BACKEND=realtime requires the OpenBMB realtime server "
            "to be running on the MiniCPM-o host. Not implemented in this "
            "plan — see the follow-up plan for the RealtimeBackend."
        )
    raise ValueError(
        f"Unknown GUARD_BACKEND: {name!r} (expected 'vllm' or 'realtime')"
    )

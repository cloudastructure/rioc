"""Live-mode SSE fan-out — the broadcast side of GET /live/stream.

Mirrors conversation_manager's SSE listener registry. The Live Session Orchestrator
is wired with `on_event=broadcast_live`; the /live/stream HTTP handler drains a queue
from register_live_listener(). Per-listener queue full => drop the event, never block.
"""
import asyncio

_listeners: list[asyncio.Queue] = []


def register_live_listener(maxsize: int = 50) -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
    _listeners.append(q)
    return q


def unregister_live_listener(q: asyncio.Queue) -> None:
    if q in _listeners:
        _listeners.remove(q)


def broadcast_live(event: dict) -> None:
    for q in list(_listeners):
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            pass  # slow/dead client — drop the event, never block the producer


def listener_count() -> int:
    return len(_listeners)

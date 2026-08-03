"""Rioc-side Live-channel WebSocket client.

Typed send/receive of protocol messages, auto-assigning seq + ts. `clock` is
injected for deterministic tests.
"""
import itertools

import websockets

from live import protocol as p


class ChannelClient:
    def __init__(self, ws=None, clock=None):
        self._ws = ws
        self._clock = clock or (lambda: 0.0)
        self._seq = itertools.count()

    async def connect(self, url):
        ssl_ctx = None
        if url.startswith("wss"):
            import ssl
            ssl_ctx = ssl.create_default_context()
            ssl_ctx.check_hostname = False
            ssl_ctx.verify_mode = ssl.CERT_NONE  # LAN/self-signed omni server
        self._ws = await websockets.connect(url, max_size=None, ssl=ssl_ctx)

    async def send(self, type, **payload):
        await self._ws.send(p.encode(p.make(type, next(self._seq), self._clock(), **payload)))

    async def messages(self):
        while True:
            try:
                raw = await self._ws.recv()
            except (StopAsyncIteration, websockets.ConnectionClosed):
                return
            yield p.decode(raw)

    async def await_admission(self):
        """Read downlink until the first admission verdict; return its type."""
        async for msg in self.messages():
            if msg["type"] in ("session_ready", "capacity_exhausted"):
                return msg["type"]
        return "capacity_exhausted"

    async def close(self):
        await self._ws.close()

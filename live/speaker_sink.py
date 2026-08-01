"""Real-time streaming speaker sink.

Plays a stream of μ-law chunks as they arrive with a small jitter buffer, and
exposes flush_and_stop() for instant local barge-in. `transport.send_chunk`
is the WS G.711 sender in production.
"""
import asyncio


class StreamingSpeakerSink:
    def __init__(self, transport, *, jitter_frames):
        self.transport = transport
        self.jitter_frames = jitter_frames
        self._q = asyncio.Queue()
        self._stopped = False

    async def feed(self, mulaw_chunk):
        await self._q.put(mulaw_chunk)

    async def run(self):
        # Prime the jitter buffer before first output.
        while self._q.qsize() < self.jitter_frames and not self._stopped:
            await asyncio.sleep(0.005)
        while not self._stopped:
            try:
                chunk = self._q.get_nowait()
            except asyncio.QueueEmpty:
                if self._stopped:
                    break
                await asyncio.sleep(0.005)
                continue
            self.transport.send_chunk(chunk)

    def flush_and_stop(self):
        dropped = self._q.qsize()
        while not self._q.empty():
            self._q.get_nowait()
        self._stopped = True
        return dropped

    @property
    def stopped(self):
        return self._stopped

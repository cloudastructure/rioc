import asyncio

from live.channel_client import ChannelClient
from live import protocol as p


class FakeWS:
    def __init__(self, inbound):
        self.sent = []
        self._in = list(inbound)

    async def send(self, s):
        self.sent.append(s)

    async def recv(self):
        if not self._in:
            raise StopAsyncIteration
        return self._in.pop(0)

    async def close(self):
        self.closed = True


def test_send_assigns_incrementing_seq_and_clock_ts():
    async def body():
        ws = FakeWS([])
        c = ChannelClient(ws=ws, clock=lambda: 7.0)
        await c.send("video", jpeg_b64="AA")
        await c.send("audio", pcm_or_opus_b64="BB")
        m0, m1 = p.decode(ws.sent[0]), p.decode(ws.sent[1])
        assert (m0["seq"], m0["ts"], m0["type"]) == (0, 7.0, "video")
        assert m1["seq"] == 1

    asyncio.run(body())


def test_messages_decodes_downlink():
    async def body():
        ws = FakeWS([p.encode(p.make("session_ready", 0, 0.0, session_id="x"))])
        c = ChannelClient(ws=ws, clock=lambda: 0.0)
        got = [m async for m in c.messages()]
        assert got[0]["type"] == "session_ready"

    asyncio.run(body())

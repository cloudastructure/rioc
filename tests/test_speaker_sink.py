import asyncio

from live.speaker_sink import StreamingSpeakerSink


class FakeTransport:
    def __init__(self):
        self.sent = []

    def send_chunk(self, mulaw):
        self.sent.append(mulaw)


def test_plays_queued_chunks_in_order():
    async def body():
        t = FakeTransport()
        s = StreamingSpeakerSink(t, jitter_frames=0)
        for c in (b"a", b"b", b"c"):
            await s.feed(c)
        task = asyncio.create_task(s.run())
        await asyncio.sleep(0.05)
        s.flush_and_stop()
        await task
        assert t.sent == [b"a", b"b", b"c"]

    asyncio.run(body())


def test_flush_and_stop_drops_pending_and_halts():
    async def body():
        t = FakeTransport()
        s = StreamingSpeakerSink(t, jitter_frames=100)  # never primes -> nothing sent yet
        for c in (b"a", b"b"):
            await s.feed(c)
        dropped = s.flush_and_stop()
        assert dropped == 2
        assert s.stopped is True

    asyncio.run(body())

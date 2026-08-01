import asyncio

from live.live_session import LiveSessionOrchestrator


class FakeChannel:
    def __init__(self, ready=True):
        self.sent = []
        self._ready = ready

    async def send(self, type, **payload):
        self.sent.append((type, payload))

    async def await_admission(self):
        return "session_ready" if self._ready else "capacity_exhausted"


class FakeSpeaker:
    def __init__(self):
        self.fed = []
        self.flushed = False

    async def feed(self, c):
        self.fed.append(c)

    def flush_and_stop(self):
        self.flushed = True
        return 0


class FakeEsc:
    def outcome(self, had_person_turn, state):
        return "Left"


def test_start_returns_false_and_emits_on_capacity():
    async def body():
        events = []
        o = LiveSessionOrchestrator(FakeChannel(ready=False), FakeSpeaker(), FakeEsc(),
                                    on_event=events.append)
        ok = await o.start(system_prompt="g", voice="default", camera_id="c1")
        assert ok is False
        assert any(e["type"] == "capacity" for e in events)

    asyncio.run(body())


def test_barge_in_flushes_speaker_and_sends_interrupt():
    async def body():
        ch, sp = FakeChannel(), FakeSpeaker()
        o = LiveSessionOrchestrator(ch, sp, FakeEsc(), on_event=lambda e: None)
        await o.barge_in()
        assert sp.flushed is True
        assert ("interrupt", {}) in ch.sent

    asyncio.run(body())


def test_speech_downlink_feeds_speaker():
    async def body():
        sp = FakeSpeaker()
        o = LiveSessionOrchestrator(FakeChannel(), sp, FakeEsc(), on_event=lambda e: None)
        await o.on_downlink({"type": "speech", "audio_b64": "QQ==", "text_delta": "hi"})
        assert sp.fed == [b"A"]   # base64 "QQ==" -> b"A"

    asyncio.run(body())

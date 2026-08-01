import asyncio

from live.escalation import EscalationOverlay
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


def test_advance_escalation_broadcasts_state_changes():
    async def body():
        events = []
        clock = {"t": 0.0}
        esc = EscalationOverlay(escalate_after=10.0, final_after=25.0)
        o = LiveSessionOrchestrator(FakeChannel(), FakeSpeaker(), esc,
                                    on_event=events.append, clock=lambda: clock["t"])
        await o.start(system_prompt="g", voice="default", camera_id="c1")  # started_at=0, state WARNING
        clock["t"] = 12.0
        assert o.advance_escalation() == "ESCALATING"
        clock["t"] = 30.0
        assert o.advance_escalation() == "FINAL"
        assert o.advance_escalation() is None   # no further change
        states = [e["state"] for e in events if e["type"] == "state"]
        assert states == ["WARNING", "ESCALATING", "FINAL"]

    asyncio.run(body())


def test_persistence_hooks_called_across_lifecycle():
    async def body():
        saved = []
        finished = []

        async def create_conv(camera_id):
            return 7

        async def save_turn(cid, speaker, text):
            saved.append((cid, speaker, text))

        async def finish_conv(cid, outcome):
            finished.append((cid, outcome))

        o = LiveSessionOrchestrator(
            FakeChannel(), FakeSpeaker(), FakeEsc(), on_event=lambda e: None,
            create_conv=create_conv, save_turn=save_turn, finish_conv=finish_conv,
        )
        await o.start(system_prompt="g", voice="default", camera_id="cam1")
        assert o.conversation_id == 7
        await o.on_downlink({"type": "turn_start"})
        await o.on_downlink({"type": "text", "delta": "Leave "})
        await o.on_downlink({"type": "text", "delta": "now."})
        await o.on_downlink({"type": "turn_end"})
        assert saved == [(7, "GUARD", "Leave now.")]
        outcome = await o.end("operator_stop")
        assert finished == [(7, outcome)]

    asyncio.run(body())

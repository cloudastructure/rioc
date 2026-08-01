from fastapi.testclient import TestClient

from omni.server import build_app
from omni import protocol as p


class FakeSession:
    def __init__(self, **kw):
        self.chunks = [("hello ", b"\x00"), ("intruder", b"\x01")]

    def prefill(self, **kw):
        pass

    def generate(self):
        for t, a in self.chunks:
            yield t, a

    def interrupt(self):
        pass

    def reset(self):
        pass


def _client(capacity=1):
    return TestClient(build_app(lambda **kw: FakeSession(**kw), capacity=capacity))


def test_session_ready_then_generate_on_speech_end():
    with _client().websocket_connect("/omni/session") as ws:
        ws.send_text(p.encode(p.make("session_start", 0, 0.0, system_prompt="g", voice="default")))
        assert p.decode(ws.receive_text())["type"] == "session_ready"
        ws.send_text(p.encode(p.make("user_speech_end", 1, 1.0)))
        types = [p.decode(ws.receive_text())["type"] for _ in range(4)]
        assert types == ["turn_start", "speech", "speech", "turn_end"]


def test_capacity_exhausted_when_full():
    client = _client(capacity=1)
    with client.websocket_connect("/omni/session") as ws1:
        ws1.send_text(p.encode(p.make("session_start", 0, 0.0, system_prompt="g", voice="default")))
        assert p.decode(ws1.receive_text())["type"] == "session_ready"
        with client.websocket_connect("/omni/session") as ws2:
            ws2.send_text(p.encode(p.make("session_start", 0, 0.0, system_prompt="g", voice="default")))
            assert p.decode(ws2.receive_text())["type"] == "capacity_exhausted"

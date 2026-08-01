import pytest
from fastapi.testclient import TestClient

from omni.server import build_app
from omni import protocol as p

pytestmark = pytest.mark.integration


class ScriptedSession:
    def __init__(self, **kw):
        self._i = 0

    def prefill(self, **kw):
        pass

    def generate(self):
        for t in ("this ", "is ", "private ", "property"):
            yield t, b"\x00\x10"

    def interrupt(self):
        self._i = 999

    def reset(self):
        pass


def test_generate_then_interrupt_midstream():
    app = build_app(lambda **kw: ScriptedSession(**kw), capacity=1)
    with TestClient(app).websocket_connect("/omni/session") as ws:
        ws.send_text(p.encode(p.make("session_start", 0, 0.0, system_prompt="g", voice="default")))
        assert p.decode(ws.receive_text())["type"] == "session_ready"
        ws.send_text(p.encode(p.make("user_speech_end", 1, 1.0)))
        assert p.decode(ws.receive_text())["type"] == "turn_start"
        # first speech frame arrives, then we barge in
        assert p.decode(ws.receive_text())["type"] == "speech"
        ws.send_text(p.encode(p.make("interrupt", 2, 1.2)))
        # server acknowledges the interrupt
        seen = [p.decode(ws.receive_text())["type"] for _ in range(2)]
        assert "interrupted" in seen

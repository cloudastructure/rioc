"""Live Session Orchestrator — sibling of the turn-based ConversationManager.

Owns the lifecycle of one live conversation: admission, downlink routing to the
speaker/captions, barge-in, and outcome classification. `on_event` is the SSE
broadcast hook to the web-ui.
"""
import base64


class LiveSessionOrchestrator:
    def __init__(self, channel, speaker, escalation, *, on_event):
        self.channel = channel
        self.speaker = speaker
        self.escalation = escalation
        self.on_event = on_event
        self._guard_speaking = False
        self._had_person_turn = False
        self._state = "WARNING"

    async def start(self, *, system_prompt, voice, camera_id):
        await self.channel.send("session_start", system_prompt=system_prompt,
                                voice=voice, camera_id=camera_id)
        verdict = await self.channel.await_admission()
        if verdict == "capacity_exhausted":
            self.on_event({"type": "capacity", "message": "Live capacity maxed out"})
            return False
        self.on_event({"type": "state", "state": self._state})
        return True

    async def on_downlink(self, msg):
        t = msg["type"]
        if t == "speech":
            await self.speaker.feed(base64.b64decode(msg["audio_b64"]))
            if msg.get("text_delta"):
                self.on_event({"type": "caption", "speaker": "GUARD", "delta": msg["text_delta"]})
        elif t == "turn_start":
            self._guard_speaking = True
            self.on_event({"type": "turn", "speaker": "GUARD", "phase": "start"})
        elif t == "turn_end":
            self._guard_speaking = False
            self.on_event({"type": "turn", "speaker": "GUARD", "phase": "end"})
        elif t == "interrupted":
            self.on_event({"type": "interrupted"})
        elif t == "error":
            self.on_event({"type": "error", **{k: msg.get(k) for k in ("code", "message")}})

    async def barge_in(self):
        self._had_person_turn = True
        self.speaker.flush_and_stop()
        await self.channel.send("interrupt")

    async def end(self, reason):
        await self.channel.send("session_end", reason=reason)
        outcome = self.escalation.outcome(self._had_person_turn, self._state)
        self.on_event({"type": "ended", "outcome": outcome})
        return outcome

"""Live Session Orchestrator — sibling of the turn-based ConversationManager.

Owns the lifecycle of one live conversation: admission, downlink routing to the
speaker/captions, barge-in, a time-driven escalation overlay, conversation
persistence, and outcome classification. `on_event` is the SSE broadcast hook to
the web-ui. Persistence + clock are injected so this stays unit-testable.
"""
import asyncio
import base64


class LiveSessionOrchestrator:
    def __init__(
        self, channel, speaker, escalation, *, on_event,
        create_conv=None, save_turn=None, finish_conv=None, clock=None,
    ):
        self.channel = channel
        self.speaker = speaker
        self.escalation = escalation
        self.on_event = on_event
        self._create_conv = create_conv      # async (camera_id) -> conversation_id
        self._save_turn = save_turn           # async (conversation_id, speaker, text) -> None
        self._finish_conv = finish_conv       # async (conversation_id, outcome) -> None
        self._clock = clock or (lambda: 0.0)
        self._guard_speaking = False
        self._had_person_turn = False
        self._state = "WARNING"
        self._conversation_id = None
        self._guard_buffer = ""
        self._started_at = 0.0

    async def start(self, *, system_prompt, voice, camera_id):
        await self.channel.send("session_start", system_prompt=system_prompt,
                                voice=voice, camera_id=camera_id)
        verdict = await self.channel.await_admission()
        if verdict == "capacity_exhausted":
            self.on_event({"type": "capacity", "message": "Live capacity maxed out"})
            return False
        self._started_at = self._clock()
        if self._create_conv is not None:
            self._conversation_id = await self._create_conv(camera_id)
        self.on_event({"type": "state", "state": self._state})
        return True

    # ── escalation (time/severity overlay) ──────────────────────────────────
    def elapsed(self):
        return self._clock() - self._started_at

    def advance_escalation(self):
        """Recompute the escalation level from elapsed time; broadcast + return it if it changed."""
        new_state = self.escalation.tick(self.elapsed())
        if new_state != self._state:
            self._state = new_state
            self.on_event({"type": "state", "state": new_state})
            return new_state
        return None

    async def run_escalation(self, interval=1.0):
        try:
            while True:
                await asyncio.sleep(interval)
                self.advance_escalation()
        except asyncio.CancelledError:
            pass

    # ── downlink routing ────────────────────────────────────────────────────
    async def on_downlink(self, msg):
        t = msg["type"]
        if t == "speech":
            await self.speaker.feed(base64.b64decode(msg["audio_b64"]))
            delta = msg.get("text_delta")
            if delta:
                self._guard_buffer += delta
                self.on_event({"type": "caption", "speaker": "GUARD", "delta": delta})
        elif t == "text":
            delta = msg.get("delta", "")
            self._guard_buffer += delta
            self.on_event({"type": "caption", "speaker": "GUARD", "delta": delta})
        elif t == "turn_start":
            self._guard_speaking = True
            self._guard_buffer = ""
            self.on_event({"type": "turn", "speaker": "GUARD", "phase": "start"})
        elif t == "turn_end":
            self._guard_speaking = False
            self.on_event({"type": "turn", "speaker": "GUARD", "phase": "end"})
            if self._guard_buffer and self._save_turn and self._conversation_id is not None:
                await self._save_turn(self._conversation_id, "GUARD", self._guard_buffer)
            self._guard_buffer = ""
        elif t == "interrupted":
            self.on_event({"type": "interrupted"})
        elif t == "error":
            self.on_event({"type": "error", **{k: msg.get(k) for k in ("code", "message")}})

    def note_person_spoke(self):
        """Called by the mic pump when the person speaks (even without a barge-in)."""
        self._had_person_turn = True

    async def barge_in(self):
        self._had_person_turn = True
        self.speaker.flush_and_stop()
        await self.channel.send("interrupt")

    async def end(self, reason):
        await self.channel.send("session_end", reason=reason)
        outcome = self.escalation.outcome(self._had_person_turn, self._state)
        if self._finish_conv is not None and self._conversation_id is not None:
            await self._finish_conv(self._conversation_id, outcome)
        self.on_event({"type": "ended", "outcome": outcome})
        return outcome

    @property
    def conversation_id(self):
        return self._conversation_id

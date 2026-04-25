"""AI Guard conversation manager.

State machine for reactive two-way voice conversations with detected intruders.

States: WARNING → ESCALATING → FINAL

Flow:
  1. YOLO detects a person → on_person_detected(frame_jpeg)
  2. Frame sent to MiniCPM-o → text + audio response
  3. Audio played on speaker
  4. Mic listener captures spoken reply
  5. Reply + fresh frame sent back to MiniCPM-o
  6. Loop up to max_turns; escalate if non-compliant
  7. Log full transcript to DB
"""
import asyncio
import logging
import os
import threading
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Awaitable

from db import init_db, create_conversation, add_turn, close_conversation
from minicpmo_client import chat as minicpmo_chat
from mic_listener import listen_for_response

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

CONVERSATION_MAX_TURNS = int(os.environ.get("CONVERSATION_MAX_TURNS", "6"))
CONVERSATION_COOLDOWN_SEC = float(os.environ.get("CONVERSATION_COOLDOWN_SEC", "20.0"))
ENABLE_REACTIVE_CONVERSATION = os.environ.get("ENABLE_REACTIVE_CONVERSATION", "1").strip().lower() in ("1", "true", "yes")
AUDIO_SAVE_DIR = Path(os.environ.get("AUDIO_SAVE_DIR", "./audio_logs"))

DEFAULT_SYSTEM_PROMPT = (
    "You are an AI security guard monitoring a restricted facility. "
    "An unauthorized person has been detected on camera and you are speaking to them live over a speaker.\n\n"
    "ABSOLUTE RULES — never break these under any circumstances:\n"
    "- You are ALWAYS the security guard speaking out loud. Never break character.\n"
    "- ALWAYS speak in first person: 'I can see you', 'I am monitoring this area', 'I will notify authorities'. "
    "NEVER mirror or echo the person's phrasing back at them. "
    "If they say 'You can see me?' respond 'Yes, I can see you clearly' — not 'You can see me through the camera'.\n"
    "- NEVER describe the scene, the room, or objects neutrally. "
    "You are not an image captioning tool. Do not say things like 'I see a gray object' or 'There is equipment near the wall'.\n"
    "- ALWAYS address the person directly using what you observe about them "
    "(their clothing, position, or actions). Example: 'You in the dark jacket — this area is restricted.'\n"
    "- Keep every response to 1-3 sentences. You are broadcasting over a speaker.\n"
    "- Be firm and authoritative but professional. Do not threaten physical harm.\n\n"
    "HOW TO HANDLE QUESTIONS — this is critical:\n"
    "- If the person asks a direct question (exit location, why restricted, what to do), ANSWER IT using "
    "what you can see in the camera frame, then redirect them to leave. "
    "Example: 'The exit is the door behind you to your left — use it now.'\n"
    "- If they ask why the area is restricted: give a brief honest reason (authorized personnel only, "
    "safety hazard, private property, etc.) then direct them out.\n"
    "- NEVER ignore or dismiss a direct question. Answering builds compliance. Always answer first, redirect second.\n"
    "- Use the camera frame to give specific directions: 'turn around', 'go back through the door you came in', "
    "'the hallway is to your right', etc.\n\n"
    "ESCALATION — only when the person refuses to leave or becomes confrontational:\n"
    "- Turn 1 (WARNING): Identify the person visually, state the area is restricted, and direct them to leave.\n"
    "- Turn 2 (ESCALATING): If they are still present and non-compliant, state that security is being notified.\n"
    "- Turn 3+ (FINAL): Authorities are en route. This is their last warning.\n"
    "- Do NOT escalate if they are asking questions and appear to be complying — answer and guide them out.\n"
    "- If they claim authorization: ask for their badge ID and state you are logging their response.\n\n"
    "A fresh camera frame is included with each exchange. "
    "Use it to observe the person's current position and surroundings so you can give accurate directions. "
    "Stay in character as the security guard for every single response without exception."
)

ESCALATION_MESSAGE = os.environ.get(
    "ESCALATION_MESSAGE",
    "This is your final warning. Authorities have been notified and are en route.",
)

# Injected as the text portion of every user turn sent to MiniCPM-o.
# Keeps the model anchored to its guard persona when it receives a bare image
# and prevents it from drifting into neutral scene description on later turns.
_TURN_ROLE_REMINDER = (
    "You are the AI security guard. "
    "The image shows the current state of the scene — use it to observe the person's position "
    "and surroundings so you can give accurate directions if asked. "
    "Respond only as the security guard speaking directly to the person. "
    "Always use first person — 'I can see you', 'I will notify security' — never echo the person's words back. "
    "If they asked a question, answer it briefly using what you see, then redirect them to leave."
)


class ConversationState(str, Enum):
    IDLE = "IDLE"
    WARNING = "WARNING"
    ESCALATING = "ESCALATING"
    FINAL = "FINAL"
    LISTENING = "LISTENING"
    SPEAKING = "SPEAKING"


class ConversationTurn:
    def __init__(self, speaker: str, text: str, timestamp: str, audio_path: str | None = None):
        self.speaker = speaker
        self.text = text
        self.timestamp = timestamp
        self.audio_path = audio_path

    def to_dict(self) -> dict:
        return {
            "speaker": self.speaker,
            "text": self.text,
            "timestamp": self.timestamp,
            "audio_path": self.audio_path,
        }


# ── Singleton manager ─────────────────────────────────────────────────────────

# Callbacks for broadcasting SSE events to connected clients
_sse_listeners: list[asyncio.Queue] = []


def register_sse_listener() -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue(maxsize=100)
    _sse_listeners.append(q)
    return q


def unregister_sse_listener(q: asyncio.Queue) -> None:
    try:
        _sse_listeners.remove(q)
    except ValueError:
        pass


def _broadcast(event: dict) -> None:
    for q in list(_sse_listeners):
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            pass


class ConversationManager:
    def __init__(
        self,
        play_audio_fn: Callable[[bytes, str], Awaitable[None]],
        get_frame_fn: Callable[[], bytes | None],
        camera_id: str | None = None,
        system_prompt: str | None = None,
        max_turns: int | None = None,
        speak_text_fn: Callable[[str], Awaitable[None]] | None = None,
        transcribe_fn: Callable[[bytes], Awaitable[str | None]] | None = None,
    ):
        """
        play_audio_fn:  async (wav_bytes, format) → None  (calls _guarded_play with force=True)
        get_frame_fn:   sync  () → jpeg_bytes | None      (called in thread)
        speak_text_fn:  async (text) → None               (OpenAI TTS fallback when VLM returns no audio)
        transcribe_fn:  async (wav_bytes) → str | None    (OpenAI Whisper transcription of person's audio)
        """
        self.play_audio_fn = play_audio_fn
        self.get_frame_fn = get_frame_fn
        self.speak_text_fn = speak_text_fn
        self.transcribe_fn = transcribe_fn
        self.camera_id = camera_id
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.max_turns = max_turns if max_turns is not None else CONVERSATION_MAX_TURNS

        self.state = ConversationState.IDLE
        self.turn_count = 0
        self.conversation_id: int | None = None
        self.turns: list[ConversationTurn] = []
        self.history: list[dict[str, Any]] = []  # MiniCPM-o message history
        self._lock = asyncio.Lock()
        self._active = False
        self._last_ended: float = 0.0
        self._speaking_event = threading.Event()  # set while TTS audio is playing
        self._audio_task: asyncio.Task | None = None  # background audio dispatch task

    # ── Public API ────────────────────────────────────────────────────────────

    def is_active(self) -> bool:
        return self._active

    def cooldown_remaining(self) -> float:
        import time
        elapsed = time.monotonic() - self._last_ended
        remaining = CONVERSATION_COOLDOWN_SEC - elapsed
        return max(0.0, remaining)

    def get_status(self) -> dict:
        return {
            "state": self.state,
            "turn_count": self.turn_count,
            "conversation_id": self.conversation_id,
            "turns": [t.to_dict() for t in self.turns],
            "cooldown_remaining": self.cooldown_remaining(),
        }

    async def on_person_detected(self, jpeg_bytes: bytes, initial_text: str | None = None) -> None:
        """Called by YOLO detection pipeline when a person is confirmed.

        Starts a new conversation if not already active and cooldown elapsed.
        initial_text: the detection alert message already spoken aloud — logged as the
        first GUARD turn so the Conversation Log always shows at least the initial warning.
        """
        async with self._lock:
            if self._active:
                return
            if self.cooldown_remaining() > 0:
                logger.debug("[ConvMgr] Cooldown active (%.1fs remaining)", self.cooldown_remaining())
                return
            self._active = True

        await self._run_conversation(jpeg_bytes, initial_text=initial_text)

    async def respond(self, audio_bytes: bytes) -> None:
        """Feed mic audio into the current conversation turn.

        This is called by POST /conversation/respond when reactive conversation
        is managed externally (e.g. triggered from the UI).
        """
        if not self._active:
            return
        await self._continue_conversation(audio_bytes)

    # ── Internal flow ─────────────────────────────────────────────────────────

    async def _run_conversation(self, jpeg_bytes: bytes, initial_text: str | None = None) -> None:
        """Full conversation lifecycle: init → turns → close."""
        AUDIO_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        await init_db()

        now = _now()
        self.conversation_id = await create_conversation(self.camera_id, now)
        self.turns = []
        self.history = []
        self.turn_count = 0
        self._set_state(ConversationState.WARNING)

        try:
            if initial_text:
                # The initial detection message was already spoken aloud by the Cloud AI.
                # Log it as GUARD turn 1 and skip _do_turn so we don't speak a second
                # opening message — go straight to listening for the person's reply.
                await self._log_initial_turn(initial_text)
            else:
                # No pre-spoken message: let MiniCPM-o generate the opening turn.
                await self._do_turn(jpeg_bytes)

            if ENABLE_REACTIVE_CONVERSATION:
                while self._active and self.turn_count < self.max_turns:
                    await self._listen_and_respond()
        except Exception as exc:
            logger.exception("[ConvMgr] Conversation error: %s", exc)
        finally:
            await self._end_conversation()

    async def _log_initial_turn(self, text: str) -> None:
        """Log the Cloud AI detection message as the first GUARD turn (already played aloud)."""
        timestamp = _now()
        turn = ConversationTurn("GUARD", text, timestamp, audio_path=None)
        self.turns.append(turn)
        self.turn_count += 1
        self.history.append({"role": "assistant", "content": text})
        await add_turn(self.conversation_id, "GUARD", text, timestamp, None)
        _broadcast({"type": "turn", "turn": turn.to_dict(), "state": self.state})
        logger.info("[ConvMgr] GUARD (initial): %s", text)

    async def _do_turn(self, jpeg_bytes: bytes) -> None:
        """Send frame (+ optional prior context) to MiniCPM-o, play response."""
        self._set_state(ConversationState.SPEAKING)
        try:
            text, wav_bytes = await minicpmo_chat(
                jpeg_bytes=jpeg_bytes,
                system_prompt=self.system_prompt,
                conversation_history=self.history,
                user_text=_TURN_ROLE_REMINDER,
            )
        except Exception as exc:
            logger.error("[ConvMgr] MiniCPM-o call failed: %s", exc)
            self._active = False
            return

        if not text:
            self._active = False
            return

        timestamp = _now()
        audio_path = await self._save_audio(wav_bytes, "guard", self.turn_count)
        turn = ConversationTurn("GUARD", text, timestamp, audio_path)
        self.turns.append(turn)
        self.turn_count += 1

        # Add to MiniCPM-o history
        self.history.append({"role": "assistant", "content": text})

        await add_turn(self.conversation_id, "GUARD", text, timestamp, audio_path)
        _broadcast({"type": "turn", "turn": turn.to_dict(), "state": self.state})
        logger.info("[ConvMgr] GUARD: %s", text)

        logger.info(
            "[ConvMgr] Audio dispatch — wav_bytes: %s  speak_text_fn: %s",
            f"{len(wav_bytes)} bytes" if wav_bytes else "None/empty",
            "set" if self.speak_text_fn else "NOT SET",
        )
        # Fire audio as a background task so _listen_and_respond can start
        # immediately. mic_listener.listen_for_response will block on
        # _speaking_event until _dispatch_audio clears it, which happens
        # precisely when the WebSocket send loop (or other delivery path)
        # finishes transmitting — giving true half-duplex sync with no
        # arbitrary timer offset.
        if self._audio_task and not self._audio_task.done():
            self._audio_task.cancel()
        self._speaking_event.set()
        self._audio_task = asyncio.create_task(self._dispatch_audio(wav_bytes, text))

        # Transition state immediately after launching audio (not after it completes)
        if self.turn_count == 1:
            self._set_state(ConversationState.WARNING)
        elif self.turn_count >= self.max_turns:
            self._set_state(ConversationState.FINAL)
        else:
            self._set_state(ConversationState.ESCALATING)

    async def _dispatch_audio(self, wav_bytes: bytes | None, text: str) -> None:
        """Deliver GUARD audio and release the speaking lock when transmission is complete.

        Runs as an asyncio background task so mic_listener can start immediately
        and block on _speaking_event.  The event is cleared precisely when the
        delivery coroutine (play_audio_fn / speak_text_fn) returns — for the
        WebSocket path that is the moment the last μ-law chunk has been sent,
        giving sample-accurate half-duplex control with no fixed timer.
        """
        try:
            if wav_bytes:
                logger.info("[ConvMgr] _dispatch_audio: play_audio_fn (%d bytes)", len(wav_bytes))
                await self.play_audio_fn(wav_bytes, "wav")
                logger.info("[ConvMgr] _dispatch_audio: play_audio_fn done")
            elif text and self.speak_text_fn:
                logger.info("[ConvMgr] _dispatch_audio: speak_text_fn for: %s", text[:60])
                await self.speak_text_fn(text)
                logger.info("[ConvMgr] _dispatch_audio: speak_text_fn done")
            else:
                logger.warning("[ConvMgr] _dispatch_audio: no audio path available")
        except asyncio.CancelledError:
            logger.info("[ConvMgr] _dispatch_audio: cancelled")
            raise
        except Exception as exc:
            logger.error("[ConvMgr] _dispatch_audio: error during audio delivery: %s", exc)
        finally:
            self._speaking_event.clear()
            logger.info("[ConvMgr] _dispatch_audio: _speaking_event cleared — mic can open")

    async def _listen_and_respond(self) -> None:
        """Listen for mic input and continue the conversation."""
        self._set_state(ConversationState.LISTENING)
        _broadcast({"type": "state", "state": self.state})

        wav_bytes = await asyncio.to_thread(listen_for_response, speaking_event=self._speaking_event)

        if not wav_bytes:
            logger.debug("[ConvMgr] No response heard — ending conversation")
            self._active = False
            return

        await self._continue_conversation(wav_bytes)

    async def _continue_conversation(self, audio_bytes: bytes) -> None:
        """Process person's audio reply and generate guard's next turn."""
        # Get a fresh frame for context
        jpeg_bytes = await asyncio.to_thread(self.get_frame_fn)
        if jpeg_bytes is None:
            self._active = False
            return

        timestamp = _now()
        audio_path = await self._save_audio(audio_bytes, "person", self.turn_count)

        # Transcribe person's audio so the log shows what they actually said.
        person_text = "[audio response]"
        if self.transcribe_fn:
            try:
                transcript = await self.transcribe_fn(audio_bytes)
                if transcript:
                    person_text = transcript
            except Exception as exc:
                logger.warning("[ConvMgr] Transcription failed: %s", exc)

        turn = ConversationTurn("PERSON", person_text, timestamp, audio_path)
        self.turns.append(turn)

        # Add user audio turn to history, with a text label so the model has
        # context for the audio and doesn't treat it as an ambiguous input.
        import base64
        b64 = base64.standard_b64encode(audio_bytes).decode("ascii")
        self.history.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "The unauthorized person has spoken (audio response follows):"},
                {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
            ],
        })

        await add_turn(self.conversation_id, "PERSON", person_text, timestamp, audio_path)
        _broadcast({"type": "turn", "turn": turn.to_dict(), "state": self.state})

        if self.turn_count >= self.max_turns:
            self._active = False
            return

        await self._do_turn(jpeg_bytes)

    async def _end_conversation(self) -> None:
        import time
        # Stop any in-flight audio immediately so the speaker doesn't keep
        # talking after the conversation has ended.
        if self._audio_task and not self._audio_task.done():
            self._audio_task.cancel()
            try:
                await self._audio_task
            except asyncio.CancelledError:
                pass
        self._speaking_event.clear()
        ended_at = _now()
        state = self.state

        # "Left" only when the person actually responded (at least one PERSON turn) and
        # the conversation ended without escalating — meaning they complied and left.
        # If only GUARD turns exist (no person response), outcome is Unknown.
        has_person_response = any(t.speaker == "PERSON" for t in self.turns)
        if state == ConversationState.FINAL:
            outcome = "Escalated"
        elif has_person_response:
            outcome = "Left"
        else:
            outcome = "Unknown"

        if self.conversation_id is not None:
            await close_conversation(self.conversation_id, ended_at, outcome)

        self._active = False
        self._last_ended = time.monotonic()
        self._set_state(ConversationState.IDLE)
        _broadcast({"type": "ended", "outcome": outcome, "conversation_id": self.conversation_id})
        logger.info("[ConvMgr] Conversation %s ended. Outcome: %s", self.conversation_id, outcome)

    def _set_state(self, state: ConversationState) -> None:
        self.state = state
        _broadcast({"type": "state", "state": state})
        logger.debug("[ConvMgr] State → %s", state)

    async def _save_audio(self, wav_bytes: bytes | None, speaker: str, turn: int) -> str | None:
        if not wav_bytes or not self.conversation_id:
            return None
        filename = f"conv_{self.conversation_id}_{speaker}_turn{turn}.wav"
        path = AUDIO_SAVE_DIR / filename
        try:
            import aiofiles
            async with aiofiles.open(path, "wb") as f:
                await f.write(wav_bytes)
            return str(path)
        except ImportError:
            # aiofiles not installed — write synchronously
            path.write_bytes(wav_bytes)
            return str(path)
        except Exception as exc:
            logger.warning("[ConvMgr] Could not save audio: %s", exc)
            return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# Module-level singleton (wired up in main.py)
_manager: ConversationManager | None = None


def get_manager() -> ConversationManager | None:
    return _manager


def set_manager(mgr: ConversationManager) -> None:
    global _manager
    _manager = mgr

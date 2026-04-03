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
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Awaitable

from db import init_db, create_conversation, add_turn, close_conversation
from minicpmo_client import chat as minicpmo_chat
from mic_listener import listen_for_response

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

CONVERSATION_MAX_TURNS = int(os.environ.get("CONVERSATION_MAX_TURNS", "3"))
CONVERSATION_COOLDOWN_SEC = float(os.environ.get("CONVERSATION_COOLDOWN_SEC", "20.0"))
ENABLE_REACTIVE_CONVERSATION = os.environ.get("ENABLE_REACTIVE_CONVERSATION", "1").strip().lower() in ("1", "true", "yes")
AUDIO_SAVE_DIR = Path(os.environ.get("AUDIO_SAVE_DIR", "./audio_logs"))

DEFAULT_SYSTEM_PROMPT = (
    "You are an AI security guard for a restricted facility. You have detected an unauthorized person on camera.\n\n"
    "Your behavior:\n"
    "- Be firm, clear, and professional. Do not be aggressive.\n"
    "- First warning: identify what you see and instruct them to leave immediately.\n"
    "- If they respond with non-compliance or excuses: escalate firmly, state that authorities are being notified.\n"
    "- If they claim authorization: ask for their badge ID and log their response.\n"
    "- Keep responses SHORT — 1-2 sentences maximum. You are speaking out loud.\n"
    "- Always address the person directly based on what you see (\"You in the red jacket...\")."
)

ESCALATION_MESSAGE = os.environ.get(
    "ESCALATION_MESSAGE",
    "This is your final warning. Authorities have been notified and are en route.",
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
    ):
        """
        play_audio_fn:  async (wav_bytes, format) → None  (calls _guarded_play with force=True)
        get_frame_fn:   sync  () → jpeg_bytes | None      (called in thread)
        speak_text_fn:  async (text) → None               (OpenAI TTS fallback when VLM returns no audio)
        """
        self.play_audio_fn = play_audio_fn
        self.get_frame_fn = get_frame_fn
        self.speak_text_fn = speak_text_fn
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

    async def on_person_detected(self, jpeg_bytes: bytes) -> None:
        """Called by YOLO detection pipeline when a person is confirmed.

        Starts a new conversation if not already active and cooldown elapsed.
        """
        async with self._lock:
            if self._active:
                return
            if self.cooldown_remaining() > 0:
                logger.debug("[ConvMgr] Cooldown active (%.1fs remaining)", self.cooldown_remaining())
                return
            self._active = True

        await self._run_conversation(jpeg_bytes)

    async def respond(self, audio_bytes: bytes) -> None:
        """Feed mic audio into the current conversation turn.

        This is called by POST /conversation/respond when reactive conversation
        is managed externally (e.g. triggered from the UI).
        """
        if not self._active:
            return
        await self._continue_conversation(audio_bytes)

    # ── Internal flow ─────────────────────────────────────────────────────────

    async def _run_conversation(self, jpeg_bytes: bytes) -> None:
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
            await self._do_turn(jpeg_bytes)

            if ENABLE_REACTIVE_CONVERSATION:
                while self._active and self.turn_count < self.max_turns:
                    await self._listen_and_respond()
        except Exception as exc:
            logger.exception("[ConvMgr] Conversation error: %s", exc)
        finally:
            await self._end_conversation()

    async def _do_turn(self, jpeg_bytes: bytes) -> None:
        """Send frame (+ optional prior context) to MiniCPM-o, play response."""
        self._set_state(ConversationState.SPEAKING)
        try:
            text, wav_bytes = await minicpmo_chat(
                jpeg_bytes=jpeg_bytes,
                system_prompt=self.system_prompt,
                conversation_history=self.history,
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

        if wav_bytes:
            await self.play_audio_fn(wav_bytes, "wav")
        elif text and self.speak_text_fn:
            # VLM returned text only (no audio) — fall back to OpenAI TTS
            logger.info("[ConvMgr] No audio from VLM — using TTS fallback for: %s", text[:80])
            await self.speak_text_fn(text)

        # Transition state
        if self.turn_count == 1:
            self._set_state(ConversationState.WARNING)
        elif self.turn_count >= self.max_turns:
            self._set_state(ConversationState.FINAL)
        else:
            self._set_state(ConversationState.ESCALATING)

    async def _listen_and_respond(self) -> None:
        """Listen for mic input and continue the conversation."""
        self._set_state(ConversationState.LISTENING)
        _broadcast({"type": "state", "state": self.state})

        wav_bytes = await asyncio.to_thread(listen_for_response)

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

        # Transcription placeholder: MiniCPM-o will transcribe internally when
        # audio is included in the next call.  We record the audio file for audit.
        timestamp = _now()
        audio_path = await self._save_audio(audio_bytes, "person", self.turn_count)
        # Use a placeholder text; the real transcript comes back in MiniCPM-o's response
        person_text = "[audio response]"
        turn = ConversationTurn("PERSON", person_text, timestamp, audio_path)
        self.turns.append(turn)

        # Add user audio turn to history
        import base64
        b64 = base64.standard_b64encode(audio_bytes).decode("ascii")
        self.history.append({
            "role": "user",
            "content": [
                {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}}
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
        ended_at = _now()
        state = self.state

        outcome_map = {
            ConversationState.FINAL: "Escalated",
            ConversationState.WARNING: "Unknown",
            ConversationState.ESCALATING: "Unknown",
        }
        outcome = outcome_map.get(state, "Unknown")
        if self.turn_count > 0 and state not in (ConversationState.FINAL, ConversationState.ESCALATING):
            outcome = "Left"

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


# Module-level singleton (wired up in webcam_stream.py)
_manager: ConversationManager | None = None


def get_manager() -> ConversationManager | None:
    return _manager


def set_manager(mgr: ConversationManager) -> None:
    global _manager
    _manager = mgr

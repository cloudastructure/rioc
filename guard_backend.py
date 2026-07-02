"""Backend abstraction for guard model calls.

Two implementations plug in behind this interface:
- vllm_backend.VllmBackend — request/response, wraps minicpmo_client.chat().
- realtime_backend.RealtimeBackend — websocket streaming (follow-up plan).

conversation_manager.ConversationManager depends on this interface only,
so switching backends is an env var change (see backend_factory.py).
"""
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class GuardEncounter(Protocol):
    """Session for a single guard/person encounter.

    Opened on person-detected. Closed on conversation end.
    """

    async def push_frame(self, jpeg_bytes: bytes) -> None:
        """Add a camera frame to the encounter.

        VllmBackend buffers up to N frames (ring; latest wins on overflow).
        RealtimeBackend sends the frame to the server immediately.
        """
        ...

    async def turn(
        self,
        audio_bytes: bytes | None,
        system_prompt: str,
        history: list[dict[str, Any]],
        user_text: str,
    ) -> tuple[str, bytes | None]:
        """Request the guard's next response.

        Returns (response_text, optional_wav_bytes). May return ("", None)
        if the model produced nothing usable.
        """
        ...

    async def close(self) -> None:
        """Release encounter resources. Must be idempotent."""
        ...


@runtime_checkable
class GuardBackend(Protocol):
    """Factory for GuardEncounter sessions."""

    async def open_encounter(self) -> GuardEncounter:
        ...

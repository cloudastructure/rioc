"""GuardBackend implementation using vLLM's OpenAI-compatible HTTP endpoint.

Wraps minicpmo_client.chat() with a session/ring-buffer shape so it fits the
GuardBackend interface. Set VLLM_FRAME_BUFFER_SIZE > 1 to buffer more frames
(currently only the newest frame is sent per turn — increasing the buffer is
a future frame-burst experiment).
"""
import os
from collections import deque
from typing import Any, Awaitable, Callable

from guard_backend import GuardBackend, GuardEncounter
from minicpmo_client import chat as minicpmo_chat


VLLM_FRAME_BUFFER_SIZE = int(os.environ.get("VLLM_FRAME_BUFFER_SIZE", "1"))


ChatFn = Callable[..., Awaitable[tuple[str, bytes | None]]]


class _VllmEncounter:
    def __init__(self, chat_fn: ChatFn, buffer_size: int) -> None:
        self._chat_fn = chat_fn
        self._frames: deque[bytes] = deque(maxlen=buffer_size)

    async def push_frame(self, jpeg_bytes: bytes) -> None:
        self._frames.append(jpeg_bytes)

    async def turn(
        self,
        audio_bytes: bytes | None,
        system_prompt: str,
        history: list[dict[str, Any]],
        user_text: str,
    ) -> tuple[str, bytes | None]:
        if not self._frames:
            raise RuntimeError(
                "VllmEncounter.turn called with no frame pushed since open_encounter()"
            )
        newest = self._frames[-1]
        return await self._chat_fn(
            jpeg_bytes=newest,
            system_prompt=system_prompt,
            conversation_history=history,
            audio_bytes=audio_bytes,
            user_text=user_text,
        )

    async def close(self) -> None:
        self._frames.clear()


class VllmBackend:
    def __init__(
        self,
        chat_fn: ChatFn | None = None,
        buffer_size: int | None = None,
    ) -> None:
        self._chat_fn = chat_fn or minicpmo_chat
        self._buffer_size = (
            buffer_size if buffer_size is not None else VLLM_FRAME_BUFFER_SIZE
        )

    async def open_encounter(self) -> GuardEncounter:
        return _VllmEncounter(self._chat_fn, self._buffer_size)

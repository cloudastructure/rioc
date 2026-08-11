"""Shared pytest fixtures for the rioc test suite."""
from typing import Any

import pytest


class FakeEncounter:
    def __init__(self, parent: "FakeBackend"):
        self._parent = parent
        self.pushed_frames: list[bytes] = []
        self.pushed_audio: list[bytes] = []
        self.turn_calls: list[dict[str, Any]] = []
        self.closed: bool = False

    async def push_frame(self, jpeg_bytes: bytes) -> None:
        self.pushed_frames.append(jpeg_bytes)

    async def turn(
        self,
        audio_bytes: bytes | None,
        system_prompt: str,
        history: list[dict[str, Any]],
        user_text: str,
    ) -> tuple[str, bytes | None]:
        if audio_bytes is not None:
            self.pushed_audio.append(audio_bytes)
        self.turn_calls.append({
            "audio_bytes": audio_bytes,
            "system_prompt": system_prompt,
            "history": list(history),
            "user_text": user_text,
        })
        return self._parent._response_text, self._parent._response_wav

    async def close(self) -> None:
        self.closed = True


class FakeBackend:
    def __init__(self) -> None:
        self._response_text: str = ""
        self._response_wav: bytes | None = None
        self.encounters: list[FakeEncounter] = []

    def script_response(self, text: str, wav: bytes | None = None) -> None:
        self._response_text = text
        self._response_wav = wav

    async def open_encounter(self) -> FakeEncounter:
        enc = FakeEncounter(self)
        self.encounters.append(enc)
        return enc


@pytest.fixture
def fake_backend() -> FakeBackend:
    return FakeBackend()

"""Tests for VllmBackend — a thin wrapper over minicpmo_client.chat()."""
import pytest

from vllm_backend import VllmBackend


class _FakeChat:
    """Records chat() calls and returns a scripted (text, wav) tuple."""

    def __init__(self, text: str = "response", wav: bytes | None = None):
        self._text = text
        self._wav = wav
        self.calls: list[dict] = []

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self._text, self._wav


async def test_open_encounter_returns_fresh_encounter_each_call():
    backend = VllmBackend(chat_fn=_FakeChat())
    e1 = await backend.open_encounter()
    e2 = await backend.open_encounter()
    assert e1 is not e2


async def test_turn_sends_newest_frame_and_returns_chat_result():
    chat = _FakeChat(text="halt", wav=b"WAV")
    backend = VllmBackend(chat_fn=chat, buffer_size=1)
    encounter = await backend.open_encounter()
    await encounter.push_frame(b"jpeg-1")

    text, wav = await encounter.turn(
        audio_bytes=None,
        system_prompt="sys",
        history=[{"role": "assistant", "content": "hi"}],
        user_text="you are the guard",
    )

    assert text == "halt"
    assert wav == b"WAV"
    assert len(chat.calls) == 1
    call = chat.calls[0]
    assert call["jpeg_bytes"] == b"jpeg-1"
    assert call["system_prompt"] == "sys"
    assert call["conversation_history"] == [{"role": "assistant", "content": "hi"}]
    assert call["audio_bytes"] is None
    assert call["user_text"] == "you are the guard"


async def test_turn_uses_newest_frame_when_buffer_overflows():
    chat = _FakeChat()
    backend = VllmBackend(chat_fn=chat, buffer_size=1)
    encounter = await backend.open_encounter()
    await encounter.push_frame(b"jpeg-old")
    await encounter.push_frame(b"jpeg-new")  # overwrites, buffer size 1

    await encounter.turn(audio_bytes=None, system_prompt="", history=[], user_text="")
    assert chat.calls[0]["jpeg_bytes"] == b"jpeg-new"


async def test_turn_forwards_audio_bytes():
    chat = _FakeChat()
    backend = VllmBackend(chat_fn=chat)
    encounter = await backend.open_encounter()
    await encounter.push_frame(b"jpeg")

    await encounter.turn(
        audio_bytes=b"WAV_IN",
        system_prompt="",
        history=[],
        user_text="",
    )
    assert chat.calls[0]["audio_bytes"] == b"WAV_IN"


async def test_turn_without_pushed_frame_raises():
    backend = VllmBackend(chat_fn=_FakeChat())
    encounter = await backend.open_encounter()
    with pytest.raises(RuntimeError, match="no frame"):
        await encounter.turn(
            audio_bytes=None, system_prompt="", history=[], user_text="",
        )


async def test_close_clears_buffer_and_is_idempotent():
    backend = VllmBackend(chat_fn=_FakeChat())
    encounter = await backend.open_encounter()
    await encounter.push_frame(b"jpeg")
    await encounter.close()
    await encounter.close()  # second close must not raise
    # After close, buffer is empty — turn must raise the empty-buffer error.
    with pytest.raises(RuntimeError):
        await encounter.turn(
            audio_bytes=None, system_prompt="", history=[], user_text="",
        )

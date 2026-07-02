"""ConversationManager uses the injected GuardBackend for each turn."""
from unittest.mock import AsyncMock

import pytest

# Avoid pulling in main.py's heavy imports (cv2, sounddevice) — import
# conversation_manager directly.
from conversation_manager import ConversationManager


@pytest.fixture
def noop_deps():
    """Return the callable deps ConversationManager expects, all no-op."""
    return {
        "play_audio_fn": AsyncMock(),
        "get_frame_fn": lambda: b"jpeg-fresh",
        "speak_text_fn": AsyncMock(),
        "transcribe_fn": AsyncMock(return_value="hello"),
    }


async def test_manager_opens_encounter_on_person_detected(fake_backend, noop_deps, tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIO_SAVE_DIR", str(tmp_path))
    monkeypatch.setenv("CONVERSATION_MAX_TURNS", "1")
    monkeypatch.setenv("ENABLE_REACTIVE_CONVERSATION", "0")
    fake_backend.script_response(text="stop where you are", wav=b"WAV")

    mgr = ConversationManager(backend=fake_backend, **noop_deps)
    await mgr.on_person_detected(b"jpeg-initial")

    assert len(fake_backend.encounters) == 1
    encounter = fake_backend.encounters[0]
    assert encounter.pushed_frames == [b"jpeg-initial"]
    assert len(encounter.turn_calls) == 1
    assert encounter.closed is True


async def test_manager_forwards_system_prompt_and_history_to_turn(fake_backend, noop_deps, tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIO_SAVE_DIR", str(tmp_path))
    monkeypatch.setenv("CONVERSATION_MAX_TURNS", "1")
    monkeypatch.setenv("ENABLE_REACTIVE_CONVERSATION", "0")
    fake_backend.script_response(text="halt", wav=None)

    mgr = ConversationManager(backend=fake_backend, system_prompt="custom sys", **noop_deps)
    await mgr.on_person_detected(b"jpeg")

    call = fake_backend.encounters[0].turn_calls[0]
    assert call["system_prompt"] == "custom sys"
    assert call["user_text"]  # non-empty role reminder
    assert call["history"] == []  # first turn, empty history

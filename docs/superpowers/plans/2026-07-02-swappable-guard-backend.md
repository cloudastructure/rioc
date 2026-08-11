# Swappable Guard Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce a session-shaped backend interface for guard model calls so the conversation manager can talk to either vLLM (request/response, today) or OpenBMB's realtime WebSocket server (streaming, future) via an env var switch, without touching conversation code.

**Architecture:** Add a `GuardBackend` / `GuardEncounter` Protocol pair. `ConversationManager` calls `backend.open_encounter()` at start of a conversation and uses `encounter.push_frame()` + `encounter.turn()` per turn instead of importing `minicpmo_chat` directly. This plan implements the interface and the `VllmBackend` (thin wrapper over the existing `minicpmo_client.chat()`). The `RealtimeBackend` is stubbed with a clear `NotImplementedError` and is scoped to a follow-up plan that requires OpenBMB's realtime server to be running on the hosted box.

**Tech Stack:** Python 3.11+, FastAPI (existing), pytest + pytest-asyncio (new dev deps), httpx (existing).

## Global Constraints

- Behavior must be identical to today when `GUARD_BACKEND` is unset — no user-visible changes on the current path.
- `VLLM_FRAME_BUFFER_SIZE` defaults to `1` (send only the newest frame per turn, matching current behavior). Increasing it is a follow-up experiment, not part of this plan.
- No new required env vars: `GUARD_BACKEND` defaults to `vllm`.
- `minicpmo_client.chat()` signature stays as-is — the backend wraps it, does not rewrite it.
- Do not add a `RealtimeBackend` implementation in this plan. It requires the OpenBMB realtime server; scope it to a follow-up plan once the server is confirmed running.
- All new modules live at the repo root next to `minicpmo_client.py` (matches existing layout).
- New tests live under `tests/`.

---

## File Structure

- Create: `guard_backend.py` — `Protocol` classes: `GuardBackend`, `GuardEncounter`.
- Create: `vllm_backend.py` — `VllmBackend` implementation wrapping `minicpmo_client.chat()`.
- Create: `backend_factory.py` — reads `GUARD_BACKEND` env, returns a `GuardBackend`.
- Create: `tests/__init__.py` — empty, marks tests dir as a package.
- Create: `tests/conftest.py` — pytest fixtures including a `FakeBackend` used across tests.
- Create: `tests/test_guard_backend_contract.py` — contract test verifying `FakeBackend` implements the Protocol correctly.
- Create: `tests/test_vllm_backend.py` — tests for `VllmBackend` with a mocked `chat_fn`.
- Create: `tests/test_backend_factory.py` — tests for env-driven backend selection.
- Create: `tests/test_conversation_manager_backend.py` — verifies `ConversationManager` uses the backend correctly with a `FakeBackend`.
- Modify: `conversation_manager.py` — take a `GuardBackend` in constructor; use `encounter.push_frame` + `encounter.turn` instead of importing `minicpmo_chat` directly.
- Modify: `main.py` — construct a backend via `backend_factory.get_backend()` in `lifespan`; pass it to `ConversationManager`.
- Modify: `requirements.txt` — add `pytest`, `pytest-asyncio`.
- Modify: `.env.example` if it exists, otherwise add a `Backend selection` section to `DEPLOY_AWS.md`.

---

## Task 1: Add pytest scaffold

**Files:**
- Modify: `requirements.txt`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_smoke.py`
- Create: `pytest.ini`

**Interfaces:**
- Consumes: none
- Produces: a working `pytest` command in the repo (`pytest tests/` returns green with one passing smoke test); asyncio mode configured so later tests can use `async def`.

- [ ] **Step 1: Add pytest dev deps**

Modify `requirements.txt`, appending:

```
pytest>=8.0.0
pytest-asyncio>=0.23.0
```

- [ ] **Step 2: Install the new deps**

Run: `pip install -r requirements.txt`
Expected: pytest and pytest-asyncio installed without errors.

- [ ] **Step 3: Create pytest config**

Create `pytest.ini`:

```ini
[pytest]
asyncio_mode = auto
testpaths = tests
```

- [ ] **Step 4: Create tests package**

Create empty file `tests/__init__.py`.

- [ ] **Step 5: Create conftest with async marker check**

Create `tests/conftest.py`:

```python
"""Shared pytest fixtures for the rioc test suite."""
import pytest
```

- [ ] **Step 6: Write a smoke test**

Create `tests/test_smoke.py`:

```python
async def test_pytest_asyncio_is_wired_up():
    assert True
```

- [ ] **Step 7: Run the smoke test**

Run: `pytest tests/test_smoke.py -v`
Expected: `1 passed`.

- [ ] **Step 8: Commit**

```bash
git add requirements.txt pytest.ini tests/__init__.py tests/conftest.py tests/test_smoke.py
git commit -m "test: add pytest scaffold with asyncio auto mode"
```

---

## Task 2: Define the GuardBackend / GuardEncounter interface

**Files:**
- Create: `guard_backend.py`
- Modify: `tests/conftest.py` — add a `FakeBackend` fixture.
- Create: `tests/test_guard_backend_contract.py`

**Interfaces:**
- Consumes: none
- Produces:
  - `guard_backend.GuardEncounter` — Protocol with `async push_frame(jpeg_bytes: bytes) -> None`, `async turn(audio_bytes: bytes | None, system_prompt: str, history: list[dict], user_text: str) -> tuple[str, bytes | None]`, `async close() -> None`.
  - `guard_backend.GuardBackend` — Protocol with `async open_encounter() -> GuardEncounter`.
  - `tests.conftest.FakeBackend` and `tests.conftest.FakeEncounter` — in-memory implementations that record calls; used by later tasks.
  - `tests.conftest.fake_backend` — pytest fixture returning a fresh `FakeBackend`.

- [ ] **Step 1: Write the contract test**

Create `tests/test_guard_backend_contract.py`:

```python
"""Verifies the FakeBackend satisfies the GuardBackend Protocol shape.

If this test breaks, the Protocol changed and every real backend needs an audit.
"""
from guard_backend import GuardBackend, GuardEncounter


async def test_fake_backend_matches_protocol(fake_backend):
    assert isinstance(fake_backend, GuardBackend)
    encounter = await fake_backend.open_encounter()
    assert isinstance(encounter, GuardEncounter)


async def test_fake_encounter_records_pushed_frames(fake_backend):
    encounter = await fake_backend.open_encounter()
    await encounter.push_frame(b"jpeg-1")
    await encounter.push_frame(b"jpeg-2")
    assert encounter.pushed_frames == [b"jpeg-1", b"jpeg-2"]


async def test_fake_encounter_turn_returns_scripted_response(fake_backend):
    fake_backend.script_response(text="stop where you are", wav=b"WAVDATA")
    encounter = await fake_backend.open_encounter()
    await encounter.push_frame(b"jpeg-1")
    text, wav = await encounter.turn(
        audio_bytes=None,
        system_prompt="sys",
        history=[],
        user_text="you are the guard",
    )
    assert text == "stop where you are"
    assert wav == b"WAVDATA"


async def test_fake_encounter_close_is_idempotent(fake_backend):
    encounter = await fake_backend.open_encounter()
    await encounter.close()
    await encounter.close()  # must not raise
    assert encounter.closed is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_guard_backend_contract.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'guard_backend'` (or fixture missing).

- [ ] **Step 3: Create the Protocol definitions**

Create `guard_backend.py`:

```python
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
```

- [ ] **Step 4: Add FakeBackend to conftest**

Modify `tests/conftest.py`:

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_guard_backend_contract.py -v`
Expected: `4 passed`.

- [ ] **Step 6: Commit**

```bash
git add guard_backend.py tests/conftest.py tests/test_guard_backend_contract.py
git commit -m "feat: define GuardBackend / GuardEncounter Protocol"
```

---

## Task 3: Implement VllmBackend

**Files:**
- Create: `vllm_backend.py`
- Create: `tests/test_vllm_backend.py`

**Interfaces:**
- Consumes: `guard_backend.GuardBackend`, `guard_backend.GuardEncounter`; `minicpmo_client.chat` (existing async function with signature `chat(jpeg_bytes, system_prompt, conversation_history, audio_bytes=None, user_text=None) -> (text, wav_bytes)`).
- Produces:
  - `vllm_backend.VllmBackend()` with optional `chat_fn` and `buffer_size` constructor args (default: `minicpmo_client.chat` and env `VLLM_FRAME_BUFFER_SIZE` (default 1)).
  - `vllm_backend.VllmBackend.open_encounter() -> _VllmEncounter`.
  - `_VllmEncounter.push_frame`, `.turn`, `.close` matching the `GuardEncounter` Protocol.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_vllm_backend.py`:

```python
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
    await encounter.close()  # must not raise
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_vllm_backend.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vllm_backend'`.

- [ ] **Step 3: Implement VllmBackend**

Create `vllm_backend.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_vllm_backend.py -v`
Expected: `6 passed`.

- [ ] **Step 5: Commit**

```bash
git add vllm_backend.py tests/test_vllm_backend.py
git commit -m "feat: add VllmBackend wrapping minicpmo_client.chat"
```

---

## Task 4: Wire ConversationManager through a GuardBackend

**Files:**
- Modify: `conversation_manager.py` — take a `backend: GuardBackend` in constructor; replace direct `minicpmo_chat` call with `encounter.push_frame` + `encounter.turn`; open encounter in `_run_conversation`, close in `_end_conversation`.
- Modify: `main.py:1451-1463` (approx.) — pass `VllmBackend()` to `ConversationManager(...)`.
- Create: `tests/test_conversation_manager_backend.py` — verify the manager drives the backend as expected using the `FakeBackend`.

**Interfaces:**
- Consumes: `guard_backend.GuardBackend`, `vllm_backend.VllmBackend`.
- Produces:
  - `ConversationManager.__init__` gains a required `backend: GuardBackend` keyword.
  - `ConversationManager` no longer imports `minicpmo_chat`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_conversation_manager_backend.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_conversation_manager_backend.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'backend'`.

- [ ] **Step 3: Update ConversationManager to accept a backend**

Modify `conversation_manager.py`.

Replace the import block near the top:

```python
from db import init_db, create_conversation, add_turn, close_conversation
from minicpmo_client import chat as minicpmo_chat
from mic_listener import listen_for_response
```

with:

```python
from db import init_db, create_conversation, add_turn, close_conversation
from guard_backend import GuardBackend, GuardEncounter
from mic_listener import listen_for_response
```

Change the `ConversationManager.__init__` signature to add `backend` as the first keyword-only argument:

```python
    def __init__(
        self,
        *,
        backend: GuardBackend,
        play_audio_fn: Callable[[bytes, str], Awaitable[None]],
        get_frame_fn: Callable[[], bytes | None],
        camera_id: str | None = None,
        system_prompt: str | None = None,
        max_turns: int | None = None,
        speak_text_fn: Callable[[str], Awaitable[None]] | None = None,
        transcribe_fn: Callable[[bytes], Awaitable[str | None]] | None = None,
    ):
```

Store `self.backend = backend` and add `self._encounter: GuardEncounter | None = None` to the body of `__init__`.

- [ ] **Step 4: Open the encounter in `_run_conversation`**

At the top of `_run_conversation`, right after `self._set_state(ConversationState.WARNING)`, add:

```python
        self._encounter = await self.backend.open_encounter()
```

- [ ] **Step 5: Replace `minicpmo_chat` call in `_do_turn`**

Locate the block in `_do_turn` that calls `minicpmo_chat(...)`:

```python
            text, wav_bytes = await minicpmo_chat(
                jpeg_bytes=jpeg_bytes,
                system_prompt=self.system_prompt,
                conversation_history=self.history,
                user_text=_TURN_ROLE_REMINDER,
            )
```

Replace with:

```python
            assert self._encounter is not None, "_do_turn called before _run_conversation opened an encounter"
            await self._encounter.push_frame(jpeg_bytes)
            text, wav_bytes = await self._encounter.turn(
                audio_bytes=None,
                system_prompt=self.system_prompt,
                history=self.history,
                user_text=_TURN_ROLE_REMINDER,
            )
```

- [ ] **Step 6: Close the encounter in `_end_conversation`**

In `_end_conversation`, after `self._speaking_event.clear()`, add:

```python
        if self._encounter is not None:
            try:
                await self._encounter.close()
            except Exception as exc:
                logger.warning("[ConvMgr] encounter.close() failed: %s", exc)
            self._encounter = None
```

- [ ] **Step 7: Run backend integration test to verify it passes**

Run: `pytest tests/test_conversation_manager_backend.py -v`
Expected: `2 passed`.

- [ ] **Step 8: Run the full test suite to catch regressions**

Run: `pytest tests/ -v`
Expected: all previously-passing tests still pass.

- [ ] **Step 9: Wire the backend into main.py's lifespan**

Modify `main.py` around lines 1457-1462 where `ConversationManager` is constructed.

Add the import inside the same `try:` block that already imports from `conversation_manager` (keeps error handling behavior — the existing block catches import failures and disables the conv manager):

```python
        from conversation_manager import ConversationManager, set_manager
        from db import init_db as _init_db
        from vllm_backend import VllmBackend
```

Change the `ConversationManager(...)` construction to prepend `backend=VllmBackend()` as a keyword arg:

```python
        _conv_manager = ConversationManager(
            backend=VllmBackend(),
            play_audio_fn=_play_for_conv,
            get_frame_fn=_get_conv_frame,
            speak_text_fn=lambda text: _speak_through_speaker(text, force=True),
            transcribe_fn=_transcribe_audio,
        )
```

- [ ] **Step 10: Smoke-check that main.py still imports cleanly**

Run: `python -c "import main"`
Expected: no import errors. (If cv2/sounddevice are unavailable in the environment, run the test app-level import from the machine where the service normally runs.)

- [ ] **Step 11: Commit**

```bash
git add conversation_manager.py main.py tests/test_conversation_manager_backend.py
git commit -m "refactor: route ConversationManager through GuardBackend interface"
```

---

## Task 5: Env-driven backend factory

**Files:**
- Create: `backend_factory.py`
- Create: `tests/test_backend_factory.py`
- Modify: `main.py` — replace direct `VllmBackend()` with `get_backend()`.

**Interfaces:**
- Consumes: `vllm_backend.VllmBackend`, `guard_backend.GuardBackend`.
- Produces: `backend_factory.get_backend() -> GuardBackend` — reads `GUARD_BACKEND` env (default `"vllm"`), returns the matching backend. Raises `NotImplementedError` for `"realtime"` with a clear message. Raises `ValueError` for anything else.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_backend_factory.py`:

```python
"""Tests for backend_factory.get_backend()."""
import pytest

from backend_factory import get_backend
from vllm_backend import VllmBackend


def test_default_backend_is_vllm(monkeypatch):
    monkeypatch.delenv("GUARD_BACKEND", raising=False)
    assert isinstance(get_backend(), VllmBackend)


def test_explicit_vllm(monkeypatch):
    monkeypatch.setenv("GUARD_BACKEND", "vllm")
    assert isinstance(get_backend(), VllmBackend)


def test_vllm_case_and_whitespace_insensitive(monkeypatch):
    monkeypatch.setenv("GUARD_BACKEND", "  VLLM  ")
    assert isinstance(get_backend(), VllmBackend)


def test_realtime_raises_not_implemented(monkeypatch):
    monkeypatch.setenv("GUARD_BACKEND", "realtime")
    with pytest.raises(NotImplementedError, match="OpenBMB realtime server"):
        get_backend()


def test_unknown_backend_raises_value_error(monkeypatch):
    monkeypatch.setenv("GUARD_BACKEND", "not-a-backend")
    with pytest.raises(ValueError, match="not-a-backend"):
        get_backend()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_backend_factory.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend_factory'`.

- [ ] **Step 3: Implement the factory**

Create `backend_factory.py`:

```python
"""Pick a GuardBackend based on the GUARD_BACKEND env var.

GUARD_BACKEND=vllm       (default) — VllmBackend, request/response HTTP.
GUARD_BACKEND=realtime            — RealtimeBackend (not implemented in this
                                    plan; see follow-up plan). Requires
                                    OpenBMB's realtime WebSocket server to
                                    be running on the MiniCPM-o host.
"""
import os

from guard_backend import GuardBackend
from vllm_backend import VllmBackend


def get_backend() -> GuardBackend:
    name = (os.environ.get("GUARD_BACKEND") or "vllm").strip().lower()
    if name == "vllm":
        return VllmBackend()
    if name == "realtime":
        raise NotImplementedError(
            "GUARD_BACKEND=realtime requires the OpenBMB realtime server "
            "to be running on the MiniCPM-o host. Not implemented in this "
            "plan — see the follow-up plan for the RealtimeBackend."
        )
    raise ValueError(
        f"Unknown GUARD_BACKEND: {name!r} (expected 'vllm' or 'realtime')"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_backend_factory.py -v`
Expected: `5 passed`.

- [ ] **Step 5: Wire factory into main.py**

Modify `main.py`. Inside the same `try:` block that imports `ConversationManager`, replace `from vllm_backend import VllmBackend` with:

```python
        from backend_factory import get_backend
```

Change the `ConversationManager(...)` construction to use `backend=get_backend()`:

```python
        _conv_manager = ConversationManager(
            backend=get_backend(),
            play_audio_fn=_play_for_conv,
            get_frame_fn=_get_conv_frame,
            speak_text_fn=lambda text: _speak_through_speaker(text, force=True),
            transcribe_fn=_transcribe_audio,
        )
```

- [ ] **Step 6: Run full test suite**

Run: `pytest tests/ -v`
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add backend_factory.py main.py tests/test_backend_factory.py
git commit -m "feat: env-driven guard backend selection (GUARD_BACKEND)"
```

---

## Task 6: Document the operational path

**Files:**
- Modify: `DEPLOY_AWS.md` — add a `Backend Selection` section.

**Interfaces:**
- Consumes: none.
- Produces: user-facing docs explaining the env switch and pointing at the follow-up plan for the realtime backend.

- [ ] **Step 1: Add a Backend Selection section to DEPLOY_AWS.md**

Append the following section to `DEPLOY_AWS.md`:

````markdown
---

## Backend Selection (GUARD_BACKEND)

The guard's model calls go through a `GuardBackend` interface so we can swap
between request/response and streaming servers without changing conversation
code.

| Value | Serving process | Transport | Status |
|-------|-----------------|-----------|--------|
| `vllm` (default) | vLLM's OpenAI-compatible HTTP server | HTTP `/v1/chat/completions` | Ready. |
| `realtime` | OpenBMB's MiniCPM-o realtime server | WebSocket | Not implemented — see follow-up plan; requires the realtime server to be running on the MiniCPM-o host. |

Set on the Mac side (or wherever `main.py` runs):

```bash
export GUARD_BACKEND=vllm   # default
```

`vllm` uses `MINICPMO_URL` / `CLOUD_AI_URL` as today. `realtime` will use a
separate `MINICPMO_REALTIME_URL` (websocket) once that backend lands.

Tunable while on `vllm`:

- `VLLM_FRAME_BUFFER_SIZE` (default `1`) — how many recent frames the
  backend buffers per encounter. Currently only the newest frame is sent per
  turn; larger buffers are reserved for a future frame-burst experiment.
````

- [ ] **Step 2: Commit**

```bash
git add DEPLOY_AWS.md
git commit -m "docs: describe GUARD_BACKEND selection and VLLM_FRAME_BUFFER_SIZE"
```

---

## Follow-up (out of scope for this plan)

- **RealtimeBackend implementation.** WebSocket client for OpenBMB's realtime MiniCPM-o server. Needs the server standing up on the hosted box (colleague action) and a review of that server's WS protocol (frame framing, audio framing, response streaming, session lifecycle). Scope to its own plan once the server is confirmed running.
- **Frame burst experiment.** Bump `VLLM_FRAME_BUFFER_SIZE` and extend `minicpmo_client.chat()` to accept multiple frames in one request. Measure latency and response quality against single-frame baseline.
- **CVR wiring for continuous video.** The current webhook sends one JPEG per detection event. Streaming backends need continuous frames — either the CVR pushes a video stream during an encounter, or the client pulls RTSP for the encounter duration. Decide during the RealtimeBackend plan.

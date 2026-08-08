# Live Multimodal Guard — Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new full-duplex "Live mode" to the AI Guard backend — a persistent MiniCPM-o omni session streamed continuous video+audio with barge-in — running alongside the existing turn-based engine.

**Architecture:** Approach B (smart edge + streaming GPU brain). A new **Omni Streaming Server** on the GPU box holds one stateful MiniCPM-o omni session and speaks a versioned bidirectional WebSocket ("the Live channel"). On the LAN, a new **Live Session Orchestrator** in Rioc bridges camera/mic/speaker to that channel and runs **barge-in detection at the edge** so interruption feels instant over a WAN. Overflow is a fixed-K admission error, not a fallback.

**Tech Stack:** Python 3 (rioc convention), FastAPI + uvicorn, `websockets` (client + server), `webrtcvad`, `sounddevice`, `numpy`, `httpx`, `aiosqlite`, `ffmpeg` (subprocess), `pytest` + `pytest-asyncio`. MiniCPM-o via its HuggingFace streaming model class on the GPU box.

## Global Constraints

- Single-process asyncio model per rioc convention; all blocking calls dispatched via `asyncio.to_thread`. (verbatim from design §3 / ARCHITECTURE §3)
- The existing turn-based `ConversationManager` path must remain **untouched and functional**. Live mode is additive.
- Mode selected by env `CONVERSATION_MODE = turn_based | live` (default `turn_based`), hot-swappable via `POST /configure`.
- Live capacity is a **static pool of GPUs behind a Session Router** (`omni/router.py`): each GPU has per-instance capacity `kᵢ` = `LIVE_MAX_SESSIONS` (> 1, provisional pending benchmark); fleet ceiling K = Σ `kᵢ`. Rioc calls `POST {router}/allocate` → least-loaded healthy GPU's **direct** `gpu_ws_url` + `session_token`; **media never proxies through the router**. Overflow **only when the whole pool is full** → `capacity_exhausted`; **no elastic autoscale, no fallback**. `Admission(capacity=kᵢ)` is capacity-agnostic; unit tests exercise the cap at a small value.
- Barge-in requires the WebSocket G.711 speaker path; live mode is **refused** for URL-only speakers at config time.
- Live conversation path uses MiniCPM-o **end-to-end** — no Whisper/OpenAI-TTS on the critical path. Whisper is logging-only and off-critical-path.
- No auth on any endpoint (LAN trust boundary), consistent with existing rioc.
- Every task ends green (`pytest` passing) and is committed.

---

## File structure

**GPU-side (new; deployed on the GPU box, not the appliance):**
- `omni/protocol.py` — Live-channel message schemas + encode/decode. Shared contract; also vendored to the Rioc side.
- `omni/omni_session.py` — thin adapter around the MiniCPM-o streaming API (`prefill`, `generate`, `interrupt`, `reset`).
- `omni/admission.py` — per-GPU session admission control.
- `omni/server.py` — FastAPI/uvicorn WebSocket server wiring protocol + admission + session.
- `omni/router.py` — control-plane Session Router: `/allocate` + occupancy tracking / leases across the static GPU pool (deployed as its own small service in front of the pool).

**Rioc-side (appliance):**
- `live/protocol.py` — copy of `omni/protocol.py` (single source vendored; keep in sync).
- `live/channel_client.py` — WebSocket client to the omni server; typed send/receive of protocol messages.
- `live/barge_in.py` — continuous edge VAD + reference-signal echo gating; emits speech/interrupt events.
- `live/speaker_sink.py` — streaming speaker playback with jitter buffer + `flush_and_stop()`; extends the WS G.711 path.
- `live/live_session.py` — Live Session Orchestrator: lifecycle, escalation timer, wiring, SSE broadcast.
- `main.py` (modify) — mode routing in the shared trigger, `/live/*` endpoints + SSE relay, `/configure` validation.
- `db.py` (modify) — `outcome` accepts `Error`; helper to append live turns.
- `whisper_log.py` (new) — async logging-only person-transcript backfill.

**Tests:** colocated under `tests/` mirroring the above.

---

### Task 1: Live-channel protocol module

**Files:**
- Create: `omni/protocol.py`
- Test: `tests/test_protocol.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `encode(msg: dict) -> str` and `decode(raw: str) -> dict` (JSON envelope with `type`, `seq`, `ts`).
  - Constant sets `UPLINK_TYPES` and `DOWNLINK_TYPES` (frozensets of the type strings).
  - `make(type: str, seq: int, ts: float, **payload) -> dict` builder that validates `type` is known and returns the envelope.
  - Raises `ProtocolError` on unknown type or missing `type`/`seq`/`ts`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_protocol.py
import pytest
from omni import protocol as p

def test_make_roundtrip():
    msg = p.make("video", seq=3, ts=1.5, jpeg_b64="AAAA")
    assert msg == {"type": "video", "seq": 3, "ts": 1.5, "jpeg_b64": "AAAA"}
    assert p.decode(p.encode(msg)) == msg

def test_make_rejects_unknown_type():
    with pytest.raises(p.ProtocolError):
        p.make("bogus", seq=1, ts=0.0)

def test_decode_rejects_missing_envelope_fields():
    with pytest.raises(p.ProtocolError):
        p.decode('{"type": "video"}')  # no seq/ts

def test_direction_sets_are_disjoint_and_known():
    assert "session_start" in p.UPLINK_TYPES
    assert "capacity_exhausted" in p.DOWNLINK_TYPES
    assert p.UPLINK_TYPES.isdisjoint(p.DOWNLINK_TYPES)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_protocol.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'omni.protocol'`

- [ ] **Step 3: Write minimal implementation**

```python
# omni/protocol.py
import json

class ProtocolError(Exception):
    pass

UPLINK_TYPES = frozenset({
    "session_start", "video", "audio",
    "user_speech_start", "user_speech_end", "interrupt", "session_end", "ping",
})
DOWNLINK_TYPES = frozenset({
    "session_ready", "capacity_exhausted", "speech", "text",
    "turn_start", "turn_end", "interrupted", "state", "error",
})
_ALL = UPLINK_TYPES | DOWNLINK_TYPES

def make(type, seq, ts, **payload):
    if type not in _ALL:
        raise ProtocolError(f"unknown type: {type}")
    return {"type": type, "seq": seq, "ts": ts, **payload}

def encode(msg):
    return json.dumps(msg, separators=(",", ":"))

def decode(raw):
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ProtocolError(f"bad json: {e}") from e
    for field in ("type", "seq", "ts"):
        if field not in msg:
            raise ProtocolError(f"missing envelope field: {field}")
    if msg["type"] not in _ALL:
        raise ProtocolError(f"unknown type: {msg['type']}")
    return msg
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_protocol.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add omni/protocol.py tests/test_protocol.py
git commit -m "feat(omni): live-channel protocol schema + codec"
```

---

### Task 2: MiniCPM-o streaming adapter (spike + adapter)

**Files:**
- Create: `omni/omni_session.py`
- Test: `tests/test_omni_session.py`

**Interfaces:**
- Consumes: nothing (owns the model handle).
- Produces: class `OmniSession` with:
  - `__init__(self, model, tokenizer, *, system_prompt: str, voice: str)`
  - `prefill(self, *, jpeg: bytes | None, pcm: bytes | None) -> None` — feed one chunk into the session context.
  - `generate(self) -> Iterator[tuple[str, bytes]]` — yields `(text_delta, audio_pcm)` chunks for one guard turn; stops when the turn completes or `interrupt()` is set.
  - `interrupt(self) -> None` — signal the active `generate()` to stop ASAP.
  - `reset(self) -> None` — clear session state.

**Spike note (do first, not a code step):** Confirm the exact MiniCPM-o streaming entry points on the pinned model (`model.streaming_prefill(...)`, `model.streaming_generate(...)`, session id handling, audio output format/sample rate) from the model card on the GPU box. Record the confirmed signatures as a comment block at the top of `omni_session.py`. The adapter below wraps whatever the real names are so downstream tasks never touch the raw model.

- [ ] **Step 1: Write the failing test (against a fake model — no GPU needed in CI)**

```python
# tests/test_omni_session.py
from omni.omni_session import OmniSession

class FakeModel:
    def __init__(self): self.prefills = []; self.aborted = False
    def streaming_prefill(self, session_id, content, tokenizer): self.prefills.append(content)
    def streaming_generate(self, session_id, tokenizer, stop_flag):
        for i in range(5):
            if stop_flag(): return
            yield {"text": f"w{i} ", "audio": b"\x00\x01"}

def test_prefill_forwards_chunks():
    m = FakeModel()
    s = OmniSession(m, tokenizer=None, system_prompt="guard", voice="default")
    s.prefill(jpeg=b"img", pcm=b"snd")
    assert len(m.prefills) == 1

def test_generate_yields_until_done():
    m = FakeModel()
    s = OmniSession(m, tokenizer=None, system_prompt="guard", voice="default")
    out = list(s.generate())
    assert [t for t, _ in out] == ["w0 ", "w1 ", "w2 ", "w3 ", "w4 "]

def test_interrupt_stops_generation_early():
    m = FakeModel()
    s = OmniSession(m, tokenizer=None, system_prompt="guard", voice="default")
    gen = s.generate()
    next(gen)                 # first chunk
    s.interrupt()
    assert list(gen) == []    # nothing more after interrupt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_omni_session.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'omni.omni_session'`

- [ ] **Step 3: Write minimal implementation**

```python
# omni/omni_session.py
# CONFIRMED MiniCPM-o streaming API (fill from spike):
#   model.streaming_prefill(session_id, {"role": "user", "content": [...]}, tokenizer)
#   model.streaming_generate(session_id, tokenizer, ...) -> iterator of {"text","audio"}
import itertools

class OmniSession:
    _ids = itertools.count()

    def __init__(self, model, tokenizer, *, system_prompt, voice):
        self.model = model
        self.tokenizer = tokenizer
        self.session_id = f"sess-{next(self._ids)}"
        self.voice = voice
        self._interrupt = False
        # Prime persona as the first prefill turn.
        self.model.streaming_prefill(
            self.session_id, {"role": "system", "content": system_prompt}, tokenizer
        )

    def prefill(self, *, jpeg=None, pcm=None):
        content = []
        if jpeg is not None:
            content.append({"type": "image", "data": jpeg})
        if pcm is not None:
            content.append({"type": "audio", "data": pcm})
        if content:
            self.model.streaming_prefill(
                self.session_id, {"role": "user", "content": content}, self.tokenizer
            )

    def generate(self):
        self._interrupt = False
        for chunk in self.model.streaming_generate(
            self.session_id, self.tokenizer, lambda: self._interrupt
        ):
            if self._interrupt:
                return
            yield chunk.get("text", ""), chunk.get("audio", b"")

    def interrupt(self):
        self._interrupt = True

    def reset(self):
        self._interrupt = False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_omni_session.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add omni/omni_session.py tests/test_omni_session.py
git commit -m "feat(omni): MiniCPM-o streaming session adapter (prefill/generate/interrupt)"
```

---

### Task 3: Fixed-K admission control

**Files:**
- Create: `omni/admission.py`
- Test: `tests/test_admission.py`

**Interfaces:**
- Consumes: nothing.
- Produces: class `Admission` with `__init__(self, capacity: int)`, `try_acquire(self) -> str | None` (returns a slot id or `None` when full), `release(self, slot_id: str) -> None`, and property `in_use: int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_admission.py
from omni.admission import Admission

def test_grants_up_to_capacity_then_refuses():
    a = Admission(capacity=1)
    s1 = a.try_acquire()
    assert s1 is not None
    assert a.try_acquire() is None      # full -> capacity_exhausted upstream
    assert a.in_use == 1

def test_release_frees_a_slot():
    a = Admission(capacity=1)
    s1 = a.try_acquire()
    a.release(s1)
    assert a.in_use == 0
    assert a.try_acquire() is not None

def test_release_unknown_slot_is_noop():
    a = Admission(capacity=1)
    a.release("nope")
    assert a.in_use == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_admission.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# omni/admission.py
import itertools

class Admission:
    def __init__(self, capacity):
        self.capacity = capacity
        self._slots = set()
        self._ids = itertools.count()

    def try_acquire(self):
        if len(self._slots) >= self.capacity:
            return None
        slot = f"slot-{next(self._ids)}"
        self._slots.add(slot)
        return slot

    def release(self, slot_id):
        self._slots.discard(slot_id)

    @property
    def in_use(self):
        return len(self._slots)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_admission.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add omni/admission.py tests/test_admission.py
git commit -m "feat(omni): fixed-K admission control"
```

---

### Task 4: Omni Streaming Server (WebSocket endpoint)

**Files:**
- Create: `omni/server.py`
- Test: `tests/test_omni_server.py`

**Interfaces:**
- Consumes: `omni.protocol`, `omni.admission.Admission`, `omni.omni_session.OmniSession`.
- Produces: `build_app(session_factory, capacity=1) -> FastAPI` exposing `GET /omni/session` (WebSocket). `session_factory(system_prompt, voice) -> OmniSession` is injected so tests pass a fake. On connect: read first message (`session_start`); if `admission.try_acquire()` is `None` → send `capacity_exhausted` and close; else send `session_ready`, then run the receive loop (prefill on `video`/`audio`, generate-on-`user_speech_end`, abort on `interrupt`), streaming `speech`/`text`/`turn_start`/`turn_end` down. Release the slot on disconnect/`session_end`.

- [ ] **Step 1: Write the failing test** (uses FastAPI `TestClient` websocket + a fake session factory)

```python
# tests/test_omni_server.py
from fastapi.testclient import TestClient
from omni.server import build_app
from omni import protocol as p

class FakeSession:
    def __init__(self, **kw): self.chunks = [("hello ", b"\x00"), ("intruder", b"\x01")]
    def prefill(self, **kw): pass
    def generate(self):
        for t, a in self.chunks: yield t, a
    def interrupt(self): pass
    def reset(self): pass

def _client(capacity=1):
    return TestClient(build_app(lambda **kw: FakeSession(**kw), capacity=capacity))

def test_session_ready_then_generate_on_speech_end():
    with _client().websocket_connect("/omni/session") as ws:
        ws.send_text(p.encode(p.make("session_start", 0, 0.0, system_prompt="g", voice="default")))
        assert p.decode(ws.receive_text())["type"] == "session_ready"
        ws.send_text(p.encode(p.make("user_speech_end", 1, 1.0)))
        types = [p.decode(ws.receive_text())["type"] for _ in range(4)]
        assert types == ["turn_start", "speech", "speech", "turn_end"]

def test_capacity_exhausted_when_full():
    client = _client(capacity=1)
    with client.websocket_connect("/omni/session") as ws1:
        ws1.send_text(p.encode(p.make("session_start", 0, 0.0, system_prompt="g", voice="default")))
        assert p.decode(ws1.receive_text())["type"] == "session_ready"
        with client.websocket_connect("/omni/session") as ws2:
            ws2.send_text(p.encode(p.make("session_start", 0, 0.0, system_prompt="g", voice="default")))
            assert p.decode(ws2.receive_text())["type"] == "capacity_exhausted"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_omni_server.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'omni.server'`

- [ ] **Step 3: Write minimal implementation**

```python
# omni/server.py
import base64, itertools
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from omni import protocol as p
from omni.admission import Admission

def build_app(session_factory, capacity=1):
    app = FastAPI()
    admission = Admission(capacity=capacity)

    @app.websocket("/omni/session")
    async def omni_session(ws: WebSocket):
        await ws.accept()
        seq = itertools.count()
        slot = None
        session = None
        try:
            start = p.decode(await ws.receive_text())
            slot = admission.try_acquire()
            if slot is None:
                await ws.send_text(p.encode(p.make("capacity_exhausted", next(seq), start["ts"])))
                await ws.close()
                return
            session = session_factory(system_prompt=start.get("system_prompt", ""),
                                      voice=start.get("voice", "default"))
            await ws.send_text(p.encode(p.make("session_ready", next(seq), start["ts"], session_id=slot)))
            while True:
                msg = p.decode(await ws.receive_text())
                t = msg["type"]
                if t == "video":
                    session.prefill(jpeg=base64.b64decode(msg["jpeg_b64"]), pcm=None)
                elif t == "audio":
                    session.prefill(jpeg=None, pcm=base64.b64decode(msg["pcm_or_opus_b64"]))
                elif t == "interrupt":
                    session.interrupt()
                    await ws.send_text(p.encode(p.make("interrupted", next(seq), msg["ts"])))
                elif t == "user_speech_end":
                    await ws.send_text(p.encode(p.make("turn_start", next(seq), msg["ts"])))
                    for text, audio in session.generate():
                        await ws.send_text(p.encode(p.make(
                            "speech", next(seq), msg["ts"],
                            audio_b64=base64.b64encode(audio).decode(), text_delta=text)))
                    await ws.send_text(p.encode(p.make("turn_end", next(seq), msg["ts"])))
                elif t == "session_end":
                    break
        except WebSocketDisconnect:
            pass
        finally:
            if slot is not None:
                admission.release(slot)
    return app
```

> Note: the test asserts `speech` frames carry the audio; `text_delta` rides on the same `speech` frame here for simplicity (a `text`-only frame is also valid per protocol and can be added when captions need finer granularity).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_omni_server.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add omni/server.py tests/test_omni_server.py
git commit -m "feat(omni): streaming server WS endpoint + admission wiring"
```

---

### Task 5: Rioc Live-channel client

**Files:**
- Create: `live/protocol.py` (vendored copy of `omni/protocol.py` — identical content)
- Create: `live/channel_client.py`
- Test: `tests/test_channel_client.py`

**Interfaces:**
- Consumes: `live.protocol`.
- Produces: class `ChannelClient` with `async connect(url)`, `async send(type, **payload)` (auto-assigns `seq`+`ts` via injected `clock()`), `async messages()` (async iterator yielding decoded downlink dicts), `async close()`. Constructor takes `clock: Callable[[], float]` for deterministic tests.

- [ ] **Step 1: Write the failing test** (against an in-process fake websocket)

```python
# tests/test_channel_client.py
import pytest
from live.channel_client import ChannelClient
from live import protocol as p

class FakeWS:
    def __init__(self, inbound): self.sent = []; self._in = list(inbound)
    async def send(self, s): self.sent.append(s)
    async def recv(self):
        if not self._in: raise StopAsyncIteration
        return self._in.pop(0)
    async def close(self): self.closed = True

@pytest.mark.asyncio
async def test_send_assigns_incrementing_seq_and_clock_ts():
    ws = FakeWS([])
    c = ChannelClient(ws=ws, clock=lambda: 7.0)
    await c.send("video", jpeg_b64="AA")
    await c.send("audio", pcm_or_opus_b64="BB")
    m0, m1 = p.decode(ws.sent[0]), p.decode(ws.sent[1])
    assert (m0["seq"], m0["ts"], m0["type"]) == (0, 7.0, "video")
    assert m1["seq"] == 1

@pytest.mark.asyncio
async def test_messages_decodes_downlink():
    ws = FakeWS([p.encode(p.make("session_ready", 0, 0.0, session_id="x"))])
    c = ChannelClient(ws=ws, clock=lambda: 0.0)
    got = [m async for m in c.messages()]
    assert got[0]["type"] == "session_ready"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_channel_client.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# live/channel_client.py
import itertools
import websockets
from live import protocol as p

class ChannelClient:
    def __init__(self, ws=None, clock=None):
        self._ws = ws
        self._clock = clock or (lambda: 0.0)
        self._seq = itertools.count()

    async def connect(self, url):
        self._ws = await websockets.connect(url, max_size=None)

    async def send(self, type, **payload):
        await self._ws.send(p.encode(p.make(type, next(self._seq), self._clock(), **payload)))

    async def messages(self):
        while True:
            try:
                raw = await self._ws.recv()
            except (StopAsyncIteration, websockets.ConnectionClosed):
                return
            yield p.decode(raw)

    async def close(self):
        await self._ws.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_channel_client.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add live/protocol.py live/channel_client.py tests/test_channel_client.py
git commit -m "feat(live): Rioc live-channel WebSocket client"
```

---

### Task 6: Edge VAD + barge-in controller (with reference-signal echo gating)

**Files:**
- Create: `live/barge_in.py`
- Test: `tests/test_barge_in.py`

**Interfaces:**
- Consumes: nothing (pure logic; audio frames pushed in).
- Produces: class `BargeInDetector` with:
  - `__init__(self, vad, *, echo_floor: float)` where `vad.is_speech(frame_pcm, rate) -> bool` is injected (webrtcvad in prod, fake in test).
  - `set_guard_speaking(self, on: bool)` — the reference signal: is our speaker currently playing guard audio.
  - `push(self, frame_pcm: bytes, rate: int, energy: float) -> str | None` — returns one of `"speech_start"`, `"speech_end"`, `"barge_in"`, or `None`. Emits `"barge_in"` (not `"speech_start"`) when guard is speaking AND energy exceeds `echo_floor` AND vad says speech. Suppresses detections at/below the echo floor while guard speaks.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_barge_in.py
from live.barge_in import BargeInDetector

class FakeVad:
    def __init__(self, verdicts): self.v = list(verdicts)
    def is_speech(self, frame, rate): return self.v.pop(0)

def test_speech_start_and_end_when_guard_silent():
    d = BargeInDetector(FakeVad([True, False]), echo_floor=100.0)
    d.set_guard_speaking(False)
    assert d.push(b"x", 16000, energy=500) == "speech_start"
    assert d.push(b"x", 16000, energy=5) == "speech_end"

def test_echo_below_floor_suppressed_while_guard_speaks():
    d = BargeInDetector(FakeVad([True]), echo_floor=100.0)
    d.set_guard_speaking(True)
    # VAD says speech, but energy under the echo floor -> it's our own voice
    assert d.push(b"x", 16000, energy=50) is None

def test_loud_speech_over_floor_triggers_barge_in():
    d = BargeInDetector(FakeVad([True]), echo_floor=100.0)
    d.set_guard_speaking(True)
    assert d.push(b"x", 16000, energy=800) == "barge_in"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_barge_in.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# live/barge_in.py
class BargeInDetector:
    def __init__(self, vad, *, echo_floor):
        self.vad = vad
        self.echo_floor = echo_floor
        self._guard_speaking = False
        self._in_speech = False

    def set_guard_speaking(self, on):
        self._guard_speaking = on

    def push(self, frame_pcm, rate, energy):
        speech = self.vad.is_speech(frame_pcm, rate)
        if self._guard_speaking:
            # Reference-signal gating: ignore anything at/below the echo floor.
            if not speech or energy <= self.echo_floor:
                return None
            return "barge_in"
        if speech and not self._in_speech:
            self._in_speech = True
            return "speech_start"
        if not speech and self._in_speech:
            self._in_speech = False
            return "speech_end"
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_barge_in.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add live/barge_in.py tests/test_barge_in.py
git commit -m "feat(live): edge VAD + barge-in with reference-signal echo gating"
```

---

### Task 7: Real-time streaming speaker sink

**Files:**
- Create: `live/speaker_sink.py`
- Test: `tests/test_speaker_sink.py`

**Interfaces:**
- Consumes: nothing in unit tests (transport injected).
- Produces: class `StreamingSpeakerSink` with:
  - `__init__(self, transport, *, jitter_frames: int)` where `transport.send_chunk(mulaw: bytes)` is injected (WS G.711 sender in prod).
  - `async feed(self, mulaw_chunk: bytes)` — enqueue a playback chunk.
  - `async run(self)` — drain the queue to the transport, respecting a jitter prime of `jitter_frames` before first output.
  - `flush_and_stop(self)` — drop all queued chunks immediately and mark stopped (barge-in). Returns count dropped.
  - property `stopped: bool`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_speaker_sink.py
import asyncio, pytest
from live.speaker_sink import StreamingSpeakerSink

class FakeTransport:
    def __init__(self): self.sent = []
    def send_chunk(self, mulaw): self.sent.append(mulaw)

@pytest.mark.asyncio
async def test_plays_queued_chunks_in_order():
    t = FakeTransport()
    s = StreamingSpeakerSink(t, jitter_frames=0)
    for c in (b"a", b"b", b"c"): await s.feed(c)
    task = asyncio.create_task(s.run())
    await asyncio.sleep(0.05)
    s.flush_and_stop()
    await task
    assert t.sent == [b"a", b"b", b"c"]

@pytest.mark.asyncio
async def test_flush_and_stop_drops_pending_and_halts():
    t = FakeTransport()
    s = StreamingSpeakerSink(t, jitter_frames=100)  # never primes -> nothing sent yet
    for c in (b"a", b"b"): await s.feed(c)
    dropped = s.flush_and_stop()
    assert dropped == 2
    assert s.stopped is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_speaker_sink.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# live/speaker_sink.py
import asyncio

class StreamingSpeakerSink:
    def __init__(self, transport, *, jitter_frames):
        self.transport = transport
        self.jitter_frames = jitter_frames
        self._q = asyncio.Queue()
        self._stopped = False

    async def feed(self, mulaw_chunk):
        await self._q.put(mulaw_chunk)

    async def run(self):
        # Prime the jitter buffer before first output.
        while self._q.qsize() < self.jitter_frames and not self._stopped:
            await asyncio.sleep(0.005)
        while not self._stopped:
            try:
                chunk = self._q.get_nowait()
            except asyncio.QueueEmpty:
                if self._stopped:
                    break
                await asyncio.sleep(0.005)
                continue
            self.transport.send_chunk(chunk)

    def flush_and_stop(self):
        dropped = self._q.qsize()
        while not self._q.empty():
            self._q.get_nowait()
        self._stopped = True
        return dropped

    @property
    def stopped(self):
        return self._stopped
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_speaker_sink.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add live/speaker_sink.py tests/test_speaker_sink.py
git commit -m "feat(live): streaming speaker sink with jitter buffer + flush/stop"
```

---

### Task 8: Escalation overlay (time/severity state machine)

**Files:**
- Create: `live/escalation.py`
- Test: `tests/test_escalation.py`

**Interfaces:**
- Consumes: nothing.
- Produces: class `EscalationOverlay` with `__init__(self, *, escalate_after: float, final_after: float)`, `tick(self, elapsed: float) -> str` returning `"WARNING" | "ESCALATING" | "FINAL"` by elapsed time, and `outcome(self, had_person_turn: bool, state: str) -> str` returning `"Escalated" | "Left" | "Unknown"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_escalation.py
from live.escalation import EscalationOverlay

def test_state_advances_by_elapsed_time():
    e = EscalationOverlay(escalate_after=10.0, final_after=25.0)
    assert e.tick(0) == "WARNING"
    assert e.tick(12) == "ESCALATING"
    assert e.tick(30) == "FINAL"

def test_outcome_classification():
    e = EscalationOverlay(escalate_after=10.0, final_after=25.0)
    assert e.outcome(had_person_turn=True, state="FINAL") == "Escalated"
    assert e.outcome(had_person_turn=True, state="WARNING") == "Left"
    assert e.outcome(had_person_turn=False, state="WARNING") == "Unknown"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_escalation.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# live/escalation.py
class EscalationOverlay:
    def __init__(self, *, escalate_after, final_after):
        self.escalate_after = escalate_after
        self.final_after = final_after

    def tick(self, elapsed):
        if elapsed >= self.final_after:
            return "FINAL"
        if elapsed >= self.escalate_after:
            return "ESCALATING"
        return "WARNING"

    def outcome(self, had_person_turn, state):
        if state == "FINAL":
            return "Escalated"
        if had_person_turn:
            return "Left"
        return "Unknown"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_escalation.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add live/escalation.py tests/test_escalation.py
git commit -m "feat(live): time/severity escalation overlay"
```

---

### Task 9: Live Session Orchestrator

**Files:**
- Create: `live/live_session.py`
- Test: `tests/test_live_session.py`

**Interfaces:**
- Consumes: `ChannelClient`, `BargeInDetector`, `StreamingSpeakerSink`, `EscalationOverlay`.
- Produces: class `LiveSessionOrchestrator` with:
  - `__init__(self, channel, speaker, escalation, *, on_event)` where `on_event(dict)` is the SSE broadcast hook.
  - `async start(self, *, system_prompt, voice, camera_id) -> bool` — sends `session_start`, awaits `session_ready`/`capacity_exhausted`; returns `False` and emits a `capacity` event on exhaustion.
  - `async on_downlink(self, msg)` — routes `speech`→`speaker.feed`, `text`→caption event, `turn_start/turn_end`→guard-speaking flag + events, `interrupted`→ack, `error`→error event.
  - `async barge_in(self)` — `speaker.flush_and_stop()` + `channel.send("interrupt")`.
  - `async end(self, reason)` — `session_end`, classify outcome via escalation, emit `ended`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_live_session.py
import pytest
from live.live_session import LiveSessionOrchestrator

class FakeChannel:
    def __init__(self, ready=True):
        self.sent = []; self._ready = ready
    async def send(self, type, **payload): self.sent.append((type, payload))
    async def await_admission(self):
        return "session_ready" if self._ready else "capacity_exhausted"

class FakeSpeaker:
    def __init__(self): self.fed = []; self.flushed = False
    async def feed(self, c): self.fed.append(c)
    def flush_and_stop(self): self.flushed = True; return 0

class FakeEsc:
    def outcome(self, had_person_turn, state): return "Left"

@pytest.mark.asyncio
async def test_start_returns_false_and_emits_on_capacity():
    events = []
    o = LiveSessionOrchestrator(FakeChannel(ready=False), FakeSpeaker(), FakeEsc(),
                                on_event=events.append)
    ok = await o.start(system_prompt="g", voice="default", camera_id="c1")
    assert ok is False
    assert any(e["type"] == "capacity" for e in events)

@pytest.mark.asyncio
async def test_barge_in_flushes_speaker_and_sends_interrupt():
    ch, sp = FakeChannel(), FakeSpeaker()
    o = LiveSessionOrchestrator(ch, sp, FakeEsc(), on_event=lambda e: None)
    await o.barge_in()
    assert sp.flushed is True
    assert ("interrupt", {}) in ch.sent

@pytest.mark.asyncio
async def test_speech_downlink_feeds_speaker():
    sp = FakeSpeaker()
    o = LiveSessionOrchestrator(FakeChannel(), sp, FakeEsc(), on_event=lambda e: None)
    await o.on_downlink({"type": "speech", "audio_b64": "QQ==", "text_delta": "hi"})
    assert sp.fed == [b"A"]   # base64 "QQ==" -> b"A"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_live_session.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# live/live_session.py
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
```

> Note: `FakeChannel.await_admission` in the test stands in for a small helper the real `ChannelClient` gains — read messages until the first `session_ready`/`capacity_exhausted`. Add that helper to `ChannelClient` when wiring Task 11, covered by its own step there.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_live_session.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add live/live_session.py tests/test_live_session.py
git commit -m "feat(live): Live Session Orchestrator (lifecycle, barge-in, downlink routing)"
```

---

### Task 10: Persistence + async Whisper backfill

**Files:**
- Modify: `db.py` (allow `Error` outcome; add `append_live_turn`)
- Create: `whisper_log.py`
- Test: `tests/test_whisper_log.py`, `tests/test_db_live.py`

**Interfaces:**
- Consumes: `db.py` existing `conversations`/`conversation_turns`.
- Produces:
  - `db.append_live_turn(conversation_id, speaker, text, audio_path) -> None` (async).
  - `whisper_log.backfill_person_turn(transcribe_fn, audio_bytes, conversation_id, db) -> None` (async) — calls `transcribe_fn(audio_bytes)` off the critical path; on exception logs an empty-text turn (non-fatal).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_whisper_log.py
import pytest
from whisper_log import backfill_person_turn

class FakeDB:
    def __init__(self): self.turns = []
    async def append_live_turn(self, conversation_id, speaker, text, audio_path):
        self.turns.append((conversation_id, speaker, text))

@pytest.mark.asyncio
async def test_backfill_writes_transcript():
    db = FakeDB()
    async def transcribe(b): return "who are you"
    await backfill_person_turn(transcribe, b"snd", conversation_id=1, db=db)
    assert db.turns == [(1, "PERSON", "who are you")]

@pytest.mark.asyncio
async def test_backfill_failure_is_nonfatal_empty_text():
    db = FakeDB()
    async def transcribe(b): raise RuntimeError("stt down")
    await backfill_person_turn(transcribe, b"snd", conversation_id=1, db=db)
    assert db.turns == [(1, "PERSON", "")]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_whisper_log.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# whisper_log.py
import logging
logger = logging.getLogger(__name__)

async def backfill_person_turn(transcribe_fn, audio_bytes, conversation_id, db):
    try:
        text = await transcribe_fn(audio_bytes)
    except Exception as e:  # logging-only path: never break the conversation
        logger.warning("[whisper_log] transcription failed: %s", e)
        text = ""
    await db.append_live_turn(conversation_id, "PERSON", text, audio_path=None)
```

Then add to `db.py` (mirroring existing async CRUD; `Error` needs no schema change — `outcome` is free text):

```python
async def append_live_turn(conversation_id, speaker, text, audio_path):
    async with aiosqlite.connect(DB_PATH) as conn:
        await conn.execute(
            "INSERT INTO conversation_turns (conversation_id, speaker, text, audio_path, timestamp)"
            " VALUES (?, ?, ?, ?, datetime('now'))",
            (conversation_id, speaker, text, audio_path),
        )
        await conn.commit()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_whisper_log.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add whisper_log.py db.py tests/test_whisper_log.py
git commit -m "feat(live): async logging-only Whisper backfill + live-turn persistence"
```

---

### Task 11: Wire into main.py — mode routing, /live endpoints, SSE relay, config validation

**Files:**
- Modify: `main.py`
- Modify: `live/channel_client.py` (add `await_admission()` helper referenced in Task 9)
- Test: `tests/test_main_live_routing.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `route_conversation(mode, live_start, turn_based_start)` helper — dispatches the shared person-detected trigger by `CONVERSATION_MODE`. Pure/testable.
  - `validate_live_config(mode, speaker_type) -> None` — raises `ValueError` when `mode=="live"` and speaker is URL-only.
  - FastAPI routes `POST /live/start`, `POST /live/stop`, `GET /live/stream` (SSE), reusing the existing SSE-subscriber deque pattern.
  - `ChannelClient.await_admission()` — reads messages until first `session_ready`/`capacity_exhausted`, returns its type.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_main_live_routing.py
import pytest
from main import route_conversation, validate_live_config

def test_route_dispatches_by_mode():
    calls = []
    route_conversation("live", lambda: calls.append("live"), lambda: calls.append("turn"))
    route_conversation("turn_based", lambda: calls.append("live"), lambda: calls.append("turn"))
    assert calls == ["live", "turn"]

def test_live_mode_refuses_url_only_speaker():
    with pytest.raises(ValueError):
        validate_live_config("live", speaker_type="ipspk_url")

def test_live_mode_accepts_ws_speaker():
    validate_live_config("live", speaker_type="fanvil_ws")  # no raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_main_live_routing.py -v`
Expected: FAIL with `ImportError: cannot import name 'route_conversation'`

- [ ] **Step 3: Write minimal implementation** (add helpers near the trigger in `main.py`)

```python
# main.py  (additions)
URL_ONLY_SPEAKER_TYPES = {"ipspk_url", "cs20_url"}

def route_conversation(mode, live_start, turn_based_start):
    (live_start if mode == "live" else turn_based_start)()

def validate_live_config(mode, speaker_type):
    if mode == "live" and speaker_type in URL_ONLY_SPEAKER_TYPES:
        raise ValueError(
            "Live mode requires a WebSocket-capable speaker; "
            f"'{speaker_type}' is URL-only and cannot be interrupted for barge-in."
        )
```

Then (no unit test, wired against existing patterns — verified by the integration task):
- Add `await_admission()` to `ChannelClient` (loop `messages()` until `session_ready`/`capacity_exhausted`).
- Add `POST /live/start` (build `ChannelClient`→`connect(MINICPMO... /omni/session)`, `LiveSessionOrchestrator.start`, spawn downlink + uplink pumps as lifespan-style tasks), `POST /live/stop` (`orchestrator.end`), and `GET /live/stream` (SSE over the same subscriber deque pattern as `/conversation/stream`).
- In the shared person-detected handler, call `validate_live_config(...)` then `route_conversation(CONVERSATION_MODE, ...)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_main_live_routing.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add main.py live/channel_client.py tests/test_main_live_routing.py
git commit -m "feat(live): mode routing, /live endpoints, SSE relay, live-config validation"
```

---

### Task 12: Lab integration test (barge-in round-trip)

**Files:**
- Create: `tests/integration/test_live_roundtrip.py` (marked `@pytest.mark.integration`, skipped in CI without a real omni server)

**Interfaces:**
- Consumes: real `omni.server` app + real `ChannelClient` over a loopback WebSocket, with a scripted intruder audio track + canned frames.

- [ ] **Step 1: Write the integration test**

```python
# tests/integration/test_live_roundtrip.py
import asyncio, pytest
from fastapi.testclient import TestClient
from omni.server import build_app
from omni import protocol as p

pytestmark = pytest.mark.integration

class ScriptedSession:
    def __init__(self, **kw): self._i = 0
    def prefill(self, **kw): pass
    def generate(self):
        for t in ("this ", "is ", "private ", "property"): yield t, b"\x00\x10"
    def interrupt(self): self._i = 999
    def reset(self): pass

def test_generate_then_interrupt_midstream():
    app = build_app(lambda **kw: ScriptedSession(**kw), capacity=1)
    with TestClient(app).websocket_connect("/omni/session") as ws:
        ws.send_text(p.encode(p.make("session_start", 0, 0.0, system_prompt="g", voice="default")))
        assert p.decode(ws.receive_text())["type"] == "session_ready"
        ws.send_text(p.encode(p.make("user_speech_end", 1, 1.0)))
        assert p.decode(ws.receive_text())["type"] == "turn_start"
        # first speech frame arrives, then we barge in
        assert p.decode(ws.receive_text())["type"] == "speech"
        ws.send_text(p.encode(p.make("interrupt", 2, 1.2)))
        # server acknowledges the interrupt
        seen = [p.decode(ws.receive_text())["type"] for _ in range(2)]
        assert "interrupted" in seen
```

- [ ] **Step 2: Run it**

Run: `pytest tests/integration/test_live_roundtrip.py -v -m integration`
Expected: PASS (with the scripted session); real-model latency runs are done manually against the GPU box.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_live_roundtrip.py
git commit -m "test(live): barge-in round-trip integration test"
```

---

### Task 13: Session Router (control-plane allocation)

**Files:**
- Create: `omni/router.py`
- Test: `tests/test_router.py`

**Interfaces:**
- Consumes: nothing (pure allocation logic; the HTTP/WS wrapper is added when deploying).
- Produces:
  - `GpuRef(ws_url: str, capacity: int)` with mutable `in_use: int` and `healthy: bool`.
  - `Router(gpus: list[GpuRef], *, lease_ttl=30.0, token_factory=None)` with:
    - `allocate(camera_id) -> dict | None` — least-loaded healthy GPU with a free slot; increments `in_use`, mints a lease token, returns `{gpu_ws_url, session_token, lease_ttl}`; `None` when the whole pool is full.
    - `release(session_token) -> None`.
    - `set_health(ws_url, healthy) -> None`.
    - properties `fleet_capacity`, `fleet_in_use`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_router.py
from omni.router import GpuRef, Router

def _pool():
    return Router([GpuRef("wss://g1", 2), GpuRef("wss://g2", 2)],
                  token_factory=lambda: "tok")

def test_allocate_prefers_least_loaded():
    r = _pool()
    assert r.allocate("c")["gpu_ws_url"] == "wss://g1"   # tie -> first
    assert r.allocate("c")["gpu_ws_url"] == "wss://g2"   # g1 now busier
    assert r.fleet_in_use == 2

def test_capacity_exhausted_when_pool_full():
    r = Router([GpuRef("wss://g1", 1)], token_factory=lambda: "tok")
    assert r.allocate("c") is not None
    assert r.allocate("c") is None                       # whole pool full

def test_release_frees_slot():
    r = Router([GpuRef("wss://g1", 1)])
    tok = r.allocate("c")["session_token"]
    r.release(tok)
    assert r.fleet_in_use == 0

def test_unhealthy_gpu_excluded():
    r = _pool()
    r.set_health("wss://g1", False)
    assert r.allocate("c")["gpu_ws_url"] == "wss://g2"
    assert r.allocate("c")["gpu_ws_url"] == "wss://g2"
    assert r.allocate("c") is None                       # g2 full, g1 unhealthy
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_router.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# omni/router.py
import itertools

class GpuRef:
    def __init__(self, ws_url, capacity):
        self.ws_url = ws_url
        self.capacity = capacity
        self.in_use = 0
        self.healthy = True

class Router:
    def __init__(self, gpus, *, lease_ttl=30.0, token_factory=None):
        self._gpus = list(gpus)
        self.lease_ttl = lease_ttl
        self._ids = itertools.count()
        self._token_factory = token_factory or (lambda: f"tok-{next(self._ids)}")
        self._leases = {}

    def allocate(self, camera_id):
        candidates = [g for g in self._gpus if g.healthy and g.in_use < g.capacity]
        if not candidates:
            return None
        gpu = min(candidates, key=lambda g: g.in_use)  # least-loaded (ties -> first)
        gpu.in_use += 1
        token = self._token_factory()
        self._leases[token] = gpu
        return {"gpu_ws_url": gpu.ws_url, "session_token": token, "lease_ttl": self.lease_ttl}

    def release(self, session_token):
        gpu = self._leases.pop(session_token, None)
        if gpu and gpu.in_use > 0:
            gpu.in_use -= 1

    def set_health(self, ws_url, healthy):
        for g in self._gpus:
            if g.ws_url == ws_url:
                g.healthy = healthy

    @property
    def fleet_capacity(self):
        return sum(g.capacity for g in self._gpus)

    @property
    def fleet_in_use(self):
        return sum(g.in_use for g in self._gpus)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_router.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add omni/router.py tests/test_router.py
git commit -m "feat(omni): Session Router — control-plane least-loaded GPU allocation"
```

> The HTTP `POST /allocate` wrapper + omni-server occupancy callbacks + lease expiry are added when the router is deployed (§4.1); this task delivers the allocation core. `LiveSessionOrchestrator.start` gains a call to `router.allocate` before connecting (folded into Task 11 wiring).

## Self-Review

**Spec coverage:**
- Omni streaming server → Tasks 2, 3, 4. ✓
- Live channel protocol → Task 1 (+ vendored Task 5). ✓
- Live Session Orchestrator → Task 9. ✓
- Edge VAD + barge-in + reference-signal gating → Task 6. ✓
- Real-time speaker sink (stream + flush/stop) → Task 7. ✓
- Escalation time/severity overlay + outcomes → Task 8. ✓
- Fixed-K admission → capacity_exhausted → Tasks 3, 4, 9, 11. ✓
- Async Whisper logging-only backfill → Task 10. ✓
- Mode routing, /live endpoints, SSE relay, URL-only-speaker refusal → Task 11. ✓
- Persistence incl. `Error` outcome → Tasks 8, 10. ✓
- Barge-in round-trip verification → Task 12. ✓
- **Out of scope (correctly absent):** web-ui Live tab (behind prototype gate), GPU autoscaling, AEC, multi-camera.

**Placeholder scan:** No TBD/TODO in code steps; the one deferred piece (MiniCPM-o exact API) is an explicit spike in Task 2 with a working fake-backed adapter so downstream tasks are unblocked. The Task 11 wiring beyond the pure helpers is verified by the Task 12 integration test rather than unit tests, and says so explicitly.

**Type consistency:** `OmniSession.generate()→(text,audio)` consumed consistently in Task 4; `ChannelClient.send/await_admission/messages` names match across Tasks 5, 9, 11; `StreamingSpeakerSink.feed/flush_and_stop/stopped` consistent across Tasks 7, 9; `EscalationOverlay.tick/outcome` consistent across Tasks 8, 9; protocol type strings identical between `omni/protocol.py` and `live/protocol.py`.

## Notes for the implementer

- `omni/` deploys to the **GPU box**; `live/` + `main.py` deploy to the **appliance**. They share only the protocol module (kept byte-identical). Do not import across the boundary.
- Real barge-in latency and audio quality are validated **manually against the GPU box**, not in CI — Task 12 proves the control flow only.
- Keep the turn-based `ConversationManager` untouched; every change here is additive and gated by `CONVERSATION_MODE`.
- **Set K from a benchmark, not a guess.** Before shipping, measure how many simultaneous full-duplex omni sessions the target GPU sustains at real-time latency (barge-in intact) and set `LIVE_MAX_SESSIONS` accordingly. The admission code is already K-agnostic; only the deployed default depends on this measurement. Re-run when GPU class or model changes.

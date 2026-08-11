# Rioc — Architecture

> Single-process Python service. Entry point: `uvicorn main:app --host 0.0.0.0 --port 8000`
> Branch of origin: `feat/cvr-frame-for-conversation-turns`

Rioc is the **AI security guard backend**. Given a camera feed (or a webhook from one) and an IP speaker, it detects unauthorized people, asks a vision LLM what's happening, and conducts a two-way voice conversation with the intruder through the speaker — escalating from WARNING → ESCALATING → FINAL and persisting the whole transcript to SQLite.

The README is the *how-to-run* document. This file is the *how-it's-built* document — what each module is responsible for, how frames and audio move through the system, what runs concurrently, and the sharp edges to know about before you touch the code.

---

## 1. System diagram

```
                          ┌─────────────────────────┐
                          │  vLLM server (GPU)       │  (self-hosted; see DEPLOY_AWS.md)
                          │  MiniCPM-o / MiniCPM-V   │
                          │  OpenAI-compatible /v1   │
                          └─────────────▲────────────┘
                                        │ HTTPS  (CLOUD_AI_URL :8100  +  MINICPMO_URL :8101)
                                        │
┌────────────┐  webhook   ┌─────────────┴────────────┐  WebSocket G.711 / HTTP   ┌──────────────┐
│  CVR /     ├───────────►│  Rioc (this service)      ├──────────────────────────►│ IP Speaker   │
│  RiocHook  │  /api/...  │  FastAPI on :8000         │  play-from-URL / VAPIX    │ Fanvil CS20  │
│  Camera    │            │                           │                           │ AXIS C1310-E │
└────────────┘  RTSP      └─────┬─────────┬───────────┘                           └──────────────┘
                                │         │ HTTPS
                                │         ▼
                                │   ┌─────────────────────┐
                                │   │  OpenAI API          │
                                │   │  STT (Whisper)       │
                                │   │  TTS (tts-1)         │
                                │   └─────────────────────┘
                                │
                                ▼
                          ┌─────────────────────┐
                          │  SQLite             │       ┌────────────┐
                          │  ai_guard.db        │       │ Mic (VAD)  │  webrtcvad
                          │  conversations +    │       │ sounddevice│
                          │  conversation_turns │       └────────────┘
                          └─────────────────────┘
```

Two distinct vLLM endpoints are used. They may point at the same physical server, but the code treats them as separate:

- **`CLOUD_AI_URL`** (default port 8100) — used for the *audit* path: "is this frame a person, and if so what should the guard say?". Called by `cloud_audit_loop` in `main.py`.
- **`MINICPMO_URL`** (default port 8101) — used for *conversation* turns by `ConversationManager`. Has audio modality enabled so it can return spoken audio alongside text. If it 4xx/5xx's, `minicpmo_client.chat()` silently falls back to `CLOUD_AI_URL`.

Everything else (OpenAI, SQLite, mic, speaker) is straightforward.

---

## 2. Module layout

```
rioc/
├── main.py                    ← FastAPI app, endpoints, audit loop, speaker/TTS, lifespan
├── conversation_manager.py    ← WARNING→ESCALATING→FINAL state machine, SSE broadcast
├── minicpmo_client.py         ← HTTP client for the conversation vLLM (with fallback)
├── mic_listener.py            ← VAD-gated audio capture (webrtcvad + sounddevice)
├── db.py                      ← SQLite schema + CRUD for conversations and turns
├── videodb_integration.py     ← Optional VideoDB "eyes and ears" pipeline
├── cloud_brain.py             ← Standalone helper script — NOT loaded by main.py
├── vision_test.py             ← Standalone Ollama vision smoke test
├── ai_guard.db                ← SQLite file (auto-created on first run)
├── audio_logs/                ← Per-turn WAV files (auto-created)
├── yolov8n.pt                 ← YOLO nano weights (~6 MB), used by local_yolo mode
├── requirements.txt
├── .env / .env.example
├── DEPLOY_AWS.md              ← vLLM GPU deployment recipe
└── README.md                  ← How to install, configure, run
```

Two files don't fit the "loaded at runtime" picture and are easy to misread:

- **`cloud_brain.py`** is a standalone CLI script that pulls Rioc's MJPEG `/video` stream and runs an old-style audit loop directly against vLLM. It was an earlier prototype of what is now the `cloud_audit_loop` inside `main.py`. It is *not* imported by the FastAPI app. Leave alone unless you're maintaining the historical RunPod recipe.
- **`vision_test.py`** is a one-off Ollama smoke test, ~40 lines. Same story: not wired in.

---

## 3. The single-process model

`main.py` instantiates one `FastAPI` app and runs everything else inside a single uvicorn event loop. There is **no external broker, queue, or worker**. All "concurrent" work is either:

- an asyncio task created in the lifespan handler, or
- a synchronous blocking call dispatched to a thread via `asyncio.to_thread()`.

This is deliberate. The service is meant to be deployed per-LAN-appliance: one camera, one speaker, one process. Horizontal scaling is not a goal; latency and operational simplicity are.

### Background tasks (started in `lifespan`)

| Task | Condition to start | What it does |
|---|---|---|
| `cloud_audit_loop` | `ENABLE_CLOUD_AI=1` and `CLOUD_AI_URL` set and `FRAME_SOURCE != "webhook"` | Drains `_detection_frame_queue`, calls vLLM, fires alerts and conversations |
| `local_yolo_source_loop` | `FRAME_SOURCE=local_yolo` | RTSP → YOLO pre-filter → enqueue person-frames |
| `live_ffmpeg_source_loop` | `FRAME_SOURCE=live_ffmpeg` | **Stub** — currently falls back to `local_yolo_source_loop` |
| `local_audit_loop` | `ENABLE_LOCAL_AUDIT=1` | Polls Ollama every `AUDIT_INTERVAL_SEC` for vision audit (dev/demo) |
| `audio_transcription_loop` | `ENABLE_AUDIO_STT=1` | Records mic, transcribes via MiniCPM-o STT, updates `latest_transcript` |
| `run_videodb_eyes` | `ENABLE_VIDEODB=1` | VideoDB WebSocket subscriber |
| `_warmup_cloud_ai` | always (fire-and-forget) | Pings vLLM on startup to surface 5xx early |

### Locks and shared state

- **`cap_lock` (threading.Lock)** — guards the `cv2.VideoCapture` handle. Required because the camera can be closed/re-opened from a request handler (`POST /configure`) while the YOLO loop is reading from it; concurrent OpenCV access segfaults.
- **`_yolo_model_lock` (threading.Lock)** — guards lazy load of the ultralytics model singleton.
- **`_tts_active` (bool)** — checked-and-set inside `_guarded_play` *before any `await`*, so the asyncio single-thread guarantee makes it effectively atomic. Prevents overlapping TTS audio without needing an explicit lock.
- **`_detection_frame_queue` (asyncio.Queue, maxsize=1)** — single-slot queue between the frame source and the audit loop. Holding only the freshest frame is intentional backpressure: if vLLM is slow, drop old frames rather than queue them up.
- **`_sse_subscribers` (list of asyncio.Queue, maxsize=50)** — fan-out to live HTTP clients. Per-client queue full means *drop the event*, never block the producer.
- **`_event_log` (deque, maxlen=500)** — in-memory timeline (`yolo_detected`, `ai_alert`, `tts_started`, `speaker_playing`), exposed at `GET /events`.
- **`detections` (deque, maxlen=100)** — recent detection dicts, exposed at `GET /detections` and replayed to new `/detections/stream` subscribers.
- **`_latest_conv_frame`, `_latest_conv_frame_ts`** — the most recent JPEG to use for conversation turns. Updated by both the RTSP loop *and* by `POST /api/frame-update` from external CVR. `ConversationManager` reads it via `_get_conv_frame`.

This is the entire shared-state surface. There is no global mutex around the conversation flow itself — `ConversationManager` owns its own `asyncio.Lock` for that.

---

## 4. Frame-source modes

`FRAME_SOURCE` (default `webhook`) picks one of three loops.

### `webhook` (default, production)

No local camera, no YOLO, no audit loop. The frame source is `POST /api/person-detected` — an external CVR / RiocHook posts JPEG bytes plus context whenever *it* believes a person is present. `_handle_person_detected_event` runs in the background, asks MiniCPM-o what to say, fires the TTS, and triggers `ConversationManager.on_person_detected`. Detection debouncing (`AUDIT_CONFIRM_FRAMES`) is *skipped* in this mode on the grounds that the upstream system already debounced.

### `local_yolo` (dev / standalone)

`local_yolo_source_loop` opens an RTSP stream (`CAMERA_RTSP_URL`), grabs frames, runs YOLO nano locally, and enqueues person-confirmed frames into `_detection_frame_queue`. `cloud_audit_loop` consumes that queue. The first detection also fires the pre-generated "Attention." chime so the operator hears something *immediately* without waiting on a vLLM round-trip.

The loop polls at a variable rate driven by `PresenceLock`:

- **IDLE** — normal scanning rate.
- **CONVERSATION_ACTIVE** — slowed to `PRESENCE_LOCK_REDUCED_FPS` (default ~1 Hz) so we don't spam vLLM with redundant frames while a conversation is in progress.
- **COOLDOWN** — same reduced rate, until `PRESENCE_LOCK_NO_DETECT_FRAMES` consecutive empty frames OR `PRESENCE_LOCK_COOLDOWN_SEC` elapse → back to IDLE.

### `live_ffmpeg` (placeholder)

The function exists but immediately delegates to `local_yolo_source_loop`. Reserved for an AGX-side live-ffmpeg interface that hasn't been wired up. Don't depend on this mode meaning anything different yet.

---

## 5. The vision pipeline (`cloud_audit_loop`)

This is the heart of the local_yolo flow. Steps, in order:

1. **Wait on `_detection_frame_queue.get()`.** No polling — the audit loop is event-driven. The frame in the queue is already YOLO-confirmed.
2. **Drain stale frames.** While vLLM was last running, fresher frames may have piled into the (size-1) queue. Take the most recent and discard the rest, so we never analyze data older than necessary.
3. **POST to `CLOUD_AI_URL/v1/chat/completions`** with `AUDIT_SYSTEM_PROMPT` + (`AUDIT_USER_PROMPT` for local_yolo / `AUDIT_WEBHOOK_PROMPT` for webhook) + base64 JPEG (resized to `AUDIT_AI_FRAME_SIZE`, default 320 px) + the most recent `latest_transcript` if any. `max_tokens=600`, audio modality off (text is faster). Auth: `Bearer ${CLOUD_AI_API_KEY}`.
4. **Strip `<think>...</think>` blocks** from the response (some MiniCPM variants emit reasoning traces).
5. **Confirm-frames debounce.** Require `AUDIT_CONFIRM_FRAMES` (default 2) consecutive non-CLEAR responses before firing an alert. Any CLEAR resets the counter. Trades latency for false-positive resistance.
6. **Alert cooldown.** Independent of TTS cooldown — at least `ALERT_COOLDOWN_SEC` (default 30 s) between alerts even if vLLM keeps flagging.
7. **PresenceLock suppression.** If we're already in CONVERSATION_ACTIVE or COOLDOWN, skip the alert — the conversation already in flight will speak.
8. **Fire the alert.** Update `latest_analysis`, push to `_event_log` and `detections`, fan out to SSE subscribers, start TTS in parallel with `ConversationManager.on_person_detected(jpeg, initial_text=msg)`.

Webhook mode does almost the same thing inside `_handle_person_detected_event` — same vLLM call, same fallback message ("A person has been detected. Please respond."), same conversation kickoff — minus the confirm-frames step.

---

## 6. Conversation flow (`conversation_manager.py`)

`ConversationManager` is a singleton (`get_manager()` / `set_manager()` at module scope) wired up in lifespan with four callbacks supplied by `main.py`:

| Callback | Provided by `main.py` |
|---|---|
| `play_audio_fn(wav_bytes, format)` | `_guarded_play(force=True)` — drops it onto the speaker bypassing TTS cooldown |
| `get_frame_fn()` | `_get_conv_frame` — returns `_latest_conv_frame` (and logs its age) |
| `speak_text_fn(text)` | `_speak_through_speaker(force=True)` — OpenAI TTS fallback when vLLM returns text but no audio |
| `transcribe_fn(audio_bytes)` | `_transcribe_audio` — OpenAI Whisper |

### State machine

```
                              max_turns
                              reached
   ┌────► WARNING ─► ESCALATING ────► FINAL ───┐
   │         │            │                    │
   │         │            │ (no reply)         ▼
   │         └────────────┴─────────────► IDLE ──► (cooldown 20s)
   │
on_person_detected()
```

Two transient labels — `LISTENING` and `SPEAKING` — are emitted on the SSE stream during turns but are not "real" states from the model's perspective; they exist so the UI can render mic-open and speaker-talking indicators.

### Per-turn flow

1. `on_person_detected(jpeg, initial_text=…)` is the entry point. It takes the manager's internal `asyncio.Lock`, refuses if already active or within `CONVERSATION_COOLDOWN_SEC`.
2. **Skip-the-greeting optimization.** If `initial_text` is passed (i.e. the audit loop already spoke the opening warning aloud), it's logged as a GUARD turn 1 and we go straight to `_listen_and_respond`. Otherwise `_do_turn` makes the model generate the opening line.
3. **`_do_turn(jpeg)`** — `minicpmo_chat(jpeg, system_prompt, history, user_text=_TURN_ROLE_REMINDER)` returns `(text, wav_bytes | None)`. The `_TURN_ROLE_REMINDER` is prepended *to every user turn* to stop the model from drifting into neutral image description by, say, turn 3 — a real failure mode of MiniCPM-o without re-anchoring.
4. **Half-duplex sync via `_speaking_event`.** The audio dispatch runs as a background task. The state transitions immediately so the SSE stream shows the new state without waiting. `mic_listener.listen_for_response()` blocks on `_speaking_event.wait()` and then sleeps `SPEAKING_BUFFER_SEC` (~0.5 s, configurable) before opening the mic, to avoid capturing the tail of the speaker's own audio as a "person reply." The event is cleared in `_dispatch_audio`'s `finally`, which means it clears precisely when the WebSocket send loop finishes the last μ-law chunk — no fixed-timer guess.
5. **`_listen_and_respond`** — runs `listen_for_response` in a thread (via `asyncio.to_thread`). VAD-gated, returns WAV bytes or None. None ends the conversation.
6. **`_continue_conversation(wav)`** — grabs a fresh JPEG, transcribes the person's reply, appends both as a user turn (with both a text label *and* the audio payload, so the model has context for the audio), persists, calls `_do_turn(fresh_jpeg)` again.
7. Loop until `turn_count >= max_turns` OR `_active` flipped false (no reply / error).
8. **`_end_conversation`** classifies outcome:
   - `FINAL` state → `"Escalated"`
   - at least one PERSON turn AND state ≠ FINAL → `"Left"`
   - GUARD-only → `"Unknown"`
   Cancels any in-flight audio task, writes outcome + ended_at to SQLite, broadcasts `{type: "ended"}`, sets `_last_ended` for the cooldown clock.

### SSE broadcast

`_broadcast(event)` pushes to every queue in `_sse_listeners`. Four event shapes:

| `type` | Fields | Emitted on |
|---|---|---|
| `state` | `state` | every `_set_state(...)` call |
| `turn` | `turn`, `state` | after every `add_turn` |
| `status` | full status dict | on new SSE connection (sent by the HTTP handler, not the manager) |
| `ended` | `outcome`, `conversation_id` | in `_end_conversation` |

Queue full → drop event. Subscribers get a 30 s keepalive comment from the HTTP handler to keep proxies happy.

---

## 7. Speaker / TTS delivery (`_guarded_play`)

This is the gnarliest part of `main.py`. There is no single speaker protocol — Rioc has to talk to whatever IP speaker the customer bought. `_guarded_play` is a multi-protocol dispatcher that picks one of five paths based on `SPEAKER_URL` and detected vendor.

| Path | When | Mechanism |
|---|---|---|
| **AXIS VAPIX** | URL host matches AXIS pattern | `POST /axis-cgi/audio/transmit.cgi` with `audio/basic` (G.711 μ-law) or `audio/mpeg`, Digest auth |
| **WebSocket G.711** | `SPEAKER_WS_URL` set or auto-derived (Fanvil/LINKVIL) | Connect to `wss://.../webtwowayaudio`, send 20 ms μ-law chunks (160 bytes @ 8 kHz), pace-matched to playback |
| **Play-from-URL** | speaker exposes `/api/play startstream` | Point speaker at `TTS_PUBLIC_URL/tts/latest.mp3?cb=...` and let it pull |
| **Upload-then-play** | fallback | POST audio as `userfile1`, then play it |
| **Local playback** | `ENABLE_LOCAL_PLAYBACK=1` (Mac dev) | `afplay` the file on the host |

### Audio format gauntlet

The TTS source is one of: pre-generated "Attention." MP3 (OpenAI tts-1), OpenAI tts-1 MP3 generated on demand, or MiniCPM-o WAV from a conversation turn. Each speaker path needs a different format, so `_guarded_play` may chain:

- `_wav_to_mp3` (ffmpeg)
- `_resample_for_speaker` — re-encode to 16 kHz mono MP3 because many embedded speakers can't decode OpenAI's 24 kHz stereo
- `_mp3_to_mulaw` — for WebSocket and AXIS audio/basic paths

This is why `ffmpeg` is a hard prerequisite.

### Rate limiting and atomicity

- **`_tts_active` flag** is set to `True` synchronously before any `await`. In a single-threaded event loop this is effectively atomic — no two coroutines can both observe it `False` at the same moment.
- **`_last_tts_time`** enforces `TTS_COOLDOWN_SEC` (default 20 s) between *normal* announcements.
- **Chime calls** (`chime=True`) bypass the cooldown and don't update `_last_tts_time`, so they don't block the real alert ~1 s later.
- **`force=True`** (conversation turns) bypasses cooldown entirely — once a conversation starts, every turn must speak.

### CS20 / "ipspk" quirk

For CS20-class speakers in play-from-URL mode: the speaker happily loops on the static URL. To stop it, Rioc estimates audio duration, schedules `_ipspk_stopstream_after()` after that estimate, and sends a stopstream. Failure mode: alerts arriving faster than stop commands → the wrong MP3 plays. There's a 150 ms wait between stop and start to mitigate, but it's a real race.

---

## 8. Persistence

### SQLite — `ai_guard.db`

`db.py` defines two tables, both created on first call to `init_db()`:

```sql
conversations(
  id, camera_id, started_at, ended_at, outcome,
  turn_count DEFAULT 0, created_at DEFAULT datetime('now')
)

conversation_turns(
  id, conversation_id REFERENCES conversations(id),
  speaker CHECK(speaker IN ('GUARD','PERSON')),
  text, audio_path, timestamp
)
CREATE INDEX idx_turns_conv ON conversation_turns(conversation_id)
```

`init_db()` is idempotent and gated by an asyncio lock + `_initialized` flag, so calling it from both `lifespan` and `_run_conversation` is safe. Each CRUD call opens its own short-lived `aiosqlite.connect` — no shared connection pool. Fine for low-throughput appliance use; would need rethinking at higher write rates.

Outcome semantics matter for the UI's conversation history view: `Escalated`, `Left`, `Unknown` (see §6 step 8). Don't add new outcomes without updating the web-services frontend.

### File system

- **`audio_logs/conv_{id}_{speaker}_turn{n}.wav`** — every per-turn WAV written by `ConversationManager._save_audio`. Uses `aiofiles` if installed, sync write otherwise. No retention policy — directory grows forever.
- **`latest_tts_audio`** — in-memory bytes, served at `GET /tts/latest.mp3`. Not on disk.

There is no migration tooling. Schema changes mean a one-time SQL or wiping `ai_guard.db`.

---

## 9. HTTP / SSE surface

All paths are relative to the FastAPI app on port 8000. Grouped by purpose; see `main.py` line ranges in the live agent map for body shapes.

### Status & video

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | HTML page embedding the MJPEG stream |
| `GET` | `/video` | MJPEG (640×640, JPEG quality 70) |
| `GET` | `/transcript` | Latest mic transcript (JSON) |
| `GET` | `/analysis` | Latest non-CLEAR audit text |

### Detections & events

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/detections` | Recent detection deque, newest first |
| `GET` | `/detections/stream` | **SSE** — replays existing, then pushes new |
| `GET` | `/events` | Event log (yolo/ai/tts/speaker), `?limit=` up to 200 |

### Conversations

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/conversation/start` | Manually trigger; refuses if active or cooldown |
| `POST` | `/conversation/respond` | Inject mic-audio reply (WAV base64) into the active turn |
| `GET` | `/conversation/status` | Current state, turn count, transcript snapshot |
| `GET` | `/conversation/stream` | **SSE** — `state` / `turn` / `ended`, plus initial `status`, 30 s keepalive |
| `POST` | `/conversation/configure` | Hot-update `systemPrompt`, `maxTurns`, `cameraId` |
| `GET` | `/conversations` | Paginated history from SQLite |
| `GET` | `/conversations/{id}` | Single conversation with all turns |

### Webhook ingress

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/person-detected` | RiocHook → JPEG + context. Returns 202; processes in background |
| `POST` | `/api/frame-update` | Live ffmpeg → fresh frame for `_latest_conv_frame`. Returns 204 |

### Runtime config

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/configure` | Hot-swap `cameraRtspUrl`, speaker URL/credentials/type, toggle `ENABLE_YOLO`. RTSP is probed via `_probe_rtsp` (subprocess ffprobe) before binding. |

### Speaker / TTS

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/tts/test` | Speak arbitrary text — bypasses `ENABLE_SPEAKER_TTS` gate |
| `GET` | `/tts/latest.mp3` | Serve the cached `latest_tts_audio` (used by play-from-URL speakers) |
| `GET` | `/speaker-test` | Same MP3, but exposed for browser-on-LAN to verify reachability |
| `GET` | `/speaker-diagnostic` | Static troubleshooting steps |
| `GET` | `/speaker-test-bell` | Trigger speaker's built-in `bell1` |

CORS middleware is on, no auth on any endpoint. Trust boundary: the LAN.

---

## 10. External services and the env-var contract

| Service | Required? | Env vars | Used in |
|---|---|---|---|
| vLLM (audit) | yes for cloud audit | `CLOUD_AI_URL`, `CLOUD_AI_API_KEY`, `CLOUD_AI_MODEL` | `main.py: cloud_audit_loop`, `_warmup_cloud_ai`, `_handle_person_detected_event` |
| vLLM (conversation) | yes for conversations | `MINICPMO_URL`, `MINICPMO_API_KEY`, `MINICPMO_MODEL`, `MINICPMO_TIMEOUT` | `minicpmo_client.chat` (falls back to `CLOUD_AI_URL`) |
| OpenAI TTS | yes for speech | `OPENAI_STT_API_KEY` (also accepts `OPENAI_API_KEY`), `OPENAI_TTS_MODEL`, `OPENAI_TTS_VOICE` | `main.py: _speak_through_speaker` |
| OpenAI Whisper / gpt-4o-transcribe | yes for STT | `OPENAI_STT_API_KEY`, `OPENAI_STT_MODEL` | `main.py: _transcribe_audio` |
| SQLite | auto | `ai_guard.db` colocated with `db.py` | `db.py` |
| Ollama | optional | `ENABLE_LOCAL_AUDIT`, `OLLAMA_URL`, `OLLAMA_VISION_MODEL` | `main.py: local_audit_loop` |
| VideoDB | optional | `ENABLE_VIDEODB`, `VIDEODB_API_KEY`, `VIDEODB_RTSP_VIDEO`, `VIDEODB_RTSP_AUDIO` | `videodb_integration.run_videodb_eyes` |
| IP speaker | optional | `SPEAKER_URL`, `SPEAKER_WS_URL`, `SPEAKER_USER`, `SPEAKER_PASS`, `SPEAKER_PLAY_PATH`, `TTS_PUBLIC_URL` | `main.py: _guarded_play` |
| Microphone | optional | `AUDIO_INPUT_DEVICE` (index or name substring), VAD_* tuning vars | `mic_listener.listen_for_response` |
| ffmpeg | yes if TTS | — (must be on `$PATH`) | `_wav_to_mp3`, `_resample_for_speaker`, `_mp3_to_mulaw`, `_probe_rtsp` |

`FRAME_SOURCE` toggles which background loops start. `ENABLE_*` flags are the master switches; check them before suspecting code bugs.

---

## 11. Concurrency model in one screen

```
       request handler          background tasks                   thread pool
       ───────────────          ────────────────                  ─────────────
        FastAPI route ─┐                                          ┌─ cv2.VideoCapture
                       │  awaits                                  │  (cap_lock)
        POST /configure├──► acquires cap_lock, swaps capture      │
                       │                                          ├─ ultralytics YOLO
        POST /api/    ─┤                                          │  (_yolo_model_lock)
        person-detected│  task spawn                              │
                       └──► _handle_person_detected_event ───────►├─ ffmpeg conversions
                                                                  │
        SSE /conversation/stream ◄─── _broadcast() ◄─── ConversationManager
        SSE /detections/stream   ◄─── _add_detection()            ├─ sounddevice mic read
                                                                  │  (mic_listener)
                                                                  │
                            cloud_audit_loop  ◄── _detection_frame_queue (size=1)
                                       │                          ├─ websockets send loop
                                       └───► spawn _speak_through_speaker
                                       └───► spawn ConversationManager.on_person_detected
                                                                  └─ aiosqlite (db.py)

       Single asyncio event loop. Blocking calls always go via asyncio.to_thread.
```

Practical implications:

- **A slow vLLM call does not block request handlers.** It blocks the audit loop, which means new frames pile up and get dropped (size-1 queue), which is fine.
- **A stuck mic read does block the conversation flow** — `listen_for_response` runs in a thread, but the conversation coroutine `await`s its completion. There's a `VAD_MAX_LISTEN_SEC` ceiling (default 20 s) so it can't hang forever.
- **`_guarded_play` is serialized by `_tts_active`.** Concurrent attempts return without playing. Conversation turns pass `force=True` and acquire the slot regardless of cooldown.
- **`cap_lock`** is the only place a *Python thread* and *the asyncio loop* both touch state. Everything else stays inside the event loop.

---

## 12. Sharp edges and known issues

Things to know before you change behavior:

- **`live_ffmpeg` is a stub.** Currently aliases `local_yolo_source_loop`. Don't ship code that branches on `FRAME_SOURCE=="live_ffmpeg"` until the real implementation lands.
- **`FRAME_SOURCE=webhook` is the default.** A first-boot operator with no external CVR sees nothing happen and may assume the service is broken. Consider whether `local_yolo` is a safer default for fresh installs.
- **AUDIT_CONFIRM_FRAMES interacts badly with the size-1 queue.** Under stale frames or flaky vLLM responses, the consecutive-non-CLEAR counter may never reach the threshold; under CONVERSATION_ACTIVE suppression, YOLO throttles and may also stall the counter. Tune both knobs together.
- **TTS-via-URL race on CS20.** Multiple alerts within the estimated playback window can cause the wrong MP3 to play. Mitigation in `_ipspk_stopstream_after` is best-effort.
- **`audio_logs/` grows unbounded.** No rotation. Add cron / systemd-tmpfiles or remember to clean up.
- **No auth, no rate limits.** Treat the service as LAN-trusted. Anything that exposes port 8000 to the internet without a fronting proxy is a mistake.
- **No retry on vLLM 5xx.** `minicpmo_client.chat` falls back from `MINICPMO_URL` to `CLOUD_AI_URL` once, then `raise_for_status`s. A flaky cloud connection will end the conversation.
- **STT hallucination filters are hardcoded.** Two-tier substring + exact-match list in `main.py` filters Whisper's known false-positive utterances ("thank you for watching", etc.). May over-filter; will under-filter on new false positives that aren't on the list.
- **`ConversationManager` init failure is unrecoverable.** If wiring fails in lifespan, `_conv_manager` stays `None`, all conversation endpoints return 503, and the only recovery is a restart.
- **Conversation cooldown is per-process and resets on restart.** Long-running uvicorn → reliable; supervisor-restart loops → cooldown gets bypassed.
- **`/configure` does not reset PresenceLock.** Toggling `ENABLE_YOLO` while in COOLDOWN won't immediately resume full-rate scanning.
- **The `_speaking_event` half-duplex sync is correct only when the audio delivery coroutine actually finishes near the audio's end.** WebSocket path is real-time-paced so this works. Play-from-URL path returns immediately after the speaker accepts the URL — the `SPEAKING_BUFFER_SEC` (0.5 s) buffer is *not enough* in that mode and you may pick up speaker echo as a person reply. Use WebSocket delivery for conversation use cases when possible.
- **`cloud_brain.py` and `vision_test.py` are not part of the runtime.** They are historical / smoke-test scripts. Don't refactor them assuming they're hot code.

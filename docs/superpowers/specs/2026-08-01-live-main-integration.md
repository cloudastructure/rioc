# Live Mode — main.py Integration Spec (on-device)

> All building blocks are built + tested (omni/, live/, whisper_log, router_http, routing, live_sse — 40 tests green). This file is the remaining wiring into `main.py`, which **must be applied and run on the appliance** (it binds the real WS-G.711 speaker + mic transports and cannot be verified off-device).

## Env vars

| Var | Purpose | Default |
|---|---|---|
| `CONVERSATION_MODE` | `turn_based` \| `live` | `turn_based` |
| `OMNI_ROUTER_URL` | Session Router base (POST /allocate) | — (required for live) |
| `LIVE_VOICE` | MiniCPM-o voice | `default` |
| `LIVE_ESCALATE_AFTER_SEC` / `LIVE_FINAL_AFTER_SEC` | escalation overlay timings | `20` / `45` |

Read once near the other config globals:
```python
from live.routing import route_conversation, validate_live_config
CONVERSATION_MODE = os.environ.get("CONVERSATION_MODE", "turn_based")
_live_orchestrator = None  # set per-session by /live/start
```

## Edit 1 — mode routing at the trigger (main.py ~1291)

Replace the direct turn-based kickoff:
```python
# BEFORE (≈1291)
if _conv_manager is not None and not _conv_manager.is_active():
    _presence_lock.on_conversation_started()
    asyncio.create_task(_conv_manager.on_person_detected(jpeg_bytes, initial_text=msg))
```
```python
# AFTER
def _start_turn_based():
    if _conv_manager is not None and not _conv_manager.is_active():
        _presence_lock.on_conversation_started()
        asyncio.create_task(_conv_manager.on_person_detected(jpeg_bytes, initial_text=msg))

def _start_live():
    _presence_lock.on_conversation_started()
    asyncio.create_task(_begin_live_session(jpeg_bytes, initial_text=msg))

route_conversation(CONVERSATION_MODE, _start_live, _start_turn_based)
```
The same webhook path (`_handle_person_detected_event`) gets the identical `route_conversation(...)` treatment.

## Edit 2 — `_begin_live_session` (new coroutine)

```python
async def _begin_live_session(jpeg_bytes: bytes, initial_text: str = "") -> None:
    global _live_orchestrator
    import httpx
    from live.channel_client import ChannelClient
    from live.speaker_sink import StreamingSpeakerSink
    from live.escalation import EscalationOverlay
    from live.live_session import LiveSessionOrchestrator
    from live.live_sse import broadcast_live

    # 0. speaker must support barge-in (WS path)
    try:
        validate_live_config("live", SPEAKER_TYPE)
    except ValueError as e:
        broadcast_live({"type": "error", "code": "speaker", "message": str(e)}); return

    # 1. allocate a GPU from the Session Router (control-plane)
    async with httpx.AsyncClient() as c:
        r = await c.post(f"{OMNI_ROUTER_URL}/allocate", json={"camera_id": CAMERA_ID})
    if r.status_code == 409:
        broadcast_live({"type": "capacity"}); return
    alloc = r.json()  # {gpu_ws_url, session_token, lease_ttl}

    # 2. connect the live channel straight to the chosen GPU
    channel = ChannelClient(clock=time.monotonic)
    await channel.connect(f"{alloc['gpu_ws_url']}?token={alloc['session_token']}")

    # 3. wire speaker sink + escalation + orchestrator
    speaker = StreamingSpeakerSink(_ws_speaker_transport(), jitter_frames=8)  # see media glue
    esc = EscalationOverlay(escalate_after=LIVE_ESCALATE_AFTER_SEC, final_after=LIVE_FINAL_AFTER_SEC)
    orch = LiveSessionOrchestrator(channel, speaker, esc, on_event=broadcast_live)
    if not await orch.start(system_prompt=_live_system_prompt(), voice=LIVE_VOICE, camera_id=CAMERA_ID):
        return  # capacity race — orch.start already broadcast 'capacity'
    _live_orchestrator = orch

    # 4. pumps (see media glue): downlink router, frame uplink, mic barge-in
    asyncio.create_task(_live_downlink_pump(orch, channel))
    asyncio.create_task(_live_uplink_pump(channel))
    asyncio.create_task(_live_barge_in_loop(orch, channel))
    asyncio.create_task(speaker.run())
```

## Edit 3 — HTTP endpoints

```python
@app.post("/live/start")
async def live_start():
    if _live_orchestrator is not None:
        return {"ok": False, "reason": "already active"}
    asyncio.create_task(_begin_live_session(_latest_conv_frame or b""))
    return {"ok": True}

@app.post("/live/stop")
async def live_stop():
    global _live_orchestrator
    if _live_orchestrator is not None:
        await _live_orchestrator.end("operator_stop")
        _live_orchestrator = None
    return {"ok": True}

@app.get("/live/stream")
async def live_stream():
    from live.live_sse import register_live_listener, unregister_live_listener
    q = register_live_listener()
    async def gen():
        try:
            while True:
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=30.0)
                    yield f"data: {json.dumps(ev)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            unregister_live_listener(q)
    return StreamingResponse(gen(), media_type="text/event-stream")
```
(Mirrors `/conversation/stream` at main.py:1846 exactly, but reads from `live.live_sse`.)

## Media glue — the on-device pieces (why this can't be verified off-device)

These bind Rioc's existing appliance internals and are the reason this step runs on the device:

1. **`_ws_speaker_transport()`** — return an object with `send_chunk(mulaw: bytes)` that writes a 20 ms μ-law frame to the **existing WS-G.711 speaker connection** inside `_guarded_play`. Requires extracting/holding the speaker websocket send loop as a reusable transport (today it's inline in `_guarded_play`).
2. **`_live_barge_in_loop(orch, channel)`** — read mic frames (as `mic_listener` does), feed `BargeInDetector(vad, echo_floor=…)`; on `"barge_in"` → `await orch.barge_in()`; on `"speech_start"/"speech_end"` → `channel.send("user_speech_start"/"user_speech_end")`. Call `detector.set_guard_speaking(True/False)` from the orchestrator's turn_start/turn_end.
3. **`_live_uplink_pump(channel)`** — sample `_latest_conv_frame` at 1–2 fps → `channel.send("video", jpeg_b64=…)`; stream mic PCM chunks → `channel.send("audio", pcm_or_opus_b64=…)`.
4. **`_live_downlink_pump(orch, channel)`** — `async for msg in channel.messages(): await orch.on_downlink(msg)`.
5. **Whisper backfill** — on each PERSON turn end, `asyncio.create_task(whisper_log.backfill_person_turn(_transcribe_audio, wav, conv_id, db))`.

## Verification (on device)

Run with a WS-capable speaker, a reachable `OMNI_ROUTER_URL`, and a GPU omni server. Walk `web-ui/.../prototype/UI-CHECKLIST.md` `AGLIVE-FLOW-001..008` end-to-end (start→active, barge-in, capacity, error, disconnected, ended). Confirm barge-in feels instant (local speaker stop) and captions stream.

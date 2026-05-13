# Rioc — AI Guard Backend

Automated AI security guard service. Detects people via camera (local YOLO or external webhook), analyzes frames with a vision LLM (MiniCPM-o / MiniCPM-V), and conducts two-way voice conversations with detected intruders through an IP speaker.

Built with **FastAPI** (Python). Runs as a single process with background tasks for vision analysis, audio transcription, and conversation management.

## Architecture

```
                          ┌─────────────────────────┐
                          │   vLLM Server (GPU)      │
                          │   MiniCPM-o / MiniCPM-V  │
                          │   OpenAI-compatible API   │
                          └────────▲──────────────────┘
                                   │ HTTPS
┌──────────┐  webhook/   ┌────────┴──────────────────┐    WebSocket/HTTP     ┌────────────┐
│ CVR /    ├────────────►│   Rioc (this service)      ├────────────────────►│ IP Speaker  │
│ Camera   │  RTSP       │   FastAPI on :8000         │                     │ (Fanvil)    │
└──────────┘             └────────┬──────────────────┘                     └────────────┘
                                  │ HTTPS
                          ┌───────▼───────────────────┐
                          │   OpenAI API               │
                          │   STT (Whisper / gpt-4o)   │
                          │   TTS (tts-1)              │
                          └────────────────────────────┘
```

## Prerequisites

- **Python 3.12+**
- **ffmpeg** — audio format conversion for speaker TTS (`brew install ffmpeg` / `apt install ffmpeg`)
- **Microphone** — for two-way conversation (optional; can use IP speaker's built-in mic via USB)

## External Services

| Service | Required? | Purpose | Notes |
|---------|-----------|---------|-------|
| **vLLM server** (self-hosted) | Yes | Vision LLM inference (MiniCPM-o-4_5 or MiniCPM-V-2_6) | OpenAI-compatible API. See [DEPLOY_AWS.md](DEPLOY_AWS.md) for GPU setup |
| **OpenAI API** | Yes (for audio) | STT via Whisper/gpt-4o-transcribe, TTS via tts-1 | Requires `OPENAI_STT_API_KEY` |
| **SQLite** | Auto | Conversation history (`ai_guard.db`, local file) | No setup needed — created automatically |
| **Ollama** | Optional | Local vision analysis (dev/demo fallback) | Only when `ENABLE_LOCAL_AUDIT=1` |
| **VideoDB** | Optional | Real-time video/audio indexing | Only when `ENABLE_VIDEODB=1` |
| **IP Speaker** (Fanvil CS20 etc.) | Optional | Loudspeaker output + mic input | WebSocket G.711 or HTTP play |

**No Redis, no external SQL database, no message queue.**

## Install

```bash
cd rioc/
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Copy the example env file and fill in values:

```bash
cp .env.example .env
```

### Required Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `OPENAI_STT_API_KEY` | OpenAI API key for STT and TTS | `sk-proj-...` |
| `CLOUD_AI_URL` | vLLM server base URL (with `/v1`) | `http://172.16.128.41:8100/v1` |
| `CLOUD_AI_API_KEY` | API key for vLLM server | `token-minicpm-v45` |

### Optional Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_CLOUD_AI` | `""` (off) | Set `1` to enable cloud vision analysis |
| `CLOUD_AI_MODEL` | `openbmb/MiniCPM-o-4_5-awq` | Model served by vLLM |
| `FRAME_SOURCE` | `webhook` | Person detection source: `webhook`, `local_yolo`, or `live_ffmpeg` |
| `CAMERA_RTSP_URL` | `""` | RTSP stream URL (if using local camera) |
| `ENABLE_LOCAL_AUDIT` | `""` (off) | Set `1` for local Ollama vision audit |
| `ENABLE_AUDIO_STT` | `""` (off) | Set `1` for microphone transcription |
| `ENABLE_SPEAKER_TTS` | `""` (off) | Set `1` to output TTS through IP speaker |
| `ENABLE_VIDEODB` | `""` (off) | Set `1` for VideoDB integration |
| `SPEAKER_URL` | `""` | IP speaker base URL (e.g. `https://192.168.10.183`) |
| `SPEAKER_WS_URL` | derived | WebSocket URL for speaker two-way audio |
| `SPEAKER_USER` / `SPEAKER_PASS` | `""` | Speaker auth credentials |
| `MINICPMO_URL` | `http://172.16.128.41:8101/` | MiniCPM-o conversation endpoint (port 8101) |
| `MINICPMO_API_KEY` | `token-minicpm-o45` | API key for MiniCPM-o conversation server |
| `AUDIO_INPUT_DEVICE` | system default | Mic device index or name substring (e.g. `Fanvil`) |
| `AUDIT_INTERVAL_SEC` | `2.0` | Seconds between vision audit cycles |
| `AUDIT_AI_FRAME_SIZE` | `320` | Frame resize before sending to cloud AI |
| `ENABLE_YOLO` | `1` | YOLO person pre-filter (reduces LLM calls) |
| `YOLO_CONFIDENCE` | `0.45` | YOLO detection confidence threshold |
| `ALERT_COOLDOWN_SEC` | `30.0` | Min seconds between detection alerts |
| `TTS_COOLDOWN_SEC` | `20.0` | Min seconds between TTS announcements |
| `CONVERSATION_MAX_TURNS` | `6` | Max conversation turns before ending |
| `CONVERSATION_COOLDOWN_SEC` | `20.0` | Cooldown between conversations |
| `TTS_PUBLIC_URL` | `""` | Public URL for speaker to fetch TTS audio (for play-from-URL mode) |
| `ENABLE_LOCAL_PLAYBACK` | `""` (off) | Play TTS through Mac speakers (testing) |

## Run

```bash
source .venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Production (webhook mode — default)

The default `FRAME_SOURCE=webhook` mode expects person-detection events from an external CVR system via `POST /api/person-detected`. No local camera or YOLO loop is needed.

```bash
ENABLE_CLOUD_AI=1 \
ENABLE_AUDIO_STT=1 \
ENABLE_SPEAKER_TTS=1 \
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Development (local YOLO)

Uses a local camera and YOLO for person detection:

```bash
FRAME_SOURCE=local_yolo \
CAMERA_RTSP_URL=rtsp://... \
ENABLE_CLOUD_AI=1 \
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### Core

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | HTML status page |
| `GET` | `/video` | MJPEG video stream |
| `GET` | `/transcript` | Latest audio transcript |
| `GET` | `/analysis` | Latest cloud AI vision analysis |

### Detections & Events

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/detections` | Recent detection history (JSON array) |
| `GET` | `/detections/stream` | SSE stream of real-time detection events |
| `GET` | `/events` | In-memory event log (last 500 events) |

### Conversations (AI Guard two-way voice)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/conversation/start` | Start a new conversation manually |
| `POST` | `/conversation/respond` | Inject a text response into active conversation |
| `GET` | `/conversation/status` | Current conversation state |
| `GET` | `/conversation/stream` | SSE stream of conversation turns |
| `POST` | `/conversation/configure` | Update system prompt / max turns at runtime |
| `GET` | `/conversations` | List past conversations (from SQLite) |
| `GET` | `/conversations/{id}` | Get a conversation with all turns |

### Webhooks & Integration

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/person-detected` | Webhook: receive person-detection events (with JPEG frame) |
| `POST` | `/api/frame-update` | Push a camera frame update |
| `POST` | `/configure` | Reconfigure camera RTSP URL at runtime |

### Speaker & TTS

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/tts/test` | Test TTS through speaker |
| `GET` | `/tts/latest.mp3` | Latest generated TTS audio file |
| `GET` | `/speaker-test` | Test speaker connectivity |
| `GET` | `/speaker-diagnostic` | Speaker connection diagnostics |
| `GET` | `/speaker-test-bell` | Play test bell sound on speaker |

## Data Storage

- **`ai_guard.db`** — SQLite database (auto-created in the project root). Stores conversation history and turns. No migrations needed; tables are created on first run.
- **`audio_logs/`** — Directory for saved audio recordings from conversations (auto-created).

## Files

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app, endpoints, YOLO detection, TTS/speaker, background loops |
| `conversation_manager.py` | Two-way voice conversation state machine (WARNING → ESCALATING → FINAL) |
| `minicpmo_client.py` | MiniCPM-o API client (vision + audio, with fallback) |
| `mic_listener.py` | VAD-gated microphone capture (WebRTC VAD) |
| `db.py` | SQLite schema and CRUD for conversations/turns |
| `vision_test.py` | Standalone Ollama vision test script |

## Cloud Brain (AWS deployment)

For deploying the vision LLM (vLLM) on AWS or RunPod GPU instances, see **[DEPLOY_AWS.md](DEPLOY_AWS.md)**.

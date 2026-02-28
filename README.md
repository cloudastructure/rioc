# Rioc

Webcam streaming over HTTP and an Ollama vision test (Rioc persona).

## Prerequisites

- Python 3.x
- Camera (default index 0)
- **ffmpeg** (for Speaker TTS: `brew install ffmpeg`)
- For **vision_test**: local [Ollama](https://ollama.ai) with `llama3.2-vision` (e.g. `ollama run llama3.2-vision`)

## Install

From the `rioc/` directory:

```bash
pip install -r requirements.txt
```

## Run

### Webcam stream

Stream the webcam at 640x640, 70% JPEG (MJPEG) for local or cloud consumption:

```bash
uvicorn webcam_stream:app --host 0.0.0.0 --port 8000
```

Then open http://localhost:8000 or http://localhost:8000/video in a browser.

**Optional: Local Visual Audit** — Run the same Visual Audit (tactical analyst, MiniCPM-V 2.6) in the same process:

1. Have Ollama running with the model: `ollama run openbmb/minicpm-v2.6` (once).
2. Enable the audit and start the stream:

```bash
ENABLE_LOCAL_AUDIT=1 uvicorn webcam_stream:app --host 0.0.0.0 --port 8000
```

Audit results print in the server console every few seconds.

**Optional: Listening (audio STT)** — Let the guard “hear” as well as see:

```bash
ENABLE_AUDIO_STT=1 OPENAI_STT_API_KEY=sk-your-key uvicorn webcam_stream:app --host 0.0.0.0 --port 8000
```

The app will capture microphone audio, send it to OpenAI’s STT (`whisper-1` by default), and expose the latest transcript at `GET /transcript`. The local Visual Audit and Cloud Brain can incorporate this transcript into their tactical warnings.

Optional env: `OLLAMA_URL` (default `http://localhost:11434`), `OLLAMA_VISION_MODEL` (default `openbmb/minicpm-v2.6`), `AUDIT_INTERVAL_SEC` (default `5.0`), `ENABLE_AUDIO_STT` (enable/disable), `OPENAI_STT_MODEL` (default `whisper-1`), `STT_SAMPLE_RATE`, `STT_DURATION_SEC`, `STT_GAP_SEC`.

**Optional: VideoDB eyes and ears** — Use [VideoDB](https://docs.videodb.io/pages/getting-started/quickstart) for real-time transcript + visual/audio indexing:

```bash
ENABLE_VIDEODB=1 VIDEODB_API_KEY=your-key uvicorn webcam_stream:app --host 0.0.0.0 --port 8000
```

With no RTSP URLs set, Rioc uses VideoDB's demo streams so you can try it immediately. For your own webcam/mic: run `mediamtx mediamtx.yml` (the config defines `cam` and `mic` paths), then FFmpeg to publish, ngrok TCP to expose, and set `VIDEODB_RTSP_VIDEO` / `VIDEODB_RTSP_AUDIO` in `.env`. See `scripts/videodb_rtsp.sh`.

**Optional: Speaker TTS** — Rioc speaks through an IP speaker:

Add to `.env`: `SPEAKER_URL`, `SPEAKER_USER`, `SPEAKER_PASS`. Then:

```bash
ENABLE_SPEAKER_TTS=1 ENABLE_LOCAL_AUDIT=1 ENABLE_AUDIO_STT=1 uvicorn webcam_stream:app --host 0.0.0.0 --port 8000
```

Rioc’s responses (to speech and Visual Audit) are converted to speech via OpenAI TTS and sent to the speaker. 
**Fanvil LINKVIL CS20 (and similar)** — Uses WebSocket at `wss://<speaker-ip>:8000/webtwowayaudio`. Add `SPEAKER_WS_URL=wss://192.168.10.183:8000/webtwowayaudio` to `.env`. The app streams TTS as **G.711 μ-law at 8 kHz** (same format as the dashboard's talk test). Requires **ffmpeg**: `brew install ffmpeg`.

**Use the speaker for both input and output** — When the Fanvil (or similar) is connected via USB, it appears as a Mac audio device. Set `AUDIO_INPUT_DEVICE=Fanvil` (or `AUDIO_INPUT_DEVICE=1` if it's the second input) in `.env` so Rioc listens through the speaker's mic instead of the Mac mic. For VideoDB's FFmpeg mic stream, use the numeric index: `AUDIO_INPUT_DEVICE=1` (find indices with `ffmpeg -f avfoundation -list_devices true -i ""`).

**Other speakers / Speaker can't reach your Mac (firewall)?** Use **Cloudflare Tunnel** (no interstitial; ngrok free tier blocks API clients):

1. Install cloudflared: `brew install cloudflared`
2. In one terminal, run the app: `uvicorn webcam_stream:app --host 0.0.0.0 --port 8000`
3. In another terminal: `cloudflared tunnel --url http://localhost:8000`
4. Copy the HTTPS URL (e.g. `https://xxx-xxx-xxx.trycloudflare.com`)
5. In `.env`, set: `TTS_PUBLIC_URL=https://xxx-xxx-xxx.trycloudflare.com`
6. Restart the app. The speaker will fetch TTS from the tunnel (no firewall changes needed).

   **Note:** Quick tunnel URLs change when you restart cloudflared—update `TTS_PUBLIC_URL` each time.

### Vision test (Rioc)

Capture frames and send them to Ollama with the Rioc persona:

```bash
python vision_test.py
```

**Note:** Both apps use the same camera (index 0). Do not run the webcam stream and vision test at the same time on one machine if they share the device, or the camera may be unavailable to one of them.

### Cloud Brain (Visual Audit on AWS)

**cloud_brain.py** runs on **AWS**. See **[DEPLOY_AWS.md](DEPLOY_AWS.md)** for the full deployment guide. It pulls frames from the Mac’s MJPEG stream and sends them to vLLM (MiniCPM-V-2_6) via an OpenAI-compatible API for a “Visual Audit.”

**Requirements:**

1. **vLLM** running on AWS serving **openbmb/MiniCPM-V-2_6** (e.g. `vllm serve openbmb/MiniCPM-V-2_6` with the appropriate vision/chat template).
2. The **Mac stream** reachable at `http://<MY_MAC_IP>:8000/video` (run the webcam stream on the Mac with `uvicorn webcam_stream:app --host 0.0.0.0 --port 8000` and ensure the AWS instance can reach the Mac’s IP).

**Environment variables:**

| Variable | Description | Default |
|----------|-------------|---------|
| `MAC_IP` | Mac’s IP (stream at `http://<MAC_IP>:8000/video`) | `localhost` |
| `STREAM_URL` | Full stream base URL; overrides `MAC_IP` if set | `http://<MAC_IP>:8000` |
| `VLLM_BASE_URL` | vLLM OpenAI-compatible API base (e.g. `http://<vllm-host>:8000/v1`) | `http://localhost:8000/v1` |
| `AUDIT_INTERVAL_SEC` | Seconds between audits | `5.0` |
| `OPENAI_API_KEY` | API key for vLLM (use `EMPTY` if none) | `EMPTY` |
| `TRANSCRIPT_URL` | URL for latest transcript JSON (default `<STREAM_URL>/transcript`) | derived |

**Run:**

```bash
# On AWS EC2 (after vLLM is running, .env has STREAM_URL and OPENAI_API_KEY)
pip install -r requirements-cloud_brain.txt
python cloud_brain.py
```

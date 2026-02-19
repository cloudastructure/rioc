# Rioc

Webcam streaming over HTTP and an Ollama vision test (Rioc persona).

## Prerequisites

- Python 3.x
- Camera (default index 0)
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

**Optional: Speaker TTS** — Rioc speaks through an IP speaker:

Add to `.env`: `SPEAKER_URL`, `SPEAKER_USER`, `SPEAKER_PASS`. Then:

```bash
ENABLE_SPEAKER_TTS=1 ENABLE_LOCAL_AUDIT=1 ENABLE_AUDIO_STT=1 uvicorn webcam_stream:app --host 0.0.0.0 --port 8000
```

Rioc’s responses (to speech and Visual Audit) are converted to speech via OpenAI TTS and POSTed to the speaker. The play endpoint defaults to `SPEAKER_PLAY_PATH=/play`; if your device uses a different path, set `SPEAKER_PLAY_PATH` in `.env`.

**Speaker returns 200 but no sound?** Check the device dashboard: **Audio** (volume, output enabled) and **Media File** (some speakers require upload-then-play by index).

**Speaker can't reach your Mac (firewall)?** Use **Cloudflare Tunnel** (no interstitial; ngrok free tier blocks API clients):

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

**cloud_brain.py** is intended to run on **AWS**. It pulls frames from the Mac’s MJPEG stream and sends them to vLLM (MiniCPM-V-2_6) via an OpenAI-compatible API for a “Visual Audit.”

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
# On AWS (Mac IP and vLLM host set for your environment)
export MAC_IP=192.168.1.100
export VLLM_BASE_URL=http://vllm-host:8000/v1
python cloud_brain.py
```

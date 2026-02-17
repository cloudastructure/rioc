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

Audit results print in the server console every few seconds. Optional env: `OLLAMA_URL` (default `http://localhost:11434`), `OLLAMA_VISION_MODEL` (default `openbmb/minicpm-v2.6`), `AUDIT_INTERVAL_SEC` (default `5.0`).

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

**Run:**

```bash
# On AWS (Mac IP and vLLM host set for your environment)
export MAC_IP=192.168.1.100
export VLLM_BASE_URL=http://vllm-host:8000/v1
python cloud_brain.py
```

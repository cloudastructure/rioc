# MiniCPM-o Streaming (Omni) Server — Setup Instructions

**Audience:** whoever set up the original MiniCPM vLLM deployment.
**Goal:** stand up a *real-time streaming* MiniCPM-o server for the AI Guard "Live" (full-duplex) mode.

---

## 0. Read this first — why vLLM can't do this

The current deployment (`vllm serve openbmb/MiniCPM-...`) exposes the **OpenAI `/v1/chat/completions`** endpoint. That is **request/response**: one prompt (with a frame or two) in, one answer out. It is *stateless* — there is no "session," so it cannot keep watching a video and listening while it talks.

The Live feature needs MiniCPM-o's **streaming omni session**: you feed it a continuous trickle of video frames *and* audio, and it streams speech back in real time, and can be interrupted. That capability is **not in vLLM** — it lives in MiniCPM-o's own model class (the HuggingFace "trust_remote_code" implementation), via methods like `streaming_prefill(...)` / `streaming_generate(...)` plus its built-in TTS.

So this is a **second, separate server process** (transformers-based), running **alongside** the existing vLLM. Keep the vLLM running — the turn-based path still uses it. We just add the streaming server for Live mode.

---

## 1. What you're standing up

A long-lived Python process that:
- loads MiniCPM-o via `transformers` with `trust_remote_code=True`,
- enables its audio input + TTS (speech output) modules,
- holds one **stateful session per live conversation**, and
- exposes a **WebSocket** endpoint that speaks the small protocol our appliance (Rioc) already implements.

We already wrote that server: **`omni/server.py`** in the Rioc repo (branch `feat/live-multimodal-guard`), with the model adapter in **`omni/omni_session.py`**. Your job is to (a) get the model loading in streaming mode on the GPU, (b) confirm the exact streaming API names for the model version you pin, and (c) run our server. Details below.

---

## 2. Pick the model variant  ⚠️ decision needed

- The reference streaming/omni + TTS path is documented for **`openbmb/MiniCPM-o-2_6`** (and newer `MiniCPM-o` releases). Use the **bf16 / full** weights.
- **The AWQ-quantized variant you may be using for vLLM (`...-4_5-awq`) is unlikely to support the streaming + TTS path** — AWQ builds are optimized for vLLM inference, and the real-time TTS/omni modules typically require the non-quantized model. **Confirm on the model card**; if in doubt, use the bf16 model for this server.
- Whatever you pick, **pin the exact `transformers` version the model card specifies** (the remote code is version-sensitive — a mismatch breaks `trust_remote_code`).

## 3. Hardware / VRAM

- MiniCPM-o 2.6 in bf16 is ~8B params → **~18–20 GB** just for weights, **plus** the audio encoder + TTS modules. Budget **≥ 24 GB VRAM** per instance (A10G 24 GB, L4 24 GB, or A100). Confirm against the model card for your chosen version.
- CUDA + a recent NVIDIA driver, same as the vLLM box. This can be the **same physical GPU** as vLLM only if there's spare VRAM headroom for both; otherwise a separate GPU/instance.

## 4. Environment + dependencies

Follow the model's official repo, not guesswork — the streaming demo has a pinned requirements file:

```bash
# reference implementation + pinned deps
git clone https://github.com/OpenBMB/MiniCPM-o
cd MiniCPM-o
python -m venv .venv && source .venv/bin/activate
pip install -r requirements_o2.6.txt        # or the file the current model card names
# TTS/omni extras are in that file: torch, torchaudio, transformers==<pinned>,
# vector_quantize_pytorch, vocos, librosa, soundfile, etc.
```

Then confirm the model loads in **omni/streaming** mode (this snippet mirrors the model card — adjust names to the pinned version):

```python
import torch
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained(
    "openbmb/MiniCPM-o-2_6", trust_remote_code=True,
    attn_implementation="sdpa", torch_dtype=torch.bfloat16,
    init_vision=True, init_audio=True, init_tts=True,   # <- audio in + TTS out
).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained("openbmb/MiniCPM-o-2_6", trust_remote_code=True)
model.init_tts()   # loads the speech-generation module
```

If that loads and `model` has `streaming_prefill` / `streaming_generate` methods, you're 90% there.

## 5. Confirm the streaming API  ⚠️ report back to us

Our adapter (`omni/omni_session.py`) is written against these reference methods:

```python
model.streaming_prefill(session_id, {"role": "user", "content": [...]}, tokenizer)
model.streaming_generate(session_id, tokenizer, ...)  # yields streaming text + audio chunks
```

For the **exact version you pin**, confirm and send us back:
1. The exact **signatures** of `streaming_prefill` / `streaming_generate` (arg names, how `session_id` is passed, how you signal "generate now").
2. The **content-chunk format** for a video frame and an audio chunk (the reference uses `{"type": "image", ...}` / `{"type": "audio", ...}` — confirm keys + whether images are PIL / bytes / base64, and audio is raw PCM / a specific sample rate).
3. The **output audio format**: sample rate + encoding of the speech it streams back (we need this to resample to the speaker — the Rioc side currently assumes **16-bit mono PCM @ 24 kHz**; correct us if it differs).

We'll drop those into `omni/omni_session.py` (there's a `CONFIRMED API` comment block waiting for exactly this) — a ~10-line change, no redesign.

## 6. Run our server

```bash
# in the Rioc repo, branch feat/live-multimodal-guard, on the GPU box
pip install fastapi uvicorn websockets   # (+ the model deps from step 4)
LIVE_MAX_SESSIONS=<from step 8> \
uvicorn omni.server:app --host 0.0.0.0 --port 8102
```

`omni/server.py` exposes `GET /omni/session` (WebSocket) and handles admission + the prefill/generate loop. If for any reason it's easier to start from the repo's own `web_demos/omni` server, that's fine too — just keep the **same WebSocket message protocol** (defined in `omni/protocol.py`: uplink `session_start` / `video` / `audio` / `user_speech_end` / `interrupt`; downlink `session_ready` / `speech` / `text` / `turn_start` / `turn_end` / `interrupted` / `capacity_exhausted`).

## 7. Networking

- Expose the WebSocket (e.g. `:8102`) so it's reachable from **Rioc** and from the **Session Router** (`omni/router_http.py`, if you run the multi-GPU router in front). Keep the existing **vLLM on `:8101`** untouched — turn-based still uses it.
- If it's remote from the sites (over VPN/WAN), that's expected — the design accounts for it.

## 8. Benchmark concurrency (sets `LIVE_MAX_SESSIONS`)

A streaming session holds GPU state for its whole duration, so one GPU serves a *bounded* number of simultaneous live conversations. **Measure it:** open N concurrent streaming sessions with a test loop and find the largest N where speech still comes back in real time (barge-in stays snappy). Set `LIVE_MAX_SESSIONS` (per-GPU) to that number. Add GPUs later to raise the fleet total.

---

## What to send back to us (unblocks the Rioc side)

1. The confirmed streaming API signatures + chunk formats (step 5.1, 5.2).
2. The **output audio sample rate + encoding** (step 5.3).
3. The server's reachable **WebSocket URL**.
4. The measured **`LIVE_MAX_SESSIONS`** (step 8).

With those four, the Rioc side is a ~10-line adapter fill + config, and we can do a real end-to-end run.

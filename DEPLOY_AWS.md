# Deploy Cloud Brain on AWS (or RunPod)

Cloud Brain pulls frames from your Mac's webcam stream and sends them to vLLM (MiniCPM-V-2_6) for Visual Audit. You can run vLLM on **AWS EC2** or **RunPod** (no AWS GPU quota needed).

## No G-type instances? Use RunPod

If you don't have access to AWS GPU instances (g5, g4dn, etc.), use **RunPod Serverless** instead—pay per second, no quota limits.

### RunPod setup (quick path)

1. **Accept the model on Hugging Face** (required—it's gated):
   - Go to [huggingface.co/openbmb/MiniCPM-V-2_6](https://huggingface.co/openbmb/MiniCPM-V-2_6)
   - Log in and click **"Agree and access repository"**.

2. **Create a Hugging Face token**:
   - [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → **New token** → Read access.

3. Sign up at [runpod.io](https://runpod.io).

4. **Serverless** → **Deploy** → choose **vLLM** worker.

5. **Environment variables** (in endpoint settings):
   - `MODEL_NAME` = `openbmb/MiniCPM-V-2_6-int4` (4-bit quantized; works without gating issues)
   - `HF_TOKEN` = your Hugging Face token (add as **Secret**; helps with download)
   - `TRUST_REMOTE_CODE` = `True`

6. Pick a GPU (e.g. 24 GB A10G or L40S). int4 uses less VRAM than the full model.

7. Deploy and copy your **Endpoint ID**.

8. Your vLLM base URL: `https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1`

9. **API key**: your RunPod API key (from RunPod dashboard).

Run `cloud_brain.py` from your Mac (or any machine):

```bash
export STREAM_URL=http://localhost:8000   # or your Cloudflare tunnel URL
export VLLM_BASE_URL=https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1
export OPENAI_API_KEY=<RUNPOD_API_KEY>
python cloud_brain.py
```

RunPod bills per second of GPU time; idle = $0.

---

## Architecture (AWS EC2)

```
[Mac] webcam_stream.py (GET /video, /transcript)
         │
         │  (expose via Cloudflare Tunnel or ngrok)
         ▼
[AWS EC2 or RunPod] vLLM (MiniCPM-V-2_6)  ←  cloud_brain.py
```

- **Mac**: Must run `webcam_stream.py` and expose it so the cloud can reach it.
- **vLLM**: Runs on AWS EC2 (below) or RunPod (above).

## Part 1: Expose Your Mac Stream

Your Mac must be reachable from the internet so AWS can fetch `GET /video` and `GET /transcript`.

### Option A: Cloudflare Tunnel (recommended)

1. On your Mac:
   ```bash
   brew install cloudflared
   cloudflared tunnel --url http://localhost:8000
   ```
2. Copy the HTTPS URL (e.g. `https://xxx-xxx.trycloudflare.com`).
3. Use this as `STREAM_URL` when running cloud_brain on AWS.

### Option B: ngrok

```bash
ngrok http 8000
```

Use the HTTPS URL (free ngrok may show interstitial for browser; API clients usually work).

---

## Part 2: Launch EC2 with vLLM (optional)

*If you don't have g5/g4dn quota, use RunPod above instead.*

### 2.1 Launch a GPU Instance

**No G-type quota?** Request a limit increase: AWS Console → Service Quotas → EC2 → "Running On-Demand G and VT instances" → Request quota increase. Or use RunPod (see top of this doc).

1. **AWS Console** → EC2 → Launch Instance.
2. **AMI**: Amazon Linux 2023 or Ubuntu 22.04.
3. **Instance type**: `g5.xlarge` (1× NVIDIA A10G, 24 GB VRAM) or `g5.2xlarge` (1× L40S, 48 GB). MiniCPM-V-2_6 fits on g5.xlarge.
4. **Storage**: 100 GB+ (for model + cache).
5. **Security group**: Allow inbound SSH (22) and HTTP (8000) from your IP or 0.0.0.0/0 if you want vLLM reachable from outside.
6. **Key pair**: Create or select one for SSH.

### 2.2 Connect and Install

```bash
ssh -i your-key.pem ec2-user@<EC2_PUBLIC_IP>
```

**Ubuntu:**

```bash
# NVIDIA drivers (if not pre-installed)
sudo apt update && sudo apt install -y nvidia-driver-535

# CUDA (if needed)
# wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
# sudo sh cuda_12.1.0_530.30.02_linux.run

# Python + pip
sudo apt install -y python3-pip python3-venv

# Create venv
python3 -m venv ~/venv
source ~/venv/bin/activate

# vLLM + MiniCPM-V-2_6
pip install vllm>=0.10.2 openai httpx python-dotenv
```

**Amazon Linux 2023:**

```bash
sudo dnf install -y python3-pip python3.11
python3.11 -m venv ~/venv
source ~/venv/bin/activate
pip install vllm>=0.10.2 openai httpx python-dotenv
```

### 2.3 Start vLLM

```bash
vllm serve openbmb/MiniCPM-V-2_6 \
  --dtype auto \
  --max-model-len 2048 \
  --api-key token-rioc \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code
```

The model will download on first run (~8 GB). vLLM listens on `http://0.0.0.0:8000` by default.

**Run in background (e.g. with tmux):**

```bash
tmux new -s vllm
vllm serve openbmb/MiniCPM-V-2_6 --dtype auto --max-model-len 2048 --api-key token-rioc --gpu-memory-utilization 0.9 --trust-remote-code
# Ctrl+B, D to detach
```

---

## Part 3: Run Cloud Brain

### 3.1 Copy Project to EC2

From your Mac:

```bash
scp -i your-key.pem cloud_brain.py requirements-cloud_brain.txt ec2-user@<EC2_PUBLIC_IP>:~/rioc/
```

Or clone the repo on EC2 if it's in a git remote.

### 3.2 Environment Variables

On the EC2 instance, create `~/rioc/.env`:

```bash
# Mac stream (use your Cloudflare/ngrok URL)
STREAM_URL=https://xxx-xxx-xxx.trycloudflare.com

# vLLM on this EC2 (localhost if cloud_brain runs on same instance)
VLLM_BASE_URL=http://localhost:8000/v1

# API key must match vLLM --api-key
OPENAI_API_KEY=token-rioc

# Optional
AUDIT_INTERVAL_SEC=5.0
```

### 3.3 Run Cloud Brain

```bash
cd ~/rioc
source ~/venv/bin/activate
pip install -r requirements-cloud_brain.txt
python cloud_brain.py
```

You should see output like:

```
[Visual Audit] You in the dark jacket, I see you. This area is restricted...
```

---

## Part 4: Mac Setup (reminder)

On your Mac, run the webcam stream:

```bash
uvicorn webcam_stream:app --host 0.0.0.0 --port 8000
```

Then start the Cloudflare Tunnel (or ngrok) so AWS can reach it.

**Optional:** Disable local Visual Audit when using Cloud Brain to avoid duplicate audits:

```bash
ENABLE_LOCAL_AUDIT=0 ENABLE_AUDIO_STT=1 ENABLE_SPEAKER_TTS=1 uvicorn webcam_stream:app --host 0.0.0.0 --port 8000
```

---

## Cloud Brain TTS (future)

Cloud Brain currently only prints to stdout. To have it speak through your Fanvil speaker, you would need to:

1. Add an HTTP endpoint on your Mac that accepts the audit text and triggers TTS.
2. Have Cloud Brain POST the audit to that endpoint (e.g. `https://your-tunnel-url/broadcast`).

This could be a follow-up enhancement.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `Stream unreachable` | Ensure STREAM_URL is correct and Cloudflare/ngrok is running. |
| `Connection refused` to vLLM | Start vLLM first; ensure VLLM_BASE_URL points to the right host/port. |
| GPU OOM | Use `--gpu-memory-utilization 0.8` or a larger instance type. |
| Model download slow | Pre-download on a machine with fast internet, then copy to EC2. |

### RunPod: "Engine core initialization failed" / EngineCore_DP0

This usually means vLLM failed to start the model. Check the **full logs** above the traceback for the real cause (often OOM or CUDA).

**Fixes to try:**

1. **Use legacy vLLM engine** – vLLM v1 can fail with vision/quantized models. Add:
   - `VLLM_USE_V1` = `0`

2. **GPU size** – Use at least **24 GB VRAM** (A10G, L40S). T4 (16 GB) can OOM with vision models.

3. **Environment variables** (in RunPod endpoint settings):
   - `GPU_MEMORY_UTILIZATION` = `0.8` (lower than default 0.95)
   - `MAX_MODEL_LEN` = `2048`
   - `MODEL_NAME` = `openbmb/MiniCPM-V-2_6-int4` (no extra spaces or characters—RunPod UI can corrupt this)

4. **Model name** – In RunPod, confirm `MODEL_NAME` has no trailing junk. If it still fails, try the full model with HF token: `openbmb/MiniCPM-V-2_6` (needs 48 GB for fp16, or use a larger GPU).

5. **Redeploy** – If rollout is stuck (e.g. "Rollout in progress" for 10+ minutes with 0 running workers), stop all workers, add the env vars above, then redeploy. Don't let a broken config keep retrying.

### RunPod: 500 "Error processing the request"

The endpoint starts but chat completions return 500. The error is generic; the **real cause is in RunPod worker logs**.

**Steps:**

1. **Check RunPod worker logs** – Endpoint → Workers → click a worker → view logs. Look for OOM, CUDA errors, or Python tracebacks when a request hits.

2. **Model name** – Cloud Brain now auto-discovers the model from `/v1/models`. If RunPod uses `OPENAI_SERVED_MODEL_NAME_OVERRIDE`, the discovered name will be used automatically.

3. **Image size** – If logs show OOM during inference, reduce frame size on the Mac: in `webcam_stream.py`, lower `FRAME_SIZE` (e.g. `(384, 384)`) and/or `JPEG_QUALITY`.

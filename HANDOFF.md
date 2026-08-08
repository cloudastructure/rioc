# Rioc (AI Guard backend) — Handoff

> **For:** Michael
> **From:** Melissa
> This is the **backend** for AI Guard. The **frontend** is in the separate `web-services` repo, branch `feat/rioc-v2` — see its `web-ui/src/admin/pages/accounts/ai-guard/HANDOFF.md`. Read this alongside that one.

---

## 1. TL;DR

- Rioc is the engine: camera frames → YOLO/vision analysis → scripted guard conversation → spoken through the IP speaker, with the person's mic replies transcribed back in.
- The work is spread across a few branches (this is a solo repo, so nothing is merged to `main` yet). **The branch map in §3 is the important part** — start there.
- **Your likely first task** is the one real blocker for the new full-duplex Live mode: standing up the MiniCPM-o streaming server. See §4.

---

## 2. The two repos

| Repo | What it is | Where |
|---|---|---|
| **rioc** (this) | Python/FastAPI backend — the AI guard engine | `~/rioc` (+ worktree `~/rioc-live`) |
| **web-services** | React frontend / operator console | branch `feat/rioc-v2` |

The FastAPI app lives in **`main.py`** (run with uvicorn). The old `webcam_stream.py` has been retired on the latest turn-based branch — `main.py` is the app.

---

## 3. Branch map (all off `cloudastructure/rioc`)

| Branch | State | What it is |
|---|---|---|
| **`feat/live-multimodal-guard`** (this branch) | pushed; 10 commits ahead of `main`; no PR yet | **Full-duplex Live mode** — persistent MiniCPM-o omni streaming session with barge-in. New `omni/` (GPU streaming server + Session Router) and `live/` (appliance engine) packages, `/live/*` endpoints wired into `main.py`, ~40 tests green. **Not yet run on real hardware.** |
| **`feat/cvr-frame-for-conversation-turns`** | pushed; PR #6 open | **Latest turn-based work.** Recent-CVR-frame webhook (`POST /api/frame-update`), STT upgraded to `gpt-4o-transcribe`, retires `webcam_stream.py`. Also holds the design docs (`docs/superpowers/`) and `ARCHITECTURE.md`. |
| `feat/swappable-guard-backend` | pushed; PR #7 open | Swappable guard backend (vLLM today, realtime hook for later). |
| `dockerize` | pushed; PR #5 open | Containerization. |
| `audio-stt` | merged (PRs #1–3) | Two-way STT guard conversation. Historical. |
| `main` | — | Behind the feature branches; not yet updated with the above. |

> These branches stack on each other (they all touch `main.py`). They have **not** been consolidated into `main` — if you want a single integrated tree, that merge is still to be done and should go in dependency order (turn-based → swappable → live).

**Design docs to read** (on `feat/cvr-frame-for-conversation-turns`):
- `ARCHITECTURE.md` — overall backend architecture.
- `docs/superpowers/specs/2026-08-01-live-multimodal-guard-design.md` (+ diagram/poster/explainer HTML) — the Live mode design.
- `docs/superpowers/plans/2026-08-01-live-multimodal-guard-backend.md` — the Live backend implementation plan.
- `docs/superpowers/plans/2026-07-02-swappable-guard-backend.md` — swappable backend plan.

On **this** (`feat/live-multimodal-guard`) branch:
- `docs/OMNI_STREAMING_SERVER.md` — how to stand up the MiniCPM-o streaming server (your first task, §4).
- `docs/superpowers/specs/2026-08-01-live-main-integration.md` — Live/`main.py` integration spec.

---

## 4. The one real blocker for Live mode

The full-duplex Live pipeline is built on the Rioc side (`omni/` + `live/`, ~40 tests green), **but it has no streaming brain to talk to yet:**

- MiniCPM-o is currently served by plain **`vllm serve`** — stateless `/v1/chat/completions` only. vLLM **cannot hold a streaming omni session**, so true full-duplex barge-in doesn't work against it.
- **What's needed:** stand up `omni/server.py` (HF MiniCPM-o `streaming_prefill` / `generate`) on the GPU box, and confirm the downlink audio rate/encoding (`MINICPMO_AUDIO_RATE`). That spike is the gate; the Rioc side is otherwise complete.
- **Step-by-step for this is in [`docs/OMNI_STREAMING_SERVER.md`](./docs/OMNI_STREAMING_SERVER.md) on this branch** — start there.

Everything else (Session Router, admission, barge-in, escalation timer, SQLite persistence, live SSE) is implemented and unit-tested — it just needs a real streaming server + a hardware run.

Also still open: the **video-feed topology** choice for Live mode — can Rioc pull the camera RTSP directly, or must the edge push frames via `POST /api/frame-update`? Both paths are built; it needs a config decision.

---

## 5. Run it locally

Turn-based backend:
```bash
source ~/rioc/.venv/bin/activate
cd ~/rioc
ENABLE_CLOUD_AI=1 uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
Then point the web-ui at it via the **Configuration tab** (the Rioc API URL is operator-configured and stored in the browser's `localStorage`, not in backend config).

Tests:
```bash
# Some websocket tests must run in a separate pytest pass — use the helper if present:
./scripts/run_tests.sh   # falls back to: pytest
```

---

## 6. Notes

- Runtime artifacts are gitignored (`ai_guard.db`, `audio_logs/`, `.claude/`). Don't commit them.
- This is a solo repo, so branch hygiene is loose — trust the branch map above over `main`.
- Questions on the frontend contract → the endpoint table in the frontend's `ARCHITECTURE.md` is the fastest reference for what the UI expects from these endpoints.

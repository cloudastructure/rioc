# Rioc — DevOps Deployment Guide

Operational reference for building, deploying, and running **rioc** (AI Guard backend) in shared infrastructure.

For app behavior, endpoints, and feature flags see [README.md](README.md). This guide only covers what DevOps needs to ship and run the service.

---

## 1. Service profile

| Field | Value |
|------|-------|
| App name (CI / k8s) | `rioc` |
| Runtime | Python 3.12, FastAPI / uvicorn (single process, async) |
| Listen port | `8000/tcp` (HTTP + SSE + WebSocket-out) |
| Protocol | HTTP/1.1 (no HTTPS termination in-pod — front with ingress/LB) |
| GPU required? | **No.** All heavy vision inference is offloaded to an external vLLM server. Local YOLOv8-nano runs on CPU. |
| State | SQLite file (`ai_guard.db`) + on-disk audio captures (`audio_logs/`) |
| Stateless? | **No** — see [§5 Persistence](#5-persistence). Single replica unless DB is moved off-box. |
| Health endpoint | `GET /events` (returns 200 once the app loop is up) |
| Slack / Logs | stdout/stderr, line-buffered (`PYTHONUNBUFFERED=1`) |

---

## 2. Build

### CI (Jenkins)

The pipeline definition is `Jenkinsfile-CI.groovy`. It:

1. Clones the shared `cloud-infrastructure` repo for build/push helper scripts.
2. Builds a single Docker image from the repo-root `Dockerfile` (no GPU SKU variants).
3. Pushes the image tagged with the short git SHA to the registry configured for `GCE_ENVIRONMENT` (default: `dev-dc-03`).
4. Posts Slack notifications on start / success / failure.

Branch + image-tag convention: `${BUILD_NUMBER}-${branch}-${git-sha}` (set as the Jenkins build display name; the registry tag is the short git SHA only).

### Local build

```bash
docker build -t rioc:dev .
docker run --rm -p 8000:8000 --env-file .env rioc:dev
```

The image is multi-stage and CPU-only. Approximate final size: **~1.6–1.8 GB** (most of which is `torch` CPU wheels + opencv + ffmpeg). Do **not** swap to the CUDA torch build — it is unnecessary and triples the image size.

---

## 3. Runtime configuration

All configuration is environment-variable driven. **No config files** are baked into the image — supply env vars via a k8s `ConfigMap` (non-secret) and `Secret` (secret).

### Required env vars

| Var | Source | Notes |
|-----|--------|-------|
| `OPENAI_STT_API_KEY` | **Secret** | OpenAI key for Whisper STT + tts-1 |
| `CLOUD_AI_URL` | ConfigMap | vLLM base URL incl. `/v1` (e.g. `http://vllm.rioc.svc:8100/v1`) |
| `CLOUD_AI_API_KEY` | **Secret** | Token for the vLLM server |
| `ENABLE_CLOUD_AI` | ConfigMap | Set to `1` in prod |

### Common optional env vars

See the full table in [README.md §Configuration](README.md#configuration). Production-relevant ones:

| Var | Recommended value | Why |
|-----|-------------------|-----|
| `FRAME_SOURCE` | `webhook` | Production path — frames arrive via `POST /api/person-detected` from CVR. No local camera needed. |
| `ENABLE_AUDIO_STT` | `1` if mic device is attached | Otherwise leave unset; the app skips mic loop. |
| `ENABLE_SPEAKER_TTS` | `1` | TTS output through IP speaker. |
| `SPEAKER_URL` / `SPEAKER_WS_URL` | ConfigMap | Per-deployment speaker endpoints. |
| `SPEAKER_USER` / `SPEAKER_PASS` | **Secret** | Fanvil basic-auth creds. |
| `MINICPMO_API_KEY` | **Secret** | Token for the MiniCPM-o conversation server. |
| `AUDIT_INTERVAL_SEC` | `2.0` | Lower = more LLM cost. |

### What **must not** ship in the image

- `.env` — excluded by `.dockerignore`. All secrets come from k8s `Secret`s at pod start.
- `ai_guard.db` — runtime state, lives on a PV (see §5).
- `*.pt` YOLO weights — auto-downloaded by ultralytics on first run; cache to a PV if cold-start matters.b

---

## 4. External dependencies

The pod has hard runtime dependencies on these services. Network policies must allow egress to each:

| Dep | Required? | Direction | Notes |
|-----|-----------|-----------|-------|
| vLLM server (MiniCPM-o / MiniCPM-V) | Yes (when `ENABLE_CLOUD_AI=1`) | egress HTTP(S) → `CLOUD_AI_URL` | Self-hosted GPU instance; not in this chart |
| OpenAI API | Yes (when audio is enabled) | egress HTTPS → `api.openai.com` | STT + TTS |
| MiniCPM-o conversation server | When conversations are enabled | egress HTTP(S) → `MINICPMO_URL` | Separate vLLM endpoint on port 8101 |
| IP speaker (Fanvil) | When `ENABLE_SPEAKER_TTS=1` | egress HTTP(S) + WSS → `SPEAKER_URL` | On-prem device on the camera VLAN |
| CVR / camera webhooks | Always (webhook mode) | **ingress** HTTP → pod `:8000` | CVR must be able to reach the rioc Service |
| VideoDB | Optional (`ENABLE_VIDEODB=1`) | egress HTTPS | Off by default |

No Redis, no managed SQL DB, no message queue.

---

## 5. Persistence

The service writes two things to disk:

| Path in container | What | Survives restart? |
|-------------------|------|-------------------|
| `/data/ai_guard.db` (symlinked from `/app/ai_guard.db`) | SQLite — conversation history | Required |
| `/app/audio_logs/` | Captured PCM/WAV from conversations | Nice-to-have |

**Mount a PV at `/data`.** A 5–10 GB PV is enough for years of conversation metadata. Audio logs grow faster; either size for retention or mount a separate PV at `/app/audio_logs` and prune on a CronJob.

Because the DB is local SQLite, **`replicas` must stay at `1`**. To horizontally scale, the database must first be migrated off-box (Postgres) — out of scope for this image.

If using a `StatefulSet`, set `volumeClaimTemplates` for `/data`. If using a `Deployment`, attach a `ReadWriteOnce` PVC and set `strategy.type: Recreate` so the next pod can claim it.

---

## 6. Kubernetes manifest sketch

Replace `<IMAGE>` with the registry path and tag produced by the CI job (`rioc:<git-sha>`).

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rioc
spec:
  replicas: 1
  strategy:
    type: Recreate          # SQLite + RWO PVC — no rolling updates
  selector:
    matchLabels: { app: rioc }
  template:
    metadata:
      labels: { app: rioc }
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
        - name: rioc
          image: <IMAGE>
          ports:
            - { containerPort: 8000, name: http }
          envFrom:
            - configMapRef: { name: rioc-config }
            - secretRef:    { name: rioc-secrets }
          resources:
            requests: { cpu: "500m", memory: "1Gi" }
            limits:   { cpu: "2",    memory: "3Gi" }
          readinessProbe:
            httpGet: { path: /events, port: http }
            initialDelaySeconds: 15
            periodSeconds: 10
          livenessProbe:
            httpGet: { path: /events, port: http }
            initialDelaySeconds: 60
            periodSeconds: 30
            failureThreshold: 3
          volumeMounts:
            - { name: data, mountPath: /data }
            - { name: audio-logs, mountPath: /app/audio_logs }
      volumes:
        - name: data
          persistentVolumeClaim: { claimName: rioc-data }
        - name: audio-logs
          persistentVolumeClaim: { claimName: rioc-audio-logs }
---
apiVersion: v1
kind: Service
metadata: { name: rioc }
spec:
  selector: { app: rioc }
  ports:
    - { name: http, port: 80, targetPort: 8000 }
```

### Resource sizing

- **CPU**: idle ~50m, ~1 vCPU during YOLO inference if local detection is on. In webhook mode, YOLO is bypassed and CPU stays low.
- **Memory**: ~700 MB resident at idle (torch + opencv weights). Bumps to ~1.5 GB during burst inference + frame buffering. Set requests = 1 Gi, limit = 3 Gi.
- **Ephemeral storage**: keep at default unless you mount audio_logs to emptyDir.

### Ingress

CVR webhooks need to reach `POST /api/person-detected`. Either:
- expose via internal Ingress/LB on the camera VLAN, or
- keep traffic inside the cluster if CVR runs in the same cluster.

The HTML status page (`GET /`) and SSE streams (`/detections/stream`, `/conversation/stream`) are useful for operators — consider a separate auth-protected ingress.

---

## 7. Observability

- **Logs**: stdout/stderr only. Standard cluster log shipping is enough. The app prefixes lines with bracketed tags (`[Cloud AI]`, `[YOLO]`, `[Conversation]`) — filter on those in your log UI.
- **Events endpoint**: `GET /events` returns the last 500 in-memory events as JSON. Useful for quick incident triage; not a metrics source.
- **Health**: `GET /events` is the readiness/liveness probe target. There is no `/healthz` — `/events` is fast, side-effect-free, and proves the event loop is running.
- **No Prometheus metrics endpoint yet.** If you need one, that's a code change, not a deploy change.

---

## 8. Image hardening notes (already applied)

- **Multi-stage build** — build toolchain stays out of the runtime image.
- **Non-root user** `rioc` (UID 1000), home `/app`, shell `nologin`.
- **`tini` as PID 1** — clean SIGTERM handling so k8s pod termination is fast.
- **CPU-only torch wheels** — installed from `download.pytorch.org/whl/cpu`, not the default index. Saves ~2 GB.
- **`.dockerignore`** keeps `.env`, `.git`, `*.db`, `*.pt`, virtualenvs, and IDE files out of the build context.
- **`HEALTHCHECK`** baked in for Docker-only deployments; k8s uses its own probes.

---

## 9. Common operational tasks

### Roll a new image

```bash
# CI does this automatically on merge to main.
# Manual override:
docker build -t <registry>/rioc:<sha> .
docker push <registry>/rioc:<sha>
kubectl set image deploy/rioc rioc=<registry>/rioc:<sha>
```

### Tail logs

```bash
kubectl logs -f deploy/rioc | grep -E '\[(Cloud AI|YOLO|Conversation|Speaker)\]'
```

### Inspect the live DB

```bash
kubectl exec -it deploy/rioc -- sqlite3 /data/ai_guard.db \
  'SELECT id, started_at, outcome, turn_count FROM conversations ORDER BY id DESC LIMIT 10;'
```

### Reset state (destructive)

```bash
kubectl exec -it deploy/rioc -- sh -c 'rm -f /data/ai_guard.db && rm -rf /app/audio_logs/*'
kubectl rollout restart deploy/rioc
```

### Verify external connectivity from inside the pod

```bash
kubectl exec -it deploy/rioc -- sh
# inside:
curl -sS -H "Authorization: Bearer $CLOUD_AI_API_KEY" "$CLOUD_AI_URL/models" | head
curl -sS https://api.openai.com/v1/models -H "Authorization: Bearer $OPENAI_STT_API_KEY" | head
```

---

## 10. Known constraints

- **Single replica only.** SQLite + local audio files. To scale horizontally, migrate the DB and move audio to object storage first.
- **No graceful drain of in-flight conversations.** Pod restart aborts any active two-way conversation. Avoid rolling during business hours; use `strategy: Recreate` and short `terminationGracePeriodSeconds` (default 30s is fine).
- **YOLO weights download on first start.** ~6 MB pulled from ultralytics' CDN. If the cluster blocks egress, mount a PV with the weights pre-seeded at `/app/yolov8n.pt` or set `ENABLE_YOLO=0` and rely entirely on cloud AI.
- **Microphone capture (`ENABLE_AUDIO_STT=1`) does not work in standard k8s pods** — there is no audio device. Use the IP speaker's built-in mic via the WebSocket path, or leave STT off.

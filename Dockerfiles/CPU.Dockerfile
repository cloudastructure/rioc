# syntax=docker/dockerfile:1.7

# ---------- Stage 1: builder ----------
# Build wheels in an isolated stage so the final image stays small and clean.
FROM python:3.12-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

# Build deps for native wheels (webrtcvad, sounddevice CFFI, opencv headers, etc.).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        python3-dev \
        portaudio19-dev \
        libsndfile1 \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps into an isolated virtualenv we will copy across.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /build

# Install CPU-only torch first so ultralytics doesn't pull the multi-GB CUDA build.
# This service offloads vision inference to a remote vLLM server; the only local
# model is YOLOv8-nano, which runs fine on CPU.
RUN pip install --upgrade pip \
 && pip install --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.2,<2.6" "torchvision>=0.17,<0.21"

COPY requirements.txt ./
RUN pip install -r requirements.txt


# ---------- Stage 2: runtime ----------
FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    APP_HOME=/app \
    DATA_DIR=/data \
    PORT=8000

# Runtime-only system deps:
#   ffmpeg          — TTS/audio format conversion (required by main.py)
#   libgl1, libglib — OpenCV runtime
#   libsndfile1     — soundfile / sounddevice runtime
#   libportaudio2   — sounddevice runtime (mic capture)
#   curl            — HEALTHCHECK
# tini handles PID 1 signal forwarding so SIGTERM cleanly stops uvicorn.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libsndfile1 \
        libportaudio2 \
        tini \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Non-root user with a fixed UID so file permissions on mounted PVs are predictable.
RUN groupadd --system --gid 1000 rioc \
 && useradd  --system --uid 1000 --gid rioc --home-dir ${APP_HOME} --shell /sbin/nologin rioc

COPY --from=builder /opt/venv /opt/venv

WORKDIR ${APP_HOME}

# Copy only application source — config/secrets are injected at runtime.
COPY --chown=rioc:rioc *.py ./
COPY --chown=rioc:rioc scripts/ ./scripts/
COPY --chown=rioc:rioc mediamtx.yml ./

# Persistent state lives outside the image so it survives restarts and rebuilds.
# Mount a PV at /data in k8s; the app's SQLite DB and audio logs are symlinked in.
RUN mkdir -p ${DATA_DIR} ${APP_HOME}/audio_logs \
 && ln -sf ${DATA_DIR}/ai_guard.db ${APP_HOME}/ai_guard.db \
 && chown -R rioc:rioc ${DATA_DIR} ${APP_HOME}

USER rioc

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/events" > /dev/null || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
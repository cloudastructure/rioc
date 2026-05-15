# syntax=docker/dockerfile:1.7

# Security-hardened CPU build
FROM python:3.12-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    UMASK=0022

# Minimal build deps with pinned versions
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential=12.9 \
        gcc=4:12.2 \
        python3-dev=3.12.0 \
        portaudio19-dev=19.7.0 \
        libsndfile1=1.0.31 \
        libgl1=1.7.0 \
        libglib2.0-0=2.76.2 \
    && rm -rf /var/lib/apt/lists/*

# Isolated virtualenv with secure permissions
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /build

# CPU-only PyTorch with pinned versions
RUN pip install --upgrade pip \
 && pip install \
        "torch==2.4.1+cpu" \
        "torchvision==0.19.1+cpu" \
    && pip install -r requirements.txt

# ---------- Stage 2: runtime ----------
FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    APP_HOME=/app \
    DATA_DIR=/data \
    PORT=8000 \
    UMASK=0022

# Minimal runtime deps with pinned versions
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg=7:7.0.1 \
        libgl1=1.7.0 \
        libglib2.0-0=2.76.2 \
        libsndfile1=1.0.31 \
        libportaudio2=19.7.0 \
        tini=0.19.0 \
        curl=8.7.1 \
        ca-certificates=20231015 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user with minimal permissions
RUN groupadd --system --gid 1000 rioc \
 && useradd --system --uid 1000 --gid rioc --home-dir ${APP_HOME} --shell /sbin/nologin rioc

# Secure file permissions
COPY --from=builder /opt/venv /opt/venv
COPY --chown=rioc:rioc *.py .
COPY --chown=rioc:rioc scripts/ ./scripts/
COPY --chown=rioc:rioc mediamtx.yml .

WORKDIR ${APP_HOME}

# Secure data directory setup
RUN mkdir -p ${DATA_DIR} ${APP_HOME}/audio_logs \
 && chown -R rioc:rioc ${DATA_DIR} ${APP_HOME} \
 && ln -sf ${DATA_DIR}/ai_guard.db ${APP_HOME}/ai_guard.db

# Security headers and restrictions
LABEL org.label-schema.vcs-url="https://github.com/your-org/rioc" \
      org.label-schema.version="1.0.0" \
      org.label-schema.license="MIT" \
      org.label-schema.build-date="2026-05-14"

USER rioc

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/events" > /dev/null || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT} --limit-concurrency 100"]
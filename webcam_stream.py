"""
Webcam stream over HTTP: 640x640, 70% JPEG, MJPEG at GET /video.
Optional local Visual Audit: set ENABLE_LOCAL_AUDIT=1 to run Ollama (MiniCPM-V 2.6) on frames.
"""
import asyncio
import base64
import logging
import os
from contextlib import asynccontextmanager

import cv2
import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response, StreamingResponse

logger = logging.getLogger(__name__)

# Optional local Visual Audit (Ollama)
ENABLE_LOCAL_AUDIT = os.environ.get("ENABLE_LOCAL_AUDIT", "").strip().lower() in ("1", "true", "yes")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_VISION_MODEL = os.environ.get("OLLAMA_VISION_MODEL", "openbmb/minicpm-v2.6")
AUDIT_INTERVAL_SEC = float(os.environ.get("AUDIT_INTERVAL_SEC", "5.0"))

AUDIT_SYSTEM_PROMPT = (
    "You are a tactical analyst. Respond in a cold, authoritative tone. Be concise and direct."
)
AUDIT_USER_PROMPT = (
    "Identify the primary person in this frame. What are they wearing? What are they doing? "
    "Give me a 1-sentence tactical warning addressing their specific outfit."
)

# Global camera; set in lifespan, released on shutdown.
cap: cv2.VideoCapture | None = None

FRAME_SIZE = (640, 640)
JPEG_QUALITY = 70
CONSECUTIVE_READ_FAILURES_MAX = 30


def get_next_frame() -> bytes | None:
    """Read one frame from the global camera; resize and encode as JPEG. Returns None on failure."""
    global cap
    if cap is None or not cap.isOpened():
        return None
    ret, frame = cap.read()
    if not ret or frame is None:
        return None
    frame = cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_LINEAR)
    _, jpeg = cv2.imencode(
        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    )
    return jpeg.tobytes()


async def video_stream_generator():
    """Async generator that yields MJPEG chunks (run frame capture in thread pool)."""
    consecutive_failures = 0
    while True:
        jpeg_bytes = await asyncio.to_thread(get_next_frame)
        if jpeg_bytes is None:
            consecutive_failures += 1
            if consecutive_failures >= CONSECUTIVE_READ_FAILURES_MAX:
                logger.warning("Too many consecutive read failures; ending stream.")
                break
            await asyncio.sleep(0.05)
            continue
        consecutive_failures = 0
        header = (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: %d\r\n\r\n" % len(jpeg_bytes)
        )
        yield header + jpeg_bytes


async def local_audit_loop() -> None:
    """Background task: every AUDIT_INTERVAL_SEC, grab a frame and send to Ollama for Visual Audit."""
    while True:
        await asyncio.sleep(AUDIT_INTERVAL_SEC)
        jpeg_bytes = await asyncio.to_thread(get_next_frame)
        if not jpeg_bytes:
            continue
        b64 = base64.standard_b64encode(jpeg_bytes).decode("ascii")
        full_prompt = AUDIT_SYSTEM_PROMPT + "\n\n" + AUDIT_USER_PROMPT
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{OLLAMA_URL.rstrip('/')}/api/chat",
                    json={
                        "model": OLLAMA_VISION_MODEL,
                        "messages": [
                            {"role": "user", "content": full_prompt, "images": [b64]},
                        ],
                        "stream": False,
                    },
                    timeout=60.0,
                )
            resp.raise_for_status()
            data = resp.json()
            msg = (data.get("message") or {}).get("content") or ""
            if msg.strip():
                print(f"[Visual Audit] {msg.strip()}")
        except Exception as e:
            logger.debug("Local audit request failed: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open camera on startup, release on shutdown. Optionally start local audit task."""
    global cap
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Failed to open camera (index 0)")
        cap = None
    audit_task = None
    if ENABLE_LOCAL_AUDIT and cap is not None:
        audit_task = asyncio.create_task(local_audit_loop())
        logger.info("Local Visual Audit enabled (model=%s, interval=%.1fs)", OLLAMA_VISION_MODEL, AUDIT_INTERVAL_SEC)
    try:
        yield
    finally:
        if audit_task is not None:
            audit_task.cancel()
            try:
                await audit_task
            except asyncio.CancelledError:
                pass
        if cap is not None:
            cap.release()
            cap = None
            logger.info("Camera released.")


app = FastAPI(title="Webcam Stream", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def root():
    """Simple page that embeds the MJPEG stream for testing."""
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Webcam Stream</title></head>
    <body>
    <h1>Webcam Stream (640x640, 70%% JPEG)</h1>
    <img src="/video" alt="Live stream" />
    </body>
    </html>
    """


@app.get("/video")
async def video():
    """Stream webcam as MJPEG (multipart/x-mixed-replace). 640x640, 70%% JPEG quality."""
    if cap is None or not cap.isOpened():
        return Response(
            content="Camera unavailable",
            status_code=503,
            media_type="text/plain",
        )
    return StreamingResponse(
        video_stream_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )

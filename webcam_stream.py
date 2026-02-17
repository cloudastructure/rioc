"""
Webcam stream over HTTP: 640x640, 70% JPEG, MJPEG at GET /video.
Optional local Visual Audit: set ENABLE_LOCAL_AUDIT=1 to run Ollama (MiniCPM-V 2.6) on frames.
Optional listening: set ENABLE_AUDIO_STT=1 to capture mic audio and transcribe via OpenAI STT.
"""
import asyncio
import base64
import io
import logging
import os
import wave
from contextlib import asynccontextmanager

import cv2
import httpx
import sounddevice as sd
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

# Optional audio transcription (cloud STT)
ENABLE_AUDIO_STT = os.environ.get("ENABLE_AUDIO_STT", "").strip().lower() in ("1", "true", "yes")
OPENAI_STT_API_KEY = os.environ.get("OPENAI_STT_API_KEY") or os.environ.get("OPENAI_API_KEY")
OPENAI_STT_MODEL = os.environ.get("OPENAI_STT_MODEL", "whisper-1")
STT_SAMPLE_RATE = int(os.environ.get("STT_SAMPLE_RATE", "16000"))
STT_DURATION_SEC = float(os.environ.get("STT_DURATION_SEC", "5.0"))
STT_GAP_SEC = float(os.environ.get("STT_GAP_SEC", "0.0"))

latest_transcript: str = ""

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


def _record_audio_chunk() -> bytes | None:
    """Record STT_DURATION_SEC of mono audio from the default mic, return WAV bytes."""
    try:
        n_samples = int(STT_SAMPLE_RATE * STT_DURATION_SEC)
        audio = sd.rec(n_samples, samplerate=STT_SAMPLE_RATE, channels=1, dtype="int16")
        sd.wait()
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # int16
            wf.setframerate(STT_SAMPLE_RATE)
            wf.writeframes(audio.tobytes())
        buf.seek(0)
        return buf.read()
    except Exception as exc:
        logger.warning("Audio capture failed: %s", exc)
        return None


async def audio_transcription_loop() -> None:
    """Background task: capture audio, send to OpenAI STT, update latest_transcript."""
    global latest_transcript
    if not OPENAI_STT_API_KEY:
        logger.warning(
            "ENABLE_AUDIO_STT is set but no OPENAI_STT_API_KEY/OPENAI_API_KEY provided; audio STT disabled."
        )
        return
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {OPENAI_STT_API_KEY}"}
    while True:
        wav_bytes = await asyncio.to_thread(_record_audio_chunk)
        if not wav_bytes:
            await asyncio.sleep(1.0)
            continue
        files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
        data = {"model": OPENAI_STT_MODEL}
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(url, headers=headers, files=files, data=data, timeout=90.0)
            resp.raise_for_status()
            text = (resp.json().get("text") or "").strip()
            if text:
                latest_transcript = text
                logger.info("Heard: %s", text)
                print(f"[Rioc] Heard: {text}")
        except Exception as exc:
            logger.warning("STT request failed: %s", exc)
        if STT_GAP_SEC > 0:
            await asyncio.sleep(STT_GAP_SEC)


async def local_audit_loop() -> None:
    """Background task: every AUDIT_INTERVAL_SEC, grab a frame and send to Ollama for Visual Audit."""
    while True:
        await asyncio.sleep(AUDIT_INTERVAL_SEC)
        jpeg_bytes = await asyncio.to_thread(get_next_frame)
        if not jpeg_bytes:
            continue
        b64 = base64.standard_b64encode(jpeg_bytes).decode("ascii")
        # Incorporate latest transcript into the prompt if available.
        context = AUDIT_USER_PROMPT
        if latest_transcript:
            context += (
                f'\n\nThe person is saying: "{latest_transcript}". '
                "Factor this audio into your tactical warning."
            )
        full_prompt = AUDIT_SYSTEM_PROMPT + "\n\n" + context
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
    """Open camera on startup, release on shutdown. Optionally start local audit + audio STT tasks."""
    global cap
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Failed to open camera (index 0)")
        cap = None
    audit_task = None
    audio_task = None
    if ENABLE_LOCAL_AUDIT and cap is not None:
        audit_task = asyncio.create_task(local_audit_loop())
        logger.info("Local Visual Audit enabled (model=%s, interval=%.1fs)", OLLAMA_VISION_MODEL, AUDIT_INTERVAL_SEC)
    if ENABLE_AUDIO_STT:
        audio_task = asyncio.create_task(audio_transcription_loop())
        logger.info(
            "Audio STT enabled (model=%s, duration=%.1fs)", OPENAI_STT_MODEL, STT_DURATION_SEC
        )
        if OPENAI_STT_API_KEY:
            print("[Rioc] Listening. Speak to the camera; you'll see '[Rioc] Heard: ...' when your speech is transcribed.")
        else:
            print("[Rioc] Audio STT is ON but no OPENAI_STT_API_KEY/OPENAI_API_KEY set — listening disabled. Set the key and restart.")
    try:
        yield
    finally:
        if audit_task is not None:
            audit_task.cancel()
            try:
                await audit_task
            except asyncio.CancelledError:
                pass
        if audio_task is not None:
            audio_task.cancel()
            try:
                await audio_task
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


@app.get("/transcript")
async def transcript():
    """Return the latest audio transcript (if any)."""
    return {"text": latest_transcript}

"""
Webcam stream over HTTP: 640x640, 70% JPEG, MJPEG at GET /video.
Optional local Visual Audit: set ENABLE_LOCAL_AUDIT=1 to run Ollama (MiniCPM-V 2.6) on frames.
Optional listening: set ENABLE_AUDIO_STT=1 to capture mic audio and transcribe via OpenAI STT.
"""
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project dir and from cwd (in case uvicorn is run from elsewhere)
_project_dir = Path(__file__).resolve().parent
load_dotenv(_project_dir / ".env")
load_dotenv(Path.cwd() / ".env")

import asyncio
import base64
import io
import logging
import os
import subprocess
import tempfile
import wave
from contextlib import asynccontextmanager
from urllib.parse import urlparse

import datetime
import json
import re
import threading
import time
from collections import deque
import numpy as np
import cv2
import httpx
import sounddevice as sd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Optional local Visual Audit (Ollama)
ENABLE_LOCAL_AUDIT = os.environ.get("ENABLE_LOCAL_AUDIT", "").strip().lower() in ("1", "true", "yes")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_VISION_MODEL = os.environ.get("OLLAMA_VISION_MODEL", "openbmb/minicpm-v2.6")
AUDIT_INTERVAL_SEC = float(os.environ.get("AUDIT_INTERVAL_SEC", "2.0"))
# Require this many consecutive non-CLEAR results before firing TTS/log (reduces false positives)
AUDIT_CONFIRM_FRAMES = int(os.environ.get("AUDIT_CONFIRM_FRAMES", "2"))
# Size to resize frames to before sending to cloud AI (smaller = faster inference, default 320x320)
AUDIT_AI_FRAME_SIZE = int(os.environ.get("AUDIT_AI_FRAME_SIZE", "320"))

# YOLO person detection pre-filter: gates VLM calls — only sends to LLM when YOLO sees a person.
# Dramatically reduces cloud costs and latency. Set ENABLE_YOLO=0 to disable.
ENABLE_YOLO = os.environ.get("ENABLE_YOLO", "1").strip().lower() in ("1", "true", "yes")
YOLO_MODEL = os.environ.get("YOLO_MODEL", "yolov8n.pt")  # nano model: fast, ~6MB
YOLO_CONFIDENCE = float(os.environ.get("YOLO_CONFIDENCE", "0.45"))
# Frame source: "local_yolo" (webcam + local YOLO) or "live_ffmpeg" (subscribe to CVR detections).
# local_yolo is the default for dev/demo. live_ffmpeg is the production path once the AGX interface is confirmed.
FRAME_SOURCE = os.environ.get("FRAME_SOURCE", "local_yolo").strip().lower()

# Optional cloud AI visual analysis (OpenAI-compatible vLLM server)
ENABLE_CLOUD_AI = os.environ.get("ENABLE_CLOUD_AI", "").strip().lower() in ("1", "true", "yes")
CLOUD_AI_URL = (os.environ.get("CLOUD_AI_URL") or "").rstrip("/")
CLOUD_AI_API_KEY = os.environ.get("CLOUD_AI_API_KEY", "token-minicpm-v45")
CLOUD_AI_MODEL = os.environ.get("CLOUD_AI_MODEL", "openbmb/MiniCPM-V-4_5")

AUDIT_SYSTEM_PROMPT = (
    "You are the Rioc Sentinel, an automated audio-broadcast security system. Do not use labels like 'Visual Data' or 'Warning'. Do not use headers or bullet points. Speak only the final warning text as it should be heard over a loudspeaker. Be cold, concise, and observational. Do not identify as an AI."
)
AUDIT_USER_PROMPT = (
    """Examine this image.

No clearly visible person → respond with the single word: CLEAR
If unsure whether a shape is a person → respond CLEAR

Person clearly visible → respond with a one-sentence spoken warning addressed directly to them. Mention their clothing. Cold, authoritative tone. Speak as if over a loudspeaker. Example: "You in the grey hoodie — this area is restricted. Leave immediately."

Do not describe. Do not use passive detection language. Speak to the person."""
)

# Optional audio transcription (cloud STT)
ENABLE_AUDIO_STT = os.environ.get("ENABLE_AUDIO_STT", "").strip().lower() in ("1", "true", "yes")
OPENAI_STT_API_KEY = os.environ.get("OPENAI_STT_API_KEY") or os.environ.get("OPENAI_API_KEY")
OPENAI_STT_MODEL = os.environ.get("OPENAI_STT_MODEL", "whisper-1")
STT_SAMPLE_RATE = int(os.environ.get("STT_SAMPLE_RATE", "16000"))
STT_DURATION_SEC = float(os.environ.get("STT_DURATION_SEC", "5.0"))
STT_GAP_SEC = float(os.environ.get("STT_GAP_SEC", "0.0"))
STT_SILENCE_THRESHOLD = float(os.environ.get("STT_SILENCE_THRESHOLD", "300"))  # RMS below this = skip STT
# Audio input device: index (e.g. 1) or name substring (e.g. Fanvil, LINKVIL, CS20). Use external speaker mic when set.
AUDIO_INPUT_DEVICE = (os.environ.get("AUDIO_INPUT_DEVICE") or "").strip() or None

# Optional VideoDB eyes and ears (real-time transcript + visual/audio indexing)
# Docs: https://docs.videodb.io/pages/getting-started/quickstart
ENABLE_VIDEODB = os.environ.get("ENABLE_VIDEODB", "").strip().lower() in ("1", "true", "yes")
VIDEODB_BATCH_SEC = int(os.environ.get("VIDEODB_BATCH_SEC", "5"))

# Optional TTS + speaker output (Rioc speaks through IP speaker)
ENABLE_SPEAKER_TTS = os.environ.get("ENABLE_SPEAKER_TTS", "").strip().lower() in ("1", "true", "yes")
SPEAKER_URL = (os.environ.get("SPEAKER_URL") or "").rstrip("/")
SPEAKER_WS_URL = os.environ.get("SPEAKER_WS_URL", "").strip()
if not SPEAKER_WS_URL and SPEAKER_URL:
    # Derive from SPEAKER_URL: https://192.168.10.183 -> wss://192.168.10.183:8000/webtwowayaudio
    p = urlparse(SPEAKER_URL)
    host = p.hostname or p.netloc.split(":")[0]
    SPEAKER_WS_URL = f"wss://{host}:8000/webtwowayaudio"
SPEAKER_USER = os.environ.get("SPEAKER_USER", "")
SPEAKER_PASS = os.environ.get("SPEAKER_PASS", "")
SPEAKER_PLAY_PATH = os.environ.get("SPEAKER_PLAY_PATH", "/play")
# Paths to try in order (first from env, then these fallbacks)
SPEAKER_PLAY_FALLBACK_PATHS = [p.strip() for p in (os.environ.get("SPEAKER_PLAY_FALLBACK_PATHS") or "/api/play,/tts,/speak,/api/tts").split(",") if p.strip()]
# Set SPEAKER_TYPE=axis to skip non-AXIS fallback paths (auto-detected from SPEAKER_URL if not set)
SPEAKER_TYPE = os.environ.get("SPEAKER_TYPE", "").strip().lower()
# Minimum seconds between TTS announcements — prevents rapid-fire warnings interrupting each other
TTS_COOLDOWN_SEC = float(os.environ.get("TTS_COOLDOWN_SEC", "20.0"))
OPENAI_TTS_MODEL = os.environ.get("OPENAI_TTS_MODEL", "tts-1")
OPENAI_TTS_VOICE = os.environ.get("OPENAI_TTS_VOICE", "onyx")  # Deep, authoritative
# Fallback: play through Mac speakers when speaker fails or for testing
ENABLE_LOCAL_PLAYBACK = os.environ.get("ENABLE_LOCAL_PLAYBACK", "").strip().lower() in ("1", "true", "yes")
# WebSocket: G.711 μ-law at 8kHz (per talk.js)
SPEAKER_WS_SAMPLE_RATE = int(os.environ.get("SPEAKER_WS_SAMPLE_RATE", "8000"))

latest_transcript: str = ""
latest_tts_audio: bytes = b""  # Served at /tts/latest.mp3 for play-from-URL mode
latest_analysis: str = ""  # Latest Cloud AI vision output (non-CLEAR)

# TTS rate-limiting: one announcement at a time with a cooldown between them.
# _tts_active is set True before the first await, so asyncio's single-threaded model
# makes this check+set atomic — no two coroutines can both see it False simultaneously.
_tts_active: bool = False
_last_tts_time: float = 0.0

# Detection history: list of {timestamp, type, text} newest-first
detections: deque = deque(maxlen=100)

# SSE subscribers: one asyncio.Queue per connected /detections/stream client
_sse_subscribers: list[asyncio.Queue] = []

# Frames that passed person detection, waiting for vLLM — size 1 ensures freshest frame is always used
_detection_frame_queue: asyncio.Queue = asyncio.Queue(maxsize=1)


def _add_detection(detection_type: str, text: str) -> None:
    d = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "type": detection_type,
        "text": text,
    }
    detections.appendleft(d)
    # Instantly push to all connected SSE clients (called from async context — put_nowait is safe)
    for q in list(_sse_subscribers):
        try:
            q.put_nowait(d)
        except asyncio.QueueFull:
            pass  # Slow client — drop rather than block

# Base URL for TTS (speaker must reach this). Set to your Mac's IP, e.g. http://192.168.10.50:8000
TTS_PUBLIC_URL = (os.environ.get("TTS_PUBLIC_URL") or "").rstrip("/")
# Optional: test with a public MP3 URL to verify speaker can play streams (e.g. https://...)
SPEAKER_TEST_URL = (os.environ.get("SPEAKER_TEST_URL") or "").strip() or None

# Whisper hallucination phrases to ignore (case-insensitive substring match)
# Comprehensive list of common Whisper video-outro / non-speech hallucinations
STT_HALLUCINATION_PHRASES = (
    # English - video outros
    "thank you for watching",
    "thanks for watching",
    "thank you so much",
    "thank you very much",
    "for watching",
    "subscribe",
    "like and subscribe",
    "don't forget to subscribe",
    "please subscribe",
    "hit the bell",
    "leave a comment",
    "let me know in the comments",
    "see you in the next",
    "see you next time",
    "until next time",
    "bye bye",
    "peace out",
    "take care",
    "goodbye",
    "subscribe to my channel",
    "youtube",
    "end of",
    "copyright",
    "all rights reserved",
    "music by",
    "sound effects",
    "translated by",
    "subtitles by",
    "subtitles",
    # Japanese
    "ありがとう",
    "視聴",
    "最後まで",
    "ございます",
    # Spanish/Portuguese
    "gracias por ver",
    "suscríbete",
    "inscreva-se",
    # Other common hallucinations
    "the end",
    "fin",
    "ciao",
    "adios",
    "bye",
    # Whisper echoing our own prompt back
    "transcribe only",
    "direct speech",
)

# Prompt to prime Whisper away from video-outro hallucinations
STT_PROMPT = "A person is speaking to a security guard. Transcribe only their direct speech."


def _strip_think_tags(text: str) -> str:
    """Extract only the final response after <think>...</think> reasoning blocks."""
    if "</think>" in text:
        # Take everything after the last closing tag
        return text.split("</think>")[-1].strip()
    if "<think>" in text:
        # Unclosed tag — model is still reasoning, no final output yet
        return ""
    return text.strip()


def _is_stt_hallucination(text: str) -> bool:
    """Return True if transcript matches known Whisper hallucination phrases."""
    t = text.lower().strip()
    if not t or len(t) < 8:
        return True  # Very short outputs are often noise
    return any(phrase in t for phrase in STT_HALLUCINATION_PHRASES)


def _is_audio_silent(wav_bytes: bytes) -> bool:
    """Return True if audio is mostly silence (skip STT to avoid hallucinations)."""
    try:
        with io.BytesIO(wav_bytes) as buf:
            with wave.open(buf, "rb") as wf:
                data = wf.readframes(wf.getnframes())
        samples = np.frombuffer(data, dtype=np.int16)
        rms = np.sqrt(np.mean(samples.astype(np.float64) ** 2))
        return rms < STT_SILENCE_THRESHOLD
    except Exception:
        return False


# Global camera; set in lifespan, released on shutdown.
cap: cv2.VideoCapture | None = None
cap_lock = threading.Lock()  # Guards all reads/writes of cap

# YOLO singleton — loaded once on first use.
_yolo_model = None
_yolo_model_lock = threading.Lock()


def _get_yolo_model():
    """Lazy-load the YOLO model (thread-safe singleton)."""
    global _yolo_model
    if _yolo_model is None:
        with _yolo_model_lock:
            if _yolo_model is None:
                from ultralytics import YOLO  # noqa: PLC0415
                print(f"[YOLO] Loading model: {YOLO_MODEL}", flush=True)
                _yolo_model = YOLO(YOLO_MODEL)
    return _yolo_model


def _yolo_person_in_frame(frame: np.ndarray) -> bool:
    """Return True if YOLO detects at least one person (COCO class 0) in frame."""
    results = _get_yolo_model()(frame, classes=[0], conf=YOLO_CONFIDENCE, verbose=False)
    detected = any(len(r.boxes) > 0 for r in results)
    if not detected:
        # Log best confidence seen so we can tune the threshold
        best = max((float(box.conf) for r in results for box in r.boxes), default=0.0)
        if best > 0.05:
            print(f"[YOLO] Best conf={best:.2f} (threshold={YOLO_CONFIDENCE}) — not detected", flush=True)
    return detected

FRAME_SIZE = (640, 640)
JPEG_QUALITY = 70
CONSECUTIVE_READ_FAILURES_MAX = 30


def get_next_frame() -> bytes | None:
    """Read one frame from the global camera; resize and encode as JPEG. Returns None on failure."""
    with cap_lock:
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


def _resolve_audio_input_device() -> int | None:
    """Resolve AUDIO_INPUT_DEVICE to a sounddevice index. Returns None for default device."""
    if not AUDIO_INPUT_DEVICE:
        return None
    try:
        # If it's a numeric string, use as index
        return int(AUDIO_INPUT_DEVICE)
    except ValueError:
        pass
    # Search by name (e.g. Fanvil, LINKVIL, CS20)
    name_lower = AUDIO_INPUT_DEVICE.lower()
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0 and name_lower in (dev.get("name") or "").lower():
            return i
    logger.warning("AUDIO_INPUT_DEVICE=%r: no matching input device found, using default", AUDIO_INPUT_DEVICE)
    return None


def _record_audio_chunk() -> bytes | None:
    """Record STT_DURATION_SEC of mono audio from the configured mic, return WAV bytes."""
    try:
        device = _resolve_audio_input_device()
        n_samples = int(STT_SAMPLE_RATE * STT_DURATION_SEC)
        rec_kw = {"samplerate": STT_SAMPLE_RATE, "channels": 1, "dtype": "int16"}
        if device is not None:
            rec_kw["device"] = device
        audio = sd.rec(n_samples, **rec_kw)
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


def _print_audio(msg: str) -> None:
    """Print with flush so it appears immediately in uvicorn output."""
    print(msg, flush=True)


def _wav_to_mp3(wav_bytes: bytes) -> bytes | None:
    """Convert WAV to MP3 via ffmpeg. Used when MiniCPM-o returns WAV and speaker needs MP3."""
    try:
        proc = subprocess.run(
            ["ffmpeg", "-y", "-i", "pipe:0", "-f", "mp3", "-q:a", "2", "pipe:1"],
            input=wav_bytes,
            capture_output=True,
            timeout=30,
        )
        if proc.returncode == 0 and proc.stdout:
            return proc.stdout
    except FileNotFoundError:
        logger.warning("ffmpeg not found. Install with: brew install ffmpeg")
    except Exception as e:
        logger.warning("WAV to MP3 conversion failed: %s", e)
    return None


def _mp3_to_mulaw(mp3_bytes: bytes, sample_rate: int = 8000) -> bytes | None:
    """Convert MP3 (or WAV) to G.711 μ-law at 8kHz - format talk.js sends to speaker."""
    try:
        proc = subprocess.run(
            [
                "ffmpeg", "-y", "-i", "pipe:0",
                "-f", "mulaw", "-ar", str(sample_rate), "-ac", "1",
                "pipe:1",
            ],
            input=mp3_bytes,
            capture_output=True,
            timeout=30,
        )
        if proc.returncode == 0 and proc.stdout:
            return proc.stdout
    except FileNotFoundError:
        logger.warning("ffmpeg not found. Install with: brew install ffmpeg")
    except Exception as e:
        logger.warning("MP3 to μ-law conversion failed: %s", e)
    return None


def _axis_transmit_sync(url: str, content_type: str, body: bytes, user: str, password: str) -> tuple[int, str]:
    """POST audio to AXIS VAPIX transmit.cgi using Digest auth.
    Runs synchronously in a thread — requests handles the 401+WWW-Authenticate challenge-response."""
    import requests
    from requests.auth import HTTPDigestAuth
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    r = requests.post(
        url,
        data=body,
        headers={"Content-Type": content_type},
        auth=HTTPDigestAuth(user, password) if user else None,
        verify=False,
        timeout=8.0,
    )
    return r.status_code, r.text[:80]


async def _guarded_play(audio_bytes: bytes, audio_format: str = "wav", force: bool = False) -> None:
    """Deliver pre-generated audio to the IP speaker with TTS guard (rate-limit + no overlaps).
    audio_format: "wav" (from MiniCPM-o) or "mp3". WAV is converted to MP3 before delivery."""
    global _tts_active, _last_tts_time, latest_tts_audio
    if not force and (not ENABLE_SPEAKER_TTS or not SPEAKER_URL):
        return
    if not audio_bytes:
        return
    if not force:
        if _tts_active:
            _print_audio("[Rioc] TTS in progress — dropping")
            return
        now = time.monotonic()
        if now - _last_tts_time < TTS_COOLDOWN_SEC:
            remaining = TTS_COOLDOWN_SEC - (now - _last_tts_time)
            _print_audio(f"[Rioc] TTS cooldown — {remaining:.0f}s remaining")
            return
    if _tts_active:
        return  # Prevent overlap even for forced calls
    _tts_active = True  # Set before first await — atomic in asyncio (no context switch possible)
    _last_tts_time = time.monotonic()
    try:
        # Normalize to MP3 — all speaker delivery paths below expect MP3 bytes
        if audio_format == "wav":
            _print_audio("[Rioc] Converting WAV → MP3...")
            mp3_bytes = await asyncio.to_thread(_wav_to_mp3, audio_bytes)
            if not mp3_bytes:
                _print_audio("[Rioc] WAV→MP3 conversion failed")
                return
        else:
            mp3_bytes = audio_bytes
        latest_tts_audio = mp3_bytes
        audio_bytes = mp3_bytes  # alias for the delivery code below
        auth = (SPEAKER_USER, SPEAKER_PASS) if SPEAKER_USER else None
        played = False
        # Try AXIS VAPIX transmit (AXIS C1310-E and similar AXIS network speakers)
        if SPEAKER_URL and not played and SPEAKER_TYPE == "axis":
            # AXIS C1310-E requires audio/basic (G.711 μ-law 8kHz mono)
            mulaw_bytes = _mp3_to_mulaw(audio_bytes, sample_rate=8000)
            for ct, body in [
                ("audio/basic", mulaw_bytes),
                ("audio/mpeg", audio_bytes),  # fallback if basic unsupported
            ]:
                if body is None:
                    continue
                try:
                    _print_audio(f"[Rioc] Trying AXIS transmit ({ct})")
                    status, resp_text = await asyncio.to_thread(
                        _axis_transmit_sync,
                        f"{SPEAKER_URL}/axis-cgi/audio/transmit.cgi",
                        ct,
                        body,
                        SPEAKER_USER,
                        SPEAKER_PASS,
                    )
                    if status < 400:
                        _print_audio(f"[Rioc] Speaker OK (AXIS transmit {ct})")
                        played = True
                        break
                    else:
                        _print_audio(f"[Rioc] AXIS transmit {ct}: {status} {resp_text}")
                except Exception as e:
                    _print_audio(f"[Rioc] AXIS transmit error ({ct}): {type(e).__name__}: {e}")
        # Try WebSocket (Fanvil/LINKVIL speakers use wss://.../webtwowayaudio)
        if SPEAKER_WS_URL and not played and SPEAKER_TYPE not in ("axis", "ipspk"):
            try:
                import ssl
                import websockets
                _print_audio(f"[Rioc] Trying WebSocket: {SPEAKER_WS_URL}")
                ssl_ctx = None
                if SPEAKER_WS_URL.startswith("wss"):
                    ssl_ctx = ssl.create_default_context()
                    ssl_ctx.check_hostname = False
                    ssl_ctx.verify_mode = ssl.CERT_NONE
                async with websockets.connect(
                    SPEAKER_WS_URL,
                    close_timeout=5,
                    open_timeout=10,
                    ping_timeout=None,
                    ssl=ssl_ctx,
                ) as ws:
                    # Speaker expects G.711 μ-law at 8kHz (per talk.js: mulawEncode → send)
                    mulaw_bytes = _mp3_to_mulaw(audio_bytes, sample_rate=SPEAKER_WS_SAMPLE_RATE)
                    if mulaw_bytes:
                        # Send in ~20ms chunks (160 bytes at 8kHz) - matches talk.js buffer flow
                        chunk_size = (SPEAKER_WS_SAMPLE_RATE // 50)  # 160 bytes
                        for i in range(0, len(mulaw_bytes), chunk_size):
                            await ws.send(mulaw_bytes[i : i + chunk_size])
                            await asyncio.sleep(0.02)
                        _print_audio("[Rioc] Speaker OK (WebSocket μ-law)")
                        played = True
                    else:
                        _print_audio("[Rioc] WebSocket: μ-law conversion failed")
            except Exception as e:
                _print_audio(f"[Rioc] WebSocket error: {e}")
        if (TTS_PUBLIC_URL or SPEAKER_TEST_URL) and not played and SPEAKER_TYPE not in ("axis", "ipspk"):
            # Prefer test URL for diagnostics (can the speaker play ANY stream?)
            audio_url = (SPEAKER_TEST_URL or f"{TTS_PUBLIC_URL}/tts/latest.mp3")
            _print_audio(f"[Rioc] Playing stream: {audio_url}")
            try:
                async with httpx.AsyncClient(verify=False) as client:
                    r = await client.get(
                        f"{SPEAKER_URL}/api/play",
                        params={
                            "action": "startstream",
                            "stream": audio_url,
                            "volume": 20,  # 0-100, per dashboard examples
                        },
                        auth=auth,
                        timeout=15.0,
                    )
                if r.status_code < 400:
                    _print_audio("[Rioc] Speaker OK (startstream)")
                    played = True
                else:
                    _print_audio(f"[Rioc] startstream: {r.status_code} {r.text[:80]}")
            except Exception as e:
                _print_audio(f"[Rioc] startstream error: {e}")
        if not played and SPEAKER_TYPE not in ("axis", "ipspk"):
            # Try upload-then-play (dashboard shows file=userfile1 for uploaded files)
            upload_paths = ["/api/upload", "/api/file/upload", "/upload", "/api/media/upload"]
            for up_path in upload_paths:
                try:
                    async with httpx.AsyncClient(verify=False) as client:
                        r = await client.post(
                            f"{SPEAKER_URL}{up_path}",
                            files={"file": ("tts.mp3", audio_bytes, "audio/mpeg")},
                            auth=auth,
                            timeout=30.0,
                        )
                    if r.status_code < 400:
                        _print_audio(f"[Rioc] Upload OK ({up_path}), playing userfile1...")
                        async with httpx.AsyncClient(verify=False) as client:
                            r2 = await client.get(
                                f"{SPEAKER_URL}/api/play",
                                params={"action": "start", "file": "userfile1", "volume": 20},
                                auth=auth,
                                timeout=10.0,
                            )
                        if r2.status_code < 400:
                            _print_audio("[Rioc] Speaker OK (upload+play)")
                            played = True
                            break
                except Exception as e:
                    pass  # Try next path
        # Generic IP speaker (Fanvil/LINKVIL and similar): upload to /cgi-bin/mediaupload, play via /api/spkplay
        if not played and SPEAKER_TYPE == "ipspk":
            try:
                _print_audio("[Rioc] Trying IP speaker upload (/cgi-bin/mediaupload + /api/spkplay)")
                async with httpx.AsyncClient(verify=False) as client:
                    r = await client.post(
                        f"{SPEAKER_URL}/cgi-bin/mediaupload?idx=1",
                        files={"upload1": ("tts.mp3", audio_bytes, "audio/mpeg")},
                        auth=auth,
                        timeout=30.0,
                    )
                if r.status_code < 400:
                    async with httpx.AsyncClient(verify=False) as client:
                        r2 = await client.get(
                            f"{SPEAKER_URL}/api/spkplay",
                            params={"action": "start", "fileid": 1},
                            auth=auth,
                            timeout=10.0,
                        )
                    if r2.status_code < 400:
                        _print_audio("[Rioc] Speaker OK (ipspk upload+spkplay)")
                        played = True
                    else:
                        _print_audio(f"[Rioc] ipspk spkplay: {r2.status_code} {r2.text[:80]}")
                else:
                    _print_audio(f"[Rioc] ipspk upload failed: {r.status_code} {r.text[:80]}")
            except Exception as e:
                _print_audio(f"[Rioc] ipspk error: {e}")

        if not played and SPEAKER_TYPE != "axis":
            paths_to_try = [SPEAKER_PLAY_PATH] + [p for p in SPEAKER_PLAY_FALLBACK_PATHS if p != SPEAKER_PLAY_PATH]
            last_err = ""
            for path in paths_to_try:
                play_url = f"{SPEAKER_URL}{path}"
                _print_audio(f"[Rioc] Trying speaker at {play_url}...")
                try:
                    async with httpx.AsyncClient(verify=False) as client:
                        r = await client.post(
                            play_url,
                            content=audio_bytes,
                            headers={"Content-Type": "audio/mpeg"},
                            auth=auth,
                            timeout=30.0,
                        )
                    if r.status_code < 400:
                        _print_audio("[Rioc] Speaker OK")
                        played = True
                        break
                    last_err = f"{r.status_code} {r.text[:100]}"
                except Exception as e:
                    last_err = str(e)
            if not played:
                _print_audio(f"[Rioc] Speaker failed (all paths): {last_err}")
        if not played and ENABLE_LOCAL_PLAYBACK:
            _play_audio_locally(audio_bytes)
    except Exception as e:
        _print_audio(f"[Rioc] Speaker error: {e}")
    finally:
        _tts_active = False


async def _speak_through_speaker(text: str, force: bool = False) -> None:
    """Generate speech for arbitrary text and play on speaker.
    Used by /tts/test, local_audit_loop, VideoDB, and as fallback when model returns no audio.

    Current backend: OpenAI TTS (model audio output not yet enabled on the vLLM server).
    TODO: switch to _speak_via_model_audio() below once vLLM is started with --enable-audio-output."""
    if not text or len(text) > 500:
        return
    if not OPENAI_STT_API_KEY:
        _print_audio("[Rioc] TTS skipped: OPENAI_STT_API_KEY not set")
        return
    try:
        def _tts():
            client = OpenAI(api_key=OPENAI_STT_API_KEY)
            resp = client.audio.speech.create(
                model=OPENAI_TTS_MODEL,
                voice=OPENAI_TTS_VOICE,
                input=text[:500],
            )
            return resp.content
        mp3_bytes = await asyncio.to_thread(_tts)
        if mp3_bytes:
            await _guarded_play(mp3_bytes, "mp3", force)
    except Exception as e:
        _print_audio(f"[Rioc] TTS generation failed: {e}")


def _play_audio_locally(audio_bytes: bytes) -> None:
    """Play MP3 through Mac speakers via afplay (fallback when IP speaker fails)."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(audio_bytes)
            path = f.name
        os.system(f'afplay "{path}" 2>/dev/null')
        os.unlink(path)
    except Exception as e:
        logger.debug("Local playback failed: %s", e)


async def audio_transcription_loop() -> None:
    """Background task: record mic audio, send to MiniCPM-o for STT + spoken response.
    MiniCPM-o handles both transcription and response generation in a single call."""
    global latest_transcript
    if not ENABLE_CLOUD_AI or not CLOUD_AI_URL:
        _print_audio("[Rioc] Audio loop disabled: ENABLE_CLOUD_AI or CLOUD_AI_URL not set")
        return
    dev_info = ""
    if AUDIO_INPUT_DEVICE:
        idx = _resolve_audio_input_device()
        if idx is not None:
            dev = sd.query_devices(idx)
            dev_info = f" (input: {dev.get('name', '?')})"
    _print_audio(f"[Rioc] Audio loop started{dev_info}. Recording {STT_DURATION_SEC}s chunks...")
    while True:
        _print_audio("[Rioc] Recording chunk...")
        wav_bytes = await asyncio.to_thread(_record_audio_chunk)
        if not wav_bytes:
            _print_audio("[Rioc] Audio capture failed — check mic permissions")
            await asyncio.sleep(1.0)
            continue
        if _is_audio_silent(wav_bytes):
            continue  # Skip when silent
        _print_audio("[Rioc] Sending audio to MiniCPM-o...")
        wav_b64 = base64.standard_b64encode(wav_bytes).decode("ascii")
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{CLOUD_AI_URL}/v1/chat/completions",
                    headers={"Authorization": f"Bearer {CLOUD_AI_API_KEY}"},
                    json={
                        "model": CLOUD_AI_MODEL,
                        "modalities": ["text", "audio"],
                        "audio": {"voice": "default", "format": "wav"},
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "You are Rioc, a cold and authoritative security guard. "
                                    "The person in front of you is speaking. Respond to them directly "
                                    "in one sentence. Do not transcribe what they said."
                                ),
                            },
                            {
                                "role": "user",
                                "content": [
                                    {"type": "input_audio", "input_audio": {"data": wav_b64, "format": "wav"}},
                                ],
                            },
                        ],
                        "max_tokens": 200,
                    },
                    timeout=60.0,
                )
            resp.raise_for_status()
            choice_msg = ((resp.json().get("choices") or [{}])[0].get("message") or {})
            audio_data = choice_msg.get("audio") or {}
            # Transcript: what the model heard / its text reply
            transcript = choice_msg.get("content") or audio_data.get("transcript") or ""
            if transcript:
                latest_transcript = transcript
                _add_detection("audio", transcript)
                _print_audio(f"[Rioc] {transcript}")
            # Play the model's spoken response
            response_wav_b64 = audio_data.get("data") or ""
            if response_wav_b64:
                asyncio.create_task(_guarded_play(base64.b64decode(response_wav_b64), "wav"))
        except Exception as exc:
            _print_audio(f"[Rioc] Audio processing failed: {exc}")
        if STT_GAP_SEC > 0:
            await asyncio.sleep(STT_GAP_SEC)


async def local_audit_loop() -> None:
    """Background task: every AUDIT_INTERVAL_SEC, grab a frame and send to Ollama for Visual Audit."""
    while True:
        await asyncio.sleep(AUDIT_INTERVAL_SEC)
        frame = await asyncio.to_thread(_get_raw_ai_frame)
        if frame is None:
            continue
        if ENABLE_YOLO:
            person_detected = await asyncio.to_thread(_yolo_person_in_frame, frame)
            if not person_detected:
                print("[Local Audit] YOLO: no person — skipping VLM", flush=True)
                continue
        _, jpeg_enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        jpeg_bytes = jpeg_enc.tobytes()
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
                asyncio.create_task(_speak_through_speaker(msg.strip()))
            else:
                print("[Visual Audit] (model returned empty — ensure Ollama has openbmb/minicpm-v2.6 loaded)")
        except Exception as e:
            print(f"[Visual Audit] Request failed: {e}")


def _get_raw_ai_frame() -> np.ndarray | None:
    """Grab one frame and resize to AUDIT_AI_FRAME_SIZE. Returns numpy array (for YOLO or encoding)."""
    with cap_lock:
        if cap is None or not cap.isOpened():
            return None
        ret, frame = cap.read()
    if not ret or frame is None:
        return None
    return cv2.resize(frame, (AUDIT_AI_FRAME_SIZE, AUDIT_AI_FRAME_SIZE), interpolation=cv2.INTER_LINEAR)


def _get_ai_frame() -> bytes | None:
    """Grab one frame and resize to AUDIT_AI_FRAME_SIZE for cloud AI (smaller = faster inference)."""
    frame = _get_raw_ai_frame()
    if frame is None:
        return None
    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return jpeg.tobytes()


YOLO_SCAN_INTERVAL_SEC = float(os.environ.get("YOLO_SCAN_INTERVAL_SEC", "1.0"))


async def local_yolo_source_loop() -> None:
    """Continuously grab webcam frames, run YOLO, and enqueue person-detected frames for vLLM.
    Runs independently of the vLLM caller so YOLO never stalls waiting on network I/O.
    Rate-limited to YOLO_SCAN_INTERVAL_SEC (default 1s) to avoid hammering vLLM."""
    await asyncio.sleep(3.0)  # Camera warmup — discard initial noisy frames
    consecutive_detections = 0
    absent_frames = 0
    while True:
        frame = await asyncio.to_thread(_get_raw_ai_frame)
        if frame is None:
            await asyncio.sleep(0.1)
            continue
        if ENABLE_YOLO:
            person_detected = await asyncio.to_thread(_yolo_person_in_frame, frame)
            if not person_detected:
                print("[YOLO] No person — skipping VLM", flush=True)
                consecutive_detections = 0
                absent_frames += 1
                await asyncio.sleep(YOLO_SCAN_INTERVAL_SEC)
                continue
        # Require 2 consecutive detections before treating as a real person entry
        consecutive_detections += 1
        absent_frames = 0
        if consecutive_detections < 2:
            await asyncio.sleep(0.1)
            continue
        _, jpeg_enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        try:
            # Snapshot transcript at detection time so the vLLM caller has the right context
            _detection_frame_queue.put_nowait((jpeg_enc.tobytes(), latest_transcript))
        except asyncio.QueueFull:
            pass  # vLLM still processing — drop, keep scanning
        # Rate-limit: wait before next YOLO scan so we don't hammer vLLM with back-to-back frames
        await asyncio.sleep(YOLO_SCAN_INTERVAL_SEC)


async def live_ffmpeg_source_loop() -> None:
    """Subscribe to live-ffmpeg person detection events from a CVR/AGX device.
    When a person-detected frame arrives, enqueue it for vLLM processing.
    TODO: implement once the live-ffmpeg external event interface is confirmed (WebSocket/HTTP/socket).
    Falls back to local_yolo_source_loop until the interface is wired up."""
    logger.warning(
        "FRAME_SOURCE=live_ffmpeg is not yet implemented — falling back to local_yolo_source_loop. "
        "Wire up the AGX live-ffmpeg interface here once confirmed."
    )
    await local_yolo_source_loop()


async def cloud_audit_loop() -> None:
    """Drain frames from the detection queue and send to cloud vLLM.
    Fires immediately when the frame source enqueues a person-detected frame — no polling interval."""
    consecutive_detections = 0
    while True:
        jpeg_bytes, transcript = await _detection_frame_queue.get()
        b64 = base64.standard_b64encode(jpeg_bytes).decode("ascii")
        context = AUDIT_USER_PROMPT
        if transcript:
            context += (
                f'\n\nThe person is saying: "{transcript}". '
                "Factor this audio into your tactical warning."
            )
        full_prompt = AUDIT_SYSTEM_PROMPT + "\n\n" + context
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{CLOUD_AI_URL}/v1/chat/completions",
                    headers={"Authorization": f"Bearer {CLOUD_AI_API_KEY}"},
                    json={
                        "model": CLOUD_AI_MODEL,
                        "modalities": ["text", "audio"],
                        "audio": {"voice": "default", "format": "wav"},
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": full_prompt},
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                                ],
                            }
                        ],
                        "max_tokens": 200,
                    },
                    timeout=60.0,
                )
            resp.raise_for_status()
            choice_msg = ((resp.json().get("choices") or [{}])[0].get("message") or {})
            audio_data = choice_msg.get("audio") or {}
            # Text for CLEAR check: prefer content, fall back to audio transcript
            raw_text = choice_msg.get("content") or audio_data.get("transcript") or ""
            msg = _strip_think_tags(raw_text)
            first_line = msg.split('\n')[0].strip().upper()
            if not msg or first_line == "CLEAR":
                consecutive_detections = 0
                print("[Cloud AI] CLEAR — no person detected", flush=True)
            else:
                consecutive_detections += 1
                print(f"[Cloud AI] (frame {consecutive_detections}/{AUDIT_CONFIRM_FRAMES}) {msg}", flush=True)
                if consecutive_detections >= AUDIT_CONFIRM_FRAMES:
                    consecutive_detections = 0
                    global latest_analysis
                    latest_analysis = msg
                    _add_detection("vision", msg)
                    # Use audio bytes from model directly — skip separate TTS generation
                    wav_b64 = audio_data.get("data") or ""
                    if wav_b64:
                        asyncio.create_task(_guarded_play(base64.b64decode(wav_b64), "wav"))
                    else:
                        # No audio in response — fall back to text TTS
                        asyncio.create_task(_speak_through_speaker(msg))
        except Exception as e:
            print(f"[Cloud AI] Request failed: {e}", flush=True)


async def _warmup_cloud_ai() -> None:
    """On startup: check vLLM is running, log model/max-model-len, and send a warmup inference
    so the model is hot before the first real detection (cold start has significant first-token latency)."""
    if not ENABLE_CLOUD_AI or not CLOUD_AI_URL:
        return
    print("[Cloud AI] Checking vLLM server...", flush=True)
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{CLOUD_AI_URL}/v1/models",
                headers={"Authorization": f"Bearer {CLOUD_AI_API_KEY}"},
                timeout=15.0,
            )
        if r.status_code == 200:
            for m in (r.json().get("data") or []):
                max_len = m.get("max_model_len") or "?"
                print(f"[Cloud AI] Model: {m.get('id')}  max_model_len={max_len}", flush=True)
        else:
            print(f"[Cloud AI] /v1/models returned {r.status_code} — server may not be ready", flush=True)
    except Exception as e:
        print(f"[Cloud AI] Could not reach vLLM server: {e}", flush=True)
        return
    # Warmup inference — forces model weights into GPU memory before first detection
    print("[Cloud AI] Sending warmup request...", flush=True)
    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(
                f"{CLOUD_AI_URL}/v1/chat/completions",
                headers={"Authorization": f"Bearer {CLOUD_AI_API_KEY}"},
                json={
                    "model": CLOUD_AI_MODEL,
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 1,
                },
                timeout=60.0,
            )
        if r.status_code < 400:
            print("[Cloud AI] Warmup complete — model is hot.", flush=True)
        else:
            print(f"[Cloud AI] Warmup failed: {r.status_code} {r.text[:80]}", flush=True)
    except Exception as e:
        print(f"[Cloud AI] Warmup request failed: {e}", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open camera on startup, release on shutdown. Optionally start local audit + audio STT + VideoDB tasks."""
    global cap, latest_transcript
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Failed to open camera (index 0)")
        cap = None
    audit_task = None
    cloud_audit_task = None
    frame_source_task = None
    audio_task = None
    videodb_task = None

    if ENABLE_CLOUD_AI and CLOUD_AI_URL:
        asyncio.create_task(_warmup_cloud_ai())
        cloud_audit_task = asyncio.create_task(cloud_audit_loop())
        if FRAME_SOURCE == "live_ffmpeg":
            frame_source_task = asyncio.create_task(live_ffmpeg_source_loop())
        else:
            frame_source_task = asyncio.create_task(local_yolo_source_loop())
        logger.info(
            "Cloud AI visual analysis enabled (model=%s, url=%s, source=%s)",
            CLOUD_AI_MODEL, CLOUD_AI_URL, FRAME_SOURCE,
        )

    if ENABLE_LOCAL_AUDIT and cap is not None:
        audit_task = asyncio.create_task(local_audit_loop())
        logger.info("Local Visual Audit enabled (model=%s, interval=%.1fs)", OLLAMA_VISION_MODEL, AUDIT_INTERVAL_SEC)
    if ENABLE_AUDIO_STT:
        audio_task = asyncio.create_task(audio_transcription_loop())
        logger.info(
            "Audio STT enabled (model=%s, duration=%.1fs)", CLOUD_AI_MODEL, STT_DURATION_SEC
        )
        _print_audio("[Rioc] Listening. Speak to the camera; you'll see '[Rioc] ...' when speech is processed.")
    if ENABLE_VIDEODB:
        from videodb_integration import run_videodb_eyes

        def _on_transcript(t: str) -> None:
            global latest_transcript
            latest_transcript = t
            _print_audio(f"[Rioc] VideoDB heard: {t}")

        def _on_visual(t: str) -> None:
            _print_audio(f"[VideoDB] Visual: {t}")
            if ENABLE_SPEAKER_TTS and SPEAKER_URL and len(t) <= 500:
                asyncio.create_task(_speak_through_speaker(t))

        videodb_task = asyncio.create_task(
            run_videodb_eyes(
                on_transcript=_on_transcript,
                on_visual_index=_on_visual,
                batch_seconds=VIDEODB_BATCH_SEC,
            )
        )
        logger.info("VideoDB eyes and ears enabled (batch=%ds)", VIDEODB_BATCH_SEC)
    if ENABLE_SPEAKER_TTS and SPEAKER_URL:
        _print_audio(f"[Rioc] Speaker TTS enabled: {SPEAKER_URL}")
    try:
        yield
    finally:
        if cloud_audit_task is not None:
            cloud_audit_task.cancel()
            try:
                await cloud_audit_task
            except asyncio.CancelledError:
                pass
        if frame_source_task is not None:
            frame_source_task.cancel()
            try:
                await frame_source_task
            except asyncio.CancelledError:
                pass
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
        if videodb_task is not None:
            videodb_task.cancel()
            try:
                await videodb_task
            except asyncio.CancelledError:
                pass
        with cap_lock:
            old_cap = cap
            cap = None
        if old_cap is not None:
            old_cap.release()
            logger.info("Camera released.")


app = FastAPI(title="Webcam Stream", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TtsTestRequest(BaseModel):
    text: str


@app.post("/tts/test")
async def tts_test(body: TtsTestRequest):
    """Send arbitrary text to the speaker (used by the AI Guard UI). Bypasses ENABLE_SPEAKER_TTS flag."""
    if not SPEAKER_URL or not CLOUD_AI_URL:
        return Response(
            content="SPEAKER_URL or CLOUD_AI_URL not configured",
            status_code=400,
            media_type="text/plain",
        )
    await _speak_through_speaker(body.text, force=True)
    return {"ok": True}


class ConfigureRequest(BaseModel):
    cameraRtspUrl: str | None = None
    speakerUrl: str | None = None
    speakerUser: str | None = None
    speakerPass: str | None = None
    speakerWsUrl: str | None = None
    speakerType: str | None = None  # e.g. "axis" — overrides SPEAKER_TYPE env var
    enableYolo: bool | None = None  # Toggle YOLO person-detection pre-filter at runtime


async def _probe_rtsp(url: str) -> bool:
    """Use ffprobe (subprocess) to check if an RTSP URL is reachable before passing to OpenCV."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffprobe", "-v", "error", "-rtsp_transport", "tcp", "-i", url,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            await asyncio.wait_for(proc.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return False
        return proc.returncode == 0
    except FileNotFoundError:
        return True  # ffprobe not installed — let OpenCV try
    except Exception:
        return False


def _open_capture(url: str) -> cv2.VideoCapture:
    """Open a VideoCapture in a thread (only called after ffprobe confirms reachability)."""
    return cv2.VideoCapture(url, cv2.CAP_FFMPEG)


@app.post("/configure")
async def configure(body: ConfigureRequest):
    """Hot-swap camera source and/or speaker settings without restarting."""
    global cap, SPEAKER_URL, SPEAKER_USER, SPEAKER_PASS, SPEAKER_WS_URL, SPEAKER_TYPE, ENABLE_YOLO

    if body.cameraRtspUrl is not None:
        # Use ffprobe to validate before touching OpenCV (prevents macOS segfault on bad RTSP)
        reachable = await _probe_rtsp(body.cameraRtspUrl)
        if not reachable:
            return Response(
                content=f"Camera stream unreachable: {body.cameraRtspUrl}",
                status_code=400,
                media_type="text/plain",
            )
        try:
            new_cap = await asyncio.wait_for(
                asyncio.to_thread(_open_capture, body.cameraRtspUrl),
                timeout=12.0,
            )
        except asyncio.TimeoutError:
            return Response(
                content=f"Camera stream timed out: {body.cameraRtspUrl}",
                status_code=400,
                media_type="text/plain",
            )
        if not new_cap.isOpened():
            new_cap.release()
            return Response(
                content=f"Failed to open camera stream: {body.cameraRtspUrl}",
                status_code=400,
                media_type="text/plain",
            )
        # Swap cap under lock so get_next_frame can't read a partially-released cap
        with cap_lock:
            old_cap = cap
            cap = new_cap
        if old_cap is not None:
            old_cap.release()
        logger.info("Camera reconfigured to: %s", body.cameraRtspUrl)

    if body.speakerUrl is not None:
        SPEAKER_URL = body.speakerUrl.rstrip("/")
        logger.info("Speaker URL updated to: %s", SPEAKER_URL)
    if body.speakerUser is not None:
        SPEAKER_USER = body.speakerUser
    if body.speakerPass is not None:
        SPEAKER_PASS = body.speakerPass
    if body.speakerWsUrl is not None:
        SPEAKER_WS_URL = body.speakerWsUrl
    if body.speakerType is not None:
        SPEAKER_TYPE = body.speakerType.strip().lower()
        logger.info("Speaker type updated to: %s", SPEAKER_TYPE)
    if body.enableYolo is not None:
        ENABLE_YOLO = body.enableYolo
        logger.info("YOLO person detection %s", "enabled" if ENABLE_YOLO else "disabled")

    return {"ok": True}


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


@app.get("/analysis")
async def analysis():
    """Return the latest Cloud AI vision analysis (if any)."""
    return {"text": latest_analysis}


@app.get("/detections")
async def get_detections():
    """Return detection history (vision + audio), newest first."""
    return {"detections": list(detections)}


@app.get("/detections/stream")
async def detections_stream():
    """Server-Sent Events stream — pushes each new detection instantly as it happens.
    On connect, replays existing detections so the client is immediately up to date."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=50)
    _sse_subscribers.append(queue)

    async def event_gen():
        try:
            # Replay existing detections (oldest first) so client starts with full history
            for d in reversed(list(detections)):
                yield f"data: {json.dumps(d)}\n\n"
            # Push new detections as they arrive; send keepalives to prevent proxy timeouts
            while True:
                try:
                    d = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"data: {json.dumps(d)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            try:
                _sse_subscribers.remove(queue)
            except ValueError:
                pass

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/tts/latest.mp3")
async def tts_latest():
    """Serve latest TTS audio for play-from-URL (speaker fetches this)."""
    if not latest_tts_audio:
        return Response(content=b"", status_code=404, media_type="audio/mpeg")
    return Response(content=latest_tts_audio, media_type="audio/mpeg")


@app.get("/speaker-test")
async def speaker_test():
    """
    Diagnostic: test if the speaker can reach this server.
    Open this URL from a browser ON THE SAME NETWORK AS THE SPEAKER (e.g. your phone on WiFi).
    If you hear audio, the speaker should be able to fetch it too.
    """
    if not latest_tts_audio:
        return Response(
            content="No TTS yet. Speak to Rioc first to generate speech.",
            status_code=404,
            media_type="text/plain",
        )
    return Response(content=latest_tts_audio, media_type="audio/mpeg")


@app.get("/speaker-diagnostic")
async def speaker_diagnostic():
    """Return diagnostic steps when startstream returns 200 but no audio plays."""
    tts_url = f"{TTS_PUBLIC_URL}/tts/latest.mp3" if TTS_PUBLIC_URL else "(set TTS_PUBLIC_URL)"
    return {
        "issue": "Speaker returns 200 but no audio - usually the speaker cannot reach your Mac",
        "tts_url": tts_url,
        "steps": [
            "1. From your PHONE (same WiFi as speaker), open the tts_url above. Can you hear it?",
            "2. If NO: Mac firewall may block port 8000. System Settings > Network > Firewall > allow Python or add port 8000.",
            "3. If YES: Speaker may not support outbound HTTP. Check speaker web UI for 'Stream' or 'Network' settings.",
            "4. Try SPEAKER_TEST_URL: set to a public MP3 (e.g. https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3) in .env, restart. If speaker plays it, the issue is Mac reachability.",
        ],
    }


@app.get("/speaker-test-bell")
async def speaker_test_bell():
    """
    Trigger the speaker's built-in bell (bell1). If you hear it, the speaker works
    and the issue is with stream/URL playback. If you don't hear it, check volume/output.
    """
    auth = (SPEAKER_USER, SPEAKER_PASS) if SPEAKER_USER else None
    try:
        async with httpx.AsyncClient(verify=False) as client:
            r = await client.get(
                f"{SPEAKER_URL}/api/play",
                params={"action": "start", "file": "bell1", "volume": 30},
                auth=auth,
                timeout=10.0,
            )
        return {"status": r.status_code, "message": "Triggered bell1. Did you hear it?"}
    except Exception as e:
        return {"status": 500, "message": str(e)}

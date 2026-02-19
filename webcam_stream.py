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

import numpy as np
import cv2
import httpx
import sounddevice as sd
from fastapi import FastAPI
from openai import OpenAI
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
STT_SILENCE_THRESHOLD = float(os.environ.get("STT_SILENCE_THRESHOLD", "300"))  # RMS below this = skip STT

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
OPENAI_TTS_MODEL = os.environ.get("OPENAI_TTS_MODEL", "tts-1")
OPENAI_TTS_VOICE = os.environ.get("OPENAI_TTS_VOICE", "onyx")  # Deep, authoritative
# Fallback: play through Mac speakers when speaker fails or for testing
ENABLE_LOCAL_PLAYBACK = os.environ.get("ENABLE_LOCAL_PLAYBACK", "").strip().lower() in ("1", "true", "yes")
# WebSocket: G.711 μ-law at 8kHz (per talk.js)
SPEAKER_WS_SAMPLE_RATE = int(os.environ.get("SPEAKER_WS_SAMPLE_RATE", "8000"))

latest_transcript: str = ""
latest_tts_audio: bytes = b""  # Served at /tts/latest.mp3 for play-from-URL mode

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


def _print_audio(msg: str) -> None:
    """Print with flush so it appears immediately in uvicorn output."""
    print(msg, flush=True)


def _mp3_to_mulaw(mp3_bytes: bytes, sample_rate: int = 8000) -> bytes | None:
    """Convert MP3 to G.711 μ-law at 8kHz - format talk.js sends to speaker."""
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


async def _speak_through_speaker(text: str) -> None:
    """Generate TTS and play through IP speaker. Runs in thread to avoid blocking."""
    if not ENABLE_SPEAKER_TTS or not SPEAKER_URL or not OPENAI_STT_API_KEY:
        return
    if not text or len(text) > 500:
        return
    try:
        _print_audio("[Rioc] Generating speech...")
        def _tts():
            client = OpenAI(api_key=OPENAI_STT_API_KEY)
            resp = client.audio.speech.create(
                model=OPENAI_TTS_MODEL,
                voice=OPENAI_TTS_VOICE,
                input=text[:500],
            )
            return resp.content

        audio_bytes = await asyncio.to_thread(_tts)
        if not audio_bytes:
            _print_audio("[Rioc] TTS returned no audio")
            return
        global latest_tts_audio
        latest_tts_audio = audio_bytes
        auth = (SPEAKER_USER, SPEAKER_PASS) if SPEAKER_USER else None
        played = False
        # Try WebSocket first (dashboard test uses wss://.../webtwowayaudio)
        if SPEAKER_WS_URL and not played:
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
        if (TTS_PUBLIC_URL or SPEAKER_TEST_URL) and not played:
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
        if not played:
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
        if not played:
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
        _print_audio(f"[Rioc] TTS/speaker error: {e}")


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
    """Background task: capture audio, send to OpenAI STT, update latest_transcript."""
    global latest_transcript
    if not OPENAI_STT_API_KEY:
        _print_audio("[Rioc] Audio STT disabled: no OPENAI_STT_API_KEY or OPENAI_API_KEY in .env")
        return
    _print_audio("[Rioc] Audio loop started. Recording 5s chunks, then sending to STT...")
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {OPENAI_STT_API_KEY}"}
    while True:
        _print_audio("[Rioc] Recording chunk (5 sec)...")
        wav_bytes = await asyncio.to_thread(_record_audio_chunk)
        if not wav_bytes:
            _print_audio("[Rioc] Audio capture failed — check mic permissions (System Settings → Privacy → Microphone)")
            await asyncio.sleep(1.0)
            continue
        if _is_audio_silent(wav_bytes):
            continue  # Skip STT when silent — avoids hallucinations
        _print_audio("[Rioc] Sending to STT...")
        files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
        data = {"model": OPENAI_STT_MODEL, "prompt": STT_PROMPT}
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(url, headers=headers, files=files, data=data, timeout=90.0)
            resp.raise_for_status()
            text = (resp.json().get("text") or "").strip()
            if text and not _is_stt_hallucination(text):
                latest_transcript = text
                _print_audio(f"[Rioc] Heard: {text}")
                # Get Rioc to respond directly to what was said (when Ollama is available)
                if ENABLE_LOCAL_AUDIT:
                    try:
                        async with httpx.AsyncClient() as client:
                            r = await client.post(
                                f"{OLLAMA_URL.rstrip('/')}/api/chat",
                                json={
                                    "model": OLLAMA_VISION_MODEL,
                                    "messages": [
                                        {
                                            "role": "user",
                                            "content": (
                                                f"You are Rioc, a tactical guard. The person in front of you just said: \"{text}\". "
                                                "Respond directly to them in one sentence. Cold, authoritative tone. Do not repeat what they said."
                                            ),
                                        },
                                    ],
                                    "stream": False,
                                },
                                timeout=30.0,
                            )
                        r.raise_for_status()
                        reply = (r.json().get("message") or {}).get("content") or ""
                        if reply.strip():
                            _print_audio(f"[Rioc] {reply.strip()}")
                            asyncio.create_task(_speak_through_speaker(reply.strip()))
                    except Exception as e:
                        logger.debug("Audio response failed: %s", e)
            elif text and _is_stt_hallucination(text):
                pass  # Silently ignore Whisper hallucinations
            else:
                _print_audio("[Rioc] STT returned empty (silence or very quiet audio)")
        except Exception as exc:
            _print_audio(f"[Rioc] STT failed: {exc}")
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
                asyncio.create_task(_speak_through_speaker(msg.strip()))
            else:
                print("[Visual Audit] (model returned empty — ensure Ollama has openbmb/minicpm-v2.6 loaded)")
        except Exception as e:
            print(f"[Visual Audit] Request failed: {e}")


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
            _print_audio("[Rioc] Listening. Speak to the camera; you'll see '[Rioc] Heard: ...' when your speech is transcribed.")
        else:
            _print_audio(
                "[Rioc] Audio STT is ON but no API key found. Create a .env file in the project root with:\n"
                "  OPENAI_STT_API_KEY=sk-proj-your-key\n"
                f"  (Project root: {_project_dir})"
            )
    if ENABLE_SPEAKER_TTS and SPEAKER_URL:
        _print_audio(f"[Rioc] Speaker TTS enabled: {SPEAKER_URL}")
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

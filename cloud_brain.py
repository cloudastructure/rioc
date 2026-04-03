"""
Cloud Brain: Visual Audit loop.
Pulls frames from Mac MJPEG stream, sends to vLLM (MiniCPM-V-2_6) via OpenAI-compatible API.
"""
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project dir and cwd
_project_dir = Path(__file__).resolve().parent
load_dotenv(_project_dir / ".env")
load_dotenv(Path.cwd() / ".env")

import base64
import logging
import os
import re
import time

import httpx
from openai import OpenAI

logger = logging.getLogger(__name__)

# Config from env
STREAM_URL = os.environ.get("STREAM_URL") or (
    f"http://{os.environ.get('MAC_IP', 'localhost')}:8000"
)
VIDEO_URL = f"{STREAM_URL.rstrip('/')}/video"
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "openbmb/MiniCPM-V-2_6-int4")
AUDIT_INTERVAL_SEC = float(os.environ.get("AUDIT_INTERVAL_SEC", "5.0"))

SYSTEM_PROMPT = (
    "You are the Rioc Sentinel, an automated audio-broadcast security system. Do not use labels like 'Visual Data' or 'Warning'. Do not use headers or bullet points. Speak only the final warning text as it should be heard over a loudspeaker. Be cold, concise, and observational. Do not identify as an AI. Always respond in English only."
)
USER_PROMPT = (
    """Start by immediately identifying the person—describe their clothing, hair, or what they're doing (e.g., 'You in the red jacket', 'You with the dark shirt', 'You by the door'). This shows them we can see them.

Tell them directly to leave — this area is restricted and they are not authorized to be here. Speak to them, not about them. Do not issue commands to a security team (e.g. do not say "secure the area" or "detain the subject").

IMPORTANT: Output the raw speech only. No labels. No 'Protocol Start'. No 'Proceed with broadcast'."""
)
USER_PROMPT_WITH_SPEECH = (
    """Start by immediately identifying the person—describe their clothing, hair, or what they're doing. This shows them we can see them.

The person just said: "{transcript}"

Respond directly to what they said in a natural, conversational way. Address their question, objection, or excuse. Do not say "I heard you"—engage with the content. Tell them to leave — this area is restricted. Speak to them directly, not to a security team.

IMPORTANT: Output the raw speech only. No labels. No 'Protocol Start'. No 'Proceed with broadcast'."""
)
MIN_TRANSCRIPT_LEN = 15  # Only use transcript when it's substantial (avoids noise, stale fragments)

BOUNDARY = b"frame"
HEADER_END = b"\r\n\r\n"
CONTENT_LENGTH_RE = re.compile(rb"Content-Length:\s*(\d+)", re.IGNORECASE)

TRANSCRIPT_URL = os.environ.get("TRANSCRIPT_URL") or f"{STREAM_URL.rstrip('/')}/transcript"


def _discover_model(client: OpenAI) -> str | None:
    """Fetch /v1/models and return the first model id. Helps when RunPod uses a different name."""
    try:
        models = client.models.list()
        for m in models.data:
            if m.id:
                return m.id
    except Exception as e:
        logger.debug("Could not list models: %s", e)
    return None


def parse_mjpeg_frames(stream_url: str):
    """
    GET stream_url (MJPEG), parse multipart boundary 'frame', yield raw JPEG bytes per frame.
    """
    with httpx.stream("GET", stream_url, timeout=30.0) as response:
        response.raise_for_status()
        boundary_marker = b"--" + BOUNDARY + b"\r\n"
        buffer = b""
        chunk_iter = response.iter_bytes(chunk_size=8192)

        def get_more() -> bytes | None:
            try:
                return next(chunk_iter)
            except StopIteration:
                return None

        while True:
            chunk = get_more()
            if chunk is None:
                break
            buffer += chunk
            while True:
                idx = buffer.find(boundary_marker)
                if idx == -1:
                    keep = len(boundary_marker) - 1
                    if len(buffer) > 64 * 1024:
                        buffer = buffer[-keep:]
                    break
                buffer = buffer[idx + len(boundary_marker) :]
                header_end = buffer.find(HEADER_END)
                if header_end == -1:
                    break
                headers = buffer[:header_end]
                buffer = buffer[header_end + len(HEADER_END) :]
                match = CONTENT_LENGTH_RE.search(headers)
                if not match:
                    continue
                n = int(match.group(1))
                while len(buffer) < n:
                    more = get_more()
                    if more is None:
                        return
                    buffer += more
                jpeg_bytes = buffer[:n]
                buffer = buffer[n:]
                yield jpeg_bytes


def run_visual_audit(
    client: OpenAI, jpeg_bytes: bytes, transcript: str | None, model: str
) -> str:
    """Send one frame (and optional transcript) to the model; return the assistant reply."""
    b64 = base64.standard_b64encode(jpeg_bytes).decode("ascii")
    use_speech = transcript and len(transcript.strip()) >= MIN_TRANSCRIPT_LEN
    text = USER_PROMPT_WITH_SPEECH.format(transcript=transcript) if use_speech else USER_PROMPT
    # MiniCPM expects text first, then image (per vLLM docs)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        },
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=256,
        temperature=0,  # Deterministic; reduces inconsistent descriptions across frames
    )
    choice = resp.choices[0] if resp.choices else None
    if choice and choice.message and choice.message.content:
        return choice.message.content.strip()
    return ""


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        base_url=VLLM_BASE_URL,
        timeout=httpx.Timeout(180.0),  # Vision inference can take 60–90+ s on cold start
    )
    logger.info("Stream URL: %s", VIDEO_URL)
    logger.info("vLLM base URL: %s", VLLM_BASE_URL)
    model = _discover_model(client) or MODEL_NAME
    logger.info("Using model: %s", model)
    last_audit_time = 0.0
    try:
        for jpeg_bytes in parse_mjpeg_frames(VIDEO_URL):
            now = time.monotonic()
            if now - last_audit_time < AUDIT_INTERVAL_SEC:
                continue
            last_audit_time = now
            try:
                transcript = ""
                try:
                    resp = httpx.get(TRANSCRIPT_URL, timeout=1.0)
                    if resp.status_code == 200:
                        transcript = (resp.json().get("text") or "").strip()
                except httpx.RequestError:
                    transcript = ""
                audit = run_visual_audit(client, jpeg_bytes, transcript or None, model)
                if audit:
                    print(f"[Visual Audit] {audit}")
                else:
                    logger.warning("Empty audit response")
            except Exception as e:
                err_str = str(e)
                if "500" in err_str or "Internal Server Error" in err_str:
                    logger.error(
                        "Audit failed (500): %s — Check RunPod worker logs for the real error (OOM, CUDA, model format).",
                        err_str,
                    )
                else:
                    logger.exception("Audit failed: %s", e)
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    except httpx.HTTPError as e:
        logger.error("Stream unreachable: %s", e)


if __name__ == "__main__":
    main()

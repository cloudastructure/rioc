"""
Cloud Brain: Visual Audit loop.
Pulls frames from Mac MJPEG stream, sends to vLLM (MiniCPM-V-2_6) via OpenAI-compatible API.
"""
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
MODEL_NAME = "openbmb/MiniCPM-V-2_6"
AUDIT_INTERVAL_SEC = float(os.environ.get("AUDIT_INTERVAL_SEC", "5.0"))

SYSTEM_PROMPT = (
    "You are a tactical analyst. Respond in a cold, authoritative tone. Be concise and direct."
)
USER_PROMPT = (
    "Identify the primary person in this frame. What are they wearing? What are they doing? "
    "Give me a 1-sentence tactical warning addressing their specific outfit."
)

BOUNDARY = b"frame"
HEADER_END = b"\r\n\r\n"
CONTENT_LENGTH_RE = re.compile(rb"Content-Length:\s*(\d+)", re.IGNORECASE)

TRANSCRIPT_URL = os.environ.get("TRANSCRIPT_URL") or f"{STREAM_URL.rstrip('/')}/transcript"


def parse_mjpeg_frames(stream_url: str):
    """
    GET stream_url (MJPEG), parse multipart boundary 'frame', yield raw JPEG bytes per frame.
    """
    with httpx.stream("GET", stream_url, timeout=30.0) as response:
        response.raise_for_status()
        boundary_marker = b"--" + BOUNDARY + b"\r\n"
        buffer = b""
        while True:
            chunk = response.read(8192)
            if not chunk:
                break
            buffer += chunk
            while True:
                # Find start of part
                idx = buffer.find(boundary_marker)
                if idx == -1:
                    # Keep last few bytes in case boundary is split
                    keep = len(boundary_marker) - 1
                    if len(buffer) > 64 * 1024:
                        buffer = buffer[-keep:]
                    break
                buffer = buffer[idx + len(boundary_marker) :]
                # Find end of headers
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
                    more = response.read(8192)
                    if not more:
                        return
                    buffer += more
                jpeg_bytes = buffer[:n]
                buffer = buffer[n:]
                yield jpeg_bytes


def run_visual_audit(client: OpenAI, jpeg_bytes: bytes, transcript: str | None = None) -> str:
    """Send one frame (and optional transcript) to the model; return the assistant reply."""
    b64 = base64.standard_b64encode(jpeg_bytes).decode("ascii")
    text = USER_PROMPT
    if transcript:
        text += (
            f'\n\nThe person is saying: "{transcript}". '
            "Combine what you see and hear in your tactical warning."
        )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": text},
            ],
        },
    ]
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=256,
    )
    choice = resp.choices[0] if resp.choices else None
    if choice and choice.message and choice.message.content:
        return choice.message.content.strip()
    return ""


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"), base_url=VLLM_BASE_URL)
    logger.info("Stream URL: %s", VIDEO_URL)
    logger.info("vLLM base URL: %s", VLLM_BASE_URL)
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
                audit = run_visual_audit(client, jpeg_bytes, transcript or None)
                if audit:
                    print(f"[Visual Audit] {audit}")
                else:
                    logger.warning("Empty audit response")
            except Exception as e:
                logger.exception("Audit failed: %s", e)
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    except httpx.HTTPError as e:
        logger.error("Stream unreachable: %s", e)


if __name__ == "__main__":
    main()

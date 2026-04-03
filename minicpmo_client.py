"""MiniCPM-o 4.5 API client.

Calls a remote MiniCPM-o 4.5 endpoint (vLLM / HuggingFace-served) with:
  - a video frame (JPEG bytes as base64)
  - optional audio input (WAV bytes as base64)
  - conversation history
Returns response text + generated audio WAV bytes.
"""
import base64
import os
from typing import Any

import httpx

MINICPMO_URL = os.environ.get("MINICPMO_URL", "http://172.16.128.41:8101/")
MINICPMO_API_KEY = os.environ.get("MINICPMO_API_KEY", "token-minicpm-o45")
MINICPMO_MODEL = os.environ.get("MINICPMO_MODEL", "openbmb/MiniCPM-o-2_6")
MINICPMO_TIMEOUT = float(os.environ.get("MINICPMO_TIMEOUT", "60.0"))


async def chat(
    jpeg_bytes: bytes,
    system_prompt: str,
    conversation_history: list[dict[str, Any]],
    audio_bytes: bytes | None = None,
) -> tuple[str, bytes | None]:
    """Send a frame (+ optional audio) to MiniCPM-o and return (text, wav_bytes).

    conversation_history is a list of {"role": "user"|"assistant", "content": ...} dicts
    representing prior turns. The current frame/audio are appended as the latest user turn.
    """
    # Build the current user message content
    b64_image = base64.standard_b64encode(jpeg_bytes).decode("ascii")
    content: list[dict[str, Any]] = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
        }
    ]
    if audio_bytes:
        b64_audio = base64.standard_b64encode(audio_bytes).decode("ascii")
        content.append(
            {
                "type": "input_audio",
                "input_audio": {"data": b64_audio, "format": "wav"},
            }
        )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        *conversation_history,
        {"role": "user", "content": content},
    ]

    base_url = MINICPMO_URL.rstrip("/")
    url = f"{base_url}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {MINICPMO_API_KEY}"}

    async with httpx.AsyncClient(timeout=MINICPMO_TIMEOUT) as client:
        resp = await client.post(
            url, headers=headers,
            json={
                "model": MINICPMO_MODEL,
                "modalities": ["text", "audio"],
                "audio": {"voice": "default", "format": "wav"},
                "messages": messages,
                "max_tokens": 200,
            },
        )
        # 405 means the server doesn't support audio modalities — retry text-only
        if resp.status_code == 405:
            resp = await client.post(
                url, headers=headers,
                json={"model": MINICPMO_MODEL, "messages": messages, "max_tokens": 200},
            )
        resp.raise_for_status()

    data = resp.json()
    choice_msg = ((data.get("choices") or [{}])[0].get("message") or {})
    audio_data = choice_msg.get("audio") or {}

    text = choice_msg.get("content") or audio_data.get("transcript") or ""

    wav_out: bytes | None = None
    wav_b64 = audio_data.get("data") or ""
    if wav_b64:
        wav_out = base64.b64decode(wav_b64)

    return text, wav_out

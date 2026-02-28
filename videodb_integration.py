"""
VideoDB eyes and ears: real-time transcript + visual/audio indexing.

When ENABLE_VIDEODB=1, connects to VideoDB RTSP streams, starts transcript and
indexing pipelines, and emits events. Use as an alternative to ENABLE_AUDIO_STT
and ENABLE_LOCAL_AUDIT for perception.

Requires: pip install videodb
Docs: https://docs.videodb.io/pages/getting-started/quickstart
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Callable

logger = logging.getLogger(__name__)

# Env
VIDEODB_API_KEY = os.environ.get("VIDEODB_API_KEY") or os.environ.get("VIDEO_DB_API_KEY")
VIDEODB_RTSP_VIDEO = os.environ.get("VIDEODB_RTSP_VIDEO", "").strip()
VIDEODB_RTSP_AUDIO = os.environ.get("VIDEODB_RTSP_AUDIO", "").strip()

# Demo streams (VideoDB hosts these) - for trying without RTSP setup
DEMO_VIDEO_URL = "rtsp://matrix.videodb.io:8554/screen"
DEMO_AUDIO_URL = "rtsp://matrix.videodb.io:8554/audio"

# Prompts for Rioc-style perception
VIDEO_INDEX_PROMPT = (
    "In one sentence, describe the person visible: their clothing, hair, and what they are doing. "
    "Be specific and observational. This is for a security guard broadcast."
)
AUDIO_INDEX_PROMPT = "Summarize what is being said or heard. Extract speech verbatim when possible."


async def run_videodb_eyes(
    *,
    on_transcript: Callable[[str], None] | None = None,
    on_visual_index: Callable[[str], None] | None = None,
    on_audio_index: Callable[[str], None] | None = None,
    batch_seconds: int = 5,
) -> None:
    """
    Connect to VideoDB RTSP streams, start transcript + indexing, emit events.

    Uses VIDEODB_RTSP_VIDEO and VIDEODB_RTSP_AUDIO if set; otherwise demo streams.
    """
    if not VIDEODB_API_KEY:
        logger.warning("VideoDB disabled: set VIDEODB_API_KEY in .env")
        return

    video_url = VIDEODB_RTSP_VIDEO or DEMO_VIDEO_URL
    audio_url = VIDEODB_RTSP_AUDIO or DEMO_AUDIO_URL

    if not VIDEODB_RTSP_VIDEO and not VIDEODB_RTSP_AUDIO:
        logger.info(
            "VideoDB using demo streams (matrix.videodb.io). "
            "Set VIDEODB_RTSP_VIDEO and VIDEODB_RTSP_AUDIO for your own webcam/mic."
        )

    try:
        import videodb
    except ImportError:
        logger.error("VideoDB not installed. Run: pip install videodb")
        return

    conn = videodb.connect(api_key=VIDEODB_API_KEY)
    coll = conn.get_collection()
    ws = conn.connect_websocket()
    await ws.connect()

    audio_stream = coll.connect_rtstream(
        url=audio_url,
        name="Rioc Audio",
        media_types=["audio"],
    )
    video_stream = coll.connect_rtstream(
        url=video_url,
        name="Rioc Video",
        media_types=["video"],
    )

    audio_stream.start_transcript(ws_connection_id=ws.connection_id)
    audio_stream.index_audio(
        prompt=AUDIO_INDEX_PROMPT,
        batch_config={"type": "time", "value": batch_seconds},
        ws_connection_id=ws.connection_id,
    )
    video_stream.index_visuals(
        prompt=VIDEO_INDEX_PROMPT,
        batch_config={"type": "time", "value": batch_seconds, "frame_count": 1},
        ws_connection_id=ws.connection_id,
    )

    logger.info("VideoDB eyes and ears active (transcript + visual/audio index)")

    try:
        async for msg in ws.receive():
            ch = msg.get("channel", "")
            if ch == "capture_session":
                continue
            data = msg.get("data") if isinstance(msg.get("data"), dict) else {}
            data = data or {}
            text = (data.get("text") or "").strip()
            if not text:
                continue

            if ch == "transcript":
                if not data.get("is_final", False):
                    continue
                if on_transcript:
                    on_transcript(text)
                logger.info("[VideoDB] Transcript: %s", text[:80])

            elif ch in ("scene_index", "visual_index"):
                if on_visual_index:
                    on_visual_index(text)
                logger.info("[VideoDB] Visual: %s", text[:80])

            elif ch == "audio_index":
                if on_audio_index:
                    on_audio_index(text)

    except asyncio.CancelledError:
        pass
    finally:
        try:
            audio_stream.stop()
            video_stream.stop()
        except Exception as e:
            logger.debug("VideoDB stop: %s", e)
        try:
            await ws.close()
        except Exception:
            pass
        logger.info("VideoDB eyes and ears stopped")

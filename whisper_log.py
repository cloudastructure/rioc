"""Async logging-only person-transcript backfill.

Runs off the real-time path: MiniCPM-o hears the person live, but for the
conversation log we transcribe the person's audio afterwards. Never fatal —
a transcription failure logs an empty-text turn rather than breaking anything.
"""
import logging

logger = logging.getLogger(__name__)


async def backfill_person_turn(transcribe_fn, audio_bytes, conversation_id, db):
    try:
        text = await transcribe_fn(audio_bytes)
    except Exception as e:  # logging-only path: never break the conversation
        logger.warning("[whisper_log] transcription failed: %s", e)
        text = ""
    await db.append_live_turn(conversation_id, "PERSON", text, audio_path=None)

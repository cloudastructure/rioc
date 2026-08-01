import asyncio

from whisper_log import backfill_person_turn


class FakeDB:
    def __init__(self):
        self.turns = []

    async def append_live_turn(self, conversation_id, speaker, text, audio_path):
        self.turns.append((conversation_id, speaker, text))


def test_backfill_writes_transcript():
    async def body():
        db = FakeDB()

        async def transcribe(b):
            return "who are you"

        await backfill_person_turn(transcribe, b"snd", conversation_id=1, db=db)
        assert db.turns == [(1, "PERSON", "who are you")]

    asyncio.run(body())


def test_backfill_failure_is_nonfatal_empty_text():
    async def body():
        db = FakeDB()

        async def transcribe(b):
            raise RuntimeError("stt down")

        await backfill_person_turn(transcribe, b"snd", conversation_id=1, db=db)
        assert db.turns == [(1, "PERSON", "")]

    asyncio.run(body())

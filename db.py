"""SQLite database for AI Guard conversations and turns."""
import aiosqlite
import asyncio
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "ai_guard.db"

_init_lock = asyncio.Lock()
_initialized = False


async def init_db() -> None:
    global _initialized
    async with _init_lock:
        if _initialized:
            return
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id TEXT,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    outcome TEXT,
                    turn_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL REFERENCES conversations(id),
                    speaker TEXT NOT NULL CHECK(speaker IN ('GUARD', 'PERSON')),
                    text TEXT NOT NULL,
                    audio_path TEXT,
                    timestamp TEXT NOT NULL
                )
            """)
            await db.execute("CREATE INDEX IF NOT EXISTS idx_turns_conv ON conversation_turns(conversation_id)")
            await db.commit()
        _initialized = True


async def create_conversation(camera_id: str | None, started_at: str) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "INSERT INTO conversations (camera_id, started_at) VALUES (?, ?)",
            (camera_id, started_at),
        )
        await db.commit()
        return cursor.lastrowid


async def add_turn(
    conversation_id: int,
    speaker: str,
    text: str,
    timestamp: str,
    audio_path: str | None = None,
) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            """INSERT INTO conversation_turns (conversation_id, speaker, text, audio_path, timestamp)
               VALUES (?, ?, ?, ?, ?)""",
            (conversation_id, speaker, text, audio_path, timestamp),
        )
        await db.execute(
            "UPDATE conversations SET turn_count = turn_count + 1 WHERE id = ?",
            (conversation_id,),
        )
        await db.commit()
        return cursor.lastrowid


async def close_conversation(conversation_id: int, ended_at: str, outcome: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE conversations SET ended_at = ?, outcome = ? WHERE id = ?",
            (ended_at, outcome, conversation_id),
        )
        await db.commit()


async def get_conversations(limit: int = 50, offset: int = 0) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """SELECT id, camera_id, started_at, ended_at, outcome, turn_count, created_at
               FROM conversations ORDER BY started_at DESC LIMIT ? OFFSET ?""",
            (limit, offset),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


async def get_conversation(conversation_id: int) -> dict | None:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        conv = dict(row)
        cursor2 = await db.execute(
            "SELECT * FROM conversation_turns WHERE conversation_id = ? ORDER BY timestamp",
            (conversation_id,),
        )
        turns = await cursor2.fetchall()
        conv["turns"] = [dict(t) for t in turns]
        return conv

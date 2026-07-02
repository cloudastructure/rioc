"""Verifies the FakeBackend satisfies the GuardBackend Protocol shape.

If this test breaks, the Protocol changed and every real backend needs an audit.
"""
from guard_backend import GuardBackend, GuardEncounter


async def test_fake_backend_matches_protocol(fake_backend):
    assert isinstance(fake_backend, GuardBackend)
    encounter = await fake_backend.open_encounter()
    assert isinstance(encounter, GuardEncounter)


async def test_fake_encounter_records_pushed_frames(fake_backend):
    encounter = await fake_backend.open_encounter()
    await encounter.push_frame(b"jpeg-1")
    await encounter.push_frame(b"jpeg-2")
    assert encounter.pushed_frames == [b"jpeg-1", b"jpeg-2"]


async def test_fake_encounter_turn_returns_scripted_response(fake_backend):
    fake_backend.script_response(text="stop where you are", wav=b"WAVDATA")
    encounter = await fake_backend.open_encounter()
    await encounter.push_frame(b"jpeg-1")
    text, wav = await encounter.turn(
        audio_bytes=None,
        system_prompt="sys",
        history=[],
        user_text="you are the guard",
    )
    assert text == "stop where you are"
    assert wav == b"WAVDATA"


async def test_fake_encounter_close_is_idempotent(fake_backend):
    encounter = await fake_backend.open_encounter()
    await encounter.close()
    await encounter.close()  # must not raise
    assert encounter.closed is True

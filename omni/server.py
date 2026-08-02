"""Omni Streaming Server — WebSocket endpoint on the GPU box.

Wires the Live-channel protocol + fixed-K admission + a MiniCPM-o session.
`session_factory(system_prompt, voice) -> OmniSession` is injected so tests can
pass a fake session (no GPU needed).

Generation runs as a *background task* so the receive loop keeps reading — this is
what lets an `interrupt` be processed mid-generation (true server-side barge-in).
The sync model generator is driven via asyncio.to_thread so it never blocks the loop.
"""
import asyncio
import base64
import itertools

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from omni import protocol as p
from omni.admission import Admission


def build_app(session_factory, capacity=1):
    app = FastAPI()
    admission = Admission(capacity=capacity)

    @app.websocket("/omni/session")
    async def omni_session(ws: WebSocket):
        await ws.accept()
        seq = itertools.count()
        slot = None
        session = None
        gen_task = None

        async def _run_generation(ts):
            await ws.send_text(p.encode(p.make("turn_start", next(seq), ts)))
            it = session.generate()
            while True:
                chunk = await asyncio.to_thread(next, it, None)  # drive sync gen off the loop
                if chunk is None:
                    break
                text, audio = chunk
                await ws.send_text(p.encode(p.make(
                    "speech", next(seq), ts,
                    audio_b64=base64.b64encode(audio).decode(), text_delta=text)))
            await ws.send_text(p.encode(p.make("turn_end", next(seq), ts)))

        try:
            start = p.decode(await ws.receive_text())
            slot = admission.try_acquire()
            if slot is None:
                await ws.send_text(p.encode(p.make("capacity_exhausted", next(seq), start["ts"])))
                await ws.close()
                return
            session = session_factory(
                system_prompt=start.get("system_prompt", ""),
                voice=start.get("voice", "default"),
            )
            await ws.send_text(p.encode(p.make("session_ready", next(seq), start["ts"], session_id=slot)))
            while True:
                msg = p.decode(await ws.receive_text())
                t = msg["type"]
                if t == "video":
                    session.prefill(jpeg=base64.b64decode(msg["jpeg_b64"]), pcm=None)
                elif t == "audio":
                    session.prefill(jpeg=None, pcm=base64.b64decode(msg["pcm_or_opus_b64"]))
                elif t == "interrupt":
                    session.interrupt()  # generator checks this between chunks -> stops
                    await ws.send_text(p.encode(p.make("interrupted", next(seq), msg["ts"])))
                elif t == "user_speech_end":
                    if gen_task is None or gen_task.done():
                        gen_task = asyncio.create_task(_run_generation(msg["ts"]))
                elif t == "session_end":
                    break
        except WebSocketDisconnect:
            pass
        finally:
            if gen_task is not None:
                gen_task.cancel()
            if slot is not None:
                admission.release(slot)

    return app

"""Appliance-side glue for Live mode: persistent WS-G.711 speaker + uplink/downlink pumps.

Reuses Rioc's existing speaker transport shape (wss://.../webtwowayaudio, μ-law in 20 ms
frames — see _guarded_play in main.py). The control-flow here is structurally complete, but
the real-time audio path is only exercised against real hardware + the omni streaming server:

  VERIFY ON HARDWARE:
   - MINICPMO_AUDIO_RATE / pcm_to_mulaw assume the omni server's downlink `speech` is
     16-bit mono PCM at MINICPMO_AUDIO_RATE. Confirm against omni/server.py once the
     streaming model server is stood up (spike), and adjust the format if it differs.
   - mic_barge_in_pump's `frames` async-generator is provided by main.py from the mic; the
     barge-in *decision* (BargeInDetector) is unit-tested, the mic *source* is the hardware seam.
"""
import asyncio
import base64
import ssl
import subprocess

import websockets

# MiniCPM-o omni speech output sample rate (Hz). 24k is the reference default.
MINICPMO_AUDIO_RATE = 24000


def pcm_to_mulaw(pcm_bytes: bytes, in_rate: int, out_rate: int) -> bytes:
    """16-bit mono PCM @ in_rate -> G.711 μ-law @ out_rate via ffmpeg (audioop is gone in 3.13+)."""
    if not pcm_bytes:
        return b""
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error",
             "-f", "s16le", "-ar", str(in_rate), "-ac", "1", "-i", "pipe:0",
             "-ar", str(out_rate), "-ac", "1", "-f", "mulaw", "pipe:1"],
            input=pcm_bytes, capture_output=True, check=True,
        )
        return proc.stdout
    except Exception:
        return b""


class WsSpeaker:
    """Persistent-WebSocket G.711 speaker implementing the orchestrator's speaker interface
    (async feed / flush_and_stop / run / stopped). Streams μ-law in 20 ms chunks, interruptible.

    flush_and_stop() clears queued audio (barge-in: guard falls silent immediately) but keeps
    the socket open for the next turn; close() ends the session.
    """

    def __init__(self, ws_url: str, sample_rate: int, to_mulaw=pcm_to_mulaw):
        self._ws_url = ws_url
        self._sample_rate = sample_rate
        self._to_mulaw = to_mulaw
        self._q: asyncio.Queue = asyncio.Queue()
        self._stopped = False
        self._ws = None

    async def _connect(self):
        ssl_ctx = None
        if self._ws_url.startswith("wss"):
            ssl_ctx = ssl.create_default_context()
            ssl_ctx.check_hostname = False
            ssl_ctx.verify_mode = ssl.CERT_NONE
        self._ws = await websockets.connect(
            self._ws_url, ssl=ssl_ctx, ping_timeout=None, open_timeout=10,
        )

    async def feed(self, pcm_chunk: bytes):
        mulaw = await asyncio.to_thread(self._to_mulaw, pcm_chunk, MINICPMO_AUDIO_RATE, self._sample_rate)
        step = self._sample_rate // 50  # 160 B = 20 ms @ 8 kHz
        for i in range(0, len(mulaw), step):
            self._q.put_nowait(mulaw[i:i + step])

    async def run(self):
        await self._connect()
        while not self._stopped:
            try:
                chunk = self._q.get_nowait()
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.005)
                continue
            try:
                await self._ws.send(chunk)
            except Exception:
                break
            await asyncio.sleep(0.02)  # real-time pace
        try:
            if self._ws:
                await self._ws.close()
        except Exception:
            pass

    def flush_and_stop(self) -> int:
        dropped = self._q.qsize()
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except asyncio.QueueEmpty:
                break
        return dropped  # socket stays open — barge-in silences the guard, session continues

    @property
    def stopped(self) -> bool:
        return self._stopped

    def close(self):
        self._stopped = True


async def downlink_pump(orchestrator, channel):
    """Route every downlink message (speech / text / turn / interrupted / error) to the orchestrator."""
    async for msg in channel.messages():
        await orchestrator.on_downlink(msg)


async def frame_uplink_pump(channel, frame_getter, fps: int = 2):
    """Sample the latest JPEG frame at ~fps and stream it up the live channel."""
    interval = 1.0 / max(fps, 1)
    while True:
        frame = frame_getter()
        if frame:
            await channel.send("video", jpeg_b64=base64.b64encode(frame).decode())
        await asyncio.sleep(interval)


async def mic_barge_in_pump(detector, frames, channel, orchestrator, on_utterance=None):
    """Edge VAD + barge-in + audio uplink, accumulating each person utterance for logging.

    frames: async generator yielding (pcm_bytes, rate, energy). The barge-in decision is
    BargeInDetector (unit-tested); guard-speaking state is read from the orchestrator each frame.
    on_utterance(pcm_bytes, rate): optional async callback fired at each person speech_end,
    used by the caller to run the (off-critical-path) Whisper transcript backfill.
    """
    buffer = bytearray()
    capturing = False
    async for pcm, rate, energy in frames:
        detector.set_guard_speaking(getattr(orchestrator, "_guard_speaking", False))
        event = detector.push(pcm, rate, energy)
        if event in ("speech_start", "barge_in"):
            orchestrator.note_person_spoke()
            if event == "barge_in":
                await orchestrator.barge_in()
            else:
                await channel.send("user_speech_start")
            capturing = True
            buffer = bytearray(pcm)
        elif event == "speech_end":
            await channel.send("user_speech_end")
            if on_utterance is not None and buffer:
                await on_utterance(bytes(buffer), rate)
            capturing = False
            buffer = bytearray()
        elif capturing:
            buffer.extend(pcm)
        # Uplink the person's audio for the model regardless of gate outcome.
        await channel.send("audio", pcm_or_opus_b64=base64.b64encode(pcm).decode())


# ── continuous media sources (VERIFY ON HARDWARE) ────────────────────────────

async def mic_frames(sample_rate: int = 16000, frame_ms: int = 30, device=None):
    """Continuous mic capture for live barge-in. Yields (pcm_bytes, sample_rate, energy_rms).

    VERIFY ON HARDWARE: needs sounddevice + a real input device. This is the continuous
    counterpart to mic_listener.listen_for_response (which is one-shot half-duplex)."""
    try:
        import numpy as np
        import sounddevice as sd
    except ImportError:
        return
    frame_samples = int(sample_rate * frame_ms / 1000)
    stream = sd.RawInputStream(
        samplerate=sample_rate, blocksize=frame_samples, dtype="int16", channels=1, device=device,
    )
    stream.start()
    try:
        while True:
            data, _ = await asyncio.to_thread(stream.read, frame_samples)
            pcm = bytes(data)
            arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
            energy = float(np.sqrt(np.mean(arr * arr))) if arr.size else 0.0
            yield pcm, sample_rate, energy
    finally:
        stream.stop()
        stream.close()


async def rtsp_frame_source(rtsp_url: str, fps: int = 2, jpeg_quality: int = 70):
    """Continuous JPEG frames from an RTSP camera, for true live video uplink to the model.

    VERIFY ON HARDWARE: needs OpenCV + a reachable RTSP stream. This is what turns the
    uplink from "one frozen detection frame" into an actual live feed."""
    try:
        import cv2
    except ImportError:
        return
    cap = await asyncio.to_thread(cv2.VideoCapture, rtsp_url)
    interval = 1.0 / max(fps, 1)
    try:
        while True:
            ok, frame = await asyncio.to_thread(cap.read)
            if not ok:
                await asyncio.sleep(interval)
                continue
            ok2, buf = await asyncio.to_thread(
                cv2.imencode, ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
            )
            if ok2:
                yield buf.tobytes()
            await asyncio.sleep(interval)
    finally:
        await asyncio.to_thread(cap.release)


async def frame_source_uplink_pump(channel, frame_source):
    """Stream JPEG frames from an async frame source up the live channel."""
    async for jpeg in frame_source:
        await channel.send("video", jpeg_b64=base64.b64encode(jpeg).decode())

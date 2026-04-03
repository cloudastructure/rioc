"""VAD-gated audio capture from the configured input device.

Listens for a spoken utterance after the guard has finished playing audio,
using WebRTC VAD (webrtcvad) to detect speech start/end. Returns WAV bytes
for the detected utterance, or None if silence / timeout.

Half-duplex design: caller must pass speaking=False *after* the speaker
has finished playing before calling listen_for_response().
"""
import io
import logging
import os
import wave

import numpy as np

logger = logging.getLogger(__name__)

# VAD aggressiveness: 0 (least aggressive) to 3 (most aggressive)
VAD_AGGRESSIVENESS = int(os.environ.get("VAD_AGGRESSIVENESS", "2"))
# Sample rate for VAD — webrtcvad supports 8000, 16000, 32000, 48000 Hz
VAD_SAMPLE_RATE = int(os.environ.get("VAD_SAMPLE_RATE", "16000"))
# Frame duration in ms — webrtcvad supports 10, 20, 30 ms
VAD_FRAME_MS = int(os.environ.get("VAD_FRAME_MS", "30"))
# Max silence after speech detected before we consider the utterance done
VAD_SILENCE_FRAMES = int(os.environ.get("VAD_SILENCE_FRAMES", "20"))  # ~600ms at 30ms
# Max total duration to listen for (seconds) before giving up
VAD_MAX_LISTEN_SEC = float(os.environ.get("VAD_MAX_LISTEN_SEC", "8.0"))
# Minimum speech frames to count as a real utterance (filters noise bursts)
VAD_MIN_SPEECH_FRAMES = int(os.environ.get("VAD_MIN_SPEECH_FRAMES", "5"))

AUDIO_INPUT_DEVICE = (os.environ.get("AUDIO_INPUT_DEVICE") or "").strip() or None


def _resolve_device(device_hint: str | None) -> int | None:
    """Resolve a device name substring or index string to a sounddevice device index."""
    if device_hint is None:
        return None
    try:
        return int(device_hint)
    except ValueError:
        pass
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if device_hint.lower() in dev["name"].lower() and dev["max_input_channels"] > 0:
                return i
    except Exception:
        pass
    return None


def _bytes_to_wav(pcm_bytes: bytes, sample_rate: int, channels: int = 1) -> bytes:
    """Wrap raw 16-bit PCM bytes in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def listen_for_response(device_hint: str | None = None) -> bytes | None:
    """Block and listen for a spoken response.  Returns WAV bytes or None.

    This is a synchronous function intended to be called in a thread via
    asyncio.to_thread().  It records audio in VAD-gated chunks and returns
    as soon as the speaker stops talking or the max duration elapses.
    """
    try:
        import webrtcvad
        import sounddevice as sd
    except ImportError as exc:
        logger.warning("mic_listener: missing dependency (%s) — skipping mic input", exc)
        return None

    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    device = _resolve_device(device_hint or AUDIO_INPUT_DEVICE)

    frame_samples = int(VAD_SAMPLE_RATE * VAD_FRAME_MS / 1000)
    frame_bytes = frame_samples * 2  # 16-bit PCM

    max_frames = int(VAD_MAX_LISTEN_SEC * 1000 / VAD_FRAME_MS)
    silence_count = 0
    speech_frames = 0
    triggered = False
    voiced_chunks: list[bytes] = []

    try:
        with sd.RawInputStream(
            samplerate=VAD_SAMPLE_RATE,
            blocksize=frame_samples,
            dtype="int16",
            channels=1,
            device=device,
        ) as stream:
            for _ in range(max_frames):
                data, _ = stream.read(frame_samples)
                chunk = bytes(data)
                if len(chunk) != frame_bytes:
                    continue

                is_speech = vad.is_speech(chunk, VAD_SAMPLE_RATE)

                if not triggered:
                    if is_speech:
                        speech_frames += 1
                        voiced_chunks.append(chunk)
                        if speech_frames >= VAD_MIN_SPEECH_FRAMES:
                            triggered = True
                            logger.debug("[mic_listener] Speech triggered")
                    else:
                        # Pre-roll: keep a small buffer of frames before speech
                        voiced_chunks.append(chunk)
                        if len(voiced_chunks) > 10:
                            voiced_chunks.pop(0)
                else:
                    voiced_chunks.append(chunk)
                    if not is_speech:
                        silence_count += 1
                        if silence_count >= VAD_SILENCE_FRAMES:
                            logger.debug("[mic_listener] Silence threshold reached — done")
                            break
                    else:
                        silence_count = 0

    except Exception as exc:
        logger.warning("[mic_listener] Recording error: %s", exc)
        return None

    if not triggered or len(voiced_chunks) < VAD_MIN_SPEECH_FRAMES:
        logger.debug("[mic_listener] No speech detected")
        return None

    pcm = b"".join(voiced_chunks)
    return _bytes_to_wav(pcm, VAD_SAMPLE_RATE)

"""Edge VAD + barge-in detection with reference-signal echo gating.

Runs continuously on mic frames. When the guard is speaking, the guard's own
voice leaks into the mic; we gate against it by requiring energy above the echo
floor before treating detected speech as a genuine barge-in.
"""


class BargeInDetector:
    def __init__(self, vad, *, echo_floor):
        self.vad = vad
        self.echo_floor = echo_floor
        self._guard_speaking = False
        self._in_speech = False

    def set_guard_speaking(self, on):
        self._guard_speaking = on

    def push(self, frame_pcm, rate, energy):
        speech = self.vad.is_speech(frame_pcm, rate)
        if self._guard_speaking:
            # Reference-signal gating: ignore anything at/below the echo floor.
            if not speech or energy <= self.echo_floor:
                return None
            return "barge_in"
        if speech and not self._in_speech:
            self._in_speech = True
            return "speech_start"
        if not speech and self._in_speech:
            self._in_speech = False
            return "speech_end"
        return None

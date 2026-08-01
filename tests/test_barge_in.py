from live.barge_in import BargeInDetector


class FakeVad:
    def __init__(self, verdicts):
        self.v = list(verdicts)

    def is_speech(self, frame, rate):
        return self.v.pop(0)


def test_speech_start_and_end_when_guard_silent():
    d = BargeInDetector(FakeVad([True, False]), echo_floor=100.0)
    d.set_guard_speaking(False)
    assert d.push(b"x", 16000, energy=500) == "speech_start"
    assert d.push(b"x", 16000, energy=5) == "speech_end"


def test_echo_below_floor_suppressed_while_guard_speaks():
    d = BargeInDetector(FakeVad([True]), echo_floor=100.0)
    d.set_guard_speaking(True)
    # VAD says speech, but energy under the echo floor -> it's our own voice
    assert d.push(b"x", 16000, energy=50) is None


def test_loud_speech_over_floor_triggers_barge_in():
    d = BargeInDetector(FakeVad([True]), echo_floor=100.0)
    d.set_guard_speaking(True)
    assert d.push(b"x", 16000, energy=800) == "barge_in"

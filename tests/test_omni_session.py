from omni.omni_session import OmniSession


class FakeModel:
    def __init__(self):
        self.prefills = []
        self.aborted = False

    def streaming_prefill(self, session_id, content, tokenizer):
        self.prefills.append(content)

    def streaming_generate(self, session_id, tokenizer, stop_flag):
        for i in range(5):
            if stop_flag():
                return
            yield {"text": f"w{i} ", "audio": b"\x00\x01"}


def test_prefill_forwards_chunks():
    m = FakeModel()
    s = OmniSession(m, tokenizer=None, system_prompt="guard", voice="default")
    s.prefill(jpeg=b"img", pcm=b"snd")
    # __init__ primes the persona (system prefill), then the user chunk is forwarded.
    assert len(m.prefills) == 2
    assert m.prefills[-1]["role"] == "user"
    assert m.prefills[-1]["content"] == [
        {"type": "image", "data": b"img"},
        {"type": "audio", "data": b"snd"},
    ]


def test_generate_yields_until_done():
    m = FakeModel()
    s = OmniSession(m, tokenizer=None, system_prompt="guard", voice="default")
    out = list(s.generate())
    assert [t for t, _ in out] == ["w0 ", "w1 ", "w2 ", "w3 ", "w4 "]


def test_interrupt_stops_generation_early():
    m = FakeModel()
    s = OmniSession(m, tokenizer=None, system_prompt="guard", voice="default")
    gen = s.generate()
    next(gen)                 # first chunk
    s.interrupt()
    assert list(gen) == []    # nothing more after interrupt

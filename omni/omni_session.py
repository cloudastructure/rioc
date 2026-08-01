"""Thin adapter around the MiniCPM-o omni streaming API.

CONFIRMED MiniCPM-o streaming API (fill exact names/signatures from the spike
against the pinned model on the GPU box):
    model.streaming_prefill(session_id, {"role": "user", "content": [...]}, tokenizer)
    model.streaming_generate(session_id, tokenizer, stop_flag) -> iterator of {"text","audio"}

Downstream code depends only on this adapter, never on the raw model.
"""
import itertools


class OmniSession:
    _ids = itertools.count()

    def __init__(self, model, tokenizer, *, system_prompt, voice):
        self.model = model
        self.tokenizer = tokenizer
        self.session_id = f"sess-{next(self._ids)}"
        self.voice = voice
        self._interrupt = False
        # Prime persona as the first prefill turn.
        self.model.streaming_prefill(
            self.session_id, {"role": "system", "content": system_prompt}, tokenizer
        )

    def prefill(self, *, jpeg=None, pcm=None):
        content = []
        if jpeg is not None:
            content.append({"type": "image", "data": jpeg})
        if pcm is not None:
            content.append({"type": "audio", "data": pcm})
        if content:
            self.model.streaming_prefill(
                self.session_id, {"role": "user", "content": content}, self.tokenizer
            )

    def generate(self):
        self._interrupt = False
        for chunk in self.model.streaming_generate(
            self.session_id, self.tokenizer, lambda: self._interrupt
        ):
            if self._interrupt:
                return
            yield chunk.get("text", ""), chunk.get("audio", b"")

    def interrupt(self):
        self._interrupt = True

    def reset(self):
        self._interrupt = False

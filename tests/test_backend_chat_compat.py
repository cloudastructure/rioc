"""Regression: VllmBackend's kwargs must match minicpmo_client.chat's real signature.

Bug this catches: c1a2234 added user_text= to _do_turn's minicpmo_chat call, but
the matching parameter on chat() was never committed. Every guard reply turn
raised TypeError, silently swallowed by ConversationManager's try/except.
Unit tests missed it because _FakeChat used **kwargs.
"""
import inspect

from minicpmo_client import chat


# Kwargs VllmBackend._VllmEncounter.turn forwards to its chat_fn. If this list
# changes, update it and confirm minicpmo_client.chat still accepts every entry.
_VLLM_BACKEND_FORWARDS = {
    "jpeg_bytes",
    "system_prompt",
    "conversation_history",
    "audio_bytes",
    "user_text",
}


def test_chat_accepts_every_kwarg_vllm_backend_forwards():
    sig = inspect.signature(chat)
    missing = _VLLM_BACKEND_FORWARDS - set(sig.parameters)
    assert not missing, (
        f"minicpmo_client.chat() is missing kwargs VllmBackend forwards: {missing}"
    )

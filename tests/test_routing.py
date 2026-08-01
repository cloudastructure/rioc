import pytest

from live.routing import route_conversation, validate_live_config


def test_route_dispatches_by_mode():
    calls = []
    route_conversation("live", lambda: calls.append("live"), lambda: calls.append("turn"))
    route_conversation("turn_based", lambda: calls.append("live"), lambda: calls.append("turn"))
    assert calls == ["live", "turn"]


def test_live_mode_refuses_url_only_speaker():
    with pytest.raises(ValueError):
        validate_live_config("live", speaker_type="ipspk_url")


def test_live_mode_accepts_ws_speaker():
    validate_live_config("live", speaker_type="fanvil_ws")  # no raise


def test_turn_based_mode_accepts_any_speaker():
    validate_live_config("turn_based", speaker_type="ipspk_url")  # no raise

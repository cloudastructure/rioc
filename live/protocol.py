"""Live-channel message schema + codec (Rioc-side copy of omni/protocol.py).

Kept byte-identical to omni/protocol.py. Do not import across the GPU/appliance
boundary — vendor instead.
"""
import json


class ProtocolError(Exception):
    pass


UPLINK_TYPES = frozenset({
    "session_start", "video", "audio",
    "user_speech_start", "user_speech_end", "interrupt", "session_end", "ping",
})
DOWNLINK_TYPES = frozenset({
    "session_ready", "capacity_exhausted", "speech", "text",
    "turn_start", "turn_end", "interrupted", "state", "error",
})
_ALL = UPLINK_TYPES | DOWNLINK_TYPES


def make(type, seq, ts, **payload):
    if type not in _ALL:
        raise ProtocolError(f"unknown type: {type}")
    return {"type": type, "seq": seq, "ts": ts, **payload}


def encode(msg):
    return json.dumps(msg, separators=(",", ":"))


def decode(raw):
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ProtocolError(f"bad json: {e}") from e
    for field in ("type", "seq", "ts"):
        if field not in msg:
            raise ProtocolError(f"missing envelope field: {field}")
    if msg["type"] not in _ALL:
        raise ProtocolError(f"unknown type: {msg['type']}")
    return msg

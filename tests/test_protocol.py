import pytest

from omni import protocol as p


def test_make_roundtrip():
    msg = p.make("video", seq=3, ts=1.5, jpeg_b64="AAAA")
    assert msg == {"type": "video", "seq": 3, "ts": 1.5, "jpeg_b64": "AAAA"}
    assert p.decode(p.encode(msg)) == msg


def test_make_rejects_unknown_type():
    with pytest.raises(p.ProtocolError):
        p.make("bogus", seq=1, ts=0.0)


def test_decode_rejects_missing_envelope_fields():
    with pytest.raises(p.ProtocolError):
        p.decode('{"type": "video"}')  # no seq/ts


def test_direction_sets_are_disjoint_and_known():
    assert "session_start" in p.UPLINK_TYPES
    assert "capacity_exhausted" in p.DOWNLINK_TYPES
    assert p.UPLINK_TYPES.isdisjoint(p.DOWNLINK_TYPES)

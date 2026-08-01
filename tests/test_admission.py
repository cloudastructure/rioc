from omni.admission import Admission


def test_grants_up_to_capacity_then_refuses():
    a = Admission(capacity=1)
    s1 = a.try_acquire()
    assert s1 is not None
    assert a.try_acquire() is None      # full -> capacity_exhausted upstream
    assert a.in_use == 1


def test_release_frees_a_slot():
    a = Admission(capacity=1)
    s1 = a.try_acquire()
    a.release(s1)
    assert a.in_use == 0
    assert a.try_acquire() is not None


def test_release_unknown_slot_is_noop():
    a = Admission(capacity=1)
    a.release("nope")
    assert a.in_use == 0

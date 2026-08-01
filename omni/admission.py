"""Fixed-K admission control for live omni sessions.

Grants up to `capacity` concurrent sessions; returns None when full so the
server can send `capacity_exhausted`. No queueing, no autoscaling, no fallback.
"""
import itertools


class Admission:
    def __init__(self, capacity):
        self.capacity = capacity
        self._slots = set()
        self._ids = itertools.count()

    def try_acquire(self):
        if len(self._slots) >= self.capacity:
            return None
        slot = f"slot-{next(self._ids)}"
        self._slots.add(slot)
        return slot

    def release(self, slot_id):
        self._slots.discard(slot_id)

    @property
    def in_use(self):
        return len(self._slots)

"""Session Router — control-plane allocation across a static GPU pool.

Allocates the least-loaded healthy GPU for a new live conversation and returns
that GPU's DIRECT WebSocket URL — media never proxies through the router. Fleet
ceiling K = sum of per-GPU capacities; `allocate` returns None (→ caller sends
capacity_exhausted) only when the whole pool is full.

This module is the pure allocation core. The HTTP `POST /allocate` wrapper,
omni-server occupancy callbacks, and lease-expiry reclamation are added at
deploy time (see design §4.1).
"""
import itertools


class GpuRef:
    def __init__(self, ws_url, capacity):
        self.ws_url = ws_url
        self.capacity = capacity
        self.in_use = 0
        self.healthy = True


class Router:
    def __init__(self, gpus, *, lease_ttl=30.0, token_factory=None):
        self._gpus = list(gpus)
        self.lease_ttl = lease_ttl
        self._ids = itertools.count()
        self._token_factory = token_factory or (lambda: f"tok-{next(self._ids)}")
        self._leases = {}

    def allocate(self, camera_id, now=0.0):
        candidates = [g for g in self._gpus if g.healthy and g.in_use < g.capacity]
        if not candidates:
            return None
        gpu = min(candidates, key=lambda g: g.in_use)  # least-loaded (ties -> first)
        gpu.in_use += 1
        token = self._token_factory()
        self._leases[token] = (gpu, now + self.lease_ttl)
        return {"gpu_ws_url": gpu.ws_url, "session_token": token, "lease_ttl": self.lease_ttl}

    def release(self, session_token):
        entry = self._leases.pop(session_token, None)
        if entry:
            gpu, _expires = entry
            if gpu.in_use > 0:
                gpu.in_use -= 1

    def touch(self, session_token, now):
        """Renew a lease (call on heartbeat / omni-server occupancy report)."""
        entry = self._leases.get(session_token)
        if entry:
            gpu, _expires = entry
            self._leases[session_token] = (gpu, now + self.lease_ttl)

    def reap(self, now):
        """Release any lease whose expiry has passed (reclaims slots leaked by a crashed session)."""
        expired = [t for t, (_gpu, exp) in self._leases.items() if exp <= now]
        for t in expired:
            self.release(t)
        return len(expired)

    def set_health(self, ws_url, healthy):
        for g in self._gpus:
            if g.ws_url == ws_url:
                g.healthy = healthy

    @property
    def fleet_capacity(self):
        return sum(g.capacity for g in self._gpus)

    @property
    def fleet_in_use(self):
        return sum(g.in_use for g in self._gpus)

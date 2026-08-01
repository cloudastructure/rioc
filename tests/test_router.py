from omni.router import GpuRef, Router


def _pool():
    return Router([GpuRef("wss://g1", 2), GpuRef("wss://g2", 2)],
                  token_factory=lambda: "tok")


def test_allocate_prefers_least_loaded():
    r = _pool()
    assert r.allocate("c")["gpu_ws_url"] == "wss://g1"   # tie -> first
    assert r.allocate("c")["gpu_ws_url"] == "wss://g2"   # g1 now busier
    assert r.fleet_in_use == 2


def test_capacity_exhausted_when_pool_full():
    r = Router([GpuRef("wss://g1", 1)], token_factory=lambda: "tok")
    assert r.allocate("c") is not None
    assert r.allocate("c") is None                       # whole pool full


def test_release_frees_slot():
    r = Router([GpuRef("wss://g1", 1)])
    tok = r.allocate("c")["session_token"]
    r.release(tok)
    assert r.fleet_in_use == 0


def test_unhealthy_gpu_excluded():
    r = _pool()
    r.set_health("wss://g1", False)
    assert r.allocate("c")["gpu_ws_url"] == "wss://g2"
    assert r.allocate("c")["gpu_ws_url"] == "wss://g2"
    assert r.allocate("c") is None                       # g2 full, g1 unhealthy


def test_fleet_capacity_sums_per_gpu():
    r = Router([GpuRef("wss://g1", 2), GpuRef("wss://g2", 3)])
    assert r.fleet_capacity == 5

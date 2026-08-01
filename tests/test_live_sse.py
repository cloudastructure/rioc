import asyncio

from live.live_sse import (
    broadcast_live,
    listener_count,
    register_live_listener,
    unregister_live_listener,
)


def test_broadcast_reaches_registered_listener():
    async def body():
        q = register_live_listener()
        broadcast_live({"type": "state", "state": "WARNING"})
        event = await q.get()
        assert event["state"] == "WARNING"
        unregister_live_listener(q)
        assert listener_count() == 0

    asyncio.run(body())


def test_full_queue_drops_without_blocking():
    async def body():
        q = register_live_listener(maxsize=1)
        broadcast_live({"a": 1})
        broadcast_live({"a": 2})  # queue full -> dropped, no raise
        assert q.qsize() == 1
        unregister_live_listener(q)

    asyncio.run(body())

#!/usr/bin/env python3
"""Benchmark concurrent live sessions against the omni streaming server to set LIVE_MAX_SESSIONS.

Opens N concurrent streaming sessions, drives each (video+audio -> user_speech_end -> await first
speech chunk), and reports time-to-first-speech latency per concurrency level. Ramp N up until p95
first-speech latency crosses your real-time budget; the largest N still under budget is the per-GPU
LIVE_MAX_SESSIONS.

Usage:
  python scripts/omni_benchmark.py --url ws://GPU_HOST:8102/omni/session --levels 1,2,4,8
"""
import argparse
import asyncio
import base64
import json
import statistics
import time


def _msg(mtype, seq, **payload):
    return json.dumps({"type": mtype, "seq": seq, "ts": time.monotonic(), **payload})


async def _one_session(url, frames=3, insecure=False):
    import ssl
    import websockets
    ssl_ctx = None
    if url.startswith("wss"):
        ssl_ctx = ssl.create_default_context()
        if insecure:
            ssl_ctx.check_hostname = False
            ssl_ctx.verify_mode = ssl.CERT_NONE
    async with websockets.connect(url, max_size=None, ssl=ssl_ctx) as ws:
        await ws.send(_msg("session_start", 0, system_prompt="benchmark", voice="default"))
        ready = json.loads(await ws.recv())
        if ready.get("type") != "session_ready":
            return None  # capacity_exhausted or error
        blank = base64.b64encode(b"\x00" * 1024).decode()
        for i in range(frames):
            await ws.send(_msg("video", i + 1, jpeg_b64=blank))
            await ws.send(_msg("audio", i + 1, pcm_or_opus_b64=blank))
        t_send = time.monotonic()
        await ws.send(_msg("user_speech_end", 99))
        while True:
            m = json.loads(await ws.recv())
            if m.get("type") == "speech":
                return time.monotonic() - t_send
            if m.get("type") in ("error", "turn_end"):
                return None


async def _level(url, n, insecure=False):
    results = await asyncio.gather(*[_one_session(url, insecure=insecure) for _ in range(n)], return_exceptions=True)
    latencies = [r for r in results if isinstance(r, float)]
    rejected = sum(1 for r in results if r is None)
    return latencies, rejected


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="ws://host:port/omni/session")
    ap.add_argument("--levels", default="1,2,4,8")
    ap.add_argument("--budget-ms", type=float, default=1000.0, help="real-time first-speech budget")
    ap.add_argument("--insecure", action="store_true", help="skip TLS verification (self-signed wss)")
    args = ap.parse_args()

    print(f"{'N':>4}  {'ok':>4}  {'rej':>4}  {'p50_ms':>8}  {'p95_ms':>8}  verdict")
    for n in [int(x) for x in args.levels.split(",")]:
        latencies, rejected = await _level(args.url, n, insecure=args.insecure)
        if not latencies:
            print(f"{n:>4}  {0:>4}  {rejected:>4}  {'-':>8}  {'-':>8}  all rejected/failed")
            continue
        p50 = statistics.median(latencies) * 1000
        p95 = (sorted(latencies)[int(len(latencies) * 0.95) - 1] if len(latencies) > 1 else latencies[0]) * 1000
        verdict = "OK" if p95 <= args.budget_ms else "OVER BUDGET"
        print(f"{n:>4}  {len(latencies):>4}  {rejected:>4}  {p50:>8.0f}  {p95:>8.0f}  {verdict}")


if __name__ == "__main__":
    asyncio.run(main())

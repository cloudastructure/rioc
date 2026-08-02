#!/usr/bin/env bash
# Run the test suite in two passes.
#
# The Starlette TestClient websocket tests deadlock on event-loop teardown when they run in the
# SAME process as the asyncio.run-based unit tests, so they get their own pytest invocation.
# Each pass is green on its own; this script is the single entry point for CI.
set -e
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"

echo "== pass 1: unit tests (no Starlette TestClient) =="
$PY -m pytest tests/ \
    --ignore=tests/test_omni_server.py \
    --ignore=tests/test_router_http.py \
    --ignore=tests/integration \
    -q -p no:asyncio

echo "== pass 2: TestClient websocket tests =="
$PY -m pytest tests/test_omni_server.py tests/test_router_http.py -q -p no:asyncio

echo "All passes green."

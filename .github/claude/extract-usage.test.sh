#!/usr/bin/env bash
# Tests for extract-usage.sh, against both execution-file shapes and against
# the failures that must never take a posted review down with them.
#
# Runnable alone:
#   bash .github/claude/extract-usage.test.sh
#
# The two fixtures:
#
#   fixtures/claude-result.json — recorded verbatim from the local CLI
#     (`claude -p --output-format json`, 2.1.258, 2026-09-02). It is the single
#     result-object shape. The run was not authenticated, so its counts are all
#     zero; the fixture is here for the FIELD NAMES and the shape, and case 2
#     asserts exactly that.
#   fixtures/claude-execution-output.json — the array-of-messages shape the
#     action writes (base-action/src/execution-file.ts:15-26, whose `messages`
#     come from run-claude-sdk.ts:186,191,222). Built from the field names in
#     the recorded fixture, with counts chosen so the result totals and the
#     per-turn sums differ — case 3 would pass on either if they matched.
#
# Negative controls:
#   case 4 fails if a garbage file is allowed to abort the step
#   case 5 fails if a zero count is written as `unknown` (or the reverse)
set -uo pipefail

if ! command -v jq > /dev/null 2>&1; then
  echo "jq is required for this test (macOS: brew install jq)" >&2
  exit 1
fi

HERE=$(cd "$(dirname "$0")" && pwd)
SCRIPT="${SCRIPT:-$HERE/extract-usage.sh}"
FIXTURES="$HERE/fixtures"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

export REVIEW_OUT="$WORK/out"
mkdir -p "$REVIEW_OUT"

pass=0
fail=0
check() { # check <name> <actual> <expected>
  if [ "$2" = "$3" ]; then
    echo "  PASS: $1"
    pass=$((pass + 1))
  else
    echo "  FAIL: $1 (expected '$3', got '$2')"
    fail=$((fail + 1))
  fi
}

run() { # run <label> <execution file>
  rm -f "$REVIEW_OUT/usage-$1.json"
  EXECUTION_FILE="$2" bash "$SCRIPT" "$1" > "$WORK/log.txt" 2>&1
  echo "$?"
}
field() { jq -r ".$2" "$REVIEW_OUT/usage-$1.json"; }

echo "== 1. The array shape reads the result message, not the turn sum =="
# The result message is authoritative: the SDK's per-turn numbers double-count
# nothing but also do not cover turns the stream dropped.
STATUS=$(run review "$FIXTURES/claude-execution-output.json")
check "exits 0" "$STATUS" "0"
check "the file is where the ledger looks for it" \
  "$([ -f "$REVIEW_OUT/usage-review.json" ] && echo yes || echo no)" "yes"
check "input comes from the result message" "$(field review input)" "1967"
check "output too" "$(field review output)" "1616"
check "cache reads too" "$(field review cache_read)" "22721"
check "cache writes too" "$(field review cache_write)" "23990"
check "turns is num_turns, not the assistant-message count" \
  "$(field review turns)" "7"
check "the model comes from modelUsage" \
  "$(field review model)" "claude-opus-5"

echo "== 2. The single result object shape reads the same fields =="
# Recorded from the CLI. Unauthenticated, so every count is a real zero — which
# is exactly what case 5 says must not be rewritten to `unknown`.
STATUS=$(run verify "$FIXTURES/claude-result.json")
check "exits 0" "$STATUS" "0"
check "the label picks the filename" \
  "$([ -f "$REVIEW_OUT/usage-verify.json" ] && echo yes || echo no)" "yes"
check "input is the recorded zero" "$(field verify input)" "0"
check "turns is the recorded num_turns" "$(field verify turns)" "1"
check "an empty modelUsage falls through to unknown" \
  "$(field verify model)" "unknown"

echo "== 3. A stream cut off before the result falls back to the turns =="
# A run killed at the step cap still spent tokens, and the ledger is the only
# place that spend is recorded.
jq 'map(select(.type != "result"))' \
  "$FIXTURES/claude-execution-output.json" > "$WORK/no-result.json"
STATUS=$(run review "$WORK/no-result.json")
check "exits 0" "$STATUS" "0"
check "input is the sum of the assistant turns" "$(field review input)" "1967"
check "output is the sum too" "$(field review output)" "1616"
check "cache writes are the sum" "$(field review cache_write)" "23990"
check "turns is the assistant-message count" "$(field review turns)" "2"
check "the model comes from the last assistant message" \
  "$(field review model)" "claude-opus-5"

echo "== 4. Nothing here can fail a review that already posted =="
printf 'not json {{{\n' > "$WORK/garbage.json"
STATUS=$(run review "$WORK/garbage.json")
check "garbage exits 0" "$STATUS" "0"
check "and records unknown" "$(field review input)" "unknown"
check "with a warning, not an error" \
  "$(grep -c '::warning::' "$WORK/log.txt")" "1"
check "and no ::error::" "$(grep -c '::error::' "$WORK/log.txt")" "0"

STATUS=$(run review "$WORK/no-such-file.json")
check "a missing file exits 0" "$STATUS" "0"
check "and records unknown" "$(field review turns)" "unknown"

STATUS=$(EXECUTION_FILE="" bash "$SCRIPT" review > "$WORK/log.txt" 2>&1; echo $?)
check "an unset EXECUTION_FILE exits 0" "$STATUS" "0"
check "and records unknown" "$(field review model)" "unknown"

printf '[{"type":"system"}]\n' > "$WORK/no-usage.json"
STATUS=$(run review "$WORK/no-usage.json")
check "a file with no result or assistant message exits 0" "$STATUS" "0"
check "and records unknown" "$(field review input)" "unknown"

echo "== 5. A real zero is not rewritten as unknown, nor unknown as zero =="
# The ledger totals only the rows it can read and marks the rest a lower bound,
# so conflating the two under-reports the seat's usage in one direction and
# invents spend in the other.
jq '{type: "result", num_turns: 3,
     usage: {input_tokens: 0, output_tokens: 0,
             cache_read_input_tokens: 0, cache_creation_input_tokens: 0}}' \
  <<< '{}' > "$WORK/real-zero.json"
run review "$WORK/real-zero.json" > /dev/null
check "a zero count stays 0" "$(field review input)" "0"
check "and is a number, not a string" \
  "$(jq -r '.input | type' "$REVIEW_OUT/usage-review.json")" "number"

jq '{type: "result", num_turns: 3, usage: {output_tokens: 5}}' <<< '{}' \
  > "$WORK/partial.json"
run review "$WORK/partial.json" > /dev/null
check "a missing count is unknown, not 0" "$(field review input)" "unknown"
check "while the count that is there survives" "$(field review output)" "5"

echo "== 6. An interrupted stream never invents missing counts =="
printf '%s\n' '[{"type":"assistant","message":{"model":"m","usage":{"output_tokens":5}}}]' > "$WORK/partial-stream.json"
run review "$WORK/partial-stream.json" > /dev/null
check "missing input stays unknown" "$(field review input)" "unknown"
check "reported output survives" "$(field review output)" "5"
check "missing cache stays unknown" "$(field review cache_read)" "unknown"

echo
echo "passed: $pass  failed: $fail"
[ "$fail" -eq 0 ]

#!/usr/bin/env bash
# Pull one run's token usage out of the action's execution file and write it
# where update-usage-ledger.sh can read it.
#
# Shape, with receipts read 2026-09-02:
#
#   anthropics/claude-code-action, base-action/src/execution-file.ts:6,15-26 —
#   the action writes `JSON.stringify(messages, null, 2)` to
#   $RUNNER_TEMP/claude-execution-output.json and exposes that path as the
#   `execution_file` output (execution-file.ts:34-41). `messages` is the SDK
#   message stream (run-claude-sdk.ts:186,191,222), whose last element is the
#   `type: "result"` message carrying `usage` and `num_turns`
#   (run-claude-sdk.ts:140-151,198-199).
#
#   The same result object is what `claude -p --output-format json` prints, so
#   a file holding one object rather than an array is the other shape this has
#   to read. Recorded from the local CLI (2.1.258) into
#   fixtures/claude-result.json: `usage.input_tokens`,
#   `usage.output_tokens`, `usage.cache_read_input_tokens`,
#   `usage.cache_creation_input_tokens`, `num_turns`, `modelUsage`.
#
# Both shapes are handled, plus the case where the stream was cut off before a
# result message: there the assistant messages' own `message.usage` is summed
# instead, so a run killed at the step cap still reports what it spent.
#
# Never fatal. A review that posted fine must not be failed by an accounting
# step, so a missing or unreadable file writes `unknown` values and exits 0
# with a ::warning::. The ledger distinguishes unknown from zero.
#
# Env in: EXECUTION_FILE (path from the action's `execution_file` output),
#         REVIEW_OUT.
# Arg 1:  label — `review` or `verify`.
# Writes: $REVIEW_OUT/usage-<label>.json
#         {"input","output","cache_read","cache_write","turns","model"}
set -euo pipefail

: "${REVIEW_OUT:?REVIEW_OUT is required}"

LABEL="${1:-review}"
case "$LABEL" in
  *[!a-z-]* | '') echo "::error::usage label '$LABEL' must be lowercase letters and dashes"; exit 1 ;;
esac
OUT="$REVIEW_OUT/usage-$LABEL.json"

write_unknown() {
  printf '{"input":"unknown","output":"unknown","cache_read":"unknown","cache_write":"unknown","turns":"unknown","model":"unknown"}\n' \
    > "$OUT"
  echo "::warning::$1; recording the $LABEL pass as unknown usage"
  exit 0
}

FILE="${EXECUTION_FILE:-}"
if [ -z "$FILE" ] || [ ! -s "$FILE" ]; then
  write_unknown "No execution file at '${FILE:-<unset>}'"
fi
if ! jq -e . "$FILE" > /dev/null 2>&1; then
  write_unknown "The execution file at $FILE is not readable JSON"
fi

# One filter for both shapes: normalise to an array, prefer the last result
# message, fall back to summing the assistant turns. A field that is not a
# number stays "unknown" rather than becoming 0 — an unread count and a real
# zero are different facts, and conflating them under-reports the seat's usage.
if ! jq -e '
  def num(x): if (x | type) == "number" and x >= 0 and x == (x | floor)
              then x else "unknown" end;
  def sum_usage(k):
    [ .[] | num(.message.usage[k]) ]
    | if any(. == "unknown") then "unknown" else add end;
  ( if type == "array" then . else [.] end ) as $msgs
  | ( [ $msgs[] | select((.type? // "") == "result") ] | last ) as $result
  | ( [ $msgs[] | select((.type? // "") == "assistant") ] ) as $turns
  | if $result != null and ($result.usage | type) == "object" then
      {
        input:       num($result.usage.input_tokens),
        output:      num($result.usage.output_tokens),
        cache_read:  num($result.usage.cache_read_input_tokens),
        cache_write: num($result.usage.cache_creation_input_tokens),
        turns:       num($result.num_turns),
        model:       ( ($result.modelUsage // {} | keys | first)
                       // ( [$turns[] | .message?.model? // empty] | last )
                       // "unknown" )
      }
    elif ($turns | length) > 0 then
      {
        input:       ( $turns | sum_usage("input_tokens") ),
        output:      ( $turns | sum_usage("output_tokens") ),
        cache_read:  ( $turns | sum_usage("cache_read_input_tokens") ),
        cache_write: ( $turns | sum_usage("cache_creation_input_tokens") ),
        turns:       ( $turns | length ),
        model:       ( ( [$turns[] | .message?.model? // empty] | last ) // "unknown" )
      }
    else
      empty
    end' "$FILE" > "$OUT" 2>/dev/null; then
  write_unknown "The execution file at $FILE carries no result or assistant message"
fi

echo "Usage for the $LABEL pass: $(cat "$OUT")"

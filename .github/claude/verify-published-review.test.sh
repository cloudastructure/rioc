#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
cat > "$WORK/gh" <<'STUB'
#!/usr/bin/env bash
[ "${FAIL_LIST:-}" = true ] && exit 1
cat "$FIXTURE"
STUB
chmod +x "$WORK/gh"
export PATH="$WORK:$PATH" FIXTURE="$WORK/comments.json" GITHUB_OUTPUT="$WORK/output"
export MODEL_OUTCOME=success CAPTURE_OUTCOME=success PUBLISH_OUTCOME=success
export GITHUB_REPOSITORY=acme/repo PR_NUMBER=1 HEAD_SHA=123456789abcdef RUN_STARTED_AT=2026-09-05T00:00:00Z IS_PR=true
pass=0
run() {
  : > "$GITHUB_OUTPUT"
  if bash "$HERE/verify-published-review.sh" > "$WORK/log" 2>&1; then result=0; else result=$?; fi
}
check() {
  [ "$result" -eq "$1" ] || { cat "$WORK/log"; echo "FAIL: $2"; exit 1; }
  if [ "$1" != 0 ]; then grep -q '^response_missing=true$' "$GITHUB_OUTPUT"; fi
  pass=$((pass + 1))
}
scored() {
  jq -n --arg body "<!-- claude-sticky-review -->
<!-- score: 5 -->
$HEAD_SHA" '[{user:{login:"claude[bot]"},updated_at:"2026-09-05T00:01:00Z",body:$body}]' > "$FIXTURE"
}
scored
run; check 0 'fresh published review'
PUBLISH_OUTCOME=failure run; check 1 'failed publisher cannot reuse a fresh-looking comment'
CAPTURE_OUTCOME=failure run; check 1 'missing model output is not a workflow guard skip'
FAIL_LIST=true run; check 1 'listing failure is not success'
jq '.[0].updated_at = "2026-09-04T00:01:00Z"' "$FIXTURE" > "$WORK/old"
mv "$WORK/old" "$FIXTURE"
run; check 1 'old review cannot satisfy current run'
scored
HEAD_SHA=999999999999999 run; check 1 'other commit cannot satisfy current run'
scored
jq '.[0].user.login = "github-actions[bot]"' "$FIXTURE" > "$WORK/old"
mv "$WORK/old" "$FIXTURE"
run; check 1 'history from another publisher is not this response'
printf '[]\n' > "$FIXTURE"
run; check 1 'empty listing'
printf 'malformed\n' > "$FIXTURE"
run; check 1 'invalid listing'
printf '%s\n' '[{"user":{"login":"claude[bot]"},"updated_at":"2026-09-05T00:01:00Z","body":"Here is the answer"}]' > "$FIXTURE"
IS_PR=false run; check 0 'plain issue answer has no score or SHA requirement'
printf 'passed: %s  failed: 0\n' "$pass"

#!/usr/bin/env bash
# Execute the gate against outcome combinations; no GitHub or network calls.
# Run: bash .github/claude/review-outcome-gate.test.sh
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
SCRIPT="${SCRIPT:-$HERE/review-outcome-gate.sh}"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
SHA="1234567890abcdef1234567890abcdef12345678"
pass=0
fail=0

# '-' represents an empty input. Compare the entire output file, so errors
# cannot leak infra_failure/head_sha and healthy runs cannot report failures.
while IFS='|' read -r name mode model capture publisher missing score head status message outputs; do
  : > "$WORK/output"
  GATE_MODE="${mode#-}" MODEL_OUTCOME="${model#-}" CAPTURE_OUTCOME="${capture#-}" \
    PUBLISHER_OUTCOME="${publisher#-}" REVIEW_MISSING="${missing#-}" \
    SCORE_MISSING="${score#-}" HEAD_SHA="${head:+${SHA}}" \
    GITHUB_OUTPUT="$WORK/output" bash "$SCRIPT" > "$WORK/message" 2>&1
  actual_status=$?
  actual_message=$(< "$WORK/message")
  actual_outputs=$(< "$WORK/output")
  expected_outputs=""
  if [ "$outputs" = "infra" ]; then
    expected_outputs=$(printf 'infra_failure=true\nhead_sha=%s' "$SHA")
  fi
  case "$message" in
    auth) expected_message="::error::Claude publisher authentication did not succeed" ;;
    head) expected_message="::error::Reviewer setup/model did not capture a commit (HEAD_SHA is unset); cannot report a neutral review-status check" ;;
    infra) expected_message="Infrastructure failure — neutral review-status check required" ;;
    structural) expected_message="::error::Claude review or readable score is missing" ;;
    response) expected_message="::error::Claude model succeeded but the review response is missing" ;;
    mode) expected_message="::error::Invalid GATE_MODE '${mode#-}'; expected auto or mention" ;;
    *) expected_message="" ;;
  esac
  if [ "$actual_status" = "$status" ] && [ "$actual_message" = "$expected_message" ] \
    && [ "$actual_outputs" = "$expected_outputs" ]; then
    echo "  PASS: $name"
    pass=$((pass + 1))
  else
    echo "  FAIL: $name"
    printf '    exit: expected %s, got %s\n    message: expected <%s>, got <%s>\n    outputs: expected <%s>, got <%s>\n' \
      "$status" "$actual_status" "$expected_message" "$actual_message" "$expected_outputs" "$actual_outputs"
    fail=$((fail + 1))
  fi
done <<'CASES'
failed auth precedes absent HEAD|auto|failure|skipped|failure|true|true||1|auth|
skipped auth precedes absent HEAD|auto|skipped|skipped|skipped|true|true||1|auth|
empty auth precedes absent HEAD|auto|-|-|-|-|-||1|auth|
failed auth with HEAD stays an error|auto|failure|failure|failure|true|true|head|1|auth|
skipped auth with healthy model|auto|success|success|skipped|false|false|head|1|auth|
empty auth with healthy model|auto|success|success|-|false|false|head|1|auth|
provider failure is neutral|auto|failure|skipped|success|false|false|head|0|infra|infra
capture failure is neutral|auto|success|failure|success|true|true|head|0|infra|infra
skipped model is neutral with HEAD|auto|skipped|skipped|success|-|-|head|0|infra|infra
empty model is neutral with HEAD|auto|-|-|success|-|-|head|0|infra|infra
skipped capture is neutral with HEAD|auto|success|skipped|success|-|-|head|0|infra|infra
empty capture is neutral with HEAD|auto|success|-|success|-|-|head|0|infra|infra
provider failure without HEAD is setup error|auto|failure|skipped|success|true|true||1|head|
capture failure without HEAD is setup error|auto|success|failure|success|true|true||1|head|
missing review is structural|auto|success|success|success|true|false|head|1|structural|
missing score is structural|auto|success|success|success|false|true|head|1|structural|
healthy auto|auto|success|success|success|false|false|head|0||
healthy auto with unset missing flags|auto|success|success|success|-|-|head|0||
mention provider failure preserves behavior|mention|failure|failure|failure|true|true||0||
mention skipped model preserves behavior|mention|skipped|skipped|skipped|true|true||0||
mention empty model preserves behavior|mention|-|-|-|true|true||0||
mention capture failure with missing response|mention|success|failure|success|true|false|head|1|response|
mention publish failure with missing response|mention|success|success|failure|true|false|head|1|response|
healthy mention|mention|success|success|success|false|false|head|0||
mention only gates on response after model success|mention|success|failure|failure|false|true||0||
invalid mode|invalid|success|success|success|false|false|head|1|mode|
empty mode|-|-|-|-|-|-||1|mode|
CASES

echo "passed: $pass  failed: $fail"
[ "$fail" -eq 0 ]

#!/usr/bin/env bash
# Tests for update-progress.sh. The script runs top-level against `gh`, so `gh`
# is stubbed on PATH and records every call it is given; jq and grep are the
# real ones, as in the script.
#
# Runnable alone:
#   bash .github/claude/update-progress.test.sh
#
# What the cases defend, and the failure each was observed to catch when run
# against the pre-fix script (2026-09-02):
#
#   1  The comment listing was read as `gh api ... 2>/dev/null | jq ... || echo
#      ""`, so a failed read produced an empty score, which is the same state
#      as "the review posted no score": the indicator was DELETED and
#      score_missing=true told the workflow to report a structurally malformed
#      review for a review that had posted fine.
#   2  The success path still PATCHes the indicator with the stars and score.
#   3  Canonical review selection follows the publisher: trusted authors, a
#      leading sticky marker, updated_at, then id; an unmarked duplicate is
#      only the fallback when no marked review exists.
#   4  A review from another HEAD_SHA or before RUN_STARTED_AT is never
#      announced as this run's result.
#   5  A failed plain-issue request reports its real phase without presenting
#      the request as a PR review or consulting an earlier score.
#
# GNU grep is required: the script parses with `grep -oP`.
set -uo pipefail

if ! command -v jq > /dev/null 2>&1; then
  echo "jq is required for this test (macOS: brew install jq)" >&2
  exit 1
fi
if ! echo x | grep -qP x 2> /dev/null; then
  echo "GNU grep (grep -P) is required for this test; the script uses grep -oP" >&2
  exit 1
fi

HERE=$(cd "$(dirname "$0")" && pwd)
SCRIPT="${SCRIPT:-$HERE/update-progress.sh}"

STUB=$(mktemp -d)
trap 'rm -rf "$STUB"' EXIT

# gh stub: logs the verb and target of every call, serves the comment listing
# from $STUB_DIR/comments.json, and fails that listing when STUB_LIST_FAILS is
# set — the transient read error the script has to keep separate from "no
# score".
cat > "$STUB/gh" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$STUB_DIR/calls.log"
case "$*" in
  *--paginate*/comments*)
    [ -n "${STUB_LIST_FAILS:-}" ] && exit 1
    cat "$STUB_DIR/comments.json"
    exit 0
    ;;
esac
exit 0
EOF
chmod +x "$STUB/gh"

export STUB_DIR="$STUB"
export PATH="$STUB:$PATH"
export GITHUB_REPOSITORY="acme/repo" GH_TOKEN="x"
export PROGRESS_COMMENT_ID="99" PR_NUMBER="188" REVIEW_NUMBER="3"
export SHORT_SHA="8c66408" HEAD_SHA="8c66408deadbeefdeadbeefdeadbeefdeadbeef1"
export RUN_STARTED_AT="2026-09-05T00:00:00Z"
export GITHUB_RUN_ID="42" PROGRESS_VERB="Review"

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

OUTF="$STUB/github_output"
LOG="$STUB/calls.log"

run_update() { # run_update [outcome] [response_missing] [review_incomplete]
  local outcome="${1:-success}" response_missing="${2:-}" review_incomplete="${3:-}"
  : > "$OUTF"
  : > "$LOG"
  GITHUB_OUTPUT="$OUTF" REVIEW_OUTCOME="$outcome" RESPONSE_MISSING="$response_missing" \
    REVIEW_INCOMPLETE="$review_incomplete" \
    bash "$SCRIPT" > /dev/null 2>&1
  echo "$?"
}
calls_matching() { grep -F -c -- "$1" "$LOG"; }

# Build one GitHub comment object. The optional marker is deliberately the
# first line so the fixtures can distinguish a real leading marker from a
# quoted decoy.
review_comment() { # review_comment <id> <login> <score> <marker> <created> <updated> <sha> <label>
  local id="$1" login="$2" score="$3" marker="$4" created="$5" updated="$6" sha="$7" label="$8" body
  if [ -n "$marker" ]; then
    body=$(printf '%s\n<!-- score: %s -->\n## 🔍 Review #3\n\n%s\nAnalyzed %s\n' \
      "$marker" "$score" "$label" "$sha")
  else
    body=$(printf '<!-- score: %s -->\n## 🔍 Review #3\n\n%s\nAnalyzed %s\n' \
      "$score" "$label" "$sha")
  fi
  jq -n --argjson id "$id" --arg login "$login" --arg body "$body" \
    --arg created "$created" --arg updated "$updated" \
    '{id:$id, created_at:$created, updated_at:$updated,
      user:{login:$login}, body:$body}'
}

comments_from() {
  jq -s '.' "$@"
}

# A current sticky review carrying a readable score.
scored() { # scored <n>
  comments_from <(review_comment 7 github-actions[bot] "$1" \
    "<!-- claude-sticky-review -->" "2026-09-05T00:01:00Z" \
    "2026-09-05T00:01:00Z" "$HEAD_SHA" "Some findings.")
}

legacy_scored() { # legacy_scored <n>
  comments_from <(review_comment 7 claude[bot] "$1" "" \
    "2026-09-05T00:01:00Z" "2026-09-05T00:01:00Z" "$HEAD_SHA" \
    "Legacy review.")
}

echo "== 1. A failed comment listing is an infra failure, not a missing score =="
scored 4 > "$STUB/comments.json"
STATUS=$(STUB_LIST_FAILS=1 run_update)
check "exits non-zero" "$([ "$STATUS" != "0" ] && echo yes || echo no)" "yes"
check "does not delete the indicator" "$(calls_matching 'DELETE')" "0"
check "does not PATCH an outcome onto the indicator" "$(calls_matching 'PATCH')" "0"
check "does not set score_missing" "$(grep -c 'score_missing=true' "$OUTF")" "0"

echo "== 2. Failure and missing-response outcomes retain their early exits =="
scored 4 > "$STUB/comments.json"
STATUS=$(run_update failure)
check "a reviewer failure exits zero" "$STATUS" "0"
check "a reviewer failure PATCHes the indicator" \
  "$(calls_matching 'issues/comments/99 -X PATCH')" "1"
check "a reviewer failure does not announce a score" "$(calls_matching '⭐⭐⭐⭐ 4/5')" "0"

check "failure explicitly says no review completed" "$(calls_matching 'No review completed')" "1"
check "failure never invents an API error" "$(calls_matching 'API error or timeout')" "0"
check "failure explains the old score" "$(calls_matching 'Any earlier score belongs to an earlier review')" "1"
# shellcheck disable=SC2016
check "failure offers only an explicit manual retry" "$(calls_matching 'comment `@claude`')" "1"

for scenario in setup startup execution capture publish; do
  export MODEL_OUTCOME=success CAPTURE_OUTCOME=success PUBLISH_OUTCOME=success MODEL_EXECUTION_FILE=""
  case "$scenario" in
    setup) MODEL_OUTCOME=skipped; expected='Reviewer setup failed' ;;
    startup) MODEL_OUTCOME=failure; expected='failed without producing execution data' ;;
    execution)
      MODEL_OUTCOME=failure
      MODEL_EXECUTION_FILE="$STUB/execution.json"
      printf '%s' '{"error":"private diagnostic text"}' > "$MODEL_EXECUTION_FILE"
      expected='failed during execution' ;;
    capture) CAPTURE_OUTCOME=failure; expected='output could not be captured' ;;
    publish) PUBLISH_OUTCOME=failure; expected='publication failed' ;;
  esac
  STATUS=$(run_update failure)
  check "$scenario has its own failure explanation" "$(calls_matching "$expected")" "1"
  check "$scenario exports the same explanation for the neutral check" "$(grep -c "$expected" "$OUTF")" "1"
  check "$scenario never lists or reuses old reviews" "$(calls_matching --paginate)" "0"
  check "$scenario never leaks execution contents" "$(calls_matching 'private diagnostic text')" "0"
done

STATUS=$(PROGRESS_COMMENT_ID='' run_update failure)
check "missing progress comment still exports failure reason" "$(grep -c failure_summary= "$OUTF")" "1"
check "missing progress comment does not attempt a write" "$(calls_matching PATCH)" "0"
unset MODEL_OUTCOME CAPTURE_OUTCOME PUBLISH_OUTCOME MODEL_EXECUTION_FILE

export MODEL_OUTCOME=failure CAPTURE_OUTCOME=skipped PUBLISH_OUTCOME=skipped
STATUS=$(IS_PR=false run_update failure)
check "a failed issue request exits zero" "$STATUS" "0"
check "an issue failure says the request failed" "$(calls_matching 'Request failed')" "1"
check "an issue failure identifies the execution phase" \
  "$(calls_matching 'Request processing failed without producing execution data')" "1"
check "an issue failure links the run logs" "$(calls_matching 'actions/runs/42')" "1"
# shellcheck disable=SC2016
check "an issue failure offers a manual retry" "$(calls_matching 'comment `@claude` to retry the request')" "1"
check "an issue failure does not call itself a review" "$(calls_matching 'review')" "0"
check "an issue failure does not mention approval" "$(calls_matching 'approval')" "0"
check "an issue failure does not mention an earlier score" "$(calls_matching 'score')" "0"
check "an issue failure never lists old scores" "$(calls_matching --paginate)" "0"
unset MODEL_OUTCOME CAPTURE_OUTCOME PUBLISH_OUTCOME

STATUS=$(run_update success true)
check "a missing response exits zero" "$STATUS" "0"
check "a missing response PATCHes the indicator" \
  "$(calls_matching 'issues/comments/99 -X PATCH')" "1"
check "a missing response does not announce a score" "$(calls_matching '⭐⭐⭐⭐ 4/5')" "0"

echo "== 3. A published review with incomplete verification stays non-final =="
scored 4 > "$STUB/comments.json"
STATUS=$(run_update success "" true)
check "an incomplete publication exits zero" "$STATUS" "0"
check "PATCHes the progress comment" \
  "$(calls_matching 'issues/comments/99 -X PATCH')" "1"
check "uses the explicit incomplete-verification notice" \
  "$(calls_matching 'Review published — verification incomplete')" "1"
check "includes the current SHA" "$(calls_matching "$HEAD_SHA")" "1"
check "includes the current run link" "$(calls_matching 'actions/runs/42')" "1"
check "does not claim complete" "$(calls_matching '✅ **Review')" "0"
check "does not announce stars" "$(calls_matching '⭐⭐⭐⭐ 4/5')" "0"
check "does not delete the indicator" "$(calls_matching 'DELETE')" "0"
check "does not set score_missing" "$(grep -c 'score_missing=true' "$OUTF")" "0"

echo "== 4. A listing carrying a current sticky score still PATCHes the indicator =="
scored 4 > "$STUB/comments.json"
STATUS=$(run_update)
check "exits zero" "$STATUS" "0"
check "PATCHes the progress comment" \
  "$(calls_matching 'issues/comments/99 -X PATCH')" "1"
check "with the stars and the score" "$(calls_matching '⭐⭐⭐⭐ 4/5')" "1"
check "does not delete the indicator" "$(calls_matching 'DELETE')" "0"
check "does not set score_missing" "$(grep -c 'score_missing=true' "$OUTF")" "0"

echo "== 5. A legacy Claude-authored score still completes the indicator =="
legacy_scored 5 > "$STUB/comments.json"
STATUS=$(run_update)
check "exits zero" "$STATUS" "0"
check "uses the legacy score" "$(calls_matching '⭐⭐⭐⭐⭐ 5/5')" "1"
check "does not delete the indicator" "$(calls_matching 'DELETE')" "0"

echo "== 6. Updated_at beats created_at among marked reviews =="
comments_from \
  <(review_comment 20 github-actions[bot] 4 "<!-- claude-sticky-review -->" \
    "2026-09-05T00:20:00Z" "2026-09-05T00:05:00Z" "$HEAD_SHA" \
    "Created later, updated earlier.") \
  <(review_comment 10 github-actions[bot] 5 "<!-- claude-sticky-review -->" \
    "2026-09-05T00:10:00Z" "2026-09-05T00:15:00Z" "$HEAD_SHA" \
    "Created earlier, updated later.") > "$STUB/comments.json"
STATUS=$(run_update)
check "exits zero" "$STATUS" "0"
check "uses the newest updated_at" "$(calls_matching '⭐⭐⭐⭐⭐ 5/5')" "1"
check "does not use created_at ordering" "$(calls_matching '⭐⭐⭐⭐ 4/5')" "0"

echo "== 7. A higher id wins an updated_at tie among marked reviews =="
comments_from \
  <(review_comment 8 github-actions[bot] 4 "<!-- claude-sticky-review -->" \
    "2026-09-05T00:20:00Z" "2026-09-05T00:20:00Z" "$HEAD_SHA" \
    "Lower id.") \
  <(review_comment 9 github-actions[bot] 5 "<!-- claude-sticky-review -->" \
    "2026-09-05T00:21:00Z" "2026-09-05T00:20:00Z" "$HEAD_SHA" \
    "Higher id.") > "$STUB/comments.json"
STATUS=$(run_update)
check "uses the higher id on a timestamp tie" "$(calls_matching '⭐⭐⭐⭐⭐ 5/5')" "1"
check "does not use the lower id" "$(calls_matching '⭐⭐⭐⭐ 4/5')" "0"

echo "== 8. The marked canonical review beats a newer unmarked duplicate =="
comments_from \
  <(review_comment 8 github-actions[bot] 4 "<!-- claude-sticky-review -->" \
    "2026-09-05T00:01:00Z" "2026-09-05T00:02:00Z" "$HEAD_SHA" \
    "Canonical sticky.") \
  <(review_comment 9 github-actions[bot] 5 "" \
    "2026-09-05T00:03:00Z" "2026-09-05T00:30:00Z" "$HEAD_SHA" \
    "Newer duplicate.") > "$STUB/comments.json"
STATUS=$(run_update)
check "uses the marked canonical score" "$(calls_matching '⭐⭐⭐⭐ 4/5')" "1"
check "does not use the duplicate score" "$(calls_matching '⭐⭐⭐⭐⭐ 5/5')" "0"

echo "== 9. A quoted sticky marker is a decoy, not a canonical marker =="
comments_from \
  <(review_comment 8 github-actions[bot] 4 "<!-- claude-sticky-review -->" \
    "2026-09-05T00:01:00Z" "2026-09-05T00:02:00Z" "$HEAD_SHA" \
    "Canonical sticky.") \
  <(review_comment 9 github-actions[bot] 5 "> <!-- claude-sticky-review -->" \
    "2026-09-05T00:03:00Z" "2026-09-05T00:30:00Z" "$HEAD_SHA" \
    "Quoted marker decoy.") > "$STUB/comments.json"
STATUS=$(run_update)
check "keeps the real leading marker canonical" "$(calls_matching '⭐⭐⭐⭐ 4/5')" "1"
check "ignores the quoted marker as canonical" "$(calls_matching '⭐⭐⭐⭐⭐ 5/5')" "0"

echo "== 10. The fallback chooses the latest trusted score without markers =="
comments_from \
  <(review_comment 8 claude[bot] 4 "" \
    "2026-09-05T00:01:00Z" "2026-09-05T00:02:00Z" "$HEAD_SHA" \
    "Older legacy score.") \
  <(review_comment 9 github-actions[bot] 5 "" \
    "2026-09-05T00:03:00Z" "2026-09-05T00:30:00Z" "$HEAD_SHA" \
    "Latest legacy score.") > "$STUB/comments.json"
STATUS=$(run_update)
check "uses the latest trusted fallback" "$(calls_matching '⭐⭐⭐⭐⭐ 5/5')" "1"
check "does not use the older fallback" "$(calls_matching '⭐⭐⭐⭐ 4/5')" "0"

echo "== 11. An untrusted score cannot become the canonical review =="
comments_from \
  <(review_comment 8 someone 1 "<!-- claude-sticky-review -->" \
    "2026-09-05T00:01:00Z" "2026-09-05T00:30:00Z" "$HEAD_SHA" \
    "Untrusted score.") \
  <(review_comment 9 github-actions[bot] 5 "" \
    "2026-09-05T00:03:00Z" "2026-09-05T00:20:00Z" "$HEAD_SHA" \
    "Trusted score.") > "$STUB/comments.json"
STATUS=$(run_update)
check "uses the trusted score" "$(calls_matching '⭐⭐⭐⭐⭐ 5/5')" "1"
check "does not use the untrusted score" "$(calls_matching '⭐ 1/5')" "0"

echo "== 12. A wrong SHA or stale run never announces its score =="
WRONG_SHA="1111111111111111111111111111111111111111"
comments_from \
  <(review_comment 8 github-actions[bot] 1 "<!-- claude-sticky-review -->" \
    "2026-09-05T00:10:00Z" "2026-09-05T00:20:00Z" "$WRONG_SHA" \
    "Wrong head SHA.") \
  <(review_comment 9 github-actions[bot] 5 "<!-- claude-sticky-review -->" \
    "2026-09-04T23:50:00Z" "2026-09-04T23:55:00Z" "$HEAD_SHA" \
    "Stale run.") > "$STUB/comments.json"
STATUS=$(run_update)
check "exits zero after finding no eligible score" "$STATUS" "0"
check "does not announce the wrong SHA score" "$(calls_matching '⭐ 1/5')" "0"
check "does not announce the stale score" "$(calls_matching '⭐⭐⭐⭐⭐ 5/5')" "0"
check "deletes the indicator instead" "$(calls_matching 'DELETE')" "1"
check "sets score_missing" "$(grep -c 'score_missing=true' "$OUTF")" "1"

echo
echo "passed: $pass  failed: $fail"
[ "$fail" -eq 0 ]

#!/usr/bin/env bash
# Tests for prepare-mention-metadata.sh and prepare-review-metadata.sh.
# Both scripts run top-level against `gh`, so `gh` is stubbed on PATH and the
# event payload is a fixture file; jq is the real one, as in the scripts.
#
# Runnable alone:
#   bash .github/claude/prepare-metadata.test.sh
#
# What the cases defend, and the failure each was observed to catch when run
# against the pre-fix scripts (2026-09-02):
#
#   1, 4  Attacker-controlled text reaches $GITHUB_ENV as `NAME=value`. A
#         comment body or a commit subject can carry a bare CR, which the
#         old `tr -d '\n'` left in place, so the entry could continue into a
#         second one (`BASH_ENV=<path in the PR>`) and later bash steps would
#         source PR-controlled code beside the review secrets.
#   2, 3, 5  REVIEW_NUMBER came from counting score-bearing comments, but
#         every re-score PATCHes the same sticky comment, so the sequence was
#         #1, #2, #2 forever.
#   6     The `issues` trigger matches @claude in the TITLE as well as the
#         body, so a title-only request used to reach the reviewer as an empty
#         request file.
#   7     IS_PR came from `gh pr view`, so a rate-limited or otherwise failed
#         lookup was indistinguishable from "this is a plain issue": the PR
#         checkout was skipped and reviewer assignment could act on an older
#         5/5 for the PR. The event payload already says which it is.
#   8     REVIEW_MODE tells prepare-review-inputs.sh whether to feed the prior
#         review back. A first pass that read it as a re-score would review
#         against its own earlier text.
#
# GNU grep is required: the scripts parse with `grep -oP`.
set -uo pipefail

if ! command -v jq > /dev/null 2>&1; then
  echo "jq is required for this test (macOS: brew install jq)" >&2
  exit 1
fi
if ! echo x | grep -qP x 2> /dev/null; then
  echo "GNU grep (grep -P) is required for this test; the scripts use grep -oP" >&2
  exit 1
fi

HERE=$(cd "$(dirname "$0")" && pwd)
MENTION_SCRIPT="${MENTION_SCRIPT:-$HERE/prepare-mention-metadata.sh}"
REVIEW_SCRIPT="${REVIEW_SCRIPT:-$HERE/prepare-review-metadata.sh}"

STUB=$(mktemp -d)
trap 'rm -rf "$STUB"' EXIT

# gh stub: serves the comment listing from $STUB_DIR/comments.json, the head
# SHA and the commit subject from env, and accepts the progress-comment writes.
cat > "$STUB/gh" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$STUB_DIR/calls.log"
case "$*" in
  *--paginate*/comments*) cat "$STUB_DIR/comments.json"; exit 0 ;;
  # Fails on purpose: IS_PR comes from the event payload, and a lookup that
  # decided it again would treat this error as a confirmed plain issue.
  "pr view "*--json\ number*) exit 1 ;;
  "pr view "*headRefOid*) printf '%s\n' "${STUB_HEAD_SHA:-}"; exit 0 ;;
  *"/commits/"*) printf '%b\n' "${STUB_COMMIT_MSG:-}"; exit 0 ;;
esac
echo '7'
exit 0
EOF
chmod +x "$STUB/gh"

export STUB_DIR="$STUB"
export PATH="$STUB:$PATH"
export GITHUB_REPOSITORY="acme/repo" GH_TOKEN="x"
export STUB_HEAD_SHA="8c66408deadbeefdeadbeefdeadbeefdeadbeef1"
# The mention step writes the request under RUNNER_TEMP, which the runner
# clears between jobs, rather than /tmp, which a self-hosted runner keeps.
export RUNNER_TEMP="$STUB"
REQUEST_FILE="$STUB/claude-request.txt"

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

ENVF="$STUB/github_env"

# The value of the last $1= entry, with CR treated as an entry separator the
# way the Actions env-file parser would see it.
env_value() { tr '\r' '\n' < "$ENVF" | grep "^$1=" | tail -1 | cut -d= -f2-; }
# Entries the parser would see that the script never meant to write.
env_injected() { tr '\r' '\n' < "$ENVF" | grep -c '^BASH_ENV='; }
env_crs() { grep -c $'\r' "$ENVF"; }

write_event() { printf '%s' "$1" > "$STUB/event.json"; }
run_mention() {
  : > "$ENVF"
  # /tmp too, so the "not /tmp" check below means "this run did not write it".
  rm -f "$REQUEST_FILE" /tmp/claude-request.txt
  GITHUB_ENV="$ENVF" GITHUB_EVENT_PATH="$STUB/event.json" \
    bash "$MENTION_SCRIPT" > /dev/null 2>&1
}
run_review() {
  : > "$ENVF"
  GITHUB_ENV="$ENVF" GITHUB_EVENT_PATH="$STUB/event.json" \
    bash "$REVIEW_SCRIPT" > /dev/null 2>&1
}

# A sticky review comment whose header says which review it is.
sticky() { # sticky <n>
  jq -n --arg body "<!-- score: 4 -->
## 🔍 Review #$1

Some findings.

*Review #$1 · Comment \`@claude\` to re-score*" \
    '[{"id":7,"user":{"login":"claude[bot]"},"body":$body}]'
}

# The first publisher wrote the sticky as claude[bot]. This is the migration
# fixture: the current publisher must still find and update it.
legacy_sticky() { # legacy_sticky <n>
  jq -n --arg body "<!-- score: 4 -->
## 🔍 Review #$1

Legacy findings.

*Review #$1 · Comment \`@claude\` to re-score*" \
    '[{"id":7,"user":{"login":"claude[bot]"},"body":$body}]'
}

# A marked comment is canonical even when a later duplicate exists. The marker
# is inserted by publish-review.sh after the model response is validated.
marked_and_duplicate() {
  jq -n --arg old "<!-- score: 4 -->
## 🔍 Review #1

Old sticky." \
    --arg marked "<!-- claude-sticky-review -->
<!-- score: 4 -->
## 🔍 Review #4

Canonical sticky." \
    '[
      {"id":7,"created_at":"2026-09-05T00:00:00Z","user":{"login":"claude[bot]"},"body":$old},
      {"id":8,"created_at":"2026-09-05T00:01:00Z","user":{"login":"claude[bot]"},"body":$marked}
    ]'
}

# A prior migration can leave two marked comments. The newest marked comment
# is the latest canonical state to PATCH; id breaks a timestamp tie.
multiple_marked() {
  jq -n --arg first "<!-- claude-sticky-review -->
<!-- score: 4 -->
## 🔍 Review #4

First marked body." \
    --arg second "<!-- claude-sticky-review -->
<!-- score: 5 -->
## 🔍 Review #5

Second marked body." \
    '[
      {"id":8,"created_at":"2026-09-05T00:01:00Z","updated_at":"2026-09-05T00:01:00Z","user":{"login":"claude[bot]"},"body":$first},
      {"id":9,"created_at":"2026-09-05T00:02:00Z","updated_at":"2026-09-05T00:03:00Z","user":{"login":"claude[bot]"},"body":$second}
    ]'
}

echo "== 1. A CRLF comment body cannot add an entry to GITHUB_ENV =="
echo '[]' > "$STUB/comments.json"
write_event '{"action":"created","issue":{"number":188,"pull_request":{"html_url":"https://github.com/acme/repo/pull/188"}},"comment":{"body":"@claude review\r\nBASH_ENV=/tmp/evil\r\n"}}'
STUB_COMMIT_MSG='fix: thing\rBASH_ENV=/tmp/evil2' run_mention
check "no CR survives anywhere in GITHUB_ENV" "$(env_crs)" "0"
check "no injected BASH_ENV entry" "$(env_injected)" "0"
check "TRIGGER_COMMENT is the one-line remainder" \
  "$(env_value TRIGGER_COMMENT)" "reviewBASH_ENV=/tmp/evil"
check "COMMIT_MSG is the one-line subject" \
  "$(env_value COMMIT_MSG)" "fix: thingBASH_ENV=/tmp/evil2"

echo "== 2. REVIEW_NUMBER follows the sticky comment's header, not the count =="
sticky 2 > "$STUB/comments.json"
write_event '{"action":"created","issue":{"number":188,"pull_request":{"html_url":"https://github.com/acme/repo/pull/188"}},"comment":{"body":"@claude take another look at the retry loop"}}'
STUB_COMMIT_MSG='fix: thing' run_mention
check "third review is numbered 3" "$(env_value REVIEW_NUMBER)" "3"

echo "== 2b. A legacy Claude-authored sticky remains the canonical review =="
legacy_sticky 1 > "$STUB/comments.json"
write_event '{"action":"created","issue":{"number":188,"pull_request":{"html_url":"https://github.com/acme/repo/pull/188"}},"comment":{"body":"@claude review"}}'
STUB_COMMIT_MSG='fix: thing' run_mention
check "legacy sticky advances to review 2" "$(env_value REVIEW_NUMBER)" "2"
check "legacy sticky id is carried forward" "$(env_value EXISTING_REVIEW_ID)" "7"

echo "== 2c. The explicit sticky marker wins over later duplicates =="
marked_and_duplicate > "$STUB/comments.json"
write_event '{"action":"created","issue":{"number":188,"pull_request":{"html_url":"https://github.com/acme/repo/pull/188"}},"comment":{"body":"@claude review"}}'
STUB_COMMIT_MSG='fix: thing' run_mention
check "marked sticky header advances from review 4" "$(env_value REVIEW_NUMBER)" "5"
check "marked sticky id is selected" "$(env_value EXISTING_REVIEW_ID)" "8"

echo "== 2d. The newest marked comment is canonical after a duplicate race =="
multiple_marked > "$STUB/comments.json"
write_event '{"action":"created","issue":{"number":188,"pull_request":{"html_url":"https://github.com/acme/repo/pull/188"}},"comment":{"body":"@claude review"}}'
STUB_COMMIT_MSG='fix: thing' run_mention
check "numbering still follows the highest review header" "$(env_value REVIEW_NUMBER)" "6"
check "the newest marked id is selected" "$(env_value EXISTING_REVIEW_ID)" "9"

echo "== 3. With no prior review the number is 1 =="
echo '[]' > "$STUB/comments.json"
write_event '{"action":"created","issue":{"number":188,"pull_request":{"html_url":"https://github.com/acme/repo/pull/188"}},"comment":{"body":"@claude take a look"}}'
STUB_COMMIT_MSG='fix: thing' run_mention
check "first review is numbered 1" "$(env_value REVIEW_NUMBER)" "1"

echo "== 4. The first-pass script sanitizes the commit subject too =="
echo '[]' > "$STUB/comments.json"
write_event '{"action":"opened","pull_request":{"number":188}}'
STUB_COMMIT_MSG='fix: thing\rBASH_ENV=/tmp/evil2' run_review
check "no CR survives anywhere in GITHUB_ENV" "$(env_crs)" "0"
check "no injected BASH_ENV entry" "$(env_injected)" "0"
check "COMMIT_MSG is the one-line subject" \
  "$(env_value COMMIT_MSG)" "fix: thingBASH_ENV=/tmp/evil2"

echo "== 5. workflow_dispatch re-runs read the header too =="
sticky 2 > "$STUB/comments.json"
write_event '{"inputs":{"pr_number":"188"}}'
STUB_COMMIT_MSG='fix: thing' run_review
check "re-run after Review #2 is numbered 3" "$(env_value REVIEW_NUMBER)" "3"
multiple_marked > "$STUB/comments.json"
STUB_COMMIT_MSG='fix: thing' run_review
check "first-pass metadata uses the newest marked id" "$(env_value EXISTING_REVIEW_ID)" "9"
check "first-pass metadata advances from the highest marked header" "$(env_value REVIEW_NUMBER)" "6"
echo '[]' > "$STUB/comments.json"
STUB_COMMIT_MSG='fix: thing' run_review
check "with no prior review it is 1" "$(env_value REVIEW_NUMBER)" "1"

echo "== 6. A title-only @claude issue still reaches the reviewer =="
echo '[]' > "$STUB/comments.json"
write_event '{"action":"opened","issue":{"number":42,"title":"@claude fix the flaky encoder test","body":null}}'
STUB_COMMIT_MSG='fix: thing' run_mention
check "the request file carries the title" \
  "$(head -1 "$REQUEST_FILE")" "@claude fix the flaky encoder test"
check "and it sits under RUNNER_TEMP, not /tmp" \
  "$([ -e /tmp/claude-request.txt ] && echo present || echo gone)" "gone"

echo "== 6b. A payload with no issue number leaves no request behind =="
# A persistent runner keeps RUNNER_TEMP between jobs on some setups, and
# prepare-review-inputs.sh reads this path. A leftover would be another PR's
# instructions.
printf 'stale request from another PR\n' > "$REQUEST_FILE"
write_event '{"action":"created","comment":{"body":"@claude"}}'
: > "$ENVF"
GITHUB_ENV="$ENVF" GITHUB_EVENT_PATH="$STUB/event.json" \
  bash "$MENTION_SCRIPT" > /dev/null 2>&1
check "the stale request is gone" \
  "$(wc -c < "$REQUEST_FILE" | tr -d ' ')" "0"

echo "== 7. IS_PR comes from the event payload, not an API lookup =="
sticky 2 > "$STUB/comments.json"
write_event '{"action":"created","issue":{"number":188,"pull_request":{"html_url":"https://github.com/acme/repo/pull/188"}},"comment":{"body":"@claude take another look"}}'
STUB_COMMIT_MSG='fix: thing' run_mention
check "issue_comment carrying issue.pull_request is a PR" "$(env_value IS_PR)" "true"

write_event '{"action":"created","issue":{"number":188},"comment":{"body":"@claude take another look"}}'
STUB_COMMIT_MSG='fix: thing' run_mention
check "the same comment without it is a plain issue" "$(env_value IS_PR)" "false"
check "the plain-issue path posts the progress comment" "$(env_value PROGRESS_COMMENT_ID)" "7"
check "and stops before the PR-only metadata" "$(env_value REVIEW_NUMBER)" ""

write_event '{"action":"submitted","pull_request":{"number":188},"review":{"body":"@claude take another look"}}'
STUB_COMMIT_MSG='fix: thing' run_mention
check "pull_request_review is a PR" "$(env_value IS_PR)" "true"

echo "== 8. Each entry point declares its REVIEW_MODE =="
echo '[]' > "$STUB/comments.json"
write_event '{"action":"opened","pull_request":{"number":188}}'
STUB_COMMIT_MSG='fix: thing' run_review
check "the first pass reviews fresh" "$(env_value REVIEW_MODE)" "first"

write_event '{"action":"created","issue":{"number":188,"pull_request":{"html_url":"https://github.com/acme/repo/pull/188"}},"comment":{"body":"@claude take another look"}}'
STUB_COMMIT_MSG='fix: thing' run_mention
check "a mention re-scores against the prior review" "$(env_value REVIEW_MODE)" "rescore"

write_event '{"action":"created","issue":{"number":188},"comment":{"body":"@claude take another look"}}'
STUB_COMMIT_MSG='fix: thing' run_mention
check "and a plain issue still declares one" "$(env_value REVIEW_MODE)" "rescore"

echo "== 9. Quoted markers cannot turn a review into a progress or canonical target =="
jq -n '[
 {id:7,created_at:"2026-09-05T00:00:00Z",user:{login:"claude[bot]"},body:"<!-- claude-sticky-review -->\n## 🔍 Review #3\n<!-- score: 4 -->"},
 {id:8,user:{login:"claude[bot]"},body:"<!-- progress-indicator -->\nPrevious progress"},
 {id:9,created_at:"2026-09-05T00:01:00Z",user:{login:"claude[bot]"},body:"## 🔍 Review #1\nConsider the `<!-- claude-sticky-review -->` and `<!-- progress-indicator -->` markers.\n<!-- score: 3 -->"}
]' > "$STUB/comments.json"
write_event '{"action":"created","issue":{"number":188,"pull_request":{}},"comment":{"body":"@claude review details"}}'
run_mention
check "quoted canonical marker is ignored" "$(env_value EXISTING_REVIEW_ID)" "7"
check "quoted progress marker is ignored" "$(env_value PROGRESS_COMMENT_ID)" "8"
write_event '{"inputs":{"pr_number":"188"}}'
: > "$STUB/calls.log"
run_review
check "dispatch reuses progress" "$(env_value PROGRESS_COMMENT_ID)" "8"
check "dispatch PATCHes rather than duplicates progress" "$(grep -c 'issues/comments/8 -X PATCH' "$STUB/calls.log")" "1"
check "dispatch does not POST another indicator" "$(grep -c ' -X POST' "$STUB/calls.log")" "0"

echo "== 10. Actions-authored comments supply history, never a cross-app PATCH target =="
jq -n '[{id:9,user:{login:"github-actions[bot]"},body:"<!-- claude-sticky-review -->\n## 🔍 Review #4\n<!-- score: 5 -->"}]' > "$STUB/comments.json"
run_review
check "Actions-only history advances numbering" "$(env_value REVIEW_NUMBER)" "5"
check "a new Claude sticky will be created" "$(env_value EXISTING_REVIEW_ID)" ""

echo "== 11. An incomplete same-head review remains retryable =="
# Bare @claude exercises the stale guard with both BSD and GNU sed.
jq -n --arg sha "$STUB_HEAD_SHA" '[{id:7,user:{login:"claude[bot]"},body:("<!-- claude-sticky-review -->\n<!-- claude-review-incomplete -->\n\n## 🔍 Review #2\n<!-- score: 5 -->\n*Analyzed [`" + $sha[0:7] + "`](https://github.com/acme/repo/commit/" + $sha + ")*")}]' > "$STUB/comments.json"
write_event '{"action":"created","issue":{"number":188,"pull_request":{}},"comment":{"body":"@claude"}}'
run_mention
check "partial review is not a stale request" "$(env_value STALE_REQUEST)" "false"

echo "== 12. Quoted incomplete markers do not retry an unchanged head =="
# shellcheck disable=SC2016 # Markdown backticks are literal review content.
for quoted_marker in \
  'Consider the `<!-- claude-review-incomplete -->` marker.' \
  $'```html\n<!-- claude-review-incomplete -->\n```' \
  $'```html\n<!-- claude-sticky-review -->\n<!-- claude-review-incomplete -->\n```'; do
  jq -n --arg sha "$STUB_HEAD_SHA" --arg quote "$quoted_marker" \
    '[{id:7,user:{login:"claude[bot]"},body:("<!-- claude-sticky-review -->\n\n## 🔍 Review #2\n<!-- score: 5 -->\n" + $quote + "\n*Analyzed [`" + $sha[0:7] + "`](https://github.com/acme/repo/commit/" + $sha + ")*")}]' \
    > "$STUB/comments.json"
  write_event '{"action":"created","issue":{"number":188,"pull_request":{}},"comment":{"body":"@claude"}}'
  run_mention
  check "quoted marker is declined: $quoted_marker" "$(env_value STALE_REQUEST)" "true"
done

echo
echo "passed: $pass  failed: $fail"
[ "$fail" -eq 0 ]

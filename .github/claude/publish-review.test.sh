#!/usr/bin/env bash
# Tests for publish-review.sh. Stubs gh, records every call it makes with the
# body it would have posted, and asserts on the log.
#
# Runnable alone:
#   bash .github/claude/publish-review.test.sh
#
# The cases that matter are 2, 3 and 7: a review body that fails validation
# must post NOTHING, because a half-posted review reads as a real one while
# the workflow's verify step reports success. Case 5 is the opposite rule —
# GitHub refuses any inline comment on a line outside the diff hunks, and that
# must never cost the review itself.
#
# Negative controls, each observed failing before the fix:
#   case 2  fails if the score-marker check does not exit
#   case 3  fails if the HEAD_SHA check does not exit
#   case 7  fails if the size check does not exit
#   case 9  fails if the not-posted list is appended without measuring it
#   case 10 fails if an unsupported verdict still posts the finding inline
#   case 10b fails if a finding the verifier never ruled on is posted anyway
#   case 12 fails if a malformed review.json is treated as an empty review
#   case 10d fails if a duplicate verdict for one index is not decided by the
#           last entry
#   case 13 fails if a token-shaped string in the answer is posted
#   case 13b fails if a token reaches the comment through a verifier reason
set -uo pipefail

if ! command -v jq > /dev/null 2>&1; then
  echo "jq is required for this test (macOS: brew install jq)" >&2
  exit 1
fi

HERE=$(cd "$(dirname "$0")" && pwd)
SCRIPT="${SCRIPT:-$HERE/publish-review.sh}"
STUB=$(mktemp -d)
WORK=$(mktemp -d)
LOG="$STUB/gh.log"

export REVIEW_IN="$WORK/in" REVIEW_OUT="$WORK/out"
mkdir -p "$REVIEW_IN" "$REVIEW_OUT"
REVIEW_JSON="$REVIEW_OUT/review.json"
VERDICTS_JSON="$REVIEW_OUT/verdicts.json"
RESPONSE_MD="$REVIEW_OUT/response.md"

# gh stub: appends the whole argument list, plus the contents of any
# `body=@file` argument, so the test can assert on what would have been
# posted. STUB_INLINE_FAIL / STUB_STICKY_FAIL inject API failures.
cat > "$STUB/gh" <<'EOF'
#!/usr/bin/env bash
{
  printf 'CALL:'
  for a in "$@"; do printf ' %s' "$a"; done
  printf '\n'
  for a in "$@"; do
    case "$a" in
      body=@*)
        printf 'BODY<<\n'
        cat "${a#body=@}"
        printf '\nBODY>>\n'
        ;;
    esac
  done
} >> "$STUB_DIR/gh.log"
case "$*" in
  *pulls/*/comments*)
    if [ -n "${STUB_INLINE_FAIL:-}" ]; then echo "HTTP 422" >&2; exit 1; fi ;;
  *)
    if [ -n "${STUB_STICKY_FAIL:-}" ]; then echo "HTTP 500" >&2; exit 1; fi ;;
esac
echo '{"id":1}'
EOF
chmod +x "$STUB/gh"

export STUB_DIR="$STUB"
export PATH="$STUB:$PATH"
# Long enough that the publisher's literal-value check treats it as a real
# token: a one-character stub would match almost any review body.
export GH_TOKEN="ghs_stubtokenvalue0123456789abcdefghij"
export GITHUB_REPOSITORY="acme/repo" PR_NUMBER="188"
export HEAD_SHA="1234567890abcdef1234567890abcdef12345678"
export SHORT_SHA="1234567"
export IS_PR="true"

printf 'web-ui/src/App.tsx\nuser-api/src/index.js\n' > "$REVIEW_IN/files.txt"

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

run() {
  : > "$LOG"
  bash "$SCRIPT" > "$STUB/out.txt" 2>&1
}

# A well-formed answer: the sticky body and two findings. jq builds it so the
# SHA and the newlines land as real JSON string content.
good_review() { # good_review [jq filter applied to the object]
  rm -f "$VERDICTS_JSON"
  jq -n --arg sha "$HEAD_SHA" '{
    review_markdown: ("## 🔍 Review #2\n\nLooks mostly fine.\n\n<!-- score: 4 -->\n\n*Reviewed [1234567](https://github.com/acme/repo/commit/" + $sha + ")*\n"),
    findings: [
      {path: "web-ui/src/App.tsx", line: 12, severity: "P1", body: "P1: this is wrong"},
      {path: "user-api/src/index.js", line: 30, start_line: 28, side: "RIGHT", severity: "P2", body: "P2: consider X"}
    ]
  }' | jq "${1:-.}" > "$REVIEW_JSON"
}

echo "== 1. A valid review posts its findings inline, then the sticky comment =="
good_review
EXISTING_REVIEW_ID="777" run
STATUS=$?
check "exits 0" "$STATUS" "0"
check "two inline comments were posted" \
  "$(grep -c '^CALL:.*repos/acme/repo/pulls/188/comments' "$LOG")" "2"
check "the first finding carries its path" \
  "$(grep -c 'path=web-ui/src/App.tsx' "$LOG")" "1"
check "and its line, as a number field" "$(grep -c -- '-F line=12' "$LOG")" "1"
check "and the reviewed commit" \
  "$(grep -c -- "-f commit_id=$HEAD_SHA" "$LOG")" "2"
check "side defaults to RIGHT when the model omits it" \
  "$(grep -c -- '-f side=RIGHT' "$LOG")" "2"
check "a multi-line finding sends start_line" \
  "$(grep -c -- '-F start_line=28' "$LOG")" "1"
# GitHub rejects a multi-line comment that carries start_line without
# start_side, so omitting it turns every multi-line finding into a 422.
check "and the start_side GitHub requires beside it" \
  "$(grep -c -- '-f start_side=RIGHT' "$LOG")" "1"
check "the finding body came from a file, not the command line" \
  "$(grep -c 'P1: this is wrong' "$LOG")" "1"
check "the sticky comment was PATCHed onto the existing one" \
  "$(grep -c '^CALL:.*repos/acme/repo/issues/comments/777.*-X PATCH' "$LOG")" "1"
check "and nothing was posted as a new issue comment" \
  "$(grep -c '^CALL:.*repos/acme/repo/issues/188/comments' "$LOG")" "0"
check "the sticky body keeps the model's score marker" \
  "$(grep -c '<!-- score: 4 -->' "$LOG")" "1"
check "the sticky body carries the canonical marker" \
  "$(grep -c '<!-- claude-sticky-review -->' "$LOG")" "1"
check "and carries no Low confidence section" \
  "$(grep -c 'Low confidence' "$LOG")" "0"

echo "== 1a. A failed sticky PATCH fails the publisher =="
# A legacy Claude-authored comment may be selected during migration. If the
# workflow token cannot edit it, the publisher must fail loudly instead of
# reporting a successful review after the inline comments have gone out.
good_review
EXISTING_REVIEW_ID="777" STUB_STICKY_FAIL=1 run
STATUS=$?
check "exits 1 when the sticky update fails" "$STATUS" "1"
check "reports the failed sticky id" \
  "$(grep -c 'Could not update the sticky review comment 777' "$STUB/out.txt")" "1"
check "does not fall back to a new top-level comment" \
  "$(grep -c '^CALL:.*repos/acme/repo/issues/188/comments.*-X POST' "$LOG")" "0"

echo "== 1b. With no existing sticky comment, the review is posted fresh =="
good_review
EXISTING_REVIEW_ID="" run
STATUS=$?
check "exits 0" "$STATUS" "0"
check "one new issue comment" \
  "$(grep -c '^CALL:.*repos/acme/repo/issues/188/comments.*-X POST' "$LOG")" "1"
check "and no PATCH" "$(grep -c -- '-X PATCH' "$LOG")" "0"

echo "== 2. A body with no score marker posts nothing =="
# The score marker is what every downstream step keys on: the metadata step
# finds the sticky comment by it, and the reviewer-assignment step reads the
# score out of it. A review without one is not a review.
good_review '.review_markdown |= gsub("<!-- score: 4 -->"; "")'
EXISTING_REVIEW_ID="777" run
STATUS=$?
check "exits 1" "$STATUS" "1"
check "no API call at all" "$(grep -c '^CALL:' "$LOG")" "0"
check "and says which rule failed" \
  "$(grep -c 'score' "$STUB/out.txt")" "1"

echo "== 3. A body that does not name the reviewed commit posts nothing =="
good_review ".review_markdown |= gsub(\"$HEAD_SHA\"; \"deadbeef\")"
EXISTING_REVIEW_ID="777" run
STATUS=$?
check "exits 1" "$STATUS" "1"
check "no API call at all" "$(grep -c '^CALL:' "$LOG")" "0"

echo "== 4. A finding on a file the PR does not touch is skipped =="
good_review '.findings[1].path = "etc/passwd"'
EXISTING_REVIEW_ID="777" run
STATUS=$?
check "exits 0" "$STATUS" "0"
check "only the valid finding was posted" \
  "$(grep -c '^CALL:.*pulls/188/comments' "$LOG")" "1"
check "the invalid path never reached the API" \
  "$(grep -c 'etc/passwd' "$LOG")" "0"
check "and the skip was announced" \
  "$(grep -c '::warning::Skipping a finding' "$STUB/out.txt")" "1"
check "the sticky comment still went out" \
  "$(grep -c -- '-X PATCH' "$LOG")" "1"

echo "== 5. GitHub refusing an inline comment does not cost the review =="
good_review
EXISTING_REVIEW_ID="777" STUB_INLINE_FAIL=1 run
STATUS=$?
check "exits 0" "$STATUS" "0"
check "the sticky comment still went out" \
  "$(grep -c -- '-X PATCH' "$LOG")" "1"
check "the refused finding is listed in the review body instead" \
  "$(grep -c 'web-ui/src/App.tsx:12' "$LOG")" "1"
check "under its own heading, after the model's footer" \
  "$(grep -c '### Findings not posted inline' "$LOG")" "1"
# A path and a line say where, not what. Without the severity and the first
# line of the body, a reader cannot triage the entry or check it against the
# diff, which is the whole point of listing it rather than dropping it.
check "the entry carries its severity and the first line of its body" \
  "$(grep -c '`web-ui/src/App.tsx:12` P1: P1: this is wrong' "$LOG")" "1"
check "and so does the second one" \
  "$(grep -c '`user-api/src/index.js:30` P2: P2: consider X' "$LOG")" "1"

echo "== 5b. A long first line is bounded before it is listed =="
# A body has no length limit of its own below MAX_FINDING_BYTES, so an unbounded
# summary could push the sticky comment past what GitHub accepts.
LONG_LINE=$(head -c 300 /dev/zero | tr '\0' 'z')
good_review ".findings[1].body = \"$LONG_LINE\""
EXISTING_REVIEW_ID="777" STUB_INLINE_FAIL=1 run
check "the summary is cut at 160 characters" \
  "$(grep -cE '^- `user-api/src/index.js:30` P2: z{160}$' "$LOG")" "1"
check "and not one character more" \
  "$(grep -cE 'z{161}' "$LOG")" "1"

echo "== 6. A plain issue posts response.md as one comment =="
printf 'Here is the answer to your question.\n' > "$RESPONSE_MD"
IS_PR="false" EXISTING_REVIEW_ID="" run
STATUS=$?
check "exits 0" "$STATUS" "0"
check "one issue comment" \
  "$(grep -c '^CALL:.*repos/acme/repo/issues/188/comments' "$LOG")" "1"
check "carrying the message verbatim" \
  "$(grep -c 'Here is the answer to your question.' "$LOG")" "1"
check "and no inline comments" "$(grep -c 'pulls/188/comments' "$LOG")" "0"
rm -f "$RESPONSE_MD"

echo "== 7. A body over the size limit posts nothing =="
# GitHub rejects a comment body over 65536 bytes. Finding that out from the
# API means the inline comments are already posted and the review is not.
FILLER=$(head -c 61000 /dev/zero | tr '\0' 'x')
jq -n --arg sha "$HEAD_SHA" --arg filler "$FILLER" \
  '{review_markdown: ("## Review\n\n<!-- score: 4 -->\n\n" + $sha + "\n" + $filler + "\n"), findings: []}' \
  > "$REVIEW_JSON"
rm -f "$VERDICTS_JSON"
EXISTING_REVIEW_ID="777" run
STATUS=$?
check "exits 1" "$STATUS" "1"
check "no API call at all" "$(grep -c '^CALL:' "$LOG")" "0"
check "and names the limit" "$(grep -c '60000' "$STUB/out.txt")" "1"

echo "== 8. The harness is live: a copy without the score check behaves =="
# Without this, cases 2 and 3 would pass against a script that validates
# nothing at all — they assert an absence.
BROKEN="$WORK/broken.sh"
sed 's|fail_body "the review body carries no|: "the review body carries no|' \
  "$SCRIPT" > "$BROKEN"
good_review '.review_markdown |= gsub("<!-- score: 4 -->"; "")'
: > "$LOG"
EXISTING_REVIEW_ID="777" bash "$BROKEN" > "$STUB/out.txt" 2>&1
STATUS=$?
check "the broken copy exits 0 where the real one exits 1" "$STATUS" "0"
check "and posts, proving case 2 tests the check and not the stub" \
  "$(grep -c -- '-X PATCH' "$LOG")" "1"

echo "== 9. A not-posted list that would overflow the comment is replaced by a count =="
# The review body may legitimately sit just under its own 60000-byte limit, and
# GitHub rejects the whole comment over 65536. Paths have no length limit to
# derive a safe entry count from, so the list is measured before it is appended.
LONG_PATH="web-ui/src/$(head -c 230 /dev/zero | tr '\0' 'd')/App.tsx"
printf '%s\n' "$LONG_PATH" >> "$REVIEW_IN/files.txt"
FILLER=$(head -c 59800 /dev/zero | tr '\0' 'x')
jq -n --arg sha "$HEAD_SHA" --arg filler "$FILLER" --arg p "$LONG_PATH" \
  '{review_markdown: ("## Review\n\n<!-- score: 4 -->\n\n" + $sha + "\n" + $filler + "\n"),
    findings: [range(30) | {path: $p, line: (. + 1), severity: "P2", body: "x"}]}' \
  > "$REVIEW_JSON"
rm -f "$VERDICTS_JSON"
EXISTING_REVIEW_ID="777" STUB_INLINE_FAIL=1 run
STATUS=$?
check "exits 0" "$STATUS" "0"
check "the sticky comment still went out" "$(grep -c -- '-X PATCH' "$LOG")" "1"
check "the paths were dropped for the count" \
  "$(grep -c '30 findings did not fit' "$LOG")" "1"
check "so the comment stays under GitHub's limit" \
  "$(awk '/^BODY<</{f=1;n=0;next} /^BODY>>/{f=0} f{n+=length($0)+1} END{print (n<65536)?"yes":"no"}' "$LOG")" "yes"
# Restore the two-path files.txt the other cases assume.
printf 'web-ui/src/App.tsx\nuser-api/src/index.js\n' > "$REVIEW_IN/files.txt"

echo "== 10. A finding the verifier rejects is listed, not posted =="
# A silent drop would make the verifier unfalsifiable: nobody could see what it
# threw away, so nobody could tell a good filter from a broken one.
good_review
jq -n '{verdicts: [{index: 0, supported: false, reason: "the diff does not show that call"},
                   {index: 1, supported: true, reason: "confirmed"}]}' \
  > "$VERDICTS_JSON"
EXISTING_REVIEW_ID="777" run
STATUS=$?
check "exits 0" "$STATUS" "0"
check "only the supported finding was posted inline" \
  "$(grep -c '^CALL:.*pulls/188/comments' "$LOG")" "1"
check "and it is the second one" \
  "$(grep -c 'path=user-api/src/index.js' "$LOG")" "1"
check "the rejected finding was never sent as an inline comment" \
  "$(grep -c 'path=web-ui/src/App.tsx' "$LOG")" "0"
check "the Low confidence section names it" \
  "$(grep -c '### Low confidence' "$LOG")" "1"
check "with its path, line and severity" \
  "$(grep -c '`web-ui/src/App.tsx:12` P1:' "$LOG")" "1"
check "and the verifier's reason" \
  "$(grep -c 'verifier: the diff does not show that call' "$LOG")" "1"
check "the rejection is announced in the run log" \
  "$(grep -c '::warning::The verifier rejected a finding' "$STUB/out.txt")" "1"

echo "== 10b. A finding the verifier never ruled on is not supported =="
# The verifier ran, so it had the chance to rule on this finding and did not.
# Posting it inline anyway presents an unchecked finding as a checked one.
good_review
jq -n '{verdicts: [{index: 1, supported: true, reason: "confirmed"}]}' > "$VERDICTS_JSON"
EXISTING_REVIEW_ID="777" run
check "only the finding with a verdict was posted" \
  "$(grep -c '^CALL:.*pulls/188/comments' "$LOG")" "1"
check "and it is the one the verifier confirmed" \
  "$(grep -c 'path=user-api/src/index.js' "$LOG")" "1"
check "the unruled finding is listed under Low confidence" \
  "$(grep -c '`web-ui/src/App.tsx:12` P1:' "$LOG")" "1"
check "with the reason naming the silence" \
  "$(grep -c 'verifier: no verdict' "$LOG")" "1"

echo "== 10bb. With no verdicts file at all, every finding is supported =="
# The opposite case, and the one this must not punish: a verifier that never
# ran would otherwise silence the entire review.
good_review
EXISTING_REVIEW_ID="777" run
check "both findings were posted" \
  "$(grep -c '^CALL:.*pulls/188/comments' "$LOG")" "2"
check "and no Low confidence section appears" \
  "$(grep -c 'Low confidence' "$LOG")" "0"

echo "== 10c. A malformed verdicts file does not suppress anything =="
# The verifier is a filter on the review, not a gate on posting it.
good_review
printf 'not json at all\n' > "$VERDICTS_JSON"
EXISTING_REVIEW_ID="777" run
STATUS=$?
check "exits 0" "$STATUS" "0"
check "both findings were posted" \
  "$(grep -c '^CALL:.*pulls/188/comments' "$LOG")" "2"
check "and the run log says the file was ignored" \
  "$(grep -c 'treating every finding as supported' "$STUB/out.txt")" "1"
rm -f "$VERDICTS_JSON"

echo "== 10d. Two verdicts for one index: the last entry wins =="
# The rule the publisher's header states and the workflow's recompose count
# cites. Both sides must break the tie the same way, or the count in the
# workflow disagrees with the comment the reader is looking at.
good_review
jq -n '{verdicts: [{index: 0, supported: true, reason: "first pass"},
                   {index: 0, supported: false, reason: "retracted on re-read"},
                   {index: 1, supported: true, reason: "confirmed"}]}' \
  > "$VERDICTS_JSON"
EXISTING_REVIEW_ID="777" run
check "[true, false] does not post the finding inline" \
  "$(grep -c 'path=web-ui/src/App.tsx' "$LOG")" "0"
check "and the last entry's reason is the one listed" \
  "$(grep -c 'verifier: retracted on re-read' "$LOG")" "1"

good_review
jq -n '{verdicts: [{index: 0, supported: false, reason: "first pass"},
                   {index: 0, supported: true, reason: "confirmed on re-read"},
                   {index: 1, supported: true, reason: "confirmed"}]}' \
  > "$VERDICTS_JSON"
EXISTING_REVIEW_ID="777" run
check "[false, true] posts the finding inline" \
  "$(grep -c 'path=web-ui/src/App.tsx' "$LOG")" "1"
check "and lists nothing under Low confidence" \
  "$(grep -c 'Low confidence' "$LOG")" "0"
rm -f "$VERDICTS_JSON"

echo "== 11. A finding with an unusable severity is skipped with a warning =="
# The severity is what a reader triages on and what the Low confidence list
# prints; an unrecognised one is a finding nobody can act on.
good_review '.findings[0].severity = "CRITICAL"'
EXISTING_REVIEW_ID="777" run
STATUS=$?
check "exits 0" "$STATUS" "0"
check "only the valid finding was posted" \
  "$(grep -c '^CALL:.*pulls/188/comments' "$LOG")" "1"
check "and the skip names the severity" \
  "$(grep -c "severity 'CRITICAL' is not P0, P1 or P2" "$STUB/out.txt")" "1"
check "the sticky comment still went out" \
  "$(grep -c -- '-X PATCH' "$LOG")" "1"

echo "== 12. A malformed review.json posts nothing =="
printf 'this is not JSON\n' > "$REVIEW_JSON"
EXISTING_REVIEW_ID="777" run
STATUS=$?
check "exits 1" "$STATUS" "1"
check "no API call at all" "$(grep -c '^CALL:' "$LOG")" "0"
check "and says the file is the problem" \
  "$(grep -c 'review_markdown string' "$STUB/out.txt")" "1"

echo '{"findings": []}' > "$REVIEW_JSON"
EXISTING_REVIEW_ID="777" run
STATUS=$?
check "a review.json with no review_markdown also posts nothing" "$STATUS" "1"
check "and still makes no API call" "$(grep -c '^CALL:' "$LOG")" "0"

rm -f "$REVIEW_JSON"
EXISTING_REVIEW_ID="777" run
STATUS=$?
check "a missing review.json posts nothing" "$STATUS" "1"
check "and still makes no API call" "$(grep -c '^CALL:' "$LOG")" "0"

echo "== 13. A token-shaped string anywhere in the answer posts NOTHING =="
# The model runs beside GH_TOKEN, GITHUB_TOKEN and CLAUDE_CODE_OAUTH_TOKEN --
# the action exports all three into the CLI's environment whatever the step
# declares -- so this comment is the last place a credential can still be
# caught before it is public. A partial post cannot be taken back.
FAKE_TOKEN="ghs_$(head -c 32 /dev/zero | tr '\0' 'A')"

good_review ".review_markdown += \"\n$FAKE_TOKEN\n\""
EXISTING_REVIEW_ID="777" run
STATUS=$?
check "a token in the review body exits 1" "$STATUS" "1"
check "with no API call at all" "$(grep -c '^CALL:' "$LOG")" "0"
check "and the error names the class, not the value" \
  "$(grep -c 'token shape' "$STUB/out.txt")" "1"
check "the value itself is never printed" \
  "$(grep -c "$FAKE_TOKEN" "$STUB/out.txt")" "0"

good_review ".findings[0].body = \"P1: the header holds $FAKE_TOKEN\""
EXISTING_REVIEW_ID="777" run
STATUS=$?
check "a token in a finding body exits 1" "$STATUS" "1"
check "with no API call at all" "$(grep -c '^CALL:' "$LOG")" "0"

good_review ".review_markdown += \"\n$GH_TOKEN\n\""
EXISTING_REVIEW_ID="777" run
STATUS=$?
check "the literal value of GH_TOKEN exits 1" "$STATUS" "1"
check "with no API call at all" "$(grep -c '^CALL:' "$LOG")" "0"

printf 'Here you go: %s\n' "$FAKE_TOKEN" > "$RESPONSE_MD"
IS_PR="false" EXISTING_REVIEW_ID="" run
STATUS=$?
check "a token in response.md exits 1" "$STATUS" "1"
check "with no API call at all" "$(grep -c '^CALL:' "$LOG")" "0"
rm -f "$RESPONSE_MD"

echo "== 13b. A token reaching the comment through a verifier reason posts NOTHING =="
# The reason is quoted verbatim in the Low confidence list, so it is answer text
# with the same reach as the review body -- and it is the one string that enters
# the comment after the body and the finding bodies were scanned.
good_review
jq -n --arg t "$FAKE_TOKEN" \
  '{verdicts: [{index: 0, supported: false, reason: ("the header holds " + $t)},
               {index: 1, supported: true, reason: "confirmed"}]}' \
  > "$VERDICTS_JSON"
EXISTING_REVIEW_ID="777" run
STATUS=$?
check "a token in a verifier reason exits 1" "$STATUS" "1"
check "with no API call at all" "$(grep -c '^CALL:' "$LOG")" "0"
check "and the value itself is never printed" \
  "$(grep -c "$FAKE_TOKEN" "$STUB/out.txt")" "0"
rm -f "$VERDICTS_JSON"

good_review
EXISTING_REVIEW_ID="777" run
STATUS=$?
check "and a clean answer still posts" "$STATUS" "0"
check "sticky comment and both findings went out" \
  "$(grep -c '^CALL:' "$LOG")" "3"

echo "== 14. A quoted sticky marker cannot suppress the trusted leading marker =="
good_review '.review_markdown += "\nDiscuss `<!-- claude-sticky-review -->` as code.\n" | .findings = []'
run
check "a leading marker is inserted even when the body quotes it" \
  "$(grep -A1 '^BODY<<' "$LOG" | tail -1)" "<!-- claude-sticky-review -->"

echo "== 15. Incomplete verification is visible and remains a canonical sticky =="
good_review '.findings = []'
REVIEW_INCOMPLETE=true run
check "incomplete notice is present" "$(grep -c 'Verification incomplete' "$LOG")" "1"
check "incomplete state is machine-readable" "$(grep -c '<!-- claude-review-incomplete -->' "$LOG")" "1"
check "there is only one leading canonical marker" "$(grep -c '<!-- claude-sticky-review -->' "$LOG")" "1"
check "incomplete status occupies the exact two-line prefix" \
  "$(grep -A2 '^BODY<<' "$LOG" | tail -2)" \
  $'<!-- claude-sticky-review -->\n<!-- claude-review-incomplete -->'

rm -rf "$STUB" "$WORK"
echo
echo "passed: $pass  failed: $fail"
[ "$fail" -eq 0 ]

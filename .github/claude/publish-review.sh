#!/usr/bin/env bash
# Post the review the model wrote. The model itself is denied `gh` and every
# write tool (the action still puts a GitHub token in its environment, which
# is why the leak refusal below exists): its structured answer is the review,
# and this trusted step is the only thing that reaches the API.
#
# That split closes two findings at once. The model can no longer be talked
# into calling `gh api` against the repository by text it is reviewing, and it
# no longer spends turns posting: review #2 used about 20 of its 46 tool calls
# on comment-posting, and posted one batch twice.
#
# Input is two JSON files the action writes from its --json-schema output, so
# the shape is the schema's problem and not a marker-splitting exercise here:
#
#   $REVIEW_OUT/review.json     review-schema.json
#     {"review_markdown": "...", "findings": [{path, line, severity, body,
#      start_line?, side?}, ...]}
#   $REVIEW_OUT/verdicts.json   verdicts-schema.json, OPTIONAL
#     {"verdicts": [{"index": <position in findings>, "supported": <bool>,
#      "reason": "..."}]}
#
# THE LAST VERDICT ENTRY FOR AN INDEX WINS. Nothing stops the verifier from
# ruling on one index twice, so the tie is broken here rather than left to
# whichever consumer reads the file: a verifier that revises itself mid-answer
# means the later line, and "any supported entry accepts" would let a retracted
# acceptance outvote its own retraction. `verdict_of` applies it, and the
# workflow's recompose count cites this paragraph so both sides count the same
# verdicts.
#
# A finding the verifier marks unsupported is NOT posted inline. It is listed
# at the bottom of the sticky comment under "### Low confidence" with the
# verifier's reason, so a human can audit what the verifier throws away — a
# silent drop would make the verifier unfalsifiable. A finding the verifier did
# not rule on is listed the same way, with the reason `no verdict`: once a
# verdicts file exists the verifier ran, and a finding it passed over was not
# checked. When there is no verdicts file at all the verifier did not run, and
# every finding is supported — that is the case this must not punish, because a
# verifier that fails to start would otherwise silence the whole review.
#
# Nothing is posted at all if the text carries something token-shaped. The model
# runs beside GH_TOKEN, GITHUB_TOKEN and CLAUDE_CODE_OAUTH_TOKEN — the action
# exports all three into the CLI's environment whatever the step declares — so a
# review body is the last place a credential can still be caught before it is
# public. Every string that can reach the API is scanned, not only the review
# body: finding bodies, the verifier's reasons (they are quoted in the Low
# confidence list), and the assembled comment itself just before it goes out.
#
# Nothing is posted unless review_markdown passes every check below. A
# malformed review is not repaired here: the workflow's verify step reports
# "review not posted", which is a loud, correct outcome, where a half-posted
# review is not.
#
# Env in: REVIEW_IN, REVIEW_OUT, PR_NUMBER, HEAD_SHA, SHORT_SHA,
#         EXISTING_REVIEW_ID (empty on the first review), IS_PR, GH_TOKEN,
#         GITHUB_REPOSITORY.
set -euo pipefail

: "${REVIEW_OUT:?REVIEW_OUT is required}"
: "${PR_NUMBER:?PR_NUMBER is required}"
: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"

IS_PR="${IS_PR:-}"
EXISTING_REVIEW_ID="${EXISTING_REVIEW_ID:-}"
REVIEW_JSON="$REVIEW_OUT/review.json"
VERDICTS_JSON="$REVIEW_OUT/verdicts.json"
RESPONSE_MD="$REVIEW_OUT/response.md"

MAX_BODY_BYTES=60000
MAX_FINDING_BYTES=20000
MAX_FINDINGS=50
MAX_LISTED=30
# The first line of a body, in the not-posted list. A path and a line number say
# where, not what, and a reader who cannot act on the entry cannot check it
# either.
MAX_SUMMARY_CHARS=160
# GitHub rejects a comment body over 65536 bytes. review_markdown is allowed to
# reach MAX_BODY_BYTES on its own, so the appended lists are measured against
# what is left rather than trusted to a line count: a path has no length limit
# that MAX_LISTED could be derived from.
MAX_STICKY_BYTES=65000
STICKY_MARKER='<!-- claude-sticky-review -->'

# Token shapes, not token values: a credential that never entered this shell's
# environment is still a credential. The literal values of the three tokens the
# action exports are compared separately, with grep -F, and are never printed.
TOKEN_SHAPES='(ghs_[A-Za-z0-9]{20,}|ghp_|gho_|ghu_|ghr_|github_pat_[A-Za-z0-9_]{20,}|sk-ant-[A-Za-z0-9_-]{20,})'

# secret_class <file> -> prints the CLASS that matched and returns 0, or 1.
# The class alone, never the match: an error annotation is as public as the
# comment this refuses to post.
secret_class() {
  local f="$1" v
  if grep -Eq "$TOKEN_SHAPES" "$f"; then
    printf 'a GitHub or Anthropic token shape'
    return 0
  fi
  if grep -iqF 'AUTHORIZATION: basic' "$f"; then
    printf 'an Authorization header'
    return 0
  fi
  # Every real value of these three is far longer than 16 bytes. A short one is
  # a stub or a placeholder, and matching it would refuse any review that
  # happens to contain that letter.
  for v in "${GH_TOKEN:-}" "${GITHUB_TOKEN:-}" "${CLAUDE_CODE_OAUTH_TOKEN:-}"; do
    [ "${#v}" -ge 16 ] || continue
    if grep -qF -- "$v" "$f"; then
      printf "the literal value of a token in this step's environment"
      return 0
    fi
  done
  return 1
}

refuse_secret() { # refuse_secret <where> <class>
  echo "::error::Nothing posted: $1 carries $2"
  exit 1
}

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
BODY="$WORK/body.md"
FINDINGS="$WORK/findings.json"
VERDICTS="$WORK/verdicts.json"
STICKY="$WORK/sticky.md"

# --- plain issue: response.md is the whole comment ----------------------------
if [ "$IS_PR" != "true" ]; then
  if [ ! -s "$RESPONSE_MD" ] || ! grep -q '[^[:space:]]' "$RESPONSE_MD"; then
    echo "::error::The reviewer produced no response ($RESPONSE_MD is missing, empty or blank)"
    exit 1
  fi
  if CLASS=$(secret_class "$RESPONSE_MD"); then
    refuse_secret "the response" "$CLASS"
  fi
  gh api "repos/$GITHUB_REPOSITORY/issues/$PR_NUMBER/comments" \
    -X POST -F "body=@$RESPONSE_MD" > /dev/null
  echo "Posted the response to #$PR_NUMBER"
  exit 0
fi

# Only reached for a PR, which is the only mode with a changed-file list to
# check inline findings against.
: "${REVIEW_IN:?REVIEW_IN is required}"
if [ ! -f "$REVIEW_IN/files.txt" ]; then
  echo "::error::$REVIEW_IN/files.txt is missing; prepare-review-inputs.sh did not run"
  exit 1
fi

fail_body() {
  echo "::error::Review not posted: $1"
  exit 1
}

# --- read review.json ---------------------------------------------------------
#
# A file that is not an object with a string review_markdown is a malformed
# review, not a review with a missing part: the model's whole answer came
# through one schema, so a shape failure means the answer did not survive.
if [ ! -s "$REVIEW_JSON" ]; then
  fail_body "$REVIEW_JSON is missing or empty"
fi
if ! jq -e 'type == "object" and (.review_markdown | type == "string")' \
    "$REVIEW_JSON" > /dev/null 2>&1; then
  fail_body "$REVIEW_JSON is not an object carrying a review_markdown string"
fi
jq -r '.review_markdown' "$REVIEW_JSON" > "$BODY"
jq -c 'if (.findings | type) == "array" then .findings else [] end' \
  "$REVIEW_JSON" > "$FINDINGS"

# --- validate the review body -------------------------------------------------
if ! grep -q '[^[:space:]]' "$BODY"; then
  fail_body "the review body is empty"
fi
if ! grep -Eq '<!-- score:[[:space:]]*[1-5][[:space:]]*-->' "$BODY"; then
  fail_body "the review body carries no <!-- score: N --> marker with N in 1..5"
fi
# The footer's commit link carries the full SHA. Requiring it is how a review
# written against a stale checkout, or a body the model invented wholesale, is
# caught before it is attributed to this commit.
if [ -z "${HEAD_SHA:-}" ] || ! grep -qF "$HEAD_SHA" "$BODY"; then
  fail_body "the review body does not name the reviewed commit ${HEAD_SHA:-<unset>}"
fi
BODY_BYTES=$(wc -c < "$BODY" | tr -d ' ')
if [ "$BODY_BYTES" -gt "$MAX_BODY_BYTES" ]; then
  fail_body "the review body is $BODY_BYTES bytes, over the $MAX_BODY_BYTES limit"
fi

# Mark the canonical sticky after the model body passes validation. The marker
# is inserted by this trusted step, rather than requested from the model, so it
# cannot be omitted by a malformed re-score and cannot make an empty body pass
# the checks above. Existing legacy claude[bot] comments have no marker and are
# selected by the metadata step's migration fallback; this first patch gives
# all later runs an unambiguous identity.
if [ "$(head -1 "$BODY")" != "$STICKY_MARKER" ]; then
  {
    printf '%s\n\n' "$STICKY_MARKER"
    cat "$BODY"
  } > "$WORK/body-marked.md"
  mv "$WORK/body-marked.md" "$BODY"
fi
if [ "${REVIEW_INCOMPLETE:-}" = "true" ]; then
  {
    printf '%s\n\n' '<!-- claude-review-incomplete -->'
    printf '%s\n\n' '> ⚠️ **Verification incomplete.** This review was published, but a verification or summary pass did not complete. Findings and the score below are provisional; request another review to finish verification.'
    tail -n +2 "$BODY"
  } > "$WORK/body-incomplete.md"
  # The first two consecutive lines are the status prefix consumed by the
  # stale-request and assignment gates; examples elsewhere are ordinary prose.
  { printf '%s\n' "$STICKY_MARKER"; cat "$WORK/body-incomplete.md"; } > "$BODY"
fi
BODY_BYTES=$(wc -c < "$BODY" | tr -d ' ')

COUNT=$(jq 'length' "$FINDINGS")

# Before any API call, and before the per-finding shape checks: a finding that
# would have been skipped anyway still proves the review text is contaminated,
# and a partial post is not recoverable.
if CLASS=$(secret_class "$BODY"); then
  refuse_secret "the review body" "$CLASS"
fi
scan_i=0
while [ "$scan_i" -lt "$COUNT" ]; do
  jq -r ".[$scan_i].body // empty" "$FINDINGS" > "$WORK/scan-$scan_i.md"
  if CLASS=$(secret_class "$WORK/scan-$scan_i.md"); then
    refuse_secret "a finding body" "$CLASS"
  fi
  scan_i=$((scan_i + 1))
done

# --- read verdicts.json -------------------------------------------------------
#
# Optional, and a malformed one degrades to "everything is supported" rather
# than taking the review down: the verifier is a filter on the review, not a
# gate on posting it. Only entries that name an integer index and a boolean
# survive, so a half-written verdict cannot suppress a finding by accident.
#
# VERIFIER_RAN records the difference the verdict rule turns on: a readable
# verdicts file means the verifier ran and had the chance to rule on every
# finding, so silence about one is silence from a verifier that saw it.
VERIFIER_RAN=false
echo '[]' > "$VERDICTS"
if [ -s "$VERDICTS_JSON" ]; then
  if jq -e 'type == "object" and ((.verdicts | type) == "array")' \
      "$VERDICTS_JSON" > /dev/null 2>&1; then
    jq -c '[.verdicts[]
            | select((.index | type) == "number" and (.supported | type) == "boolean")]' \
      "$VERDICTS_JSON" > "$VERDICTS"
    VERIFIER_RAN=true
  else
    echo "::warning::$VERDICTS_JSON is not a verdicts object; treating every finding as supported"
  fi
fi

# The reasons are answer text: they are quoted in the Low confidence list and
# reach the comment. Scanned here, before the first API call, so a token in a
# reason costs the run rather than being caught after inline comments are public.
# Entries jq dropped above cannot be printed, so only the surviving ones matter.
if CLASS=$(secret_class "$VERDICTS"); then
  refuse_secret "a verifier reason" "$CLASS"
fi

# supported <index> -> "yes" | "no <reason>". A finding with no verdict entry is
# NOT supported once the verifier ran: an unchecked finding posted inline reads
# as a checked one. With no verdicts file at all it is supported, because there
# was no verifier to check it. `last` is the duplicate rule from the header: the
# last verdict entry for an index wins.
verdict_of() {
  jq -r --argjson i "$1" --arg ran "$VERIFIER_RAN" \
    'map(select(.index == $i)) | last
     | if . == null then (if $ran == "true" then "no no verdict" else "yes" end)
       elif .supported then "yes"
       else "no " + ((.reason // "no reason given") | gsub("[\r\n]+"; " "))
       end' "$VERDICTS"
}

NOT_POSTED="$WORK/not-posted.txt"
LOW_CONFIDENCE="$WORK/low-confidence.txt"
: > "$NOT_POSTED"
: > "$LOW_CONFIDENCE"

post_finding() { # post_finding <index>
  local i="$1" fpath line start_line side severity body_file
  fpath=$(jq -r ".[$i].path // empty" "$FINDINGS")
  line=$(jq -r ".[$i].line // empty" "$FINDINGS")
  start_line=$(jq -r ".[$i].start_line // empty" "$FINDINGS")
  side=$(jq -r ".[$i].side // \"RIGHT\"" "$FINDINGS")
  severity=$(jq -r ".[$i].severity // empty" "$FINDINGS")
  body_file="$WORK/finding-$i.md"
  jq -r ".[$i].body // empty" "$FINDINGS" > "$body_file"

  # Only paths the PR actually changed. Without this the model could be
  # steered into commenting on a file the PR never touched, and GitHub would
  # accept it.
  if [ -z "$fpath" ] || ! grep -qxF "$fpath" "$REVIEW_IN/files.txt"; then
    echo "::warning::Skipping a finding: '$fpath' is not a file this PR changes"
    return 0
  fi
  case "$line" in
    '' | *[!0-9]*) echo "::warning::Skipping a finding on $fpath: line '$line' is not a number"; return 0 ;;
  esac
  if [ "$line" -lt 1 ]; then
    echo "::warning::Skipping a finding on $fpath: line $line is below 1"
    return 0
  fi
  if [ -n "$start_line" ]; then
    case "$start_line" in
      *[!0-9]*) echo "::warning::Skipping a finding on $fpath: start_line '$start_line' is not a number"; return 0 ;;
    esac
    if [ "$start_line" -lt 1 ] || [ "$start_line" -gt "$line" ]; then
      echo "::warning::Skipping a finding on $fpath: start_line $start_line is not within 1..$line"
      return 0
    fi
  fi
  if [ "$side" != "LEFT" ] && [ "$side" != "RIGHT" ]; then
    echo "::warning::Skipping a finding on $fpath:$line: side '$side' is neither LEFT nor RIGHT"
    return 0
  fi
  # The severity is what a reader triages on and what the Low confidence list
  # prints, so an unrecognised one is a finding nobody can act on rather than a
  # cosmetic problem.
  case "$severity" in
    P0 | P1 | P2) : ;;
    *) echo "::warning::Skipping a finding on $fpath:$line: severity '$severity' is not P0, P1 or P2"; return 0 ;;
  esac
  if ! grep -q '[^[:space:]]' "$body_file"; then
    echo "::warning::Skipping a finding on $fpath:$line: its body is empty"
    return 0
  fi
  local bytes
  bytes=$(wc -c < "$body_file" | tr -d ' ')
  if [ "$bytes" -gt "$MAX_FINDING_BYTES" ]; then
    echo "::warning::Skipping a finding on $fpath:$line: $bytes bytes, over the $MAX_FINDING_BYTES limit"
    return 0
  fi

  # The verifier's judgement is applied after the shape checks, so a rejected
  # finding is only listed once it is known to be a real, postable one.
  local verdict
  verdict=$(verdict_of "$i")
  if [ "${verdict%% *}" = "no" ]; then
    printf '%s\t%s\t%s\t%s\n' "$fpath:$line" "$severity" \
      "$(head -1 "$body_file")" "${verdict#no }" >> "$LOW_CONFIDENCE"
    echo "::warning::The verifier rejected a finding on $fpath:$line; listing it under Low confidence"
    return 0
  fi

  local args
  args=(-X POST -F "body=@$body_file" -f "commit_id=$HEAD_SHA"
    -f "path=$fpath" -F "line=$line" -f "side=$side")
  # GitHub requires start_side on a multi-line comment. Without it the request
  # is a 422 and the finding falls to the sticky list, which reads as a line
  # outside the diff rather than as a field this step forgot to send.
  if [ -n "$start_line" ]; then
    args+=(-F "start_line=$start_line" -f "start_side=$side")
  fi
  # A per-finding failure is expected traffic, not an incident: GitHub answers
  # 422 for any line outside the diff hunks, and the model cannot always tell.
  # The finding still reaches the reader, in the sticky comment's list.
  if ! gh api "repos/$GITHUB_REPOSITORY/pulls/$PR_NUMBER/comments" "${args[@]}" > /dev/null 2>&1; then
    echo "::warning::GitHub refused an inline comment on $fpath:$line; listing it in the review instead"
    printf '%s\t%s\t%s\n' "$fpath:$line" "$severity" \
      "$(head -1 "$body_file" | cut -c "1-$MAX_SUMMARY_CHARS")" >> "$NOT_POSTED"
  fi
}

HANDLED=0
i=0
while [ "$i" -lt "$COUNT" ]; do
  if [ "$HANDLED" -ge "$MAX_FINDINGS" ]; then
    fpath=$(jq -r ".[$i].path // \"?\"" "$FINDINGS")
    line=$(jq -r ".[$i].line // \"?\"" "$FINDINGS")
    severity=$(jq -r ".[$i].severity // \"?\"" "$FINDINGS")
    summary=$(jq -r ".[$i].body // \"\"" "$FINDINGS" | head -1 | cut -c "1-$MAX_SUMMARY_CHARS")
    printf '%s\t%s\t%s\n' "$fpath:$line" "$severity" "$summary" >> "$NOT_POSTED"
  else
    post_finding "$i"
    HANDLED=$((HANDLED + 1))
  fi
  i=$((i + 1))
done

# --- assemble and post the sticky review --------------------------------------
#
# Both lists are appended after the footer rather than woven into the body: the
# model owns the review's structure, and a heading inserted mid-document would
# break it.
cp "$BODY" "$STICKY"
STICKY_BYTES="$BODY_BYTES"

append_section() { # append_section <file> <bytes-so-far> -> new byte count
  local section="$1" have="$2" bytes
  bytes=$(wc -c < "$section" | tr -d ' ')
  if [ "$((have + bytes))" -le "$MAX_STICKY_BYTES" ]; then
    cat "$section" >> "$STICKY"
    echo "$((have + bytes))"
  else
    echo "$have"
  fi
}

if [ -s "$NOT_POSTED" ]; then
  LISTED_TOTAL=$(wc -l < "$NOT_POSTED" | tr -d ' ')
  {
    printf '\n\n### Findings not posted inline\n\n'
    head -n "$MAX_LISTED" "$NOT_POSTED" \
      | awk -F'\t' '{printf "- `%s` %s: %s\n", $1, $2, $3}'
    if [ "$LISTED_TOTAL" -gt "$MAX_LISTED" ]; then
      printf -- '- …and %s more\n' "$((LISTED_TOTAL - MAX_LISTED))"
    fi
  } > "$WORK/list.md"
  BEFORE="$STICKY_BYTES"
  STICKY_BYTES=$(append_section "$WORK/list.md" "$STICKY_BYTES")
  if [ "$STICKY_BYTES" = "$BEFORE" ]; then
    # The list itself would push the comment past what GitHub accepts, so the
    # count goes in and the paths stay in the run log.
    printf '\n\n### Findings not posted inline\n\n- %s findings did not fit; see the run log.\n' \
      "$LISTED_TOTAL" > "$WORK/list-short.md"
    STICKY_BYTES=$(append_section "$WORK/list-short.md" "$STICKY_BYTES")
  fi
fi

if [ -s "$LOW_CONFIDENCE" ]; then
  # Listed rather than dropped: a verifier whose rejections are invisible
  # cannot be checked, and this is the only place a human sees what it threw
  # away.
  LOW_TOTAL=$(wc -l < "$LOW_CONFIDENCE" | tr -d ' ')
  {
    printf '\n\n### Low confidence\n\n'
    head -n "$MAX_LISTED" "$LOW_CONFIDENCE" \
      | awk -F'\t' '{printf "- `%s` %s: %s (verifier: %s)\n", $1, $2, $3, $4}'
    if [ "$LOW_TOTAL" -gt "$MAX_LISTED" ]; then
      printf -- '- …and %s more\n' "$((LOW_TOTAL - MAX_LISTED))"
    fi
  } > "$WORK/low.md"
  BEFORE="$STICKY_BYTES"
  STICKY_BYTES=$(append_section "$WORK/low.md" "$STICKY_BYTES")
  if [ "$STICKY_BYTES" = "$BEFORE" ]; then
    printf '\n\n### Low confidence\n\n- %s findings were rejected by the verifier; see the run log.\n' \
      "$LOW_TOTAL" > "$WORK/low-short.md"
    STICKY_BYTES=$(append_section "$WORK/low-short.md" "$STICKY_BYTES")
  fi
fi

# The last thing that happens before the comment is public. The body was scanned
# on its own, but the appended lists carry finding summaries and verifier reasons
# that were not part of it, and this file is exactly what GitHub will publish.
if CLASS=$(secret_class "$STICKY"); then
  refuse_secret "the assembled review comment" "$CLASS"
fi

if [ -n "$EXISTING_REVIEW_ID" ]; then
  if ! gh api "repos/$GITHUB_REPOSITORY/issues/comments/$EXISTING_REVIEW_ID" \
      -X PATCH -F "body=@$STICKY" > /dev/null; then
    echo "::error::Could not update the sticky review comment $EXISTING_REVIEW_ID"
    exit 1
  fi
  echo "Updated the sticky review comment $EXISTING_REVIEW_ID on #$PR_NUMBER"
else
  if ! gh api "repos/$GITHUB_REPOSITORY/issues/$PR_NUMBER/comments" \
      -X POST -F "body=@$STICKY" > /dev/null; then
    echo "::error::Could not post the review comment on #$PR_NUMBER"
    exit 1
  fi
  echo "Posted the review comment on #$PR_NUMBER"
fi

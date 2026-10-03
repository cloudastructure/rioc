#!/usr/bin/env bash
# Gather everything the @claude re-score needs, detect a stale re-request, and
# post or update the progress indicator.
#
# Ported from the reviewer's inline step. Two changes:
#
#   - Reviews from the optimization rollout came from github-actions[bot];
#     older reviews and all future publishing use claude[bot]. Both trusted
#     authors supply history. New writes are Claude-owned; the progress and
#     ledger markers identify their own comment types.
#   - The request body is written to a FILE rather than interpolated into the
#     prompt. It is attacker-supplied text, and a static prompt file cannot
#     safely carry it inline.
#
# Env in: GH_TOKEN, GITHUB_REPOSITORY, GITHUB_ENV, GITHUB_EVENT_PATH.
set -euo pipefail

# GitHub returns 30 comments per page by default and the sticky review is the
# LAST comment, so an unpaginated read loses it on any PR past 30 comments
# (measured 2026-09-01 on PR #3942: 43 comments, plain query returned 30).
# Here that also lets a bare "@claude" slip past stale-request detection and
# spend a run on a commit that was already reviewed.
#
# --paginate emits one array PER PAGE, so pages are slurped and added before
# any filter runs. A listing error is fatal rather than an empty list, which
# would read as "nothing here yet" and duplicate comments.
all_comments() {
  local listing
  if ! listing=$(gh api --paginate \
      "repos/$GITHUB_REPOSITORY/issues/$1/comments?per_page=100" 2>/dev/null); then
    echo "::error::Could not list comments for #$1" >&2
    return 1
  fi
  printf '%s' "$listing" | jq -s -c '(add // [])'
}

# Under RUNNER_TEMP, which the runner clears between jobs, rather than /tmp,
# which a self-hosted runner keeps across them. prepare-review-inputs.sh reads
# the same path. Truncated here rather than only at the write below, so the
# early exit on a payload with no issue number cannot leave another PR's
# request behind.
REQUEST_FILE="${RUNNER_TEMP:-/tmp}/claude-request.txt"
: > "$REQUEST_FILE"

ISSUE_NUMBER=$(jq -r '.issue.number // .pull_request.number // empty' "$GITHUB_EVENT_PATH")
{
  echo "PR_NUMBER=$ISSUE_NUMBER"
  echo "RUN_STARTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "STALE_REQUEST=false"
  echo "IS_PR=false"
  # A mention arrives after a first pass has already run, so the model is shown
  # the prior review and the inline comments already on the PR and re-scores
  # against them rather than starting over.
  echo "REVIEW_MODE=rescore"
} >> "$GITHUB_ENV"

if [ -z "$ISSUE_NUMBER" ]; then
  exit 0
fi

# The request that triggered this run, as data. `issues` carries it on the
# issue body; the comment triggers carry it on the comment or review.
jq -r 'if .action == "opened" or .action == "assigned" then ((.issue.title // "") + "\n\n" + (.issue.body // ""))
       else (.comment.body // .review.body // "") end' \
  "$GITHUB_EVENT_PATH" > "$REQUEST_FILE"

COMMENTS=$(all_comments "$ISSUE_NUMBER")

# Only a leading progress marker identifies an indicator. A review can quote
# the marker without becoming a PATCH target. Update only Claude-owned output.
EXISTING_PROGRESS_ID=$(printf '%s' "$COMMENTS" \
  | jq -r '[.[] | select(.user.login == "claude[bot]" and ((.body // "") | startswith("<!-- progress-indicator -->")))] | last | .id // empty')

# Upsert helper: update the existing progress comment if found, otherwise create a new one
upsert_progress() {
  local body="$1"
  if [ -n "$EXISTING_PROGRESS_ID" ]; then
    gh api "repos/$GITHUB_REPOSITORY/issues/comments/$EXISTING_PROGRESS_ID" \
      -X PATCH -f body="$(printf '<!-- progress-indicator -->\n%s' "$body")" > /dev/null || true
    echo "PROGRESS_COMMENT_ID=$EXISTING_PROGRESS_ID" >> "$GITHUB_ENV"
  else
    local NEW_ID
    NEW_ID=$(gh api "repos/$GITHUB_REPOSITORY/issues/$ISSUE_NUMBER/comments" \
      -X POST -f body="$(printf '<!-- progress-indicator -->\n%s' "$body")" --jq '.id')
    echo "PROGRESS_COMMENT_ID=$NEW_ID" >> "$GITHUB_ENV"
  fi
}

# Check if this event refers to a PR (not a plain issue), from the event
# payload rather than an API lookup: a transient `gh` error would otherwise
# read as a confirmed plain issue, skip the PR checkout, and let the reviewer
# assignment act on an older 5/5 for this PR. `pull_request` is the payload
# root on pull_request_review and pull_request_review_comment; GitHub puts
# `pull_request` under `issue` only when that issue is a PR, so an
# issue_comment on a plain issue and an `issues` event both have neither.
# https://docs.github.com/en/webhooks/webhook-events-and-payloads#issue_comment
# (verified 2026-09-02: `gh api repos/cli/cli/issues/<n> --jq '.pull_request
# != null'` is true for a PR number and false for a plain issue number.)
IS_PR=$(jq -r 'if (.pull_request // .issue.pull_request) then "true" else "" end' "$GITHUB_EVENT_PATH")
if [ -z "$IS_PR" ]; then
  upsert_progress "👀 **@claude is on it...** (~2 min) Processing your request"
  exit 0
fi
echo "IS_PR=true" >> "$GITHUB_ENV"

# Capture existing review comment ID, review number, commit msg and trigger comment.
# New output is owned by Claude. Older Actions comments remain history;
# do not PATCH another app's comment or make a refused cross-app edit sticky.
# A marked comment is canonical after the publisher migration. If a previous
# race left more than one marked comment, the newest marked body is the latest
# canonical state; ties use the highest comment id. Before any marker exists,
# migrate the oldest Claude score comment, which is the original sticky.
REVIEW_COMMENTS=$(printf '%s' "$COMMENTS" | jq -c '
  [.[]
   | select((.user.login == "github-actions[bot]" or .user.login == "claude[bot]")
            and ((.body // "") | contains("<!-- score:")))]
  | sort_by([(.updated_at // .created_at // ""), (.id // 0)])
')
EXISTING_REVIEW_ID=$(printf '%s' "$REVIEW_COMMENTS" | jq -r '
  map(select(.user.login == "claude[bot]"))
  |   [ .[] | select((.body // "") | startswith("<!-- claude-sticky-review -->")) ] as $marked
  | if ($marked | length) > 0
    then ($marked | sort_by([(.updated_at // .created_at // ""), (.id // 0)]) | last)
    else (sort_by([(.created_at // ""), (.id // 0)]) | first)
    end
  | .id // empty
')
echo "EXISTING_REVIEW_ID=$EXISTING_REVIEW_ID" >> "$GITHUB_ENV"

REVIEW_COUNT=$(printf '%s' "$REVIEW_COMMENTS" | jq 'length')
# Canonical PATCH target and review history are separate concerns. A legacy
# sticky (or the newest marked comment) is the one we update, but bare
# `@claude review` must compare the request with the latest body that was
# actually analyzed, including a later duplicate left by an earlier race.
LAST_REVIEW=$(printf '%s' "$REVIEW_COMMENTS" \
  | jq -r 'sort_by([(.updated_at // .created_at // ""), (.id // 0)]) | last | .body // empty')
# Every re-score PATCHes the same sticky comment, so the comment count stops
# growing after the first review (#1, #2, #2, ...). The sticky review's own
# header carries the real number; the count is the fallback for the first
# review and for a body whose header does not parse.
# Number from the highest parsed header across trusted comments. The canonical
# sticky is selected separately for patching, so a duplicate cannot make the
# next number go backwards merely because the canonical comment is older.
MAX_REVIEW_N=$(printf '%s' "$REVIEW_COMMENTS" \
  | jq -r '.[].body // empty' \
  | grep -oP '^(?:## 🔍 Review #|\*Review #)\K[0-9]+' \
  | sort -n | tail -1 || true)
if [ -n "$MAX_REVIEW_N" ]; then
  echo "REVIEW_NUMBER=$((MAX_REVIEW_N + 1))" >> "$GITHUB_ENV"
else
  echo "REVIEW_NUMBER=$((REVIEW_COUNT + 1))" >> "$GITHUB_ENV"
fi

CURRENT_HEAD=$(gh pr view "$ISSUE_NUMBER" --json headRefOid -q '.headRefOid' 2>/dev/null || echo "")
{
  echo "HEAD_SHA=$CURRENT_HEAD"
  echo "SHORT_SHA=${CURRENT_HEAD:0:7}"
} >> "$GITHUB_ENV"

COMMIT_MSG=$(gh api "repos/$GITHUB_REPOSITORY/commits/$CURRENT_HEAD" \
  --jq '.commit.message | split("\n")[0]' 2>/dev/null | cut -c1-72 | tr -d '\r\n"' || echo "")
echo "COMMIT_MSG=$COMMIT_MSG" >> "$GITHUB_ENV"

TRIGGER_COMMENT=$(sed 's/@claude[[:space:]]*//' "$REQUEST_FILE" | tr -d '\r\n' | cut -c1-80 || echo "")
echo "TRIGGER_COMMENT=$TRIGGER_COMMENT" >> "$GITHUB_ENV"

# Normalize: strip @claude, the word "review", and whitespace to detect bare re-requests
STRIPPED=$(sed 's/@claude//g' "$REQUEST_FILE" | sed 's/\breview\b//gi' | tr -d '[:space:]')

# If there's any extra context beyond "@claude review", proceed normally — the reviewer handles it
if [ -n "$STRIPPED" ]; then
  upsert_progress "👀 **@claude is on it...** (~2 min) Processing your request"
  exit 0
fi

# Bare "@claude review" — check if there are new commits since the last review
CURRENT_SHORT="${CURRENT_HEAD:0:7}"

if [ -z "$LAST_REVIEW" ]; then
  # No prior review — proceed normally
  upsert_progress "👀 **@claude is on it...** (~2 min) Processing your request"
  exit 0
fi

# Extract last reviewed SHA from footer (pattern: `abc1234`](https://github.com)
LAST_SHA=$(echo "$LAST_REVIEW" | grep -oP '`[a-f0-9]{7}`\]\(https://github\.com' | grep -oP '[a-f0-9]{7}' | head -1 || echo "")
LAST_SCORE=$(echo "$LAST_REVIEW" | grep -oP '<!-- score:\s*\K[0-9]+' || echo "?")

# Only the publisher's two-line envelope marks an incomplete review. Quoted
# markers, including standalone lines inside a code fence, are review content.
if [ -n "$LAST_SHA" ] && [ "$CURRENT_SHORT" = "$LAST_SHA" ] \
    && [[ "$LAST_REVIEW" != $'<!-- claude-sticky-review -->\n<!-- claude-review-incomplete -->\n'* ]]; then
  # Stale re-request: same SHA, no extra context — post explanation and skip the reviewer
  echo "STALE_REQUEST=true" >> "$GITHUB_ENV"
  STALE_BODY="$(printf '👋 **No new commits since the last review** (score: %s/5 · SHA `%s`).\n\nTo trigger a re-score, either:\n- **Push a fix commit** addressing the open findings, then comment `@claude`\n- **Justify a finding** you disagree with: `@claude dismiss P1: <finding title> — <your reason>`' \
    "$LAST_SCORE" "$CURRENT_SHORT")"
  upsert_progress "$STALE_BODY"
else
  # New commits since last review — proceed normally
  upsert_progress "👀 **@claude is on it...** (~2 min) Processing your request"
fi

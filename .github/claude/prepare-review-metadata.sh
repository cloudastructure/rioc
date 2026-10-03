#!/usr/bin/env bash
# Gather everything the first-pass review needs, and post the progress
# indicator before the slow steps so an opened PR shows something immediately.
#
# The trusted scripts post through `gh` with a dedicated Claude app token.
# Both Claude and Actions comments
# supply review history, but only a Claude comment is updated in place.
#
# Env in: GH_TOKEN, GITHUB_REPOSITORY, GITHUB_ENV, GITHUB_EVENT_PATH,
#         PR number from the event or the workflow_dispatch input.
set -euo pipefail

# GitHub returns 30 comments per page by default, and the sticky review is the
# LAST comment, so an unpaginated read loses it on any PR past 30 comments.
# Measured 2026-09-01 on PR #3942: 43 comments, of which the plain query
# returned 30. Losing it means EXISTING_REVIEW_ID comes back empty and the bot
# posts a second sticky review instead of updating the first.
#
# --paginate emits one array PER PAGE, so the result is slurped and added
# before any filter runs; `last` over per-page arrays would otherwise mean
# "last of some page". A listing error is fatal rather than an empty list: an
# empty list reads as "no review yet" and silently duplicates the comment.
all_comments() {
  local listing
  if ! listing=$(gh api --paginate \
      "repos/$GITHUB_REPOSITORY/issues/$1/comments?per_page=100" 2>/dev/null); then
    echo "::error::Could not list comments for #$1" >&2
    return 1
  fi
  printf '%s' "$listing" | jq -s -c '(add // [])'
}

PR=$(jq -r '.pull_request.number // .inputs.pr_number // empty' "$GITHUB_EVENT_PATH")
if [ -z "$PR" ]; then
  echo "::error::Could not determine the PR number from the event payload"
  exit 1
fi

{
  echo "PR_NUMBER=$PR"
  echo "RUN_STARTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >> "$GITHUB_ENV"

# From the event payload where there is one: on a pull_request event the
# checkout is the event's head, and a push between the event and this step would
# otherwise report a SHA that does not describe the tree being reviewed. A
# workflow_dispatch payload has no pull_request, so it falls back to the lookup.
HEAD_SHA=$(jq -r '.pull_request.head.sha // empty' "$GITHUB_EVENT_PATH")
if [ -z "$HEAD_SHA" ]; then
  HEAD_SHA=$(gh pr view "$PR" --json headRefOid -q '.headRefOid')
fi
SHORT_SHA="${HEAD_SHA:0:7}"
{
  echo "HEAD_SHA=$HEAD_SHA"
  echo "SHORT_SHA=$SHORT_SHA"
} >> "$GITHUB_ENV"

COMMENTS=$(all_comments "$PR")
# `claude-sticky-review` is added by publish-review.sh. It makes canonical
# comments explicit after the first post following the publisher migration. If
# a race left more than one marked comment, the newest marked body is the
# latest canonical state; ties use the highest comment id. For an older PR
# without that marker, migrate the oldest Claude score comment, which is the
# original sticky, while later comments may be accidental duplicates.
REVIEW_COMMENTS=$(printf '%s' "$COMMENTS" | jq -c '
  [.[]
   | select((.user.login == "github-actions[bot]" or .user.login == "claude[bot]")
            and ((.body // "") | contains("<!-- score:")))]
  | sort_by([(.updated_at // .created_at // ""), (.id // 0)])
')
REVIEW_COUNT=$(printf '%s' "$REVIEW_COMMENTS" | jq 'length')
# Every re-score PATCHes the same sticky comment, so the comment count stops
# growing after the first review (#1, #2, #2, ...). The sticky review's own
# header carries the real number. Numbering uses the highest parsed header
# across trusted comments so an accidental duplicate cannot make a later run
# go backwards just because the canonical sticky is the oldest comment.
MAX_REVIEW_N=$(printf '%s' "$REVIEW_COMMENTS" \
  | jq -r '.[].body // empty' \
  | grep -oP '^(?:## 🔍 Review #|\*Review #)\K[0-9]+' \
  | sort -n | tail -1 || true)
if [ -n "$MAX_REVIEW_N" ]; then
  REVIEW_N=$((MAX_REVIEW_N + 1))
else
  REVIEW_N=$((REVIEW_COUNT + 1))
fi
echo "REVIEW_NUMBER=$REVIEW_N" >> "$GITHUB_ENV"

# The first pass reviews fresh: no prior review is fed back to the model, so
# prepare-review-inputs.sh blanks prior-review.md and review-comments.json.
echo "REVIEW_MODE=first" >> "$GITHUB_ENV"

COMMIT_MSG=$(gh api "repos/$GITHUB_REPOSITORY/commits/$HEAD_SHA" \
  --jq '.commit.message | split("\n")[0]' 2>/dev/null | cut -c1-72 | tr -d '\r\n"' || echo "")
echo "COMMIT_MSG=$COMMIT_MSG" >> "$GITHUB_ENV"

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

# Reuse the indicator on dispatch/ready-for-review just as a mention does.
# A score comment quoting this marker must never become a progress target.
PROGRESS_COMMENT_ID=$(printf '%s' "$COMMENTS" | jq -r '
  [.[] | select(.user.login == "claude[bot]"
    and ((.body // "") | startswith("<!-- progress-indicator -->")))]
  | last | .id // empty')
PROGRESS_BODY=$(printf '<!-- progress-indicator -->\n👀 **Review #%d in progress...** (~4 min) Analyzing [%s](https://github.com/%s/commit/%s)' \
  "$REVIEW_N" "$SHORT_SHA" "$GITHUB_REPOSITORY" "$HEAD_SHA")
if [ -n "$PROGRESS_COMMENT_ID" ]; then
  gh api "repos/$GITHUB_REPOSITORY/issues/comments/$PROGRESS_COMMENT_ID" \
    -X PATCH -f body="$PROGRESS_BODY" > /dev/null
else
  PROGRESS_COMMENT_ID=$(gh api "repos/$GITHUB_REPOSITORY/issues/$PR/comments" \
    -X POST -f body="$PROGRESS_BODY" --jq '.id')
fi
echo "PROGRESS_COMMENT_ID=$PROGRESS_COMMENT_ID" >> "$GITHUB_ENV"

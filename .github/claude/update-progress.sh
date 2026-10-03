#!/usr/bin/env bash
# Turn the "in progress" comment into the outcome: stars and the score on a
# completed review, an error note when the reviewer failed, and deletion when
# the review landed without a readable score.
#
# Ported unchanged from the reviewer's two inline copies, which differed only
# in the word "Review" vs "Re-score" — that word is now PROGRESS_VERB.
#
# Env in:  PROGRESS_COMMENT_ID, REVIEW_OUTCOME, PROGRESS_VERB, PR_NUMBER, IS_PR,
#          REVIEW_NUMBER, SHORT_SHA, HEAD_SHA, RUN_STARTED_AT,
#          REVIEW_INCOMPLETE, MODEL_OUTCOME, MODEL_EXECUTION_FILE,
#          CAPTURE_OUTCOME, PUBLISH_OUTCOME, GH_TOKEN, GITHUB_REPOSITORY, GITHUB_RUN_ID.
# Env out (GITHUB_OUTPUT): score_missing — true when a run that did not fail
#          left no readable score behind.
set -euo pipefail

VERB="${PROGRESS_VERB:-Review}"
COMMENT="repos/$GITHUB_REPOSITORY/issues/comments/${PROGRESS_COMMENT_ID:-}"

if [ "${REVIEW_OUTCOME:-}" = "failure" ]; then
  # Outcomes identify the failed stage, not its root cause. In particular, an
  # action can fail during Bun installation or actor validation before it ever
  # returns execution data. Never label all of those failures as API timeouts.
  if [ "${MODEL_OUTCOME:-}" = "skipped" ]; then
    PR_REASON="Reviewer setup failed; the review action did not run."
    ISSUE_REASON="Request setup failed before processing began."
  elif [ "${MODEL_OUTCOME:-}" = "failure" ] || [ "${MODEL_OUTCOME:-}" = "cancelled" ]; then
    if [ -n "${MODEL_EXECUTION_FILE:-}" ] && [ -s "$MODEL_EXECUTION_FILE" ]; then
      PR_REASON="The review action failed during execution. The underlying error is in the run logs."
      ISSUE_REASON="Request processing failed during execution. The underlying error is in the run logs."
    else
      PR_REASON="The review action failed without producing execution data. Check the run logs for dependency installation, authorization, or startup errors; the exact cause is not available here."
      ISSUE_REASON="Request processing failed without producing execution data. Check the run logs for dependency installation, authorization, or startup errors; the exact cause is not available here."
    fi
  elif [ "${MODEL_OUTCOME:-}" = "success" ] && [ "${CAPTURE_OUTCOME:-}" != "success" ]; then
    PR_REASON="The review action finished, but its output could not be captured as a usable review."
    ISSUE_REASON="The request was processed, but its response could not be captured."
  elif [ "${CAPTURE_OUTCOME:-}" = "success" ] && [ "${PUBLISH_OUTCOME:-}" = "failure" ]; then
    PR_REASON="A review was generated, but publication failed. Some comments may have been posted; the result is not complete."
    ISSUE_REASON="A response was generated, but publication failed. Some output may have been posted; the result is not complete."
  else
    PR_REASON="The review attempt failed; see the run logs for the cause."
    ISSUE_REASON="The request failed; see the run logs for the cause."
  fi

  if [ "${IS_PR:-true}" = "true" ]; then
    REASON="$PR_REASON"
  else
    REASON="$ISSUE_REASON"
  fi
  # Only fixed text is exported, never model output or raw logs. The isolated
  # status job uses the same explanation without holding an AI-accessible token.
  if [ -n "${GITHUB_OUTPUT:-}" ]; then
    printf 'failure_summary=%s\n' "$REASON" >> "$GITHUB_OUTPUT"
  fi
  if [ -n "${PROGRESS_COMMENT_ID:-}" ]; then
    # Markdown backticks are literal text.
    # shellcheck disable=SC2016
    if [ "${IS_PR:-true}" = "true" ]; then
      gh api "$COMMENT" -X PATCH \
        -f body="$(printf '<!-- progress-indicator -->\n❌ **No review completed — %s failed.** %s\n\nThis attempt is not an approval. Any earlier score belongs to an earlier review. [View run logs](https://github.com/%s/actions/runs/%s). After the failure is addressed, comment `@claude` to request another review.' \
          "$VERB" "$REASON" "$GITHUB_REPOSITORY" "$GITHUB_RUN_ID")" \
        || true
    else
      gh api "$COMMENT" -X PATCH \
        -f body="$(printf '<!-- progress-indicator -->\n❌ **Request failed.** %s\n\n[View run logs](https://github.com/%s/actions/runs/%s). After the failure is addressed, comment `@claude` to retry the request.' \
          "$REASON" "$GITHUB_REPOSITORY" "$GITHUB_RUN_ID")" \
        || true
    fi
  fi
  exit 0
fi

if [ -z "${PROGRESS_COMMENT_ID:-}" ]; then
  exit 0
fi

# A run that completed but posted nothing must not be scored from the comment
# list: the newest score there belongs to the PREVIOUS review, and reading it
# would announce a stale number as this run's result. The caller has already
# said so on the PR; the indicator only has to stop claiming success.
if [ "${RESPONSE_MISSING:-}" = "true" ]; then
  gh api "$COMMENT" -X PATCH \
    -f body="$(printf '<!-- progress-indicator -->\n⚠️ **%s produced no response** — the run finished without errors but posted nothing. The score above, if any, is from the previous review. [View run logs](https://github.com/%s/actions/runs/%s)' \
      "$VERB" "$GITHUB_REPOSITORY" "$GITHUB_RUN_ID")" \
    || true
  exit 0
fi

# A published review whose verifier or recompose pass did not finish is a
# successful publication with an incomplete result. Do not inspect the comment
# list here: a score from an earlier run must not turn this state into a false
# completion. Keep score_missing unset so the caller can report the distinct
# verification state rather than a missing publication.
if [ "${REVIEW_INCOMPLETE:-}" = "true" ]; then
  gh api "$COMMENT" -X PATCH \
    -f body="$(printf '<!-- progress-indicator -->\n⚠️ **%s published — verification incomplete** — the review was posted, but verification did not finish. Analyzed [%s](https://github.com/%s/commit/%s) · [View run logs](https://github.com/%s/actions/runs/%s)' \
      "$VERB" "$SHORT_SHA" "$GITHUB_REPOSITORY" "$HEAD_SHA" "$GITHUB_REPOSITORY" "$GITHUB_RUN_ID")" \
    || true
  exit 0
fi

# Paginated: an unpaginated read misses a sticky review that has scrolled off
# the first comment page, which would delete the indicator and mark an
# otherwise good run as structurally failed.
#
# The listing is captured on its own so a failed read is not collapsed into an
# empty score: that would delete the indicator and report a structurally
# malformed review for a review that posted fine and a GitHub read that did
# not. An infra failure is loud instead — the indicator stays, score_missing
# stays unset, and the step exits non-zero.
if ! LISTING=$(gh api --paginate "repos/$GITHUB_REPOSITORY/issues/$PR_NUMBER/comments?per_page=100" 2>/dev/null); then
  echo "::error::Could not list comments for #$PR_NUMBER; the progress indicator is left as it stands and no score is claimed" >&2
  exit 1
fi

SCORE_BODY=$(printf '%s' "$LISTING" \
  | jq -s -r \
    --arg head_sha "${HEAD_SHA:-}" \
    --arg run_started_at "${RUN_STARTED_AT:-}" \
    --arg sticky_marker '<!-- claude-sticky-review -->' \
    '
      def body_text:
        if (.body? | type) == "string" then .body else "" end;
      def trusted_author:
        .user.login == "github-actions[bot]" or .user.login == "claude[bot]";
      def score_comment:
        body_text | contains("<!-- score:");
      def current_head:
        $head_sha == "" or (body_text | contains($head_sha));
      def current_run:
        $run_started_at == "" or ((.updated_at // "") > $run_started_at);

      (add // [])
      | [.[]
         | select(trusted_author and score_comment and current_head and current_run)]
      | (map(select(body_text | startswith($sticky_marker))) as $marked
         | (if ($marked | length) > 0 then $marked else . end
            | max_by([(.updated_at // .created_at // ""), (.id // 0)])
            | body_text))
    ' \
  )

SCORE=$(printf '%s' "$SCORE_BODY" \
  | grep -oP '<!-- score:\s*\K[0-9]+' || echo "")

case "$SCORE" in
  5) STARS="⭐⭐⭐⭐⭐" ;;
  4) STARS="⭐⭐⭐⭐" ;;
  3) STARS="⭐⭐⭐" ;;
  2) STARS="⭐⭐" ;;
  1) STARS="⭐" ;;
  *) STARS="" ;;
esac

if [ -z "$STARS" ]; then
  # No score to show. Delete the indicator rather than leave "in progress"
  # standing forever, and tell the caller so the job can go red.
  if [ -n "${GITHUB_OUTPUT:-}" ]; then
    echo "score_missing=true" >> "$GITHUB_OUTPUT"
  fi
  gh api "$COMMENT" -X DELETE || true
  exit 0
fi

gh api "$COMMENT" -X PATCH \
  -f body="$(printf '<!-- progress-indicator -->\n✅ **%s #%d complete** — %s %s/5 · [%s](https://github.com/%s/commit/%s)' \
    "$VERB" "${REVIEW_NUMBER:-1}" "$STARS" "$SCORE" "$SHORT_SHA" "$GITHUB_REPOSITORY" "$HEAD_SHA")" \
  || true

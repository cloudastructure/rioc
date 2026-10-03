#!/usr/bin/env bash
# Confirm that this run actually published its response. Workflow-file changes
# alone are not evidence of an upstream guard: publishing and API failures must
# remain failures rather than being classified as successful skips.
set -euo pipefail

missing() {
  printf 'review_missing=true\nresponse_missing=true\n' >> "$GITHUB_OUTPUT"
  echo "::error::$1"
  exit 1
}

if [ "${MODEL_OUTCOME:-}" != "success" ] \
    || [ "${CAPTURE_OUTCOME:-}" != "success" ] \
    || [ "${PUBLISH_OUTCOME:-}" != "success" ]; then
  missing 'This run did not finish generating and publishing a response.'
fi

if ! LISTING=$(gh api --paginate "repos/$GITHUB_REPOSITORY/issues/$PR_NUMBER/comments?per_page=100"); then
  missing 'Could not verify the published response because the comment listing failed.'
fi

if ! COUNT=$(printf '%s' "$LISTING" | jq -s -r \
    --arg started "$RUN_STARTED_AT" --arg sha "${HEAD_SHA:-}" --arg is_pr "${IS_PR:-true}" '
  (add // []) | [.[]
    | select(.user.login == "claude[bot]" and (.updated_at // "") > $started)
    | select(if $is_pr == "true"
      then ((.body // "") | startswith("<!-- claude-sticky-review -->")
        and contains("<!-- score:") and contains($sha))
      else ((.body // "") | startswith("<!-- progress-indicator -->")
        or startswith("<!-- claude-usage-ledger -->") | not)
      end)] | length'); then
  missing 'The comment listing was unreadable; no successful response is claimed.'
fi
if [ "$COUNT" = "0" ]; then
  missing 'No current Claude response was found for this run and reviewed commit.'
fi

echo 'Verified the current Claude response.'

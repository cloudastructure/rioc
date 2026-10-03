#!/usr/bin/env bash
# Classify reviewer outcomes for the auto-review and mention jobs.
# Env in: GATE_MODE (auto|mention), MODEL_OUTCOME, CAPTURE_OUTCOME,
#         PUBLISHER_OUTCOME (publisher authentication step), REVIEW_MISSING,
#         SCORE_MISSING, HEAD_SHA, GITHUB_OUTPUT.
# Env out: infra_failure=true and head_sha only for an auto infrastructure
#          failure with a captured commit. Structural/setup failures exit 1.
set -euo pipefail

case "${GATE_MODE:-}" in
  auto)
    # Authentication can fail before metadata captures HEAD_SHA. Report the
    # actual cause before considering a neutral check against that commit.
    if [ "${PUBLISHER_OUTCOME:-}" != "success" ]; then
      echo "::error::Claude publisher authentication did not succeed" >&2
      exit 1
    fi

    if [ "${MODEL_OUTCOME:-}" != "success" ] || [ "${CAPTURE_OUTCOME:-}" != "success" ]; then
      if [ -z "${HEAD_SHA:-}" ]; then
        echo "::error::Reviewer setup/model did not capture a commit (HEAD_SHA is unset); cannot report a neutral review-status check" >&2
        exit 1
      fi
      {
        echo "infra_failure=true"
        echo "head_sha=$HEAD_SHA"
      } >> "${GITHUB_OUTPUT:?GITHUB_OUTPUT is required for an infrastructure failure}"
      echo "Infrastructure failure — neutral review-status check required"
      exit 0
    fi

    if [ "${REVIEW_MISSING:-}" = "true" ] || [ "${SCORE_MISSING:-}" = "true" ]; then
      echo "::error::Claude review or readable score is missing" >&2
      exit 1
    fi
    ;;
  mention)
    # Preserve the mention job's provider-failure behavior. A successful model
    # run with no published response is a separate structural failure.
    if [ "${MODEL_OUTCOME:-}" != "success" ]; then
      exit 0
    fi
    if [ "${REVIEW_MISSING:-}" = "true" ]; then
      echo "::error::Claude model succeeded but the review response is missing" >&2
      exit 1
    fi
    ;;
  *)
    echo "::error::Invalid GATE_MODE '${GATE_MODE:-}'; expected auto or mention" >&2
    exit 1
    ;;
esac

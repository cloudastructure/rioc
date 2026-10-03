#!/usr/bin/env bash
# Resolve an @claude comment with the outcome when the run finishes. The
# Claude action itself acknowledges a mention with the eyes reaction; posting
# another eyes reaction here made the first run display a duplicate count.
# Completion reactions remain owned by this workflow so the custom publisher
# can distinguish success from failure.
#
# Best effort throughout: a missing reaction is never worth failing a review
# over.
#
# Env in: COMMENT_ID, GH_TOKEN, GITHUB_REPOSITORY, COMMENT_SCOPE.
# Arg 1: success | failure. `seen` is retained as a no-op for callers from an
# older checkout while the workflow migrates away from the duplicate ack.
#
# COMMENT_SCOPE is `issues` for a PR conversation comment and `pulls` for a
# review comment on a diff line. They are different REST collections and the
# ids do not overlap, so reacting on the wrong one 404s.
set -euo pipefail

if [ -z "${COMMENT_ID:-}" ]; then
  exit 0
fi

REACTIONS="repos/$GITHUB_REPOSITORY/${COMMENT_SCOPE:-issues}/comments/$COMMENT_ID/reactions"

case "${1:-}" in
  seen)
    # The upstream action posts the single acknowledgement reaction.
    :
    ;;
  success)
    gh api "$REACTIONS" -X POST -f content='+1' > /dev/null 2>&1 || true
    ;;
  failure)
    # Not a thumbs-up: a run that died still gets a reaction, but one that
    # does not claim the review happened.
    gh api "$REACTIONS" -X POST -f content='confused' > /dev/null 2>&1 || true
    ;;
esac

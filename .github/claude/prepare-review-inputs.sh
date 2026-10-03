#!/usr/bin/env bash
# Prefetch everything the reviewer reads into plain files, so the model can
# review with no GitHub token and no `gh` in its allowlist at all.
#
# Review #2 on the PR that introduced this bot spent 548k fresh input tokens,
# partly because the model fetched the same diff three different ways. It also
# held GH_TOKEN while `gh api *` was allowed, which is a write and
# exfiltration primitive the moment a PR's diff contains instructions. Both go
# away when a trusted step does the fetching once and hands over files.
#
# Where each file comes from: diff.patch and files.txt are computed from git at
# $HEAD_SHA, the commit the workflow checked out, because the API answers for
# the PR's current head and a push can move that after the checkout. Everything
# else -- pr.json, the comment listings, check-runs.txt -- comes from the API.
#
# Env in:  GH_TOKEN, GITHUB_REPOSITORY, PR_NUMBER, HEAD_SHA, IS_PR
#          ("true" for a PR, anything else for a plain issue),
#          REVIEW_MODE ("first" or "rescore"; defaults to rescore),
#          REQUEST_FILE (optional, defaults to
#          $RUNNER_TEMP/claude-request.txt),
#          RUNNER_TEMP (optional), GITHUB_ENV (optional),
#          SHORT_SHA, REVIEW_NUMBER, COMMIT_MSG, TRIGGER_COMMENT (optional;
#          copied into meta.json for the model, which cannot read the
#          environment itself).
# Env out: REVIEW_IN and REVIEW_OUT, absolute paths, also exported to
#          GITHUB_ENV when it is set.
set -euo pipefail

: "${PR_NUMBER:?PR_NUMBER is required}"
: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"

IS_PR="${IS_PR:-}"
REQUEST_FILE="${REQUEST_FILE:-${RUNNER_TEMP:-/tmp}/claude-request.txt}"
# The first pass reviews fresh by decision: it is not shown the prior review or
# the inline comments already on the PR, so it cannot anchor on its own earlier
# text or on a reviewer's note. A re-score is the opposite — its whole job is to
# judge whether the open findings were addressed — so it gets both.
REVIEW_MODE="${REVIEW_MODE:-rescore}"

# Recreated rather than reused. A re-run on a persistent runner would otherwise
# inherit the previous attempt's files, and a PR that managed to write into
# RUNNER_TEMP would have its plant read as trusted input.
BASE="${RUNNER_TEMP:-$(mktemp -d)}/claude-review"
REVIEW_IN="$BASE/in"
REVIEW_OUT="$BASE/out"
rm -rf "$REVIEW_IN" "$REVIEW_OUT"
mkdir -p "$REVIEW_IN" "$REVIEW_OUT"
export REVIEW_IN REVIEW_OUT

if [ -n "${GITHUB_ENV:-}" ]; then
  {
    echo "REVIEW_IN=$REVIEW_IN"
    echo "REVIEW_OUT=$REVIEW_OUT"
  } >> "$GITHUB_ENV"
fi

# --- pr.json, diff.patch, files.txt ------------------------------------------
#
# pr.json comes from the API: title, body, base and author are not SHA-bound.
# The diff and the file list are NOT taken from the API. `gh pr diff` and the
# files endpoint both answer for the PR's CURRENT head, while this job has
# $HEAD_SHA checked out and publishes its review under $HEAD_SHA: a push
# between the checkout and this step would have the model read a newer diff
# against an older tree. Both are computed from git instead, over the same
# base...head range `gh pr diff` reports.
#
# A failed fetch here is fatal rather than an empty file. files.txt is what
# publish-review.sh checks every inline finding's path against, so an empty one
# silently drops the entire inline half of the review while the run still
# reports success; and a review written without the diff is worse than no
# review at all.
if [ "$IS_PR" = "true" ]; then
  : "${HEAD_SHA:?HEAD_SHA is required when IS_PR=true}"
  if ! gh pr view "$PR_NUMBER" \
      --json title,body,baseRefName,headRefName,headRefOid,author,labels,additions,deletions,files \
      > "$REVIEW_IN/pr.json"; then
    echo "::error::Could not read PR #$PR_NUMBER metadata"
    exit 1
  fi
  # The workflow checks the head SHA again after the model has run. This one is
  # earlier and cheaper: a PR that has already moved should stop before the
  # review is paid for, not after.
  PR_HEAD_OID=$(jq -r '.headRefOid // empty' "$REVIEW_IN/pr.json")
  if [ "$PR_HEAD_OID" != "$HEAD_SHA" ]; then
    echo "::error::PR #$PR_NUMBER moved past $HEAD_SHA (head is now ${PR_HEAD_OID:-unknown})"
    exit 1
  fi
  BASE_REF=$(jq -r '.baseRefName // empty' "$REVIEW_IN/pr.json")
  if [ -z "$BASE_REF" ]; then
    echo "::error::PR #$PR_NUMBER metadata carries no base branch name"
    exit 1
  fi
  # No token of its own: the workflow ran `gh auth setup-git`, so git fetches
  # through the credential helper that wrote.
  if ! git fetch --no-tags origin \
      "+refs/heads/$BASE_REF:refs/remotes/origin/$BASE_REF"; then
    echo "::error::Could not fetch base branch $BASE_REF for PR #$PR_NUMBER"
    exit 1
  fi
  if ! MERGE_BASE=$(git merge-base "origin/$BASE_REF" "$HEAD_SHA"); then
    echo "::error::Could not find the merge base of origin/$BASE_REF and $HEAD_SHA"
    exit 1
  fi
  # base...head: what the PR changes, with commits the base branch gained since
  # the branch point left out. Same range `gh pr diff` reports.
  if ! git diff "$MERGE_BASE" "$HEAD_SHA" > "$REVIEW_IN/diff.patch"; then
    echo "::error::Could not diff $MERGE_BASE..$HEAD_SHA for PR #$PR_NUMBER"
    exit 1
  fi
  # Neither `gh pr view --json files` nor the files endpoint: the first caps its
  # list at 100 entries, and both describe the PR's current head rather than the
  # tree this diff was taken from. files.txt is what publish-review.sh checks
  # every inline finding's path against, so a list that disagrees with the diff
  # silently drops findings.
  if ! git diff --name-only "$MERGE_BASE" "$HEAD_SHA" > "$REVIEW_IN/files.txt"; then
    echo "::error::Could not list the files changed by PR #$PR_NUMBER"
    exit 1
  fi
else
  echo '{}' > "$REVIEW_IN/pr.json"
  : > "$REVIEW_IN/diff.patch"
  : > "$REVIEW_IN/files.txt"
fi

# --- diff.patch trimming, skipped.txt ----------------------------------------
#
# Every turn re-sends the diff, so a byte that carries nothing reviewable is
# paid for on every step: review #2 spent 548k fresh input tokens on a 282 KB
# diff. Three deterministic rules run here rather than in the prompt, because a
# model told to skip a file has already read it by the time it decides.
# Lockfiles and generated output are dropped whole. An ordinary deletion keeps
# only its headers, so the reviewer still sees the name and can grep for its
# callers; hand-written test assets keep their full diff for coverage review.
#
# Skipped paths stay in files.txt on purpose. publish-review.sh posts an
# inline finding only on a path listed there, and "this lockfile does not
# belong in the PR" is a finding worth being able to make. skipped.txt is the
# signal that the diff itself was withheld.
DIFF_BEFORE=$(wc -c < "$REVIEW_IN/diff.patch" | tr -d ' ')
: > "$REVIEW_IN/skipped.txt"
awk -v skipped="$REVIEW_IN/skipped.txt" '
function basename(p,   parts, count) {
  count = split(p, parts, "/")
  return parts[count]
}
function is_test_asset(p,   name) {
  name = basename(p)
  return p ~ /(^|\/)(__tests__|e2e|spec|test|tests)\// \
    || name ~ /(^|[._-])(e2e|spec|test)\.([cm]?[jt]sx?|sh)$/ \
    || name ~ /(Spec|Tests?)\.([cm]?[jt]sx?|java)$/ \
    || name ~ /\.feature$/
}
# The b/ path, or a/ for a deletion, where "+++ /dev/null" leaves bpath empty.
# The diff --git line is the fallback for a block with no ---/+++ header at all,
# which is what a mode-only change looks like.
function block_path(   p, f, count) {
  if (bpath != "") return bpath
  if (apath != "") return apath
  count = split(gitline, f, " ")
  if (count >= 4) { p = f[4]; sub(/^b\//, "", p); if (p != "") return p }
  if (count >= 3) { p = f[3]; sub(/^a\//, "", p); if (p != "") return p }
  return ""
}
# *.lock covers yarn.lock, Gemfile.lock, poetry.lock, Cargo.lock,
# composer.lock and Pipfile.lock, so only the three that do not end in .lock
# are named.
function classify(p) {
  if (p ~ /\.lock$/) return "lockfile"
  if (basename(p) ~ /^(package-lock\.json|npm-shrinkwrap\.json|pnpm-lock\.yaml)$/) return "lockfile"
  if (p ~ /(^|\/)(dist|build)\//) return "generated"
  if (p ~ /\.min\.(js|css)$/ || p ~ /\.map$/ || p ~ /\.snap$/) return "generated"
  if (is_test_asset(p)) return ""
  if (deleted || (in_hunk && !nonremoval)) return "deletion-only"
  return ""
}
function flush(   i, p, reason, last) {
  if (n == 0) return
  p = block_path()
  reason = classify(p)
  last = in_hunk ? header_n : n
  if (reason == "") {
    for (i = 1; i <= n; i++) print buf[i]
  } else {
    if (reason == "deletion-only") { for (i = 1; i <= last; i++) print buf[i] }
    print p "\t" reason > skipped
  }
  n = 0
}
{
  if ($0 ~ /^diff --git /) {
    flush()
    header_n = 0; in_hunk = 0; nonremoval = 0; deleted = 0
    apath = ""; bpath = ""; gitline = $0
  }
  buf[++n] = $0
  if (!in_hunk) {
    if ($0 ~ /^@@/) { in_hunk = 1; header_n = n - 1 }
    else if ($0 ~ /^deleted file mode /) deleted = 1
    else if ($0 ~ /^--- a\//) apath = substr($0, 7)
    else if ($0 ~ /^\+\+\+ b\//) bpath = substr($0, 7)
  } else if ($0 !~ /^@@/ && $0 !~ /^-/ && $0 !~ /^\\/) nonremoval = 1
}
END { flush() }
' "$REVIEW_IN/diff.patch" > "$REVIEW_IN/diff.trimmed"
mv "$REVIEW_IN/diff.trimmed" "$REVIEW_IN/diff.patch"
DIFF_AFTER=$(wc -c < "$REVIEW_IN/diff.patch" | tr -d ' ')

# --- check-runs.txt ----------------------------------------------------------
#
# The job token gets a 403 on this endpoint today (seen in the run log), so the
# unavailable path is the common one, not the exception. gh's output is
# captured before the filter runs: with the grep in the pipeline, a PR whose
# only checks are the two filtered ones would exit non-zero and be reported as
# an API failure.
#
# The filter drops this workflow's own check runs, which are named after its
# jobs (claude.yml: "auto-review (advisory — not a merge gate)" and "Claude
# auto-review status"). Reading its own in-progress status back is noise at
# best and a self-reference the model reasons about at worst.
#
# --paginate because this repo runs well past a single page of jobs per SHA and
# an unpaginated read would silently drop everything after the first page. With
# --jq the filter runs per page and the pages concatenate into one line stream,
# which is what this consumer wants; nothing here reads the enclosing object.
if [ -n "${HEAD_SHA:-}" ] && CHECKS=$(gh api --paginate \
    "repos/$GITHUB_REPOSITORY/commits/$HEAD_SHA/check-runs?per_page=100" \
    --jq '.check_runs[] | "\(.name): \(.conclusion // .status)"' 2> /dev/null); then
  printf '%s\n' "$CHECKS" \
    | grep -Ev '^(auto-review|Claude )' > "$REVIEW_IN/check-runs.txt" || true
else
  printf 'CI results unavailable\n' > "$REVIEW_IN/check-runs.txt"
fi

# --- comments.json, review-comments.json, prior-review.md --------------------
#
# --paginate emits one array PER PAGE, so the pages are slurped and added into
# one flat array before anything filters or takes `last`; over per-page arrays
# `last` means "last of some page". Same reasoning as
# prepare-review-metadata.sh, and so is the fatal-on-failure rule: an empty
# comment list reads as "no review yet", and prior-review.md is what stops a
# re-score from repeating a finding it already made.
if LISTING=$(gh api --paginate \
    "repos/$GITHUB_REPOSITORY/issues/$PR_NUMBER/comments?per_page=100" 2> /dev/null); then
  printf '%s' "$LISTING" | jq -s -c '(add // [])' > "$REVIEW_IN/comments.json"
elif [ "$IS_PR" = "true" ]; then
  echo "::error::Could not list comments for #$PR_NUMBER"
  exit 1
else
  echo "::warning::Could not list comments for #$PR_NUMBER; continuing with none"
  echo '[]' > "$REVIEW_IN/comments.json"
fi

# Inline review comments are context, not a dedupe key, so a failure here
# degrades the review rather than ending it. On a first pass they are not
# fetched at all: an empty array is what "review this fresh" means on disk.
if [ "$IS_PR" = "true" ] && [ "$REVIEW_MODE" != "first" ]; then
  if LISTING=$(gh api --paginate \
      "repos/$GITHUB_REPOSITORY/pulls/$PR_NUMBER/comments?per_page=100" 2> /dev/null); then
    printf '%s' "$LISTING" | jq -s -c '(add // [])' > "$REVIEW_IN/review-comments.json"
  else
    echo "::warning::Could not list inline review comments for #$PR_NUMBER"
    echo '[]' > "$REVIEW_IN/review-comments.json"
  fi
else
  echo '[]' > "$REVIEW_IN/review-comments.json"
fi

# The score marker is paired with one of the two trusted publisher identities:
# new comments come from github-actions[bot], while existing PRs can carry the
# legacy claude[bot] sticky from before the publisher migration. The metadata
# step chooses the canonical comment to PATCH separately. For model context we
# want the most recently analyzed body, including a later duplicate created by
# an earlier race, so the model sees the latest findings rather than stale
# text from the oldest migration candidate.
#
# Empty on a first pass, for the reason at the top of the file: the prior
# review is the previous run's own words, and feeding them back is how a second
# opinion turns into an echo.
if [ "$REVIEW_MODE" = "first" ]; then
  : > "$REVIEW_IN/prior-review.md"
else
  jq -r '[.[]
          | select((.user.login == "github-actions[bot]" or .user.login == "claude[bot]")
                   and ((.body // "") | contains("<!-- score:")))]
         | sort_by([(.updated_at // .created_at // ""), (.id // 0)])
         | last
         | .body // empty' \
    "$REVIEW_IN/comments.json" > "$REVIEW_IN/prior-review.md"
fi

# --- request.txt -------------------------------------------------------------
#
# What the human asked for on an @claude mention. Absent on a plain
# pull_request trigger, where the request is "review this PR".
#
# Read ONLY in rescore mode, which is the only mode a mention reaches. A first
# pass has no request of its own, and a persistent runner can still be holding
# the file another PR's mention wrote: copying it there would hand this review
# someone else's instructions.
if [ "$REVIEW_MODE" = "rescore" ] && [ -f "$REQUEST_FILE" ]; then
  cp "$REQUEST_FILE" "$REVIEW_IN/request.txt"
else
  : > "$REVIEW_IN/request.txt"
fi

# --- meta.json ---------------------------------------------------------------
#
# The model's Bash hook expands only $REVIEW_IN and $REVIEW_OUT in a command,
# so everything else this run knows about itself has to reach the model as a
# file. Built with `jq -n --arg` and not a here-doc: the commit subject and the
# trigger comment are attacker-controlled text, and --arg JSON-escapes them
# instead of letting a quote or a backslash rewrite the object around them.
jq -n \
  --arg pr_number "${PR_NUMBER:-}" \
  --arg head_sha "${HEAD_SHA:-}" \
  --arg short_sha "${SHORT_SHA:-}" \
  --arg review_number "${REVIEW_NUMBER:-}" \
  --arg commit_msg "${COMMIT_MSG:-}" \
  --arg trigger_comment "${TRIGGER_COMMENT:-}" \
  --arg github_repository "${GITHUB_REPOSITORY:-}" \
  --arg review_mode "${REVIEW_MODE:-}" \
  '{pr_number: $pr_number, head_sha: $head_sha, short_sha: $short_sha,
    review_number: $review_number, commit_msg: $commit_msg,
    trigger_comment: $trigger_comment,
    github_repository: $github_repository, review_mode: $review_mode}' \
  > "$REVIEW_IN/meta.json"

# The run log is what a human reads when a review comes back thin. Sizes tell
# them whether the model was handed an empty diff or a truncated comment list,
# which is otherwise invisible until the review itself looks wrong.
echo "Review inputs prepared in $REVIEW_IN (mode: $REVIEW_MODE):"
for f in pr.json diff.patch files.txt check-runs.txt \
  comments.json review-comments.json prior-review.md request.txt skipped.txt \
  meta.json; do
  printf '  %-20s %8s bytes\n' "$f" "$(wc -c < "$REVIEW_IN/$f" | tr -d ' ')"
done
printf 'Diff trimmed: %s files skipped, %s -> %s bytes\n' \
  "$(wc -l < "$REVIEW_IN/skipped.txt" | tr -d ' ')" "$DIFF_BEFORE" "$DIFF_AFTER"
echo "Review output directory: $REVIEW_OUT"

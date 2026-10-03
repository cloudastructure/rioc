#!/usr/bin/env bash
# Overwrite the bot's own files in the working tree with main's copies, so a
# PR cannot rewrite the instructions it is reviewed by, or the shell that runs
# beside the subscription token.
#
# How much this buys depends on the trigger, and the difference is real:
#
#   issue_comment / issues / pull_request_review* — GitHub runs the workflow
#   file from the DEFAULT branch, while the tree being reviewed is the PR
#   author's. There the workflow is trusted and this restore is the whole
#   boundary: without it, "@claude" on a hostile PR would execute that PR's
#   .claude/ and .github/claude/ next to the runner's tokens.
#
#   pull_request — the workflow file itself comes from the PR head, so anyone
#   who can push a branch here can already change this step. Against that, the
#   restore is drift protection, not a boundary. The boundary for that trigger
#   is repository push access.
#
# Env out (GITHUB_OUTPUT): pr_touches_bot_files — true when this branch's
# copies differ from main's, so the workflow can say so on the PR.
#
# Env in: GITHUB_OUTPUT (optional — skipped when run outside Actions).
set -euo pipefail

# Every path the review jobs execute from, not just the reviewer's own. The
# jobs also run .github/scripts/assign-primary-reviewer.sh, after the PR head
# is checked out, with GH_TOKEN and the Slack webhook in the environment. That
# script also reads .github/reviewer-pool.json for the GitHub-login -> Slack-ID
# mapping it assigns and pings from, so the data it acts on comes from main too.
#
# .claude and .mcp.json are here because Claude Code loads them out of the
# WORKING TREE as directives, not as data: .claude/settings.json carries hooks,
# which are shell commands the CLI runs on its own; .claude/agents, .claude/skills
# and .claude/commands are prompt files it can load unprompted; .mcp.json names
# MCP servers it will start. On a comment trigger that tree is the PR author's,
# so all four are the same class of hazard as the review command itself.
TRUSTED=(.github/claude .github/scripts .github/reviewer-pool.json .claude .mcp.json)

# No --depth=1. It turns a complete repository into a shallow one, and the
# workflow checks out with fetch-depth: 0 precisely so the reviewer's `git log`
# and `git blame` see real history; a shallow fetch truncates both, and also
# leaves origin/main sharing no merge base with HEAD (measured 2026-09-02 in
# this repository, in a worktree and in its parent checkout alike: 4 commits to
# 2). The one ref this needs is fetched by name, so the cost is a fetch of
# main's history rather than of the whole remote.
if ! git fetch --no-tags origin +refs/heads/main:refs/remotes/origin/main; then
  echo "::error::Failed to fetch origin/main; refusing to run the reviewer with unverified bot files"
  exit 1
fi

# The bootstrap case: on the PR that first adds these paths, main does not have
# them yet and there is nothing to restore. Warn rather than fail, because the
# alternative is a workflow that can never be merged. Once merged, main always
# carries them and the branch below is not taken again.
#
# The same branch covers a repository that simply has no .claude or .mcp.json on
# main: a missing path there is ordinary, not an attack, and must not stop a
# review. Unlike the source this ports from, the paths main DOES have are still
# restored — .claude and .mcp.json are ordinary repository paths that can be
# absent on their own, and letting one missing path suppress the restore of
# .github/claude would trade the whole boundary for a bootstrap case.
PRESENT=()
for path in "${TRUSTED[@]}"; do
  if git rev-parse --verify --quiet "origin/main:$path" > /dev/null; then
    PRESENT+=("$path")
  else
    echo "::warning::origin/main has no $path — using this branch's copy. Expected only on the PR that introduces the reviewer."
  fi
done

# `git checkout <tree> -- <path>` overwrites files that exist in <tree> but
# never removes ones that do not, so restoring alone leaves any file the PR
# ADDED sitting in place. That is not cosmetic: Claude Code reads
# `.claude/settings.local.json` out of the working tree alongside
# `.claude/settings.json`, and the local file wins. Such a file can set
# permissions and hooks for the review, and a PR that only ADDS it is not
# touched by the restore at all.
#
# So anything under a trusted path that this branch has and main does not is
# deleted rather than merely overwritten, by purge_extras below.

# .claude/settings.local.json is deleted outright rather than through
# purge_extras, in both branches below. purge_extras compares against
# `git ls-files --others --exclude-standard`, and this repo gitignores the file
# (AGENTS.md, "Jenkins credentials setup"), so an ignored copy is invisible to
# that listing and would survive. main can never carry one either, so there is
# nothing a delete can lose.
purge_local_settings() {
  if [ -e .claude/settings.local.json ]; then
    echo "::warning::Removing .claude/settings.local.json — a local settings override has no place in a review checkout"
    rm -f .claude/settings.local.json
  fi
}

purge_extras() {
  local path="$1" extra in_main in_tree
  in_main=$(git ls-tree -r --name-only origin/main -- "$path" 2>/dev/null | sort)
  in_tree=$( { git ls-files -- "$path"
               git ls-files --others --exclude-standard -- "$path"; } | sort -u)
  while IFS= read -r extra; do
    [ -n "$extra" ] || continue
    echo "::warning::Removing $extra — this branch has it, origin/main does not"
    rm -f -- "$extra"
  done < <(comm -13 <(printf '%s\n' "$in_main") <(printf '%s\n' "$in_tree"))
}

TOUCHED=false
if [ "${#PRESENT[@]}" -gt 0 ]; then
  if ! git diff --quiet origin/main HEAD -- "${PRESENT[@]}"; then
    TOUCHED=true
  fi
  git checkout origin/main -- "${PRESENT[@]}"
  for path in "${PRESENT[@]}"; do
    purge_extras "$path"
  done
  echo "Restored ${PRESENT[*]} from origin/main and purged additions."
fi
purge_local_settings

# AGENTS.md and CLAUDE.md are instructions to the reviewer, not data it reads.
# Claude Code loads them as directives, so a PR that edits one is telling the
# reviewer what to do while the reviewer holds both tokens. They come from main
# for the same reason the review command does.
#
# Matched by pattern rather than listed, so moving or adding one does not
# silently drop it out of the set. Restored where main has them, deleted where
# it does not, since a PR can also introduce a new one in a new directory.
#
# Deliberately not part of TRUSTED: these are ordinary files people edit, and
# folding them in would fire the "this PR changes the bot's own files" note on
# every AGENTS.md tweak.
INSTRUCTION_RE='(^|/)(AGENTS|CLAUDE)\.md$'
INSTR_MAIN=$(git ls-tree -r --name-only origin/main | grep -E "$INSTRUCTION_RE" | sort || true)
INSTR_TREE=$( { git ls-files
                git ls-files --others --exclude-standard; } | grep -E "$INSTRUCTION_RE" | sort -u || true)

if [ -n "$INSTR_MAIN" ]; then
  # Read into an array the long way: macOS ships bash 3.2, which has no
  # mapfile, and this script also runs from a developer's machine via the
  # preflight review.
  INSTR_ARRAY=()
  while IFS= read -r instr_file; do
    [ -n "$instr_file" ] && INSTR_ARRAY+=("$instr_file")
  done <<< "$INSTR_MAIN"
  git checkout origin/main -- "${INSTR_ARRAY[@]}"
  echo "Restored $(printf '%s\n' "$INSTR_MAIN" | wc -l | tr -d ' ') instruction file(s) from origin/main."
fi

while IFS= read -r extra; do
  [ -n "$extra" ] || continue
  echo "::warning::Removing $extra — an instruction file this branch adds and origin/main does not have"
  rm -f -- "$extra"
done < <(comm -13 <(printf '%s\n' "$INSTR_MAIN") <(printf '%s\n' "$INSTR_TREE"))

# Symbolic links whose target leaves the checkout are removed, on every trigger.
#
# Both things that bound what the reviewer may read judge the path the model
# SPELLS, not the file that path lands on: deny-shell-escapes.sh refuses an
# absolute operand by looking at the word in the command, and the Read tool's
# deny rules match the requested path. A relative word therefore passes both,
# and a symbolic link is the one way such a word reaches outside the tree —
# `cat link`, where the PR committed `link -> /var/run/secrets/kubernetes.io/
# serviceaccount/token`, is a read of that token spelled relatively. A PR can
# commit a link, so the tree is swept here, before the model runs.
#
# The bootstrap case is included, and this deliberately does not consult
# origin/main: the hazard is the PR's own tree, whatever main does or does not
# carry. Links that stay inside the checkout are ordinary repository content
# and are left alone; dangling ones go, since a target that does not exist
# cannot be shown to be inside.
ROOT=$(cd "$(git rev-parse --show-toplevel)" && pwd -P)

# The dangling case is settled by `[ -e ]`, which follows the whole chain, and
# not left to the resolver: GNU realpath canonicalizes a missing last component
# instead of failing, so a link to a name that does not exist would resolve to
# an inside path and be kept (measured on ubuntu:24.04).
#
# realpath then resolves what remains. It is absent on older macOS, and this
# script also runs from a developer's machine via the preflight review, so a
# portable walk backs it up; the walk follows chains itself, so a link pointing
# at a link pointing outside is caught in one pass rather than depending on the
# sweep's order.
resolve_link() { # print $1's final target, or fail when it dangles
  local p="$1" hops=0 target dir
  [ -e "$p" ] || return 1
  if command -v realpath > /dev/null 2>&1; then
    realpath -- "$p" 2> /dev/null
    return
  fi
  while [ -L "$p" ] && [ "$hops" -lt 40 ]; do
    target=$(readlink -- "$p") || return 1
    case "$target" in
      /*) p="$target" ;;
      *) p="$(dirname -- "$p")/$target" ;;
    esac
    hops=$((hops + 1))
  done
  [ -e "$p" ] || return 1
  dir=$(dirname -- "$p")
  [ -d "$dir" ] || return 1
  printf '%s/%s\n' "$(cd -- "$dir" && pwd -P)" "$(basename -- "$p")"
}

# find prints every path with a './' prefix, and that prefix is what keeps a
# committed name like '-leak.md' from being read as a flag by the commands
# below. It is stripped for the message only; every filesystem call gets the
# path find produced, plus '--', because rm and readlink parse '-leak.md' as
# options '-l -e -a -k ...' and the link would outlive the sweep.
while IFS= read -r link; do
  [ -n "$link" ] || continue
  display="${link#./}"
  inside=false
  if resolved=$(resolve_link "$link"); then
    case "$resolved" in
      "$ROOT" | "$ROOT"/*) inside=true ;;
    esac
  fi
  if [ "$inside" = true ]; then
    continue
  fi
  echo "::warning::Removing $display -> $(readlink -- "$link") — a symbolic link that leaves the checkout"
  rm -f -- "$link"
done < <(find . -type l -not -path './.git/*')

if [ -n "${GITHUB_OUTPUT:-}" ]; then
  echo "pr_touches_bot_files=$TOUCHED" >> "$GITHUB_OUTPUT"
fi

# Keep a copy outside the working tree. A job that later checks out the PR head
# replaces every file here with the PR author's version, this script included,
# so re-running the tree copy at that point would execute PR-controlled shell
# beside the runner's tokens — the precise boundary a comment-triggered job is
# supposed to hold, since its workflow file comes from the default branch even
# though its tree does not. Callers run the snapshot after such a checkout.
#
# It is the third of three layers, and each bounds something the others cannot
# see: settings.json's permissions.allow bounds WHICH commands run,
# deny-shell-escapes.sh (the PreToolUse hook on Bash) bounds what an allowed
# command can DO with the filesystem, and this snapshot means a write that gets
# past both still changes nothing the workflow later executes.
#
# Written once and never overwritten: the second call runs FROM the snapshot,
# and re-copying there would capture whatever the tree holds at that moment,
# which on a failed restore is the PR's copy. First write wins, and the first
# write happens while the tree is still the default branch's.
if [ -n "${RUNNER_TEMP:-}" ] && [ ! -e "${RUNNER_TEMP}/claude-trusted" ]; then
  SNAP="${RUNNER_TEMP}/claude-trusted"
  mkdir -p "$SNAP"
  # Repo-relative layout is mirrored rather than flattened, so a step can run
  # from the snapshot exactly as it runs from the tree: the workflow executes
  # every post-reviewer step from here, and assign-primary-reviewer.sh resolves
  # .github/reviewer-pool.json relative to its working directory.
  #
  # A plain loop rather than `cp -R --parents`: macOS ships BSD cp, which has no
  # --parents, and this script also runs from a developer's machine via the
  # preflight review.
  for path in "${TRUSTED[@]}"; do
    [ -e "$path" ] || continue
    mkdir -p "$SNAP/$(dirname "$path")"
    cp -R "$path" "$SNAP/$(dirname "$path")/"
  done
  echo "Snapshotted the trusted bot paths to $SNAP"
fi

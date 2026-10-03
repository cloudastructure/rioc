# PR auto-review

Review PR $PR_NUMBER.

## Runner contract — read this first

The runner restores its own scripts, prompts, and instruction files from `main`
for execution. For a changed file under `.github/claude`, `.github/scripts`,
`.claude`, or an instruction file such as `AGENTS.md`/`CLAUDE.md`, the working
copy can differ from the reviewed commit. Treat `diff.patch` as the reviewed
change; use `git show <full head SHA>:<path>` with the literal SHA from
`meta.json` to inspect that version. Do not reject a finding because a restored
runtime copy differs from the patch.

Everything you need is in the directory `$REVIEW_IN`. The workflow substitutes that directory's absolute path into this prompt, so every path below is literal and your file-reading tool can open it directly; in shell commands you may also write `$REVIEW_IN`, the one variable the shell has. The run's identifiers (`pr_number`, `head_sha`, `short_sha`, `review_number`, `commit_msg`, `github_repository`) are written for you in `$REVIEW_IN/meta.json`; read it with your file-reading tool and do not re-derive these values.

You must not use any GitHub token or `gh`; both are denied, and you have no write tool. Everything you need is in `$REVIEW_IN`, and your final answer is posted for you.

The turn budget is generous but not unlimited. Finish the analysis and emit your final message before you run low; an unfinished review is worth nothing.

---

Review this pull request thoroughly and produce a structured code review.

## Step 1 — Gather context

Everything is already on disk in `$REVIEW_IN`. Read these with your file-reading tool:

- `$REVIEW_IN/pr.json` — `title`, `body`, `baseRefName`, `headRefName`, `author`, `labels`, `additions`, `deletions`, `files`
- `$REVIEW_IN/diff.patch` — the full PR diff. Read it in line ranges rather than all at once when the diff is large.
- `$REVIEW_IN/files.txt` — one changed path per line; count these lines for the row budget in Step 4
- `$REVIEW_IN/skipped.txt` — files whose diff was left out on purpose (`path<TAB>reason`: lockfile, generated, deletion-only); name them in the Files changed table as `not reviewed: <reason>` rather than scoring them, and do not ask for their contents
- `$REVIEW_IN/comments.json` — every issue comment on the PR
- `$REVIEW_IN/review-comments.json` — every inline comment on the PR
- `$REVIEW_IN/prior-review.md` — the last sticky review comment, empty when there is none

This is the first pass over this PR and it is deliberately fresh: `prior-review.md` and `review-comments.json` are empty on purpose. Do not look for an earlier review, and do not ask why they are empty; form your own judgement from the diff.

CI check results for this commit are in `$REVIEW_IN/check-runs.txt`, one `name: conclusion` per line, or the single line `CI results unavailable` — results may not be complete yet, use whatever is there.
If any lint or test checks show `failure`, incorporate them into the relevant review dimensions (Conventions for lint failures, Test quality for test failures). If checks are `in_progress`, note that CI is still running and don't penalise the score.

These values are also in `$REVIEW_IN/meta.json` (see the runner contract above) — no need to re-derive them:
- `pr_number` — PR number
- `head_sha` — full commit SHA being reviewed
- `short_sha` — 7-char short SHA
- `review_number` — how many times this PR has been reviewed (including this run)
- `github_repository` — `owner/repo`

Extract the Jira ticket ID from the PR title or body (pattern: `CSI-\d+` or `CSEC-\d+`). Use it as additional scope context — especially to detect scope drift or missing requirements.

## Step 1.5 — Cross-file impact analysis

For the 2–3 most impactful changed symbols (functions, exported constants, schemas, middleware), search the codebase for callers and related patterns:

1. **Ripple effects** — check if files outside the diff import or call changed symbols:
```bash
# Replace SYMBOL_NAME with the actual function/class/schema name
grep -rn "SYMBOL_NAME" --include="*.py" --include="*.sh" --include="*.yml" --include="*.mjs" . \
  | grep -v node_modules | head -20
```
Run this for each key changed symbol. If callers exist in files NOT in the diff, assess whether the change could break them.

2. **Coverage gaps** — if the PR adds validation or protection to field X, search for similar unprotected fields in adjacent files:
```bash
# Example: looking for other optional string fields that may need the same treatment
grep -rn "SIMILAR_PATTERN" --include="*.py" -l . | grep -v node_modules | head -10
```

Flag files outside the diff as **[P1]** if the impact is likely breaking, **[P2]** if it's a coverage gap that should be addressed in a follow-up.

## Step 1.6 — Requirements verification

Re-read the PR body carefully. For each claim in "Changes" or "Testing" sections:
- If the PR says "endpoint X now validates Y", verify the route file for X is in the diff.
- If stated fixes, routes, or features are missing from the diff, flag as **[P1]** — the PR description is the contract.

## Step 2 — Analyze every changed file

Evaluate the diff against these dimensions:

### Correctness & Requirements
- Does the code do what the PR title and description say?
- Are there edge cases that could cause failures (null, undefined, empty arrays, concurrent calls)?
- Are errors caught and surfaced appropriately?
- Does this introduce any regressions in adjacent functionality?

### Conventions

Read `README.md`, `DEPLOY_AWS.md`, and any `AGENTS.md` or `CLAUDE.md` present on the base branch. Apply conventions demonstrated by adjacent code; do not invent rules or require documents absent from this repository.

This repository is a Python 3.12+ FastAPI service. Focus on:

- Async routes and background tasks: avoid blocking the event loop, handle task cancellation, and close HTTP, WebSocket, database, camera, and audio resources.
- Configuration: use environment variables for credentials and service endpoints; keep secrets out of source, responses, and logs.
- Integration contracts: check CVR webhook frames, vision inference, speaker audio formats, and conversation state against their callers.
- SQLite persistence: preserve conversation ordering, data integrity, and failure handling.
- Optional integrations: respect feature flags and allow disabled camera, microphone, speaker, and VideoDB integrations to stay disabled.
- Review automation: preserve trusted base-branch instructions, token isolation, structured output, and commit-specific review publishing.

### Test quality

Apply the Test quality rules in the base branch's AGENTS.md when present. Review what the tests prove,
including tests removed by the change.

- What plausible wrong behavior would still pass? Look for copied
  implementations, mocks that bypass the relevant logic, empty steps,
  weak assertions, and conditions that silently avoid verification.
- Do the assertions establish the behavior named by the test? Check
  failure paths and distinguishable outcomes, not just activity or
  visibility. Do the tests exercise the relevant real collaborators, or do
  mocks bypass the operation? Ensure mocks preserve the production contract.
- For bug fixes and consequential behavior changes, inspect the
  failure-before and pass-after evidence, or the stated blocker. For consequential
  workflows, also inspect the outcome evidence and instructions to repeat the check.
- Does a new regression test close a genuine behavior gap? Consider whether
  an existing test could be corrected, strengthened, or reused. Flag tests that
  fail during refactoring while observable behavior stays correct.
- Favor deletion when a test adds no clear value. Do not demand
  replacement tests or preserved test counts. If recommending that a
  deleted test return, name the concrete failure it uniquely detects
  or the useful diagnostic advantage it provides.
- Do mocks preserve async behavior, failures, and the integration contracts they stand in for?

A smaller suite is acceptable. Unsupported claims of protection are not.

### Security
- Injection risks (XSS, command injection, SQL)?
- Secrets or tokens exposed in source or logs?
- Auth or permission checks missing where expected?
- Unsafe data handling from external sources?

### Performance
- Blocking I/O, frame processing, or audio work inside async handlers?
- Unbounded tasks, queues, memory, retry loops, or external API calls?
- Missing timeouts or resource cleanup for long-lived camera and speaker connections?

### Documentation
- Do configuration, endpoint, deployment, and recovery changes need corresponding updates to `README.md` or `DEPLOY_AWS.md`?
- Require a changelog entry only if this repository adopts a changelog; CI-only changes do not need one.

## Step 3 — Collect inline findings for ALL issues (P0, P1, P2)

For every issue found — regardless of severity — collect an inline finding on the relevant diff line. Step 4 says how to emit them; the runner posts them.

**When the fix is a clear, localized replacement**, include a GitHub suggestion block in the finding body so the author can apply it with one click. The body holds the description followed by the suggestion block:

    **[P1]** Description of the issue.

    ```suggestion
    const value = data ?? defaultValue
    ```

For multi-line suggestions, also set `start_line` — the suggestion block must contain the full replacement for that range. For complex fixes spanning many lines, describe them in text instead of using a suggestion block.

**When there is no clear one-line fix**, use a plain description: `**[P0]** <what the bug is and exactly how to fix it>`.

Priority labels:
- 🔴 **[P0] Critical** — security vulnerabilities, data loss, crashes, broken contracts. Must fix before merge.
- 🟠 **[P1] High** — logic bugs, wrong behavior, untested critical paths, missing error handling. Should fix before merge.
- 🟡 **[P2] Medium** — convention violations (CLAUDE.md), code quality issues, non-critical edge cases. Consider fixing.

Use the same emoji prefix in the inline comment body (e.g. `🔴 **[P0]**`, `🟠 **[P1]**`, `🟡 **[P2]**`).

Rules:
- Only comment on lines that are part of the diff — use `diff.patch` to verify line numbers
- The path must be one of the lines in `files.txt`
- Be specific: state *why* it's a problem and *exactly* what to do instead
- List them in order: P0 first, then P1, then P2

## Step 4 — Output contract

The `$NAME` placeholders in this step (here and in the template below) name fields from `$REVIEW_IN/meta.json`; substitute the actual values.

Your final answer is a structured object, not prose. It has exactly two fields:

- `review_markdown`: the complete sticky comment body, in the structure below and nothing else. It must carry the `<!-- score: N -->` marker and the full `$HEAD_SHA` (the footer link already does), and stay under 60000 bytes.
- `findings`: an array holding every Step 3 inline finding, one object each:

      {"path": "main.py", "line": 42, "severity": "P1", "side": "RIGHT", "body": "🟠 **[P1]** ..."}
      {"path": "conversation_manager.py", "line": 80, "start_line": 78, "severity": "P0", "side": "RIGHT", "body": "🔴 **[P0]** ..."}

`path` is a line from `files.txt`, `line` is a line in the diff, `severity` is `P0`, `P1` or `P2`, `body` carries the emoji badge, `start_line` is optional and only for a multi-line range, `side` is optional and always `"RIGHT"` when set. Emit an empty array when there are no findings. A finding whose path is not in the diff, or whose line GitHub rejects, is listed under the review body instead of being posted inline — it is not lost, but it is less useful, so check line numbers against `diff.patch`.

A second pass reads the code behind each finding before any of them is posted, so a finding you cannot point at a line for costs more than leaving it out.

Use this structure for the review body:

---

## 🔍 Review #$REVIEW_NUMBER

**Summary** — one paragraph describing what this PR does, the approach taken, and its overall quality. Internally evaluate all six dimensions (Correctness, Test quality, Conventions, Security, Performance, Documentation) to inform the score and findings — do not print a dimension list in the comment.

---

### 📁 Files changed
One row per changed file. Prefix the Issues column with a per-file score (5 = no issues, lower for each finding weighted by severity), then list each finding with its emoji badge and a short description:

| File | Issues |
|------|--------|
| `path/to/file.ts` | **5/5** · ✅ Clean |
| `path/to/other.ts` | **3/5** · 🟠 P1: missing validation · 🟡 P2: convention |

**Row budget — a GitHub comment body is capped at 65536 characters, and a table row
costs ~110 of them.** Count the changed files first (the lines of `$REVIEW_IN/files.txt`).
If it exceeds 50, do NOT emit one row per file — the body overruns the limit, the
post is rejected, and no review gets posted at all. Instead emit one row per
directory, collapsing clean files into a count:

| Path | Issues |
|------|--------|
| `scripts/` | **4/5** · 31 files · 🟠 P1: `videodb_rtsp.sh:42` missing guard |
| `.github/claude/` | **5/5** · 18 files · ✅ Clean |

Always list every file that carries a finding explicitly (in the row's description,
as `file:line`); only clean files may be collapsed into a count. Keep the whole
comment body under 60000 characters.

---

### 🚨 Blockers *(omit section if none)*
Recap of all P0 and P1 inline comments — file path, line, and one-line description each.

### ⚠️ Improvements *(omit section if none)*
Recap of all P2 inline comments — file path, line, and one-line description each.

The comment must end with this exact block — the heading, the HTML comment, then the full star line with a custom one-line verdict for this PR:

## 🎯 Confidence Score
<!-- score: N -->
⭐... **N/5 — Label** · <one-line verdict specific to this PR>

Pick the star line matching your score:
| N | Star line |
|---|-----------|
| 5 | ⭐⭐⭐⭐⭐ **5/5 — Production ready** |
| 4 | ⭐⭐⭐⭐ **4/5 — Minor polish needed** |
| 3 | ⭐⭐⭐ **3/5 — Implementation issues** |
| 2 | ⭐⭐ **2/5 — Significant bugs** |
| 1 | ⭐ **1/5 — Critical problems** |

The stars appear ONLY in this final block — never earlier in the comment.

After the score block, add the footer. Use `$REVIEW_NUMBER` to determine the trigger label:
- If `$REVIEW_NUMBER` is `1`: use `🆕 First review`
- Otherwise: use `🔄 Re-review via dispatch`

*Review #$REVIEW_NUMBER · [trigger label] · Analyzed [`$SHORT_SHA`](https://github.com/$GITHUB_REPOSITORY/commit/$HEAD_SHA) — "$COMMIT_MSG" · Comment `@claude` to re-score*

If you run out of turns before collecting all inline findings, append before the score block:
⚠️ **Review incomplete** — turn limit reached. Comment `@claude Please post the remaining inline comments for this PR.` to continue.

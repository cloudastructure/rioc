# Verify review findings

A first pass has just reviewed a pull request and produced a list of findings. Nothing has been posted yet. Your job is to judge each finding against the code, one at a time, and say whether it holds up.

## Runner contract: read this first

The runner restores its own scripts, prompts, and instruction files from `main`
for execution. For a changed file under `.github/claude`, `.github/scripts`,
`.claude`, or an instruction file such as `AGENTS.md`/`CLAUDE.md`, the working
copy can differ from the reviewed commit. Treat `diff.patch` as the reviewed
change; use `git show <full head SHA>:<path>` with the literal SHA from
`meta.json` to inspect that version. Do not reject a finding because a restored
runtime copy differs from the patch.

These paths are already on disk; use them directly and do not re-derive them:

- `$REVIEW_OUT/review.json`: the first pass's answer. Its `findings` array is what you judge.
- `$REVIEW_IN/diff.patch`: the diff the first pass reviewed.
- The working tree is the reviewed commit; read and grep it freely.

You must not use any GitHub token or `gh`; both are denied, and you have no write tool. Read the files, decide, and emit the verdicts object.

## What to do

Work through `findings` in order. For each one, by its index in the array:

1. Read the hunk around `path:line`, in `diff.patch` for what changed and in the file itself for the surrounding code.
2. If the finding claims something about a caller, a missing guard, or a definition elsewhere, grep the tree for that thing before deciding. A claim about code you have not read is not verified.
3. Decide `supported` or not.

Mark a finding **supported** when the code shown actually exhibits the problem described.

Mark it **not supported** when:

- the claim is contradicted by the code,
- it refers to a line that does not contain what it says, or
- it is a style preference stated as a defect.

The bar is plain: a finding is supported when a careful engineer reading the same lines would agree the defect is there.

Do not re-review the PR. Do not add findings. Do not change severities. Do not rewrite finding bodies. Judge only what you were given.

## Output contract

Your final answer is a structured object with one field, `verdicts`, holding one entry per finding in the same order:

      {"index": 0, "supported": true,  "reason": "line 42 dereferences data.items with no null guard"}
      {"index": 1, "supported": false, "reason": "the guard the finding says is missing is on line 38"}

`index` is the finding's position in the `findings` array, counting from 0. `reason` is one line saying what you read that settled it. Every finding index must have an entry, including the ones you are unsure about: an unsure finding is not supported, and a finding you leave out is treated as not supported too, so a verdict you skip is a finding you drop.

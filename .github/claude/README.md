# Pull request reviews

Ported from the Web Services review pipeline at [5dbaaf6](https://github.com/cloudastructure/web-services/commit/5dbaaf623e3bf4c80a3c9b1885c675571d044a05). Review guidance is adapted to this Python/FastAPI service. The core review helpers and regression tests are retained. Web Services reviewer routing and Slack notifications are excluded.

## Setup

1. Install the [Claude GitHub app](https://github.com/apps/claude) with access to `cloudastructure/rioc`.
2. Make the Actions secret `CLAUDE_CODE_OAUTH_TOKEN` available to this repository. Use the same credential source as Web Services; GitHub does not allow reading an existing Actions secret back out of that repository. RIOC had no repository-level Actions secrets when this port was prepared; organization-level secret access could not be verified.
3. Merge the workflows and bot files into `main`. Publishing validates the workflow against the default branch, and runtime helpers and prompts are restored from `origin/main`. The introducing PR's automatic review skips with an explicit workflow summary until those trusted files exist on `main`.

Optional Actions variables `CLAUDE_REVIEW_MODEL` and `CLAUDE_REVIEW_EFFORT` override the source defaults (`claude-opus-5` and `high`). Jobs run on GitHub-hosted Ubuntu runners with Node 24 for publishing.

## Behavior

- Automatically review non-draft PRs on opening or marking ready. Automatic fork reviews, Dependabot PRs, and bots other than `claude[bot]` are skipped.
- Pushes do not automatically run another review. Repository owners, members, and collaborators can comment `@claude` for a follow-up, including on fork PRs. Bot-authored mentions and outside participants' mentions are ignored.
- The Actions **Claude Code** workflow can also be dispatched with a PR number after setup.
- Publish a sticky review with a score out of five, commit SHA, and inline findings. A second pass verifies findings, and recomposition removes rejected findings from the summary. Incomplete verification is marked explicitly.
- Keep progress and usage comments current and reject stale follow-up requests. Every review identifies the captured head SHA; a push during a review requires a follow-up for the new commit.
- Reviews are advisory. An infrastructure failure or skipped bootstrap review is not an approval.
- Review jobs serialize per PR. The active job finishes; GitHub retains the newest pending request.

## Validation

Run the same regression checks as **Review Bot Tests**:

```bash
node --test .github/claude/*.test.mjs
for file in .github/claude/*.sh; do
  bash -n "$file"
  shellcheck -S warning "$file"
done
for file in .github/claude/*.test.sh; do bash "$file"; done
```

After setup, open a small PR and confirm the review and score name its current head SHA. Push a change and request another review with `@claude`; confirm the sticky review updates and findings are verified before publishing. Check failure summaries and incomplete markers when any stage fails. Live review verification on the introducing PR is blocked because its trusted runtime is not on `main` yet.

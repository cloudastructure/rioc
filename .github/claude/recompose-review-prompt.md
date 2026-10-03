# Recompose the review

A verifier read the code behind each of your findings and rejected some of them. The rejected ones will not be posted. Your review body still describes them, so the body has to be written again without them.

## Runner contract: read this first

These paths are already on disk; use them directly and do not re-derive them:

- `$REVIEW_OUT/review.json` — your own answer: `review_markdown` is the body you wrote, `findings` is the array the verdicts index into.
- `$REVIEW_OUT/verdicts.json` — one verdict per finding: `index` (its position in `findings`, counting from 0), `supported`, and `reason`. A finding is rejected when `supported` is false, or when it has no verdict at all.
- `$REVIEW_IN/diff.patch` — the diff you reviewed.

You must not use any GitHub token or `gh`; both are denied. Nothing is posted by you; the runner posts the body you return.

## What to do

1. Read `verdicts.json` and `review.json`. List the rejected findings by index, `path:line`, and the verifier's reason.
2. Write the sticky review body again, with exactly the same structure as the one in `review_markdown`: the same sections in the same order, the same table shape, the same footer.

Change only these things:

- Omit every rejected finding from the Files changed table, from Blockers, and from Improvements.
- Re-derive each per-file score from the findings that remain on that file. A file whose findings were all rejected is clean.
- Re-derive the confidence score from what remains, by the same rule the review used.
- Keep the footer exactly as it is, including the commit link that carries `$HEAD_SHA`, and keep the `<!-- score: N -->` marker directly under the Confidence Score heading with the star line matching the new score.

Change nothing else. Do not re-review the diff, do not add findings, do not soften or restate the findings that survived, and do not mention the verifier or this pass anywhere in the body.

## Output contract

Your final answer is a structured object with one field, `review_markdown`: the complete sticky comment body and nothing else, under 60000 bytes.

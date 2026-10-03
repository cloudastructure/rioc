#!/usr/bin/env bash
# Per-review token usage and the running total for this PR, kept in one sticky
# comment. Ported from the cost ledger the OpenCode variant carried.
#
# Record tokens reported by the SDK, without deriving dollars or subscription
# quota consumption from them. The ledger is an execution record, not a
# measurement of the seat's remaining usage window.
#
# The rows are stored as hidden markers, not parsed back out of the rendered
# table: the table is for people, the markers are the data. Format is
#   <!-- row: <n>|<sha>|<model (effort)>|<in>|<out>|<cache_read>|<cache_write>|<turns>|<run-id>|<attempt> -->
# where every count is an integer or the literal `unknown`.
#
# Model is a column because it is a per-run variable, so a PR reviewed before
# and after a model swap shows what each one drew.
#
# Usage source: extract-usage.sh, which reads the action's execution file into
# $REVIEW_OUT/usage-review.json and $REVIEW_OUT/usage-verify.json. Both are
# summed when both are present, because the verifier pass runs on the same seat
# and its tokens come out of the same window.
#
# A count of 0 is REAL and is recorded as 0. A count that could not be READ is
# recorded as `unknown` and excluded from the total, with the total marked a
# lower bound. The two are never conflated: treating unknown as zero
# under-reports the window.
#
# Env in: REVIEW_OUT, PR_NUMBER, SHORT_SHA, CLAUDE_MODEL, CLAUDE_EFFORT,
#         GH_TOKEN, GITHUB_REPOSITORY, GITHUB_RUN_ID.
set -euo pipefail

if [ -z "${PR_NUMBER:-}" ]; then
  exit 0
fi

TOK_IN="unknown"; TOK_OUT="unknown"; TOK_READ="unknown"; TOK_WRITE="unknown"
TURNS="unknown"
MODEL_FROM_USAGE=""

# Sum the passes that reported a number. A field is `unknown` for the run as a
# whole as soon as any present pass could not report it: a partial sum
# presented as a total is the same under-report as treating unknown as zero.
#
# Skipped verifier/recompose steps do not produce usage files. Any file that
# exists represents an attempted pass; an unreadable attempt must keep the
# total marked incomplete, even when another pass reported all its counts.
if [ -n "${REVIEW_OUT:-}" ]; then
  SUMMED=$(jq -s -r '
    def field(k):
      if ([.[] | .[k]] | any(type != "number")) then "unknown"
      else ([.[] | .[k]] | add) end;
    if length == 0 then empty
    else
      [ field("input"), field("output"), field("cache_read"),
        field("cache_write"), field("turns"),
        ( [ .[] | .model? // empty | select(. != "unknown") ] | last // "" )
      ] | @tsv
    end' \
    "$REVIEW_OUT"/usage-*.json 2>/dev/null || echo "")
  if [ -n "$SUMMED" ]; then
    TOK_IN=$(printf '%s' "$SUMMED" | cut -f1)
    TOK_OUT=$(printf '%s' "$SUMMED" | cut -f2)
    TOK_READ=$(printf '%s' "$SUMMED" | cut -f3)
    TOK_WRITE=$(printf '%s' "$SUMMED" | cut -f4)
    TURNS=$(printf '%s' "$SUMMED" | cut -f5)
    MODEL_FROM_USAGE=$(printf '%s' "$SUMMED" | cut -f6)
  fi
fi

if [ "$TOK_IN" = "unknown" ]; then
  echo "::warning::Could not read token usage for this run; recording it as unknown"
fi

SHORT_SHA="${SHORT_SHA:-unknown}"; SHORT_SHA="${SHORT_SHA:0:7}"

# The workflow's configured model wins, because it is what a reader recognises;
# the execution file's is the fallback for a run whose model came from a
# default the workflow does not name.
MODEL_LABEL="${CLAUDE_MODEL:-${MODEL_FROM_USAGE:-unknown}}"
if [ -n "${CLAUDE_EFFORT:-}" ]; then
  MODEL_LABEL="${MODEL_LABEL} (${CLAUDE_EFFORT})"
fi
# The row is pipe-delimited; a pipe in a model name would split the fields.
MODEL_LABEL=${MODEL_LABEL//|/-}

MARKER="<!-- claude-usage-ledger -->"

# --paginate: the default page is 30 comments, after which the ledger would
# silently vanish and every run would start a fresh one.
#
# A failed listing must NOT look like "no ledger yet": that would post a fresh
# comment whose total omits every prior run. Fail instead, leaving the existing
# ledger untouched.
if ! LISTING=$(gh api --paginate "repos/$GITHUB_REPOSITORY/issues/$PR_NUMBER/comments?per_page=100" 2>/dev/null); then
  echo "::error::Could not list PR comments; leaving the ledger untouched rather than starting a new one"
  exit 1
fi

# `jq -s ... add`: --paginate emits one array PER PAGE, so a bare `.[]` would
# select per page and hand back several ids once a PR passes 100 comments.
# Slurping and concatenating first makes `last` mean the last overall.
#
# startswith, not contains, and the bot's own identity: the review comments are
# posted by the same account and can quote this marker while reviewing this
# very file, which would otherwise let the ledger overwrite a review.
LEDGERS=$(printf '%s' "$LISTING" | jq -s -c --arg m "$MARKER" '
  (add // []) | [.[] | select((.user.login == "github-actions[bot]" or .user.login == "claude[bot]")
    and ((.body // "") | startswith($m)))]
  | sort_by([(.updated_at // .created_at // ""), (.id // 0)])')
# Keep Actions comments as history. Only the Claude app owns future writes.
LEDGER_ID=$(printf '%s' "$LEDGERS" | jq -r '
  [.[] | select(.user.login == "claude[bot]")] | last | .id // empty')

# Recover rows from every historical ledger, not just the last comment left by
# an overlapping run. Run id + attempt is the accounting identity: repeating a
# write replaces that attempt, while a GitHub re-run consumes tokens again.
# Legacy nine-field rows predate attempt tracking and are treated as attempt 1.
CURRENT_ROW=$(jq -nc --arg sha "$SHORT_SHA" --arg model "$MODEL_LABEL" \
  --arg i "$TOK_IN" --arg o "$TOK_OUT" --arg r "$TOK_READ" --arg w "$TOK_WRITE" \
  --arg t "$TURNS" --arg run "$GITHUB_RUN_ID" --arg attempt "${GITHUB_RUN_ATTEMPT:-1}" \
  '["0", $sha, $model, $i, $o, $r, $w, $t, $run, $attempt]')
ALL_ROWS=$(printf '%s' "$LEDGERS" | jq -r --argjson current "$CURRENT_ROW" '
  [ .[].body | split("\n")[] | capture("^<!-- row: (?<row>[^>]+) -->$")? | .row
    | split("|") | select(length >= 9) | .[9] = (.[9] // "1") ] + [$current]
  | reduce .[] as $row ([];
      if any(.[]; .[8:10] == $row[8:10])
      then map(if .[8:10] == $row[8:10] then $row else . end)
      else . + [$row] end)
  | to_entries[] | .value[0] = ((.key + 1) | tostring)
  | "<!-- row: " + (.value | join("|")) + " -->"')
REVIEW_NUMBER=$(printf '%s\n' "$ALL_ROWS" | wc -l | tr -d ' ')

# Render the table and total from the markers.
TABLE=$(printf '%s\n' "$ALL_ROWS" | awk '
  # Compact so the column stays readable: cache reads run to millions.
  function human(x) {
    if (x == "" || x == "unknown") return "?"
    if (x >= 1000000) return sprintf("%.1fM", x / 1000000)
    if (x >= 1000) return sprintf("%.1fk", x / 1000)
    return sprintf("%d", x)
  }
  BEGIN { unknown = 0; t_in = 0; t_out = 0; t_read = 0; t_write = 0; t_turns = 0 }
  {
    line = $0
    sub(/^<!-- row: /, "", line)
    sub(/ -->$/, "", line)
    n = split(line, f, "|")
    if (n < 9) next
    # A row is countable only when every one of its four token fields is a
    # number: a partial row added into the total would invent the fields it
    # could not read.
    if (f[4] == "unknown" || f[5] == "unknown" || f[6] == "unknown" || f[7] == "unknown") {
      unknown++
      toks = "?"
    } else {
      t_in += f[4]; t_out += f[5]; t_read += f[6]; t_write += f[7]
      toks = human(f[4]) " / " human(f[5]) " / " human(f[6] + f[7])
    }
    if (f[8] != "unknown") { t_turns += f[8]; turns = f[8] } else { turns = "?" }
    printf "| #%s | `%s` | %s | %s | %s |\n", f[1], f[2], f[3], toks, turns
  }
  END {
    printf "| **Total** |  |  | **%s / %s / %s** | **%d** |\n",
      human(t_in), human(t_out), human(t_read + t_write), t_turns
    if (unknown > 0) printf "UNKNOWN_ROWS=%d\n", unknown
  }
')

UNKNOWN_NOTE=""
if printf '%s' "$TABLE" | grep -q '^UNKNOWN_ROWS='; then
  N=$(printf '%s' "$TABLE" | sed -n 's/^UNKNOWN_ROWS=//p')
  TABLE=$(printf '%s' "$TABLE" | grep -v '^UNKNOWN_ROWS=')
  UNKNOWN_NOTE=$(printf '\n> ⚠️ %s run(s) could not report their usage, so the total is a **lower bound**. A reported `0` is a measured zero; `?` means the count could not be read.\n' "$N")
fi

BODY=$(printf '%s\n%s\n### 🧾 Claude review usage\n\n| Run | Commit | Model | Tokens (in / out / cached) | Turns |\n| --- | --- | --- | --- | --- |\n%s\n%s\n<sub>Read from the execution file per run. Runs include failed attempts and all review, verification, and summary passes. Tokens are recorded usage, not a measurement of the subscription limit; older reviews without a ledger are not included.</sub>\n' \
  "$MARKER" "$ALL_ROWS" "$TABLE" "$UNKNOWN_NOTE")

# No `|| true`: a swallowed write loses the only record of this run's usage
# while still printing success. If the write fails, the step fails.
if [ -n "$LEDGER_ID" ]; then
  if ! gh api "repos/$GITHUB_REPOSITORY/issues/comments/$LEDGER_ID" \
       -X PATCH -f body="$BODY" > /dev/null; then
    echo "::error::Failed to update the usage ledger; review #${REVIEW_NUMBER} at ${TOK_IN} input tokens is unrecorded"
    exit 1
  fi
else
  if ! gh api "repos/$GITHUB_REPOSITORY/issues/$PR_NUMBER/comments" \
       -X POST -f body="$BODY" > /dev/null; then
    echo "::error::Failed to post the usage ledger; review #${REVIEW_NUMBER} at ${TOK_IN} input tokens is unrecorded"
    exit 1
  fi
fi

echo "Recorded review #${REVIEW_NUMBER}: in ${TOK_IN}, out ${TOK_OUT}, cache ${TOK_READ}/${TOK_WRITE}, turns ${TURNS}"

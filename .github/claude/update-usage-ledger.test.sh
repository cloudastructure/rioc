#!/usr/bin/env bash
# Tests for update-usage-ledger.sh. Stubs gh, captures the body the script
# would post, and asserts on it.
#
# Runnable alone:
#   bash .github/claude/update-usage-ledger.test.sh
#
# The case that matters is 3: an unreadable count and a real zero must never be
# conflated, because treating unknown as zero under-reports the seat's usage
# window, which is the only number this ledger exists to report. Cases 6 to 8
# cover the failure modes that lose a run silently: a failed listing must not
# restart the ledger, a decoy comment must not be mistaken for it, and a failed
# write must not report success.
#
# Ported from the cost ledger's suite. Of its 14 cases, 10 survive here (1-3,
# 5-10, 14 in the old numbering, renumbered 1-10 below); 4 dropped as
# cost-specific — the real-$0.0000 case, the no-matching-session case, the
# START_MS fallback, and the rows-predating-tokens case, none of which has a
# token-ledger equivalent since usage is read from a file rather than a
# database and every row this ledger writes has all nine fields. Cases 4 and 5
# are new: they cover the summing of the review and verify passes, which the
# cost ledger had no counterpart for.
#
# Negative controls, each observed failing before the fix:
#   case 2  fails if PRIOR_ROWS is dropped (running total restarts)
#   case 3  fails if the awk branch treats unknown as 0
#   case 6  fails if a listing error is treated as "no ledger yet"
#   case 7  fails if selection uses `contains` instead of `startswith`
#   case 8  fails if the write keeps `|| true`
#   case 5  fails if a pass with a MIX of numbers and unknowns is summed anyway
#   case 5b fails if an all-unknown pass poisons a row the other pass could read
#   case 10 fails if SHORT_SHA is sliced before falling back to unknown
set -uo pipefail

# jq is a hard requirement of both this harness and the script under test.
# Skipping quietly would report a pass for tests that never ran.
if ! command -v jq > /dev/null 2>&1; then
  echo "jq is required for this test (macOS: brew install jq)" >&2
  exit 1
fi

HERE=$(cd "$(dirname "$0")" && pwd)
SCRIPT="${SCRIPT:-$HERE/update-usage-ledger.sh}"
STUB=$(mktemp -d)
OUT="$STUB/body.txt"

# gh stub: serves $STUB/existing.json for the listing, records the written
# body. STUB_LIST_FAIL / STUB_WRITE_FAIL inject API failures.
cat > "$STUB/gh" <<'EOF'
#!/usr/bin/env bash
if [ "${1:-}" = "api" ]; then
  case "$*" in
    *--paginate*)
      if [ -n "${STUB_LIST_FAIL:-}" ]; then echo "api error" >&2; exit 1; fi
      cat "$STUB_DIR/existing.json"; exit 0 ;;
  esac
  if [ -n "${STUB_WRITE_FAIL:-}" ]; then echo "api error" >&2; exit 1; fi
  prev=""
  for a in "$@"; do
    case "$prev" in
      -f) case "$a" in body=*) printf '%s' "${a#body=}" > "$STUB_DIR/body.txt" ;; esac ;;
    esac
    prev="$a"
  done
  echo '{"id":999}'
  exit 0
fi
EOF
chmod +x "$STUB/gh"

export STUB_DIR="$STUB"
export PATH="$STUB:$PATH"
export GITHUB_REPOSITORY="acme/repo" GITHUB_RUN_ID="42" GH_TOKEN="x"
export PR_NUMBER="188"
export CLAUDE_MODEL="claude-opus-5" CLAUDE_EFFORT="high"
export REVIEW_OUT="$STUB/out"
mkdir -p "$REVIEW_OUT"

pass=0; fail=0
check() { # name expected-substring
  if grep -qF "$2" "$OUT"; then echo "  PASS: $1"; pass=$((pass+1));
  else echo "  FAIL: $1"; echo "        want: $2"; fail=$((fail+1)); fi
}
refute() { # name unwanted-substring
  if grep -qF "$2" "$OUT"; then echo "  FAIL: $1"; fail=$((fail+1));
  else echo "  PASS: $1"; pass=$((pass+1)); fi
}
assert_status() { # name expected-status actual-status
  if [ "$2" = "$3" ]; then echo "  PASS: $1"; pass=$((pass+1));
  else echo "  FAIL: $1 (want exit $2, got $3)"; fail=$((fail+1)); fi
}

usage() { # usage <label> <in> <out> <cache_read> <cache_write> <turns> [model]
  jq -n --argjson i "$2" --argjson o "$3" --argjson r "$4" --argjson w "$5" \
        --argjson t "$6" --arg m "${7:-claude-opus-5}" \
    '{input:$i, output:$o, cache_read:$r, cache_write:$w, turns:$t, model:$m}' \
    > "$REVIEW_OUT/usage-$1.json"
}
usage_unknown() { # usage_unknown <label>
  printf '{"input":"unknown","output":"unknown","cache_read":"unknown","cache_write":"unknown","turns":"unknown","model":"unknown"}\n' \
    > "$REVIEW_OUT/usage-$1.json"
}
# A pass that ran and was only partly readable: some fields are numbers.
usage_partial() { # usage_partial <label>
  printf '{"input":500,"output":"unknown","cache_read":5000,"cache_write":0,"turns":2,"model":"claude-opus-5"}\n' \
    > "$REVIEW_OUT/usage-$1.json"
}
clear_usage() { rm -f "$REVIEW_OUT"/usage-*.json; }

run() { rm -f "$OUT"; bash "$SCRIPT" >/dev/null 2>&1; }
# Wrap the current body as a ledger comment the listing will return.
as_ledger() { printf '[{"id":7,"user":{"login":"github-actions[bot]"},"body":%s}]\n' "$(jq -Rs . < "$OUT")" > "$STUB/existing.json"; }

echo "== 1. First review, real usage =="
echo '[]' > "$STUB/existing.json"
clear_usage; usage review 1000 200 40000 10000 7
SHORT_SHA="8c66408abcdef" run
check "row rendered" '| #1 | `8c66408` | claude-opus-5 (high) | 1.0k / 200 / 50.0k | 7 |'
check "total matches" '| **Total** |  |  | **1.0k / 200 / 50.0k** | **7** |'
check "footer present" '<sub>Read from the execution file'
refute "no dollar column" '$'
refute "no spurious lower-bound note" 'lower bound'

echo "== 2. Second review accumulates =="
as_ledger
clear_usage; usage review 1000 200 40000 10000 7
GITHUB_RUN_ID=43 SHORT_SHA="deadbeef1234" run
check "prior row preserved" '| #1 | `8c66408` |'
check "new row numbered 2" '| #2 | `deadbee` |'
check "total is the sum" '| **Total** |  |  | **2.0k / 400 / 100.0k** | **14** |'

echo "== 3. Unknown usage is not zero =="
echo '[]' > "$STUB/existing.json"
clear_usage; usage_unknown review
SHORT_SHA="aaaaaaa1111" run
check "shown as unreadable" '| ? |'
check "total is a lower bound" 'lower bound'
check "and the total excludes it" '**0 / 0 / 0**'

echo "== 4. The verify pass is summed into the same row =="
# It runs on the same seat, so its tokens come out of the same window.
echo '[]' > "$STUB/existing.json"
clear_usage
usage review 1000 200 40000 10000 7
usage verify 500 100 5000 0 2
SHORT_SHA="bbbbbbb2222" run
check "one row, both passes" '| #1 | `bbbbbbb` | claude-opus-5 (high) | 1.5k / 300 / 55.0k | 9 |'

echo "== 5. A pass that ran and was only partly read makes the run unreadable =="
# A partial sum presented as a total is the same under-report as calling
# unknown zero — it just hides in a plausible number.
echo '[]' > "$STUB/existing.json"
clear_usage
usage review 1000 200 40000 10000 7
usage_partial verify
SHORT_SHA="ccccccc3333" run
check "the row reports nothing rather than half" '| #1 | `ccccccc` | claude-opus-5 (high) | ? |'
check "counted as a lower bound" 'lower bound'

echo "== 5b. An attempted pass with unreadable usage keeps totals incomplete =="
# The verifier does not run on a plain issue, or when the review before it
# failed, and extract-usage.sh still writes its file with every field unknown.
# Treating that as a pass that could not be read would mark every such row
# unknown and make the lower-bound note permanent.
echo '[]' > "$STUB/existing.json"
clear_usage
usage review 1000 200 40000 10000 7
usage_unknown verify
SHORT_SHA="ccccccc4444" run
check "unreadable verifier is not silently omitted" \
  '| #1 | `ccccccc` | claude-opus-5 (high) | ? |'
check "the incomplete total is labeled" 'lower bound'

echo "== 6. A failed listing must not restart the ledger =="
echo '[]' > "$STUB/existing.json"
clear_usage; usage review 1000 200 40000 10000 7
rm -f "$OUT"
SHORT_SHA="ddddddd4444" STUB_LIST_FAIL=1 bash "$SCRIPT" >/dev/null 2>&1
assert_status "script fails loudly" 1 "$?"
[ ! -f "$OUT" ] && { echo "  PASS: nothing was written"; pass=$((pass+1)); } || { echo "  FAIL: wrote a ledger despite a failed listing"; fail=$((fail+1)); }

echo "== 7. A decoy comment quoting the marker is ignored =="
# A review comment that quotes the marker mid-body, posted by the same bot,
# and newer than the real ledger. `contains` would select this one.
printf '[{"id":7,"user":{"login":"github-actions[bot]"},"body":"<!-- claude-usage-ledger -->\\n<!-- row: 1|1111111|m|9000|1000|10000|0|4|1 -->\\nreal ledger"},{"id":8,"user":{"login":"github-actions[bot]"},"body":"## Findings\\nthis review quotes <!-- claude-usage-ledger --> while reviewing it"}]\n' > "$STUB/existing.json"
clear_usage; usage review 1000 200 40000 10000 7
SHORT_SHA="eeeeeee5555" run
check "real ledger's row was carried forward" '| #1 | `1111111` |'
check "new row appended as #2" '| #2 | `eeeeeee` |'
check "total includes the prior row" '**10.0k / 1.2k / 60.0k** | **11** |'

echo "== 8. A multi-page listing is treated as one list =="
# gh api --paginate emits one array PER PAGE. With a matching comment on more
# than one page, a per-page `last` yields several objects, so LEDGER_ID becomes
# two ids and the PATCH URL is malformed. Two ledger comments is not
# hypothetical: an interrupted run can leave a duplicate behind.
printf '[{"id":7,"user":{"login":"github-actions[bot]"},"body":"<!-- claude-usage-ledger -->\\n<!-- row: 1|1111111|m|7000|700|0|0|3|1 -->\\nolder ledger"}]\n[{"id":9,"user":{"login":"github-actions[bot]"},"body":"<!-- claude-usage-ledger -->\\n<!-- row: 1|2222222|m|2000|300|0|0|2|2 -->\\nnewer ledger"}]\n' > "$STUB/existing.json"
clear_usage; usage review 1000 200 40000 10000 7
SHORT_SHA="9999999aaaa" run
check "the first historical row survives" '| #1 | `1111111` |'
check "the second historical row survives" '| #2 | `2222222` |'
check "total combines both historical ledgers" '**10.0k / 1.2k / 50.0k** | **12** |'

echo "== 9. Huge numbers stay readable, and a failed write fails the step =="
echo '[]' > "$STUB/existing.json"
clear_usage; usage review 47206 8629 4268352 0 11
SHORT_SHA="1234567aaaa" run
check "millions render as M" '47.2k / 8.6k / 4.3M'

echo '[]' > "$STUB/existing.json"
clear_usage; usage review 1000 200 40000 10000 7
SHORT_SHA="fffffff6666" STUB_WRITE_FAIL=1 bash "$SCRIPT" >/dev/null 2>&1
assert_status "script fails loudly" 1 "$?"

echo "== 10. SHORT_SHA unset (a plain issue) does not abort the ledger step =="
echo '[]' > "$STUB/existing.json"
clear_usage; usage review 1000 200 40000 10000 7
rm -f "$OUT"
unset SHORT_SHA
bash "$SCRIPT" >/dev/null 2>&1
assert_status "script exits 0" 0 "$?"
check "sha recorded as unknown" '| #1 | `unknown` |'

echo "== 11. With no usage file at all the run is unknown, not zero =="
echo '[]' > "$STUB/existing.json"
clear_usage
SHORT_SHA="7777777cccc" run
check "shown as unreadable" '| ? |'
check "counted as a lower bound" 'lower bound'

echo "== 12. An attempt is counted once, but re-running the workflow counts new usage =="
echo '[]' > "$STUB/existing.json"
clear_usage; usage review 1000 200 40000 10000 7
SHORT_SHA="aaaaaaa" run
as_ledger
SHORT_SHA="aaaaaaa" run
check "same attempt does not double the total" '**1.0k / 200 / 50.0k** | **7** |'
refute "same attempt does not append another run" '| #2 |'
as_ledger
GITHUB_RUN_ATTEMPT=2 SHORT_SHA="aaaaaaa" run
check "a real workflow rerun adds its consumption" '**2.0k / 400 / 100.0k** | **14** |'
check "run attempt is retained in the row identity" '|42|2 -->'

rm -rf "$STUB"
echo
echo "passed: $pass  failed: $fail"
[ "$fail" -eq 0 ]

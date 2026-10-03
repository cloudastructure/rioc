#!/usr/bin/env bash
# Tests for react-to-comment.sh.
#
# Runnable alone:
#   bash .github/claude/react-to-comment.test.sh
#
# The Claude action already posts the single eyes acknowledgement for an
# @claude mention. The workflow's completion reaction is still custom, so this
# test locks down the split: `seen` must not add another eyes reaction.
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
SCRIPT="${SCRIPT:-$HERE/react-to-comment.sh}"
STUB=$(mktemp -d)
trap 'rm -rf "$STUB"' EXIT

cat > "$STUB/gh" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$STUB_DIR/calls.log"
EOF
chmod +x "$STUB/gh"

export STUB_DIR="$STUB" PATH="$STUB:$PATH"
export GITHUB_REPOSITORY="acme/repo" COMMENT_ID="42" COMMENT_SCOPE="issues"

pass=0
fail=0
check() {
  if [ "$2" = "$3" ]; then
    echo "  PASS: $1"
    pass=$((pass + 1))
  else
    echo "  FAIL: $1 (expected '$3', got '$2')"
    fail=$((fail + 1))
  fi
}

run() {
  : > "$STUB/calls.log"
  bash "$SCRIPT" "$1" > /dev/null 2>&1
}

echo "== 1. The seen compatibility path does not post a second eyes reaction =="
run seen
check "no reaction is posted at start" "$(wc -l < "$STUB/calls.log" | tr -d ' ')" "0"

echo "== 2. A successful run posts one completion reaction =="
run success
check "success reaction is posted" "$(cat "$STUB/calls.log")" \
  "api repos/acme/repo/issues/comments/42/reactions -X POST -f content=+1"

echo "== 3. A failed run posts one non-success reaction =="
run failure
check "failure reaction is posted" "$(cat "$STUB/calls.log")" \
  "api repos/acme/repo/issues/comments/42/reactions -X POST -f content=confused"

echo "== 4. Missing comment id remains a no-op =="
: > "$STUB/calls.log"
COMMENT_ID='' bash "$SCRIPT" success > /dev/null 2>&1
check "missing ids are ignored" "$(wc -l < "$STUB/calls.log" | tr -d ' ')" "0"

echo
echo "passed: $pass  failed: $fail"
[ "$fail" -eq 0 ]

